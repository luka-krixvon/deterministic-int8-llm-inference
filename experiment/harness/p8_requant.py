"""P8: build INT8 checkpoints whose per-channel scales are constrained to powers of two.

The existing probe (make_probe_pow2.py) rewrites the stored weight_scale and leaves the
int8 weights alone, so the weights remain quantized for the old scale. That conflates
three effects -- coarser scales, a weight/scale mismatch, and clipping -- and P6 measured
their sum at +157.4% perplexity. This module requantizes from the parent's bf16 weights
under a chosen scale rule, so the arms separate those effects (pre-registration A-12 and
its addendum).

The convention it reproduces was reverse-engineered from the committed base checkpoint,
not assumed: scale = max(|w|, dim=1) / 127.5 cast to bfloat16, giving int8 weights in
[-128, 127]. See artifacts/quantization_convention_2026-08-15.json. The `minmax` rule
exists to be checked against that checkpoint byte for byte; if it does not reproduce
`bc6258648cc6...` then this pipeline is not equivalent to the one that produced the study's
data, and no pow2 arm built here would be comparable to base.

Only quantized Linear weights and their scales are recomputed. Every other tensor and every
config or tokenizer file is copied from the base checkpoint, so an output differs from base
in exactly the intended way.

Usage:
  python3 p8_requant.py --parent <hf snapshot dir> --base <base ckpt> \\
                        --rule minmax --out <dir> --report <json>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import safe_open, save_file

# Confirmed against compressed_tensors' calculate_qparams, which computes
# scales = max_val_pos / (bit_range / 2) with bit_range = 127 - (-128) = 255, so the
# divisor is 127.5 exactly. Reverse engineering had already found this; the source
# confirms it rather than the other way round.
DIVISOR = 127.5
QMIN, QMAX = -128, 127
SCALE_DTYPE = torch.bfloat16
RULES = ("minmax", "pow2_nearest", "pow2_ceil", "pow2_search")
SEARCH_OFFSETS = (-2, -1, 0, 1)      # relative to ceil; negative offsets may clip


def base_scale(w: torch.Tensor) -> torch.Tensor:
    """The unconstrained per-output-channel scale, as the study's pipeline computes it."""
    amax = w.to(torch.float32).abs().amax(dim=1, keepdim=True)
    return (amax / DIVISOR).to(SCALE_DTYPE).to(torch.float32)


def quantize(w: torch.Tensor, s: torch.Tensor):
    """Quantize to int8 with the recorded convention. Returns (q, n_clipped).

    The division happens in bfloat16, which is the detail that makes this reproduce the
    committed checkpoint exactly. Both the weight and the stored scale are bf16, and the
    pipeline divides them without widening, so the quotient is rounded to bf16 -- eight
    mantissa bits, a spacing of 0.25 near 61 -- before it is rounded to an integer. That
    is why a quotient of 61.398 becomes 62, and why the resulting deviation from ideal
    rounding reaches 0.749 and varies in sign within one output channel. Dividing in
    float32 instead leaves 6.42% of the int8 weights different; see
    artifacts/p8_requant_reproduction_failure.json for the diagnosis that led here.
    """
    r = torch.round(w.to(torch.bfloat16) / s.to(torch.bfloat16)).to(torch.float32)
    n_clipped = int(((r < QMIN) | (r > QMAX)).sum())
    return torch.clamp(r, QMIN, QMAX).to(torch.int8), n_clipped


def recon_error(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Per-channel Frobenius error of the round-trip, used by the search rule."""
    q, _ = quantize(w, s)
    d = w.to(torch.float32) - q.to(torch.float32) * s
    return d.pow(2).sum(dim=1)


def scale_for(rule: str, w: torch.Tensor) -> torch.Tensor:
    """Return the per-channel scale under the chosen rule, as float32 holding a bf16 value.

    Powers of two are exactly representable in bfloat16, so casting a pow2 result to the
    stored dtype is lossless and the arms differ only in which value was chosen.
    """
    s0 = base_scale(w)
    if rule == "minmax":
        return s0
    lg = torch.log2(s0)
    if rule == "pow2_nearest":
        return torch.exp2(torch.round(lg)).to(SCALE_DTYPE).to(torch.float32)
    if rule == "pow2_ceil":
        # Never smaller than s0, so |w/s| <= DIVISOR and clipping is impossible.
        return torch.exp2(torch.ceil(lg)).to(SCALE_DTYPE).to(torch.float32)
    if rule == "pow2_search":
        k_ceil = torch.ceil(lg)
        best_s, best_e = None, None
        for off in SEARCH_OFFSETS:
            s = torch.exp2(k_ceil + off).to(SCALE_DTYPE).to(torch.float32)
            e = recon_error(w, s)
            if best_s is None:
                best_s, best_e = s.clone(), e
            else:
                take = e < best_e
                best_s = torch.where(take.unsqueeze(1), s, best_s)
                best_e = torch.where(take, e, best_e)
        return best_s
    raise ValueError(f"unknown rule {rule!r}")


def is_pow2(s: torch.Tensor) -> bool:
    f = s.to(torch.float64).flatten()
    if not bool(torch.isfinite(f).all()) or bool((f <= 0).any()):
        return False
    return bool(torch.equal(torch.log2(f), torch.round(torch.log2(f))))


def build(parent_dir: str, base_dir: str, rule: str, out_dir: str) -> dict:
    parent = {}
    for f in sorted(glob.glob(os.path.join(parent_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                parent[k] = f

    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(base_dir, "*"))):
        if not f.endswith(".safetensors"):
            shutil.copy2(f, out_dir)

    stats = {"layers": 0, "clipped_total": 0, "clipped_layers": 0,
             "int8_min": 127, "int8_max": -128, "scale_ratio_min": None,
             "scale_ratio_max": None, "all_scales_pow2": True, "per_layer": {}}

    for shard in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
        out = {}
        with safe_open(shard, framework="pt") as sf:
            keys = list(sf.keys())
            for k in keys:
                out[k] = sf.get_tensor(k)
            for k in keys:
                if not k.endswith(".weight"):
                    continue
                sk = k[: -len(".weight")] + ".weight_scale"
                if sk not in keys:
                    continue                    # not a quantized Linear
                if k not in parent:
                    raise RuntimeError(f"{k} has a scale in base but no parent weight")
                with safe_open(parent[k], framework="pt") as pf:
                    w = pf.get_tensor(k)
                s = scale_for(rule, w)
                q, n_clip = quantize(w, s)
                out[k] = q
                out[sk] = s.to(SCALE_DTYPE)
                s0 = base_scale(w)
                ratio = float((s / s0).median())
                stats["layers"] += 1
                stats["clipped_total"] += n_clip
                stats["clipped_layers"] += 1 if n_clip else 0
                stats["int8_min"] = min(stats["int8_min"], int(q.min()))
                stats["int8_max"] = max(stats["int8_max"], int(q.max()))
                for key, val in (("scale_ratio_min", ratio), ("scale_ratio_max", ratio)):
                    cur = stats[key]
                    if cur is None:
                        stats[key] = val
                    else:
                        stats[key] = min(cur, val) if key.endswith("min") else max(cur, val)
                if rule != "minmax" and not is_pow2(s):
                    stats["all_scales_pow2"] = False
                stats["per_layer"][k[: -len(".weight")]] = {
                    "n_clipped": n_clip, "scale_ratio_to_minmax": ratio,
                    "int8_min": int(q.min()), "int8_max": int(q.max()),
                }
        save_file(out, os.path.join(out_dir, os.path.basename(shard)),
                  metadata={"format": "pt"})

    bound = max(abs(stats["int8_min"]), stats["int8_max"]) * 127
    stats["derived_product_bound"] = bound
    stats["derived_product_bound_note"] = (
        f"max|weight int8| = {max(abs(stats['int8_min']), stats['int8_max'])} times the "
        "activation bound of 127, which all 196 captured layers respect. Each arm gets its "
        "own bound: a ceil rule enlarges the scale, so the weights may no longer reach -128 "
        "and the bound can fall to 16129, unlike base's 16256.")
    stats["rule"] = rule
    stats["divisor"] = DIVISOR
    stats["scale_dtype"] = str(SCALE_DTYPE)
    stats["checkpoint_digest"] = digest(out_dir)
    return stats


def digest(model_dir: str) -> str:
    """Same scope as tf_v6.checkpoint_digest, so arms are identified the study's way."""
    h = hashlib.sha256()
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(shard, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rule", required=True, choices=RULES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    st = build(a.parent, a.base, a.rule, a.out)
    st["parent"] = a.parent
    st["base"] = a.base
    st["base_digest"] = digest(a.base)
    st["reproduces_base"] = st["checkpoint_digest"] == st["base_digest"]
    json.dump(st, open(a.report, "w"), indent=2, sort_keys=True)
    print(f"rule={a.rule} layers={st['layers']} digest={st['checkpoint_digest'][:16]} "
          f"clipped={st['clipped_total']} in {st['clipped_layers']} layers "
          f"int8=[{st['int8_min']},{st['int8_max']}] bound={st['derived_product_bound']} "
          f"pow2={st['all_scales_pow2']} reproduces_base={st['reproduces_base']}")


if __name__ == "__main__":
    main()
