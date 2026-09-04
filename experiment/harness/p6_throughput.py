"""P6 throughput: what the pow2 intervention costs in prefill and decode.

Pre-registration A-10 pins that prefill and decode are measured separately, that the
batch and ISL grid is fixed before running, that the repeat count is fixed, that median
and IQR are reported rather than a best-of, and that the nine identity-contract flags
travel with the numbers.

Prefill and decode are separated by measurement, not by assumption: a prefill-only run
(max_tokens=1) is timed, then a run generating OSL tokens is timed, and decode
throughput comes from the difference. Reporting a single tokens-per-second would hide
which phase the intervention touches, and the epilogue runs in both.

Median and IQR rather than mean and standard deviation because a shared machine
produces occasional slow iterations, and a mean lets one of them decide the answer.
The full per-repeat series is recorded so a reader can see the spread we summarised.

Usage:
  python3 p6_throughput.py --model models/qwen3-1.7b-int8-w8a8 --out p6_tp_base.json
  python3 p6_throughput.py --compare p6_tp_base.json p6_tp_pow2.json --out p6_tp_cost.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import time

# Pinned grid. Chosen to bracket the study's layer measurements (K up to 32768 came
# from ISL 2048 at 1.7B) without exceeding a 24 GB card at batch 16.
BATCH_ISL_GRID = [(1, 128), (1, 2048), (4, 128), (4, 2048), (16, 128), (16, 512)]
OSL = 64          # tokens generated when measuring decode
REPEATS = 7       # odd, so the median is an observed value rather than an average
WARMUP = 2


def _digest(model_dir: str) -> str:
    h = hashlib.sha256()
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(shard, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _iqr(xs):
    if len(xs) < 4:
        return None
    q = statistics.quantiles(sorted(xs), n=4, method="inclusive")
    return {"q1": q[0], "q3": q[2], "iqr": q[2] - q[0]}


def _summarise(series, label):
    return {"label": label, "n": len(series), "median": statistics.median(series),
            "min": min(series), "max": max(series), "spread": _iqr(series),
            "series": series}


def measure(model_dir: str, grid=None, osl: int = OSL, repeats: int = REPEATS,
            warmup: int = WARMUP, max_model_len: int = 4096, gpu_frac: float = 0.85):
    from vllm import LLM, SamplingParams
    import torch

    grid = grid or BATCH_ISL_GRID
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_frac, enforce_eager=True,
              disable_log_stats=True)
    tok = llm.get_tokenizer()

    # A fixed synthetic prompt, so ISL is exact rather than approximately right and the
    # two arms see byte-identical input. Content is irrelevant to timing.
    def prompt_of(isl):
        ids = [tok.eos_token_id or 0] * 0
        base = tok("the ", add_special_tokens=False)["input_ids"]
        while len(ids) < isl:
            ids.extend(base)
        return tok.decode(ids[:isl])

    results = []
    for batch, isl in grid:
        prompts = [prompt_of(isl)] * batch
        sp_prefill = SamplingParams(max_tokens=1, temperature=0.0)
        sp_full = SamplingParams(max_tokens=osl, temperature=0.0, ignore_eos=True)

        for _ in range(warmup):
            llm.generate(prompts, sp_prefill)
        torch.cuda.synchronize()

        t_prefill, t_full = [], []
        for _ in range(repeats):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            llm.generate(prompts, sp_prefill)
            torch.cuda.synchronize(); t_prefill.append(time.perf_counter() - t0)
        for _ in range(repeats):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            llm.generate(prompts, sp_full)
            torch.cuda.synchronize(); t_full.append(time.perf_counter() - t0)

        pre_tps = [batch * isl / t for t in t_prefill]
        # decode time is what the longer run spent beyond prefill; using the paired
        # medians rather than per-repeat differences, since the two loops are not paired
        dec_secs = [f - statistics.median(t_prefill) for f in t_full]
        dec_tps = [batch * (osl - 1) / d for d in dec_secs if d > 0]

        results.append({
            "batch": batch, "isl": isl, "osl": osl,
            "prefill_seconds": _summarise(t_prefill, "prefill wall seconds"),
            "prefill_tokens_per_s": _summarise(pre_tps, "prefill tok/s"),
            "full_seconds": _summarise(t_full, "prefill+decode wall seconds"),
            "decode_tokens_per_s": (_summarise(dec_tps, "decode tok/s") if dec_tps else None),
            "decode_note": ("decode seconds are full-run wall time minus the median "
                            "prefill time for the same batch and ISL; the two loops are "
                            "separate, so this is a paired-median subtraction and not a "
                            "per-repeat difference"),
        })
    return results


def identity_flags(model_dir: str) -> dict:
    """The nine flags the study's identity contract records, as far as this script sees.

    Anything this script cannot observe is None rather than absent, so a downstream
    join fails closed instead of treating a missing flag as satisfied.
    """
    import vllm
    return {
        "checkpoint_digest": _digest(model_dir),
        "quant_config_sha256": _sha_of(os.path.join(model_dir, "config.json")),
        "tokenizer_sha256": _sha_of(os.path.join(model_dir, "tokenizer.json")),
        "parent_revision": _parent_revision(model_dir),
        "runtime": f"vllm {vllm.__version__}",
        "image_digest": os.environ.get("IMAGE_DIGEST"),
        "kernel_class": None,          # not captured here; see tf_v6 for the log capture
        "prompt_manifest": "synthetic fixed-ISL prompt, see prompt_of()",
        "treatment": "pow2 probe vs base INT8 checkpoint",
    }


def _sha_of(path: str):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _parent_revision(model_dir: str):
    for cand in ("parent_manifest.json", "parent8b_manifest.json"):
        p = os.path.join(os.path.dirname(model_dir.rstrip("/")), cand)
        if os.path.exists(p):
            try:
                return json.load(open(p)).get("revision")
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "POW2_JSON"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.compare:
        base, pow2 = (json.load(open(p)) for p in a.compare)
        if [(r["batch"], r["isl"]) for r in base["grid"]] != \
           [(r["batch"], r["isl"]) for r in pow2["grid"]]:
            raise RuntimeError("arms measured different grids; comparison refused")
        rows = []
        for b, p in zip(base["grid"], pow2["grid"]):
            def rel(kind):
                bm = b[kind]["median"] if b[kind] else None
                pm = p[kind]["median"] if p[kind] else None
                if bm in (None, 0) or pm is None:
                    return None
                return (pm - bm) / bm
            rows.append({"batch": b["batch"], "isl": b["isl"],
                         "prefill_tps_base": b["prefill_tokens_per_s"]["median"],
                         "prefill_tps_pow2": p["prefill_tokens_per_s"]["median"],
                         "prefill_relative_change": rel("prefill_tokens_per_s"),
                         "decode_tps_base": (b["decode_tokens_per_s"] or {}).get("median"),
                         "decode_tps_pow2": (p["decode_tokens_per_s"] or {}).get("median"),
                         "decode_relative_change": rel("decode_tokens_per_s")})
        out = {"schema": "p6-throughput-compare-v1",
               "commitment": ("A-10 requires this reported whatever it says. A negative "
                              "relative change means the pow2 arm is slower."),
               "rows": rows,
               "identity": {"base": base["identity"], "pow2": pow2["identity"]},
               "caveat": ("Measured on a shared machine. Median and IQR are reported and "
                          "the full per-repeat series is in the per-arm files; a reader "
                          "who suspects interference can look at the spread.")}
        json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
        for r in rows:
            print(f"b={r['batch']:2d} isl={r['isl']:5d} prefill {r['prefill_relative_change']}"
                  f" decode {r['decode_relative_change']}")
        return

    if not a.model:
        ap.error("--model is required unless --compare is given")
    out = {
        "schema": "p6-throughput-v1",
        "model": a.model,
        "grid_spec": {"batch_isl": BATCH_ISL_GRID, "osl": OSL, "repeats": REPEATS,
                      "warmup": WARMUP},
        "grid": measure(a.model),
        "identity": identity_flags(a.model),
        "env": {k: os.environ.get(k) for k in
                ("VLLM_DISABLED_KERNELS", "CUDA_VISIBLE_DEVICES",
                 "VLLM_ENABLE_V1_MULTIPROCESSING", "IMAGE_DIGEST")},
    }
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"{a.model}: {len(out['grid'])} grid points measured")


if __name__ == "__main__":
    main()
