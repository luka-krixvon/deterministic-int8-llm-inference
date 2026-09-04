"""integer-alibi P1: per-layer 2^24 representability predictions.

For an INT8-W8A8 checkpoint, every linear layer's GEMM accumulates
int8 x int8 products in INT32 — exact and order-independent. The FP32
epilogue can only round if some accumulator magnitude reaches 2^24
(above which consecutive integers are no longer representable in FP32).

This script runs the pinned calibration prompts through an EXACT emulation
(float64 holds integers < 2^53 exactly, and per-dot-product bounds keep us
far below that) and records, per linear layer, the maximum |accumulator|
observed plus a worst-case bound. Output is the pre-registered prediction
list: layers whose observed-max and bound sit below 2^24 MUST be bitwise
identical across every backend; only layers above the line are even eligible
to diverge.

Emulation detail: weights are the checkpoint's stored int8 tensors; dynamic
per-token activation quantization is re-implemented here exactly as vLLM's
dynamic path defines it (per-token absmax -> scale = absmax/127, symmetric
round-to-nearest-even to int8). The int x int matmul is computed in fp64.
Exactness bound: |sum| <= K * 127 * 127 = K * 16129; for K <= 2^38 this is
< 2^53, so fp64 emulation is exact for every realistic K.

Run inside the project venv on <host>:
  .venv/bin/python3 p1_predictions.py \
      --checkpoint models/qwen3-1.7b-int8-w8a8 \
      --prompts-file calib_prompts.json \
      --out p1_predictions_qwen3-1.7b.json

The output JSON (SHA-256-pinned in the pre-registration before any
measurement) is the falsifiable artifact.
"""
from __future__ import annotations
import argparse, hashlib, json, math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TWO24 = float(2 ** 24)


def dynamic_per_token_int8(x: torch.Tensor):
    """vLLM-style dynamic per-token symmetric int8 quantization.
    x: (tokens, K) float. Returns int8 tensor (as float64 holding integers)
    and per-token scales."""
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax / 127.0
    q = torch.round(x / scale).clamp(-127, 127)
    return q.to(torch.float64), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts-file", required=True,
                    help="JSON list of calibration prompt strings (pinned)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompts", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    # BF16 load: we only need each linear layer's *input* activations, which
    # the surrounding (non-GEMM) ops produce; the int8 GEMM itself is
    # emulated exactly from those inputs plus the stored int8 weights.
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    prompts = json.load(open(args.prompts_file))[: args.max_prompts]

    # collect stored int8 weights per linear module
    int8_w = {}
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is not None and w.dtype == torch.int8:
            int8_w[name] = w
    if not int8_w:
        raise SystemExit("no int8 weights found — is this a W8A8 checkpoint?")
    print(f"int8 linear layers: {len(int8_w)}")

    stats = {n: {"max_abs_acc": 0.0, "tokens_seen": 0} for n in int8_w}

    hooks = []
    def make_hook(name):
        # fp64 weight is materialized transiently per call: pre-materializing
        # all layers would need params*8 bytes (64 GB for an 8B model)
        def hook(mod, inputs, output):
            w64 = int8_w[name].to(torch.float64).t()  # (K, N), transient
            x = inputs[0]
            x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
            q, _ = dynamic_per_token_int8(x2)
            # exact int x int accumulation in fp64, done on GPU
            acc = q.to(w64.device) @ w64           # (tokens, N), exact
            del w64
            m = float(acc.abs().max().item())
            s = stats[name]
            if m > s["max_abs_acc"]:
                s["max_abs_acc"] = m
            s["tokens_seen"] += q.shape[0]
        return hook

    for name, mod in model.named_modules():
        if name in int8_w:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt", truncation=True,
                      max_length=args.max_len).to("cuda")
            model(**ids)
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{len(prompts)} prompts", flush=True)
    for h in hooks:
        h.remove()

    layers = []
    for name, w in int8_w.items():
        K = w.shape[1]
        worst = K * 127.0 * 127.0
        obs = stats[name]["max_abs_acc"]
        layers.append({
            "layer": name, "K": int(K),
            "worst_case_bound": worst,
            "observed_max_abs_acc": obs,
            "observed_headroom_bits": math.log2(TWO24 / obs) if obs > 0 else None,
            "prediction": ("BITWISE-SAFE" if obs < TWO24 else "ELIGIBLE-TO-DIVERGE"),
            "bound_also_safe": worst < TWO24,
            "tokens_seen": stats[name]["tokens_seen"],
        })
    safe = sum(1 for l in layers if l["prediction"] == "BITWISE-SAFE")
    result = {
        "checkpoint": args.checkpoint,
        "prompts_sha256_16": hashlib.sha256(
            "\n".join(prompts).encode()).hexdigest()[:16],
        "n_prompts": len(prompts),
        "threshold": 2 ** 24,
        "n_layers": len(layers),
        "n_bitwise_safe": safe,
        "n_eligible_to_diverge": len(layers) - safe,
        "caveat": ("BITWISE-SAFE is an observed-activation claim: an input "
                   "off the calibration distribution could exceed the "
                   "threshold. Layers with bound_also_safe=true are safe "
                   "unconditionally. The pre-registered prediction applies "
                   "to the pinned evaluation prompts."),
        "layers": layers,
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[p1] {safe}/{len(layers)} layers predicted BITWISE-SAFE "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
