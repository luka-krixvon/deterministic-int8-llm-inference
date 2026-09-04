"""W3 flagship: per-layer verification of the P1 prediction list (196 layers).

Two stages (one file, --stage picks):

capture  (runs in the project venv, HF side)
  Load the INT8-W8A8 checkpoint via transformers, hook every int8 linear
  module, run the pinned prompts, and for each layer save ONE batch of real
  inputs: the dynamically quantized activations (int8 q, fp32 per-token
  scale) exactly as the serving stack defines them (absmax/127, symmetric,
  round-half-even), capped at CAP_TOKENS tokens per layer.

verdict  (runs inside the pinned vLLM container)
  For every captured layer: load the layer's stored int8 weight W (N,K) and
  weight_scale (N,) straight from the checkpoint safetensors, feed the SAME
  captured int8 activations to cutlass_scaled_mm and triton_scaled_mm, and
  compare bitwise under two scale regimes:
    pow2  — every scale overridden to a power of two (P1a probe): for layers
            on the P1 BITWISE-SAFE list, identity is REQUIRED; any diff is a
            kernel bug (headline falsification test, 196/196 predicted safe).
    real  — the checkpoint's true scales (P1b): diffs allowed only at the
            ulp level; per-layer max relative diff and differing-element
            fraction are recorded against the pre-registered bound.

Output: w3_perlayer_verdict.json with per-layer rows and the aggregate
P1a/P1b verdict versus the locked prediction list.
"""
from __future__ import annotations
import argparse, json, os

CAP_TOKENS = 512


def stage_capture(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    prompts = json.load(open(args.prompts))[:16]

    int8_layers = [n for n, m in model.named_modules()
                   if getattr(m, "weight", None) is not None
                   and m.weight.dtype == torch.int8]
    print(f"int8 layers: {len(int8_layers)}")
    os.makedirs(args.capture_dir, exist_ok=True)
    got = {n: 0 for n in int8_layers}

    def make_hook(name):
        def hook(mod, inputs, output):
            if got[name] >= CAP_TOKENS:
                return
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).to(torch.float32)
            take = min(CAP_TOKENS - got[name], x.shape[0])
            x = x[:take]
            absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
            scale = absmax / 127.0
            q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
            f = os.path.join(args.capture_dir, name.replace("/", "_") + ".pt")
            if got[name] == 0:
                torch.save({"q": q.cpu(), "scale": scale.cpu()}, f)
            else:
                prev = torch.load(f)
                torch.save({"q": torch.cat([prev["q"], q.cpu()]),
                            "scale": torch.cat([prev["scale"], scale.cpu()])}, f)
            got[name] += take
        return hook

    hooks = [m.register_forward_hook(make_hook(n))
             for n, m in model.named_modules() if n in int8_layers]
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True,
                      max_length=512).to("cuda")
            model(**ids)
    for h in hooks:
        h.remove()
    json.dump(int8_layers, open(os.path.join(args.capture_dir, "_layers.json"), "w"))
    print(f"captured {len(int8_layers)} layers x <= {CAP_TOKENS} tokens")
    print("CAPTURE_OK")


def stage_verdict(args):
    import torch, glob, importlib
    from safetensors.torch import load_file
    from vllm import _custom_ops as ops
    triton_mm = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors"
        ".triton_scaled_mm").triton_scaled_mm

    # merge all safetensors shards' int8 weights + scales
    tensors = {}
    for shard in glob.glob(os.path.join(args.checkpoint, "*.safetensors")):
        tensors.update(load_file(shard))
    layers = json.load(open(os.path.join(args.capture_dir, "_layers.json")))
    pred = {l["layer"]: l["prediction"]
            for l in json.load(open(args.p1))["layers"]}

    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metrics_v2 import divergence_report

    rows, p1a_viol, p1b_over = [], 0, 0
    ULP_DIST_BOUND = 1                   # bf16 bit-pattern distance bound (audit metric)
    for name in layers:
        wkey = name + ".weight"
        skey = name + ".weight_scale"
        if wkey not in tensors:
            rows.append({"layer": name, "status": "weight-missing"}); continue
        W = tensors[wkey].cuda()                       # (N, K) int8
        ws = tensors[skey].float().cuda().reshape(-1)  # (N,)
        cap = torch.load(os.path.join(
            args.capture_dir, name.replace("/", "_") + ".pt"))
        A = cap["q"].cuda()                            # (M, K) int8
        sa = cap["scale"].float().cuda()               # (M, 1)
        B = W.t()                                      # column-major (K, N)

        row = {"layer": name, "M": int(A.shape[0]), "K": int(A.shape[1]),
               "N": int(W.shape[0]), "p1": pred.get(name, "?")}
        for kind in ("pow2", "real"):
            if kind == "pow2":
                sa_k = torch.full_like(sa, 2.0 ** -9)
                ws_k = torch.full_like(ws, 2.0 ** -8)
            else:
                sa_k, ws_k = sa, ws
            oc = ops.cutlass_scaled_mm(A, B, scale_a=sa_k,
                                       scale_b=ws_k.view(1, -1),
                                       out_dtype=torch.bfloat16)
            ot = triton_mm(A, B, scale_a=sa_k, scale_b=ws_k.view(-1, 1),
                           out_dtype=torch.bfloat16)
            rep = divergence_report(oc, ot)
            rep["identical"] = rep["bitwise_identical"]
            row[kind] = rep
        if not row["pow2"]["bitwise_identical"] and row["p1"] == "BITWISE-SAFE":
            p1a_viol += 1
        if row["real"]["max_ulp_distance_finite"] > ULP_DIST_BOUND:
            p1b_over += 1
        rows.append(row)

    n_pow2_ident = sum(1 for r in rows if r.get("pow2", {}).get("bitwise_identical"))
    n_real_ident = sum(1 for r in rows if r.get("real", {}).get("bitwise_identical"))
    out = {
        "n_layers": len(rows),
        "p1a_pow2_identical": n_pow2_ident,
        "p1a_violations": p1a_viol,
        "p1b_real_identical": n_real_ident,
        "p1b_over_ulp_bound": p1b_over,
        "ulp_distance_bound": ULP_DIST_BOUND,
        "layers": rows,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"P1a: {n_pow2_ident}/{len(rows)} identical under pow2 "
          f"({p1a_viol} violations of BITWISE-SAFE)")
    print(f"P1b: {n_real_ident}/{len(rows)} identical under real scales, "
          f"{p1b_over} layers exceed the 2-ulp bound")
    print("VERDICT_OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["capture", "verdict"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--capture-dir", default="perlayer_capture")
    ap.add_argument("--prompts")
    ap.add_argument("--p1")
    ap.add_argument("--out", default="w3_perlayer_verdict.json")
    a = ap.parse_args()
    (stage_capture if a.stage == "capture" else stage_verdict)(a)
