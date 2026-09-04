"""Tests for P8's requantizer. The first one is the gate: under the minmax rule this must
reproduce the committed base checkpoint exactly, which is what licenses attributing any
pow2 arm's difference to the scale constraint rather than to the quantizer.

Runs against the real parent weights and the real base checkpoint, on one layer, so it is
fast; the whole-checkpoint verification is recorded in out/p8_report_minmax.json.
Set P8_PARENT and P8_BASE to relocate.
"""
import glob, os, sys, torch
sys.path.insert(0, '.')
from safetensors.torch import safe_open
from p8_requant import (base_scale, quantize, scale_for, is_pow2, DIVISOR, QMIN, QMAX,
                        SEARCH_OFFSETS, recon_error, RULES)

P = F = 0
def ck(n, c):
    global P, F
    if c: P += 1; print(f"PASS {n}")
    else: F += 1; print(f"FAIL {n}")

PARENT = os.environ.get("P8_PARENT",
  glob.glob("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*")[0])
BASE = os.environ.get("P8_BASE", "/home/ubuntu/integer_alibi/models/qwen3-1.7b-int8-w8a8")
def get(d, k):
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, framework="pt") as s:
            if k in s.keys(): return s.get_tensor(k)

# --- the divisor must come from the library's own formula, not a guess
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.quantization.quant_args import QuantizationArgs
qa = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="channel",
                      dynamic=False, observer="memoryless_minmax")
s_lib, _ = calculate_qparams(torch.tensor([[-1.0]]), torch.tensor([[1.0]]), qa)
ck("divisor agrees with calculate_qparams", abs(float(s_lib.flatten()[0]) - 1.0 / DIVISOR) < 1e-9)
ck("int8 range is [-128, 127]", (QMIN, QMAX) == (-128, 127))

for L in ("model.layers.0.mlp.down_proj", "model.layers.10.self_attn.o_proj"):
    w = get(PARENT, L + ".weight")
    bq = get(BASE, L + ".weight")
    bs = get(BASE, L + ".weight_scale")
    s = scale_for("minmax", w)
    q, n_clip = quantize(w, s)
    # --- THE GATE
    ck(f"[{L}] minmax scale matches base exactly",
       bool(torch.equal(s.to(torch.bfloat16), bs)))
    ck(f"[{L}] minmax int8 matches base exactly",
       bool(torch.equal(q.to(torch.int32), bq.to(torch.int32))))
    # --- bf16 division is load-bearing: fp32 must NOT reproduce, or the test is vacuous
    q_fp32 = torch.clamp(torch.round(w.to(torch.float32) / s), QMIN, QMAX).to(torch.int8)
    ck(f"[{L}] fp32 division does NOT reproduce base (bf16 division is essential)",
       not bool(torch.equal(q_fp32.to(torch.int32), bq.to(torch.int32))))
    # --- base itself clips, so pow2 predictions are about the amount not the presence
    ck(f"[{L}] base clips some elements (127.5 divisor sends round(127.5) to 128)", n_clip > 0)

w = get(PARENT, "model.layers.0.mlp.down_proj.weight")
s0 = scale_for("minmax", w)
_, clip0 = quantize(w, s0)

s_ceil = scale_for("pow2_ceil", w)
ck("pow2_ceil scale is never smaller than minmax", bool((s_ceil >= s0 - 0.0).all()))
ck("pow2_ceil scales are exact powers of two", is_pow2(s_ceil))
_, clip_ceil = quantize(w, s_ceil)
ck(f"pow2_ceil clips no more than base ({clip_ceil} vs {clip0})", clip_ceil <= clip0)

s_near = scale_for("pow2_nearest", w)
ck("pow2_nearest scales are exact powers of two", is_pow2(s_near))
ck("pow2_nearest is sometimes SMALLER than minmax, so the clipping hypothesis is testable",
   bool((s_near < s0).any()))
_, clip_near = quantize(w, s_near)
ck(f"pow2_nearest clips more than pow2_ceil ({clip_near} vs {clip_ceil})", clip_near > clip_ceil)

s_srch = scale_for("pow2_search", w)
ck("pow2_search scales are exact powers of two", is_pow2(s_srch))
e_srch = recon_error(w, s_srch)
worse = 0
for off in SEARCH_OFFSETS:
    s_alt = torch.exp2(torch.ceil(torch.log2(s0)) + off).to(torch.bfloat16).to(torch.float32)
    worse += int((recon_error(w, s_alt) < e_srch - 1e-6).sum())
ck("pow2_search picks the minimum-error offset per channel", worse == 0)
ck("pow2_search never exceeds ceil's error",
   bool((e_srch <= recon_error(w, s_ceil) + 1e-6).all()))

# --- ratios are recorded so each arm's own product bound can be derived
for rule in RULES:
    sr = scale_for(rule, w)
    q, _ = quantize(w, sr)
    lo, hi = int(q.min()), int(q.max())
    bound = max(abs(lo), hi) * 127
    ck(f"{rule}: int8 range [{lo},{hi}] gives product bound {bound}",
       bound in (16129, 16256) and lo >= QMIN and hi <= QMAX)

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
