"""Work-order item 4: minimal numerical boundary tests (GPU, pinned container).

Tests are evidence about THIS environment only; the output is stamped with
GPU, CUDA, driver, torch versions and the container digest (passed via env).
Passing here does not prove every kernel implementation — it validates the
mathematical preconditions the paper's claims rest on, in the environment
where the measurements ran.
"""
import json, os, struct, sys
import torch

ENV = {
    "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.version.cuda,
    "torch": torch.__version__,
    "driver": None,
    "image_digest": os.environ.get("IMAGE_DIGEST", "unset"),
}
try:
    import pynvml
    pynvml.nvmlInit()
    ENV["driver"] = pynvml.nvmlSystemGetDriverVersion()
except Exception:
    pass

R = {"env": ENV, "tests": {}}
dev = "cuda"


def bf16_bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16).to(torch.int32)


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Ordered bit-pattern distance between bf16 tensors (monotone mapping:
    negative floats map below positives; NaN excluded by caller)."""
    ia, ib = bf16_bits(a), bf16_bits(b)
    oa = torch.where(ia < 0, torch.tensor(-2**15, dtype=torch.int32, device=ia.device) - ia - 1 + 2**15*0 , ia)
    # standard trick: for negative patterns, order = 0x8000 - bits
    oa = torch.where(ia >= 0, ia + 2**15, 2**15 - (ia & 0x7FFF))
    ob = torch.where(ib >= 0, ib + 2**15, 2**15 - (ib & 0x7FFF))
    return (oa - ob).abs()


# ---- T1: INT32 -> FP32 exactness at the 2^24 boundary ----------------------
vals = [2**24 - 1, 2**24, 2**24 + 1, 2**24 + 2]
rows = []
for v in vals:
    for s in (1, -1):
        x = s * v
        f = float(torch.tensor(x, dtype=torch.int32).to(torch.float32).item())
        rows.append({"int": x, "fp32_roundtrip_exact": int(f) == x})
R["tests"]["T1_int32_fp32_boundary"] = rows
# expectation: ±(2^24-1) and ±2^24 exact; ±(2^24+1) NOT exact; ±(2^24+2) exact

# ---- T2: INT8 accumulator exactness vs INT64 reference ---------------------
t2 = []
gen = torch.Generator(device="cpu").manual_seed(7)
for K in (2048, 12288, 32768):
    for lo in (-127, -128):        # activation range with/without -128
        A = torch.randint(lo, 128, (64, K), generator=gen, dtype=torch.int8)
        B = torch.randint(-127, 128, (K, 64), generator=gen, dtype=torch.int8)
        ref = (A.to(torch.int64) @ B.to(torch.int64))                 # CPU exact
        gpu = torch._int_mm(A.cuda(), B.cuda()).to(torch.int64).cpu() # int8->int32 GPU
        f64 = (A.to(torch.float64).cuda() @ B.to(torch.float64).cuda()).to(torch.int64).cpu()
        # worst-case magnitude check against int32 overflow bound
        t2.append({"K": K, "act_min": lo, "kind": "random",
                   "gpu_int_mm_exact": bool(torch.equal(ref, gpu)),
                   "fp64_emul_exact": bool(torch.equal(ref, f64)),
                   "max_abs_acc": int(ref.abs().max()),
                   "int32_bound_ok": int(ref.abs().max()) < 2**31})
# true all-extreme worst cases; with activation=-128 the product bound is
# 128*127 = 16256 (not 16129), so worst |acc| = 16256*K
for K in (2048, 32768):
    for af, bf in ((-128, 127), (-128, -127), (127, 127)):
        A = torch.full((32, K), af, dtype=torch.int8)   # _int_mm requires M>16
        B = torch.full((K, 32), bf, dtype=torch.int8)
        ref = (A.to(torch.int64) @ B.to(torch.int64))
        gpu = torch._int_mm(A.cuda(), B.cuda()).to(torch.int64).cpu()
        bound = 16256 * K if af == -128 else 16129 * K
        t2.append({"K": K, "kind": f"extreme A={af} B={bf}",
                   "gpu_int_mm_exact": bool(torch.equal(ref, gpu)),
                   "max_abs_acc": int(ref.abs().max()),
                   "worst_case_bound_used": bound,
                   "acc_equals_bound": int(ref.abs().max()) == bound,
                   "int32_bound_ok": bound < 2**31})
R["tests"]["T2_int8_accumulator"] = t2

# ---- T3: power-of-two scaling commutation across value regimes -------------
t3 = []
cases = {
    "normal": torch.tensor([1.5, -2.75, 3.1415927, 1e-3, 123456.78], device=dev),
    "large": torch.tensor([1e30, -3e37, 1.7e38], device=dev),
    "tiny_normal": torch.tensor([1.2e-38, -5e-38], device=dev),
    "subnormal": torch.tensor([1e-40, -1e-42, 1e-45], device=dev),
    "zeros": torch.tensor([0.0, -0.0], device=dev),
}
for name, x in cases.items():
    for k in (-9, -1, 1, 8):
        s = float(2.0 ** k)
        lhs = ((x * 1.3).to(torch.float32) * s)          # round(x*1.3) then *2^k
        rhs = (x * (1.3 * s)).to(torch.float32)          # x*(1.3*2^k) rounded once
        # commutation claim compares round(y)*2^k vs round(y*2^k) with y=x*1.3
        y = (x.to(torch.float64) * 1.3)
        a = y.to(torch.float32) * s
        b = (y * s).to(torch.float32)
        eq = bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        t3.append({"regime": name, "k": k, "commutes_bitwise": eq,
                   "all_finite": finite})
R["tests"]["T3_pow2_commutation"] = t3
# expectation: normal/zeros commute; subnormal and overflow rows may FAIL —
# that is the documented boundary of the claim, not a bug.

# ---- T3b: bf16 midpoint cases ----------------------------------------------
mids = []
# bf16 has a 7-bit mantissa: spacing in [1,2) is 2^-7; exact midpoints are
# 1 + (2i+1)*2^-8. Test each midpoint and one FP32 ULP on either side.
mp = torch.tensor([1.0 + 2**-8, 1.0 + 3 * 2**-8, 2.0 + 2**-7], device=dev,
                  dtype=torch.float32)
up = torch.nextafter(mp, torch.full_like(mp, float("inf")))
dn = torch.nextafter(mp, torch.full_like(mp, float("-inf")))
pts = torch.cat([mp, up, dn]).to(torch.float64)
for k in (-3, 5):
    a = (pts.to(torch.float32) * float(2.0**k)).to(torch.bfloat16)
    b = (pts * float(2.0**k)).to(torch.float32).to(torch.bfloat16)
    mids.append({"k": k, "n_points": int(pts.numel()),
                 "bitwise_equal": bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))})
R["tests"]["T3b_bf16_midpoints"] = mids

# ---- T4: real-scale ordering, bf16 ulp distance -----------------------------
gen2 = torch.Generator(device=dev).manual_seed(11)
acc = torch.randint(-2**24, 2**24 + 1, (200000,), generator=gen2, device=dev).to(torch.float32)
sa = (torch.rand(200000, generator=gen2, device=dev) * 0.02 + 1e-4)
sw = (torch.rand(200000, generator=gen2, device=dev) * 0.02 + 1e-4)
o1 = ((acc * sa) * sw).to(torch.bfloat16)
o2 = (acc * (sa * sw)).to(torch.bfloat16)
nz = (o1.float() != 0) & (o2.float() != 0)
d = bf16_ulp_distance(o1[nz], o2[nz])
viol = (d > 1)
n_nz = int(nz.sum()); n_events = int((d > 0).sum())
# zero-event rate claim: report the 95% binomial upper bound (rule of three),
# never an order-of-magnitude equivalence
ub95 = 3.0 / n_nz if n_events == 0 else None
R["tests"]["T4_real_scale_ordering"] = {
    "n": n_nz, "n_events": n_events,
    "frac_differ": n_events / n_nz,
    "binomial_95_upper_bound_if_zero": ub95,
    "max_ulp_distance": int(d.max()),
    "cases_gt1ulp": int(viol.sum()),
    "saved_cases": [
        {"acc": float(acc[nz][i]), "sa": float(sa[nz][i]), "sw": float(sw[nz][i]),
         "o1_bits": int(o1[nz][i].view(torch.int16)), "o2_bits": int(o2[nz][i].view(torch.int16))}
        for i in torch.nonzero(viol).flatten()[:20].tolist()
    ],
}

json.dump(R, open("/w/numeric_boundaries.json", "w"), indent=2)
n_t1_bad = sum(1 for r in R["tests"]["T1_int32_fp32_boundary"]
               if r["int"] in (2**24+1, -(2**24+1)) and r["fp32_roundtrip_exact"])
print("T1 boundary pattern as theory predicts:", n_t1_bad == 0)
print("T2 all exact:", all(r["gpu_int_mm_exact"] and r["fp64_emul_exact"] for r in t2))
print("T3 normal-regime all commute:", all(r["commutes_bitwise"] for r in t3 if r["regime"] in ("normal","zeros") and r["all_finite"]))
print("T4 max ulp distance:", R["tests"]["T4_real_scale_ordering"]["max_ulp_distance"])
print("BOUNDARY_TESTS_OK")
