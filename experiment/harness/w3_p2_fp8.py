"""W3/P2 formal test: FP8 cross-kernel divergence vs reduction depth K.

Two independent FP8 GEMM implementations on the same inputs:
  - vLLM ops.cutlass_scaled_mm (fp8e4m3)
  - torch._scaled_mm (fp8e4m3)

Operationalization of pre-registered P2 (stated BEFORE running): e4m3 x e4m3
products are exactly representable in FP32, so both kernels' accumulation is
FP32 with implementation-specific summation order. Order differences produce
relative accumulator deltas ~ sqrt(K) * 2^-24, which surface at the bf16
output only when they cross a rounding boundary. Therefore:
  P2-metric-1: fraction of differing output elements grows ~ sqrt(K)
  P2-metric-2: magnitude of each difference stays ~ 1 bf16 ulp (flat)
INT8 contrast on the same grid (from w3_ksweep.json): frac flat in K.

Per-tensor scales for both ops (the common denominator both support).
"""
import json, torch
from vllm import _custom_ops as ops

torch.manual_seed(20260813)
dev = "cuda"
M, N = 256, 2048
rows = []
for K in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    a = (torch.randn(M, K, device=dev) * 0.5).clamp(-3, 3)
    b = (torch.randn(N, K, device=dev) * 0.5).clamp(-3, 3)
    A = a.to(torch.float8_e4m3fn)
    B = b.to(torch.float8_e4m3fn).t()          # column-major (K, N)
    sa = torch.tensor(1.7e-2, device=dev, dtype=torch.float32)
    sb = torch.tensor(2.3e-2, device=dev, dtype=torch.float32)

    oc = ops.cutlass_scaled_mm(A, B, scale_a=sa.view(1, 1),
                               scale_b=sb.view(1, 1), out_dtype=torch.bfloat16)
    ot = torch._scaled_mm(A, B, scale_a=sa, scale_b=sb,
                          out_dtype=torch.bfloat16)
    if isinstance(ot, tuple):
        ot = ot[0]

    d = (oc.float() - ot.float()).abs()
    rel = d / oc.float().abs().clamp(min=1e-9)
    nz = rel[d > 0]
    rows.append({
        "K": K,
        "fp8_frac_differ": float((d > 0).float().mean().item()),
        "fp8_rms_rel": float(rel.pow(2).mean().sqrt().item()),
        "fp8_max_rel": float(rel.max().item()),
        "fp8_median_rel_of_differing": (float(nz.median().item())
                                        if nz.numel() else 0.0),
        "bitwise_identical": bool(torch.equal(oc.view(torch.int16),
                                              ot.view(torch.int16))),
    })
    print("K", K, "done", flush=True)

json.dump(rows, open("/w/w3_p2_fp8.json", "w"), indent=2)
print("W3_P2_OK")
