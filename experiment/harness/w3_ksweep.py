"""W3/P2: divergence magnitude vs reduction depth K.

Pre-registered prediction P2: for order-dependent paths the inter-kernel
divergence grows ~sqrt(K) (random-rounding model). For the INT8 path the
accumulation is exact, so divergence (real scales) comes only from the
epilogue and should be K-INDEPENDENT (one or two multiplies per element,
regardless of K) — a sharper prediction unique to the integer path.
Both are tested here: INT8 inter-kernel divergence vs K at real scales,
plus, as the floating-point contrast, the same GEMM computed in bf16
(torch matmul) vs fp32 reference to show K-dependent growth.
Fixed M=256 (the regime where W2 caught real-scale divergence), N=2048.
"""
import json, math, torch
from vllm import _custom_ops as ops
import importlib
triton_mm = importlib.import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm").triton_scaled_mm

torch.manual_seed(20260813)
dev = "cuda"
M, N = 256, 2048
out = []
for K in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    A = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
    B = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8).t()
    sa = torch.rand(M, 1, device=dev, dtype=torch.float32) * 0.01 + 0.001
    sbv = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.001
    oc = ops.cutlass_scaled_mm(A, B, scale_a=sa, scale_b=sbv.view(1, -1), out_dtype=torch.bfloat16)
    ot = triton_mm(A, B, scale_a=sa, scale_b=sbv.view(-1, 1), out_dtype=torch.bfloat16)
    d = (oc.float() - ot.float()).abs()
    rel = d / oc.float().abs().clamp(min=1e-9)
    # floating-point contrast: bf16 matmul vs fp64 reference on same values
    Af = (A.float() * sa)
    Bf = (B.float() * sbv.view(1, -1))
    ref = (Af.double() @ Bf.double())
    bf = (Af.bfloat16() @ Bf.bfloat16()).float()
    fp_rel = ((bf - ref.float()).abs() / ref.float().abs().clamp(min=1e-9))
    out.append({
        "K": K,
        "int8_interkernel_frac_differ": float((d > 0).float().mean().item()),
        "int8_interkernel_max_rel": float(rel.max().item()),
        "int8_interkernel_p999_rel": float(rel.flatten().kthvalue(
            int(rel.numel() * 0.999)).values.item()),
        "bf16_vs_fp64_median_rel": float(fp_rel.median().item()),
        "bf16_vs_fp64_p99_rel": float(fp_rel.flatten().kthvalue(
            int(fp_rel.numel() * 0.99)).values.item()),
    })
    print("K", K, "done", flush=True)
json.dump(out, open("/w/w3_ksweep.json", "w"), indent=2)
print("W3_KSWEEP_OK")
