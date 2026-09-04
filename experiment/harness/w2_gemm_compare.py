"""W2: controlled op-level comparison. Same int8 inputs -> CUTLASS vs Triton
scaled-MM. Separates act-quant differences from GEMM+epilogue differences,
and runs the op-level P1a probe (power-of-two scales => bitwise identity
REQUIRED)."""
import json, torch

results = {"discovery": {}, "quant_compare": [], "gemm_compare": []}

from vllm import _custom_ops as ops
results["discovery"]["scaled_int8_quant"] = hasattr(ops, "scaled_int8_quant")
results["discovery"]["cutlass_scaled_mm"] = hasattr(ops, "cutlass_scaled_mm")

triton_mm = None
import pkgutil, importlib, vllm
for m in pkgutil.walk_packages(vllm.__path__, "vllm."):
    if "triton_scaled_mm" in m.name:
        try:
            mod = importlib.import_module(m.name)
            if hasattr(mod, "triton_scaled_mm"):
                triton_mm = mod.triton_scaled_mm
                results["discovery"]["triton_mm"] = m.name
                break
        except Exception:
            pass
assert triton_mm is not None, "triton_scaled_mm not found"

torch.manual_seed(20260813)
dev = "cuda"

def bitwise_eq(a, b):
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))

for M, K in [(1, 2048), (16, 2048), (256, 2048), (16, 6144)]:
    x = (torch.randn(M, K, device=dev, dtype=torch.float32) * 3).to(torch.bfloat16)
    q1, s1, _ = ops.scaled_int8_quant(x)
    q2, s2, _ = ops.scaled_int8_quant(x)
    results["quant_compare"].append({
        "M": M, "K": K,
        "quant_self_consistent": bool(torch.equal(q1, q2) and torch.equal(s1, s2)),
    })

for M, K, N in [(1, 2048, 2048), (16, 2048, 6144), (256, 2048, 2048), (64, 6144, 2048)]:
    A = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
    # cutlass_scaled_mm requires column-major B
    B = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8).t()
    for scale_kind in ("real", "pow2"):
        if scale_kind == "real":
            sa = torch.rand(M, 1, device=dev, dtype=torch.float32) * 0.01 + 0.001
            sbv = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.001
        else:
            sa = torch.full((M, 1), 2.0 ** -9, device=dev, dtype=torch.float32)
            sbv = torch.full((N,), 2.0 ** -8, device=dev, dtype=torch.float32)
        # same numeric scales, each op's own layout convention:
        # cutlass wants scale_b (1,N); triton asserts scale_b (N,1)
        out_c = ops.cutlass_scaled_mm(A, B, scale_a=sa, scale_b=sbv.view(1, -1),
                                      out_dtype=torch.bfloat16)
        out_t = triton_mm(A, B, scale_a=sa, scale_b=sbv.view(-1, 1),
                          out_dtype=torch.bfloat16)
        d = (out_c.float() - out_t.float()).abs()
        rel = (d / out_c.float().abs().clamp(min=1e-9)).max().item()
        results["gemm_compare"].append({
            "M": M, "K": K, "N": N, "scales": scale_kind,
            "bitwise_identical": bitwise_eq(out_c, out_t),
            "max_abs_diff": float(d.max().item()),
            "max_rel_diff": rel,
            "frac_elems_differ": float((d > 0).float().mean().item()),
        })

json.dump(results, open("/w/w2_gemm_compare.json", "w"), indent=2)
print("W2_OK")
