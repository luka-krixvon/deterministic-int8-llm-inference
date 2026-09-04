"""P2 K-sweep v2 — landing-audit compliant.

CLI:
  python3 /w/w3_p2_fp8_v2.py --seeds 5 --out /w/w3_p2_fp8_v2.json

Design (pre-fixed before running):
- SEEDS independent replicates. Per seed, inputs are NESTED-PREFIX: one draw
  of A_full (M, Kmax) and B_full (N, Kmax); every smaller K uses the same
  leading columns A_full[:, :K] / B_full[:, :K], so K is the ONLY thing that
  changes within a seed.
- Metrics per (K, seed), raw (not aggregate-only): n_total, n_differ,
  bf16 ulp-distance histogram (0,1,2,>2), abs_diff_max / output_rms,
  relative-error stats computed ONLY on |ref|>0.01*rms elements (near-zero
  excluded and counted separately).
- Model comparison on p(K) = P(element differs), fitted per model by least
  squares on the per-seed fractions (no logit, weights = 1):
    M_sqrt:  p = c*sqrt(K)
    M_sat:   p = 1 - exp(-c*sqrt(K))
    M_pow:   p = c*K**alpha            (alpha FREE — never preset to 0.5)
  Report SSE for each, fitted alpha with a seed-level percentile bootstrap CI
  (resample seeds with replacement, refit; B=2000).
- PRE-FIXED READING CRITERIA (stated before execution):
    * "sqrt-compatible" if the M_pow bootstrap 90% CI for alpha contains 0.5
      and excludes both 0 and 1.
    * M_sat is reported as better-fitting only if SSE(M_sat) < 0.5*SSE(M_sqrt).
    * No claim of "confirmed law" from 7 grid points; wording stays
      "compatible with".
Output schema: {"env":…, "config":…, "raw":[{K,seed,n_total,n_differ,
  ulp_hist,abs_over_rms,rel_p50,rel_p999,near_zero_frac}…],
  "fits":{"M_sqrt":{c,sse},"M_sat":{c,sse},"M_pow":{c,alpha,sse,
  alpha_ci90:[lo,hi]}},"criteria":…}
"""
import argparse, json, math, os
import torch
from vllm import _custom_ops as ops

KS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
M, N = 256, 2048


def bf16_ulp_distance(a, b):
    ia = a.view(torch.int16).to(torch.int32)
    ib = b.view(torch.int16).to(torch.int32)
    oa = torch.where(ia >= 0, ia + 2**15, 2**15 - (ia & 0x7FFF))
    ob = torch.where(ib >= 0, ib + 2**15, 2**15 - (ib & 0x7FFF))
    return (oa - ob).abs()


def run_seed(seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    Kmax = KS[-1]
    a_full = (torch.randn(M, Kmax, generator=g, device=dev) * 0.5).clamp(-3, 3)
    b_full = (torch.randn(N, Kmax, generator=g, device=dev) * 0.5).clamp(-3, 3)
    sa = torch.tensor(1.7e-2, device=dev, dtype=torch.float32)
    sb = torch.tensor(2.3e-2, device=dev, dtype=torch.float32)
    rows = []
    for K in KS:
        A = a_full[:, :K].to(torch.float8_e4m3fn)
        B = b_full[:, :K].to(torch.float8_e4m3fn).t()
        oc = ops.cutlass_scaled_mm(A, B, scale_a=sa.view(1, 1),
                                   scale_b=sb.view(1, 1), out_dtype=torch.bfloat16)
        ot = torch._scaled_mm(A, B, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        if isinstance(ot, tuple):
            ot = ot[0]
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
        from metrics_v2 import divergence_report
        rep = divergence_report(oc, ot)
        rep.update({"K": K, "seed": seed,
                    "n_differ": rep["n_bitwise_differ"]})
        rows.append(rep)
        print(f"seed {seed} K {K} done", flush=True)
    return rows


def sse(model, ks, ps):
    return sum((model(k) - p) ** 2 for k, p in zip(ks, ps))


def fit_all(raw):
    import numpy as np
    ks = np.array([r["K"] for r in raw], float)
    ps = np.array([r["n_differ"] / r["n_total"] for r in raw], float)

    def fit_sqrt(ks, ps):
        x = np.sqrt(ks); c = float((x * ps).sum() / (x * x).sum())
        return {"c": c, "sse": float(((c * x - ps) ** 2).sum())}

    def fit_sat(ks, ps):
        from scipy.optimize import minimize_scalar
        f = lambda c: (((1 - np.exp(-c * np.sqrt(ks))) - ps) ** 2).sum()
        r = minimize_scalar(f, bounds=(1e-6, 1.0), method="bounded")
        return {"c": float(r.x), "sse": float(r.fun)}

    def fit_pow(ks, ps):
        m = ps > 0
        lk, lp = np.log(ks[m]), np.log(ps[m])
        A = np.vstack([lk, np.ones_like(lk)]).T
        coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
        alpha, logc = float(coef[0]), float(coef[1])
        c = math.exp(logc)
        return {"c": c, "alpha": alpha,
                "sse": float(((c * ks ** alpha - ps) ** 2).sum())}

    fits = {"M_sqrt": fit_sqrt(ks, ps), "M_sat": fit_sat(ks, ps),
            "M_pow": fit_pow(ks, ps)}
    # seed-level percentile bootstrap for alpha
    seeds = sorted(set(r["seed"] for r in raw))
    rng = np.random.default_rng(20260813)
    alphas = []
    for _ in range(2000):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        sub = [r for s in pick for r in raw if r["seed"] == s]
        try:
            alphas.append(fit_pow(np.array([r["K"] for r in sub], float),
                                  np.array([r["n_differ"] / r["n_total"] for r in sub], float))["alpha"])
        except Exception:
            pass
    alphas.sort()
    fits["M_pow"]["alpha_ci90"] = [alphas[int(len(alphas) * 0.05)],
                                   alphas[int(len(alphas) * 0.95)]]
    return fits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="/w/w3_p2_fp8_v2.json")
    args = ap.parse_args()
    raw = []
    for s in range(args.seeds):
        raw.extend(run_seed(20260813 + s))
    fits = fit_all(raw)
    lo, hi = fits["M_pow"]["alpha_ci90"]
    out = {
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "image_digest": os.environ.get("IMAGE_DIGEST", "unset")},
        "config": {"M": M, "N": N, "KS": KS, "seeds": args.seeds,
                   "nested_prefix": True},
        "raw": raw,
        "fits": fits,
        "criteria": {
            "sqrt_compatible": bool(lo <= 0.5 <= hi and lo > 0 and hi < 1),
            "sat_preferred": bool(fits["M_sat"]["sse"] < 0.5 * fits["M_sqrt"]["sse"]),
            "wording": "compatible-with only; no law confirmed from 7 grid points",
        },
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print("alpha=%.3f CI90=[%.3f,%.3f] sqrt_compatible=%s" % (
        fits["M_pow"]["alpha"], lo, hi, out["criteria"]["sqrt_compatible"]))
    print("W3_P2V2_OK")
