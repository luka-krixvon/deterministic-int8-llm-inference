"""K-sweep model re-fit v3 — re-analysis of the EXISTING raw artifact
(w3_p2_fp8_v2.json). No GPU. Fixes per Codex review #1.5 R2:

- ALL candidate models minimize the SAME objective: unweighted linear SSE on
  per-(K,seed) fractions. No log-space fitting; p=0 rows are kept.
- Adds the generalized saturating model p = 1 - exp(-c*K^alpha).
- 5-seed bootstrap CI is labelled exploratory conditional interval; adds
  seed jackknife, leave-one-K-out alpha range, and K-range sensitivity.
- No binary sqrt-compatibility criterion from a misspecified pure power CI;
  verdict wording is descriptive and model-conditional.

CLI: python3 w3_p2_refit_v3.py <raw_json_in> <out_json>
"""
import json, math, sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar

MODELS = ("M_sqrt", "M_sat_sqrt", "M_pow", "M_gensat")


def fit(ks, ps, model):
    ks = np.asarray(ks, float); ps = np.asarray(ps, float)
    if model == "M_sqrt":
        x = np.sqrt(ks); c = float((x * ps).sum() / (x * x).sum())
        return {"c": c, "sse": float(((c * x - ps) ** 2).sum())}
    if model == "M_sat_sqrt":
        f = lambda c: (((1 - np.exp(-c * np.sqrt(ks))) - ps) ** 2).sum()
        r = minimize_scalar(f, bounds=(1e-8, 1.0), method="bounded")
        return {"c": float(r.x), "sse": float(r.fun)}
    if model == "M_pow":
        def f(v):
            c, a = math.exp(v[0]), v[1]
            return ((c * ks ** a - ps) ** 2).sum()
        r = min((minimize(f, [math.log(c0), a0], method="Nelder-Mead")
                 for c0 in (1e-3, 1e-2) for a0 in (0.3, 0.5, 0.7)),
                key=lambda r: r.fun)
        return {"c": math.exp(r.x[0]), "alpha": float(r.x[1]), "sse": float(r.fun)}
    if model == "M_gensat":
        def f(v):
            c, a = math.exp(v[0]), v[1]
            return ((1 - np.exp(-c * ks ** a) - ps) ** 2).sum()
        r = min((minimize(f, [math.log(c0), a0], method="Nelder-Mead")
                 for c0 in (1e-3, 1e-2) for a0 in (0.3, 0.5, 0.7)),
                key=lambda r: r.fun)
        return {"c": math.exp(r.x[0]), "alpha": float(r.x[1]), "sse": float(r.fun)}


def main():
    raw = json.load(open(sys.argv[1]))["raw"]
    ks = [r["K"] for r in raw]
    ps = [r["n_differ"] / r["n_total"] for r in raw]
    seeds = sorted(set(r["seed"] for r in raw))
    Ks = sorted(set(ks))

    fits = {m: fit(ks, ps, m) for m in MODELS}

    # seed jackknife on the generalized model's alpha
    jk = []
    for s in seeds:
        sub = [(r["K"], r["n_differ"] / r["n_total"]) for r in raw if r["seed"] != s]
        jk.append(fit([k for k, _ in sub], [p for _, p in sub], "M_gensat")["alpha"])
    # leave-one-K-out
    loko = []
    for K in Ks:
        sub = [(r["K"], r["n_differ"] / r["n_total"]) for r in raw if r["K"] != K]
        loko.append({"left_out_K": K,
                     "alpha": fit([k for k, _ in sub], [p for _, p in sub],
                                  "M_gensat")["alpha"]})
    # K-range sensitivity: low half / high half
    lo = [(r["K"], r["n_differ"] / r["n_total"]) for r in raw if r["K"] <= Ks[len(Ks)//2]]
    hi = [(r["K"], r["n_differ"] / r["n_total"]) for r in raw if r["K"] >= Ks[len(Ks)//2]]
    krange = {"low_half_alpha": fit([k for k, _ in lo], [p for _, p in lo], "M_gensat")["alpha"],
              "high_half_alpha": fit([k for k, _ in hi], [p for _, p in hi], "M_gensat")["alpha"]}
    # exploratory conditional bootstrap (seed resample) on gensat alpha
    rng = np.random.default_rng(20260813)
    boots = []
    for _ in range(2000):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        sub = [(r["K"], r["n_differ"] / r["n_total"]) for s in pick for r in raw if r["seed"] == s]
        boots.append(fit([k for k, _ in sub], [p for _, p in sub], "M_gensat")["alpha"])
    boots.sort()

    out = {
        "input": sys.argv[1],
        "objective": "unweighted linear SSE on per-(K,seed) fractions; identical for all models; p=0 kept",
        "fits": fits,
        "gensat_alpha_jackknife": {"values": jk, "min": min(jk), "max": max(jk)},
        "gensat_leave_one_K_out": loko,
        "gensat_K_range_sensitivity": krange,
        "gensat_alpha_seed_bootstrap_ci90_EXPLORATORY": [
            boots[int(len(boots) * 0.05)], boots[int(len(boots) * 0.95)]],
        "labels": {
            "ci_status": ("exploratory conditional interval: 5 seeds, fixed K grid/"
                          "shape/kernel pair/input distribution; does NOT cover model "
                          "misspecification or design choices"),
            "verdict_wording": ("descriptive, model-conditional; no binary "
                                "sqrt-compatibility criterion is applied"),
        },
    }
    json.dump(out, open(sys.argv[2], "w"), indent=2)
    g = fits["M_gensat"]
    print(f"gensat: alpha={g['alpha']:.3f} sse={g['sse']:.2e} | "
          f"pow: alpha={fits['M_pow']['alpha']:.3f} sse={fits['M_pow']['sse']:.2e} | "
          f"sat_sqrt sse={fits['M_sat_sqrt']['sse']:.2e} | sqrt sse={fits['M_sqrt']['sse']:.2e}")
    print(f"jackknife alpha range [{min(jk):.3f},{max(jk):.3f}] | "
          f"LOKO range [{min(l['alpha'] for l in loko):.3f},{max(l['alpha'] for l in loko):.3f}] | "
          f"K-range low/high {krange['low_half_alpha']:.3f}/{krange['high_half_alpha']:.3f}")
    print("REFIT_V3_OK")


if __name__ == "__main__":
    main()
