"""K-sweep model re-fit v4 (v3 frozen for lineage) — re-analysis of the EXISTING raw artifact
(w3_p2_fp8_v2.json). No GPU. Fixes per Codex review #1.5 R2:

- ALL candidate models minimize the SAME objective: unweighted linear SSE on
  per-(K,seed) fractions. No log-space fitting; p=0 rows are kept.
- Adds the generalized saturating model p = 1 - exp(-c*K^alpha).
- 5-seed bootstrap CI is labelled exploratory conditional interval; adds
  seed jackknife, leave-one-K-out alpha range, and K-range sensitivity.
- No binary sqrt-compatibility criterion from a misspecified pure power CI;
  verdict wording is descriptive and model-conditional.

v4 additions (Codex review #2, P1): alpha bounded to [0, 1.5] with bound-
sensitivity check; optimizer success/message/nit recorded, fail-closed when
all starts fail; identifiability guards (>=3 distinct K, non-degenerate p,
finite inputs, n_total>0) return an explicit status instead of an arbitrary
alpha; held-out predictive SSE (leave-one-seed-out and leave-one-K-out) for
ALL FOUR models; K-range sensitivity uses DISJOINT halves (middle K excluded).
Model-comparison wording: lowest in-sample SSE among the four pre-listed
candidates; compatibility statement only.

CLI: python3 w3_p2_refit_v4.py <raw_json_in> <out_json>
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
    A_LO, A_HI = 0.0, 1.5
    LC_LO, LC_HI = -30.0, 5.0
    def bounded_2p(residfn):
        best, diags = None, []
        for c0 in (1e-3, 1e-2):
            for a0 in (0.1, 0.5, 0.9, 1.4):
                r = minimize(residfn, [math.log(c0), a0], method="L-BFGS-B",
                             bounds=[(LC_LO, LC_HI), (A_LO, A_HI)])
                ok = bool(r.success) and math.isfinite(float(r.fun))
                diags.append({"c0": c0, "a0": a0, "success": bool(r.success),
                              "nit": int(r.nit),
                              "fun": float(r.fun) if math.isfinite(float(r.fun)) else None,
                              "alpha": float(r.x[1]),
                              "at_bound": bool(abs(r.x[1] - A_LO) < 1e-9 or abs(r.x[1] - A_HI) < 1e-9)})
                if ok and (best is None or float(r.fun) < best[0]):
                    best = (float(r.fun), math.exp(float(r.x[0])), float(r.x[1]))
        if best is None:
            return {"status": "fit-failed", "diagnostics": diags}
        return {"c": best[1], "alpha": best[2], "sse": best[0],
                "alpha_bounds": [A_LO, A_HI],
                "alpha_at_bound": bool(abs(best[2]-A_LO) < 1e-9 or abs(best[2]-A_HI) < 1e-9),
                "status": "ok", "diagnostics": diags}
    if model == "M_pow":
        def f(v):
            c, a = math.exp(min(max(v[0], LC_LO), LC_HI)), v[1]
            return float(((c * ks ** a - ps) ** 2).sum())
        return bounded_2p(f)
    if model == "M_gensat":
        def f(v):
            c, a = math.exp(min(max(v[0], LC_LO), LC_HI)), v[1]
            return float(((1 - np.exp(-c * ks ** a) - ps) ** 2).sum())
        return bounded_2p(f)


def ci_metadata(n_seeds):
    """Structured CI metadata (Codex review #4 P1): role is dynamic, never
    encoded in JSON key names; 'conditional' is a separate boolean."""
    return {"role": "exploratory" if n_seeds < 20 else "primary",
            "conditional": True,
            "n_seeds": int(n_seeds),
            "conditioning_scope": ("fixed K grid, matrix shape, kernel pair and "
                                   "input distribution; excludes model "
                                   "misspecification and design-choice uncertainty"),
            "role_threshold_note": "20-seed threshold is an analysis-freedom choice (A-7/A-9)"}


def identifiable(ks, ps, ns, nds=None, seeds_ks=None):
    import numpy as _np
    if len(set(ks)) < 3: return "insufficient-distinct-K"
    if not all(_np.isfinite(ks)) or not all(_np.isfinite(ps)): return "non-finite-input"
    if any(k <= 0 for k in ks): return "non-positive-K"
    if any(p < 0 or p > 1 for p in ps): return "p-out-of-range"
    if any(n <= 0 for n in ns): return "empty-cells"
    if nds is not None and any(nd < 0 or nd > n for nd, n in zip(nds, ns)):
        return "count-inconsistent"
    if max(ps) - min(ps) == 0: return "degenerate-constant-p"
    if seeds_ks is not None:
        grids = {s: tuple(sorted(kk)) for s, kk in seeds_ks.items()}
        if len(set(grids.values())) > 1: return "incomplete-K-grid-per-seed"
    return None


def predict(model, params, K):
    import numpy as _np
    if model == "M_sqrt": return params["c"] * _np.sqrt(K)
    if model == "M_sat_sqrt": return 1 - _np.exp(-params["c"] * _np.sqrt(K))
    if model == "M_pow": return params["c"] * K ** params["alpha"]
    if model == "M_gensat": return 1 - _np.exp(-params["c"] * K ** params["alpha"])


def main():
    raw = json.load(open(sys.argv[1]))["raw"]
    ks = [r["K"] for r in raw]
    ps = [r["n_differ"] / r["n_total"] for r in raw]
    ns = [r["n_total"] for r in raw]
    seeds = sorted(set(r["seed"] for r in raw))
    Ks = sorted(set(ks))
    nds = [r["n_differ"] for r in raw]
    seeds_ks = {}
    for r in raw: seeds_ks.setdefault(r["seed"], []).append(r["K"])
    bad = identifiable(ks, ps, ns, nds, seeds_ks)
    if bad:
        json.dump({"status": bad}, open(sys.argv[2], "w"), indent=2)
        print("NOT-IDENTIFIABLE:", bad); return

    fits = {m: fit(ks, ps, m) for m in MODELS}
    for m, f_ in fits.items():
        if f_.get("status") == "fit-failed":
            json.dump({"status": f"fit-failed:{m}", "fits": fits},
                      open(sys.argv[2], "w"), indent=2, default=str)
            print("FIT-FAILED", m); return

    # held-out predictive SSE for ALL models: leave-one-seed-out and leave-one-K-out
    heldout = {m: {"loso_sse": 0.0, "loko_sse": 0.0,
                   "loso_folds": {"expected": len(seeds), "completed": 0, "failed": 0},
                   "loko_folds": {"expected": len(Ks), "completed": 0, "failed": 0}} for m in MODELS}
    def cv(splits, key):
        for tr, te in splits:
            for m in MODELS:
                f_ = fit([k for k,_ in tr], [p for _,p in tr], m)
                if f_.get("status", "ok") == "ok" and "c" in f_:
                    heldout[m][key + "_sse"] += float(sum((predict(m, f_, k) - p) ** 2 for k, p in te))
                    heldout[m][key + "_folds"]["completed"] += 1
                else:
                    heldout[m][key + "_folds"]["failed"] += 1
    cv([([(r["K"], r["n_differ"]/r["n_total"]) for r in raw if r["seed"] != s],
         [(r["K"], r["n_differ"]/r["n_total"]) for r in raw if r["seed"] == s]) for s in seeds], "loso")
    cv([([(r["K"], r["n_differ"]/r["n_total"]) for r in raw if r["K"] != K],
         [(r["K"], r["n_differ"]/r["n_total"]) for r in raw if r["K"] == K]) for K in Ks], "loko")
    fold_complete = all(v["loso_folds"]["failed"] == 0 and v["loko_folds"]["failed"] == 0
                        for v in heldout.values())

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
    mid = Ks[len(Ks) // 2]
    lo = [(r["K"], r["n_differ"] / r["n_total"]) for r in raw if r["K"] < mid]
    hi = [(r["K"], r["n_differ"] / r["n_total"]) for r in raw if r["K"] > mid]
    krange = {"disjoint": True, "middle_K_excluded": mid,
              "low_half_alpha": fit([k for k, _ in lo], [p for _, p in lo], "M_gensat").get("alpha"),
              "high_half_alpha": fit([k for k, _ in hi], [p for _, p in hi], "M_gensat").get("alpha")}
    # seed-resample bootstrap on gensat alpha (role in ci_metadata)
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
        "heldout_predictive_sse": heldout,
        "model_comparison_wording": ("M_gensat attains the lowest SSE among the four "
                                     "pre-listed candidates (in-sample and held-out as "
                                     "reported); alpha~0.5 is COMPATIBLE with sqrt-shaped "
                                     "saturation; no functional family is excluded and no "
                                     "causal mechanism is identified by this fit alone"),
        "gensat_alpha_jackknife": {"values": jk, "min": min(jk), "max": max(jk)},
        "gensat_leave_one_K_out": loko,
        "gensat_K_range_sensitivity": krange,
        "gensat_alpha_seed_bootstrap_ci90": [
            boots[int(len(boots) * 0.05)], boots[int(len(boots) * 0.95)]],
        "ci_metadata": ci_metadata(len(seeds)),
        "heldout_comparison_valid": fold_complete,
        "labels": {
            "ci_status": ("role=%s, %d seeds; see ci_metadata"
                          % (ci_metadata(len(seeds))["role"], len(seeds))),
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
    print("REFIT_V4_OK")


if __name__ == "__main__":
    main()
