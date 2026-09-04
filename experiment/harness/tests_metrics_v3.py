"""Regression tests v3 — Codex review #2 mandated cases, in ADDITION to the
19 checks of tests_metrics_v2.py (kept frozen). CPU only.
Env: see requirements-test.txt. Run: .venv/bin/python3 tests_metrics_v3.py
"""
import json, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from metrics_v3 import (divergence_report, strict_pair_join_v3,
                        average_precision_grouped, mann_whitney_auc)

fails = []
def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        fails.append(name)

def all_json_finite(obj):
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(all_json_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(all_json_finite(v) for v in obj)
    return True

# ---- non-finite inputs: no JSON NaN anywhere, exclusive classes ------------
a = torch.tensor([float("inf"), float("nan"), 1.0, 65280.0], dtype=torch.bfloat16)
b = torch.tensor([float("inf"), 1.0, float("nan"), float("inf")], dtype=torch.bfloat16)
r = divergence_report(a, b)
check("non-finite case: every JSON value finite or None", all_json_finite(r))
check("non-finite case: no finite/finite pair -> main hist strictly zero",
      sum(r["ulp_hist_main"].values()) == 0)
cls = r["finiteness_exclusive"]
check("exclusive classes partition n_total",
      sum(cls.values()) == r["n_total"])
check("rms uses finite ref only",
      r["rms_finite_ref"] is not None and math.isfinite(r["rms_finite_ref"]))

# ---- zero reference RMS -----------------------------------------------------
a = torch.zeros(8, dtype=torch.bfloat16)
b = torch.zeros(8, dtype=torch.bfloat16); b[3] = 1e-3
r = divergence_report(a, b)
check("zero-RMS -> explicit status", r["status"] == "zero-reference-rms")
check("zero-RMS -> abs_over_rms all None",
      all(v is None for v in r["abs_over_rms"].values()))
check("zero-RMS -> absolute summary still present",
      r["abs_err_finite_pairs"]["max"] is not None)
check("zero-RMS output json-safe", all_json_finite(r))

# ---- no finite reference ----------------------------------------------------
a = torch.tensor([float("nan"), float("inf")], dtype=torch.bfloat16)
b = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
r = divergence_report(a, b)
check("no-finite-reference status", r["status"] == "no-finite-reference")
check("no-finite-reference json-safe", all_json_finite(r))

# ---- dtype/shape guards ------------------------------------------------------
try:
    divergence_report(torch.zeros(4, dtype=torch.float32),
                      torch.zeros(4, dtype=torch.bfloat16))
    check("dtype guard raises", False)
except ValueError:
    check("dtype guard raises", True)
try:
    divergence_report(torch.zeros(4, dtype=torch.bfloat16),
                      torch.zeros(2, 2, dtype=torch.bfloat16))
    check("shape guard raises (no silent broadcast)", False)
except ValueError:
    check("shape guard raises (no silent broadcast)", True)

# ---- join v3: duplicate keys fail closed ------------------------------------
Cr = [[{"pos": 5, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1},
       {"pos": 5, "margin": 2.0, "top1_id": 2, "chosen_lp": -0.2},
       {"pos": 6, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]]
Tr = [[{"pos": 5, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1},
       {"pos": 6, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]]
pairs, diag = strict_pair_join_v3(Cr, Tr)
check("duplicate key invalidated, not last-write-wins",
      diag["duplicate_keys_invalidated_a"] == 1
      and all(k != 5 for k, *_ in [(p[0],) for p in pairs]) or diag["n_pairs_used"] == 1)
check("legacy records flagged identity-unverified",
      diag["identity_verified"] is False)

# ---- join v3: prompt reorder DETECTED with prompt_sha ------------------------
Cn = [{"prompt_sha": "aaa", "steps": [{"pos": 1, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]},
      {"prompt_sha": "bbb", "steps": [{"pos": 1, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]}]
Tn = [Cn[1], Cn[0]]
pairs, diag = strict_pair_join_v3(Cn, Tn)
check("prompt reorder detected via sha", "error" in diag)
pairs, diag = strict_pair_join_v3(Cn, Cn)
check("sha-verified join reports identity_verified", diag["identity_verified"] is True)

# ---- join v3: missing-margin diagnostics -------------------------------------
Cm = [[{"pos": 1, "margin": None, "top1_id": 1, "chosen_lp": -0.1},
       {"pos": 2, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]]
pairs, diag = strict_pair_join_v3(Cm, Cm)
check("n_common_keys vs n_pairs_used vs skipped are separated",
      diag["n_common_keys"] == 2 and diag["n_pairs_used"] == 1
      and diag["n_skipped_missing_margin"] == 1)

# ---- calibration v4: tail-bin support guarantee -------------------------------
from w4_p4_stats_v4 import build_bins_v4, MIN_BIN_N
cal = [(0, float(m), False, None) for m in range(25)]   # 25 distinct margins
bins = build_bins_v4(cal)
occ = [x["cal_n"] for x in bins["meta"] if x["cal_n"] > 0]
check("every occupied merged bin >= MIN_BIN_N (tail merged back)",
      all(nn >= MIN_BIN_N for nn in occ))
tiny = [(0, float(m), False, None) for m in range(5)]
bins2 = build_bins_v4(tiny)
check("total below MIN_BIN_N -> single-bin small-sample status",
      bins2["status"] == "small-sample-single-bin" and bins2["n_bins"] == 1)

# ---- refit v4: degenerate and bounded ----------------------------------------
from w3_p2_refit_v4 import fit, identifiable
check("all-zero p flagged not-identifiable",
      identifiable([512, 1024, 2048, 4096], [0, 0, 0, 0], [10] * 4) == "degenerate-constant-p")
check("single distinct K flagged", identifiable([512, 512], [0.1, 0.2], [10, 10]) == "insufficient-distinct-K")
f = fit([512, 1024, 2048, 4096], [1 - math.exp(-0.004 * k ** 0.5) for k in [512, 1024, 2048, 4096]], "M_gensat")
check("gensat v4 recovers alpha~0.5 within bounds, with diagnostics",
      f.get("status") == "ok" and abs(f["alpha"] - 0.5) < 0.02
      and all("success" in dgn for dgn in f["diagnostics"]))
check("alpha bounds recorded", f.get("alpha_bounds") == [0.0, 1.5])

# ---- bootstrap edge: no positives / no negatives ------------------------------
check("AUC None when single-class", mann_whitney_auc([(0, 1.0, True, None)] * 3) is None)
check("AP None when no positives", average_precision_grouped([(0, 1.0, False, None)] * 3) is None)

# ---- PAVA decreasing ----------------------------------------------------------
from w4_p4_stats_v4 import pava_decreasing
fitv = pava_decreasing([1, 2, 3, 4], [0.5, 0.6, 0.2, 0.1], [1, 1, 1, 1])
check("PAVA output is non-increasing",
      all(fitv[i] >= fitv[i + 1] - 1e-12 for i in range(len(fitv) - 1)))

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
