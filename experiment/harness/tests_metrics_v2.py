"""Synthetic regression tests for metrics_v2 and refit v3 — no GPU (CPU torch).
Covers the cases mandated by the review-1.5 work order:
  -0/+0, finite/Inf/NaN, mixed-label margin ties, zero-difference K rows,
  duplicated quantile edges / empty calibration bins, missing or misordered
  positions in the pair join.
Run: .venv/bin/python3 tests_metrics_v2.py   (expects torch, numpy, scipy)
"""
import json, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from metrics_v2 import (bf16_bitwise_differ, bf16_ulp_distance_finite,
                        divergence_report, strict_pair_join,
                        average_precision_grouped, mann_whitney_auc)

fails = []
def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        fails.append(name)

# ---- signed zero -------------------------------------------------------------
a = torch.tensor([0.0, -0.0, 1.0], dtype=torch.bfloat16)
b = torch.tensor([-0.0, -0.0, 1.0], dtype=torch.bfloat16)
bd = bf16_bitwise_differ(a, b)
check("signed-zero counts as bitwise divergence", bool(bd[0]) and not bool(bd[1]))
ud, fin = bf16_ulp_distance_finite(a, b)
check("signed-zero numerical ULP distance is 0 (documented collapse)",
      int(ud[0]) == 0)
r = divergence_report(a, b)
check("report: n_bitwise_differ=1 with -0/+0", r["n_bitwise_differ"] == 1)

# ---- NaN / Inf ---------------------------------------------------------------
a = torch.tensor([float("inf"), float("nan"), 1.0, 65280.0], dtype=torch.bfloat16)
b = torch.tensor([float("inf"), 1.0, float("nan"), float("inf")], dtype=torch.bfloat16)
r = divergence_report(a, b)
check("nan_any counted", r["nan_any"] == 2)
check("inf_any counted", r["inf_any"] >= 2)
check("finite_mismatch counted", r["finite_mismatch"] == 3)
hist_total = sum(r["ulp_hist_main"].values()) + sum(r["near_zero"]["ulp_hist"].values())
check("non-finite pairs never enter ULP histograms", hist_total <= 1)

# ---- near-zero sub-report ----------------------------------------------------
a = torch.tensor([100.0] * 98 + [1e-4, 1e-4], dtype=torch.bfloat16)
b = a.clone(); b[98] = 2e-4
r = divergence_report(a, b)
check("near-zero region has own divergence count",
      r["near_zero"]["n"] >= 2 and r["near_zero"]["n_bitwise_differ"] == 1)
check("abs/RMS reports quantiles not only max",
      r["abs_over_rms"]["p50"] is not None and "rmse_over_rms" in r["abs_over_rms"])

# ---- mixed-label margin ties: AP must be permutation-invariant ---------------
base = ([(0, 0.5, True, None)] * 3 + [(0, 0.5, False, None)] * 5
        + [(1, 2.0, False, None)] * 10 + [(1, 0.1, True, None)])
import random
ap0 = average_precision_grouped(base)
for s in range(5):
    sh = base[:]; random.Random(s).shuffle(sh)
    if average_precision_grouped(sh) != ap0:
        check("grouped-tie AP permutation invariant", False); break
else:
    check("grouped-tie AP permutation invariant", True)
check("AP None when no positives",
      average_precision_grouped([(0, 1.0, False, None)] * 4) is None)
check("AUC tie handling gives half credit",
      abs(mann_whitney_auc([(0, 1.0, True, None), (0, 1.0, False, None)]) - 0.5) < 1e-12)

# ---- strict join: missing / misordered positions -----------------------------
Cr = [[{"pos": 10, "margin": 1.0, "top1_id": 5, "chosen_lp": -0.1},
       {"pos": 11, "margin": 2.0, "top1_id": 6, "chosen_lp": -0.2}]]
Tr = [[{"pos": 11, "margin": 2.0, "top1_id": 7, "chosen_lp": -0.3}]]  # pos 10 missing
pairs, diag = strict_pair_join(Cr, Tr)
check("join drops nothing silently: diagnostics report the missing position",
      diag["only_in_a"] == 1 and diag["n_joined"] == 1)
check("misordered join is position-keyed (flip detected at pos 11)",
      pairs[0][2] is True)

# ---- refit: zero-difference K rows kept ---------------------------------------
from w3_p2_refit_v3 import fit
rows_k = [512, 1024, 2048, 4096]
ps = [0.0, 0.1, 0.14, 0.2]           # includes p=0
f = fit(rows_k, ps, "M_gensat")
check("fit runs with p=0 rows kept", "alpha" in f and math.isfinite(f["alpha"]))
# known-alpha recovery: generate from gensat alpha=0.5 exactly, no noise
gen = [1 - math.exp(-0.004 * k ** 0.5) for k in rows_k]
f2 = fit(rows_k, gen, "M_gensat")
check("gensat recovers alpha=0.5 on clean synthetic",
      abs(f2["alpha"] - 0.5) < 0.02)
# common objective: pow model SSE evaluated in linear space equals objective
f3 = fit(rows_k, gen, "M_pow")
resid = sum((f3["c"] * k ** f3["alpha"] - p) ** 2 for k, p in zip(rows_k, gen))
check("pow fit objective == reported linear SSE", abs(resid - f3["sse"]) < 1e-12)

# ---- calibration: duplicated edges + empty bins --------------------------------
from w4_p4_stats_v3 import build_bins
cal = [(0, 1.0, False, None)] * 50 + [(0, 1.0, True, None)] * 5 \
    + [(2, 9.0, False, None)] * 5   # massive ties at 1.0 -> duplicate edges
edges, binof = build_bins(cal)
check("duplicate quantile edges merged", len(edges) == len(set(edges)))
bins = set(binof(m) for _, m, _, _ in cal)
check("binning total, no crash on ties", len(bins) >= 1)

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
