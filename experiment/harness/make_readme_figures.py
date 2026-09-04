#!/usr/bin/env python3
"""Regenerate the two README figures from the committed artifacts. CPU only.

Usage: python3 experiment/harness/make_readme_figures.py
Writes figures/fig_divergence.png and figures/fig_margin.png.
"""
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)
OK = {"blue": "#0072B2", "orange": "#E69F00", "verm": "#D55E00", "green": "#009E73"}
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- figure 1: divergence prevalence vs reduction depth ---------------------
raw = json.load(open(os.path.join(ART, "w3_p2_raw_20seeds.json")))["raw"]
fp8 = defaultdict(list)
for r in raw:
    fp8[r["K"]].append(r["n_bitwise_differ"] / r["n_total"])
ks = sorted(fp8)
fp8_mean = [sum(fp8[k]) / len(fp8[k]) for k in ks]
i8 = json.load(open(os.path.join(ART, "w3_ksweep.json")))
i8_k = [r["K"] for r in i8]
i8_f = [r["int8_interkernel_frac_differ"] for r in i8]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.loglog(ks, fp8_mean, "o-", color=OK["verm"], lw=2, ms=6,
          label="FP8, CUTLASS vs torch._scaled_mm (20 seeds)")
ax.loglog(i8_k, i8_f, "s--", color=OK["blue"], lw=1.6, ms=6,
          label="INT8, CUTLASS vs Triton (single sweep)")
ax.set_xlabel("reduction depth $K$")
ax.set_ylabel("fraction of differing output elements")
ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
ax.set_ylim(5e-7, 1.2)
ax.legend(frameon=False, loc="center left", fontsize=9.5)
ax.grid(True, which="major", alpha=0.15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_divergence.png"))

# ---- figure 2: margin-conditioned flip rate --------------------------------
C = json.load(open(os.path.join(ART, "tf64v6b_CUTLASS.json")))["records"]
T = json.load(open(os.path.join(ART, "tf64v6b_TRITON.json")))["records"]
pairs = []
for ci, ti in zip(C, T):
    for c, t in zip(ci["steps"], ti["steps"]):
        if c and t and c.get("margin") is not None:
            pairs.append((c["margin"], c["top1_id"] != t["top1_id"]))
pairs.sort()
n = len(pairs)
xs, ys = [], []
for i in range(10):
    seg = pairs[i * n // 10:(i + 1) * n // 10]
    xs.append(sorted(m for m, _ in seg)[len(seg) // 2])
    ys.append(sum(1 for _, f in seg if f) / len(seg))

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.semilogx(xs, [y * 100 for y in ys], "o-", color=OK["green"], lw=2, ms=6)
ax.set_xlabel("median logit margin of decile")
ax.set_ylabel("cross-kernel flip rate (%)")
ax.set_xticks([0.25, 1, 4, 14]); ax.set_xticklabels(["0.25", "1", "4", "14"])
ax.grid(True, which="major", alpha=0.15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_margin.png"))
print(f"wrote 2 figures; {n} teacher-forced positions, "
      f"{sum(1 for _, f in pairs if f)} flips")
