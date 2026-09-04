"""Build the three figures for the conformance paper, in the house style.

Style follows the routing-oracle figures: DejaVu Sans at 9.5pt, white ground, a restrained
palette, left-aligned bold panel titles, faint x-grid, value labels beside bars, Type-42
fonts, and stripped PDF metadata so a rebuild is byte-comparable and carries no local
provenance.

Only three figures are drawn, deliberately. Per-layer agreement is four numbers and the
+157% decomposition is three; both are stated in prose and belong in a table if anywhere.
What survives is what prose cannot carry: the overview a reader needs before Section II,
the ULP composition that IS the blindness argument, and the cost intervals whose position
relative to zero differs by model size.

Usage: python3 build_figures_conformance.py <artifacts-dir> <out-dir>
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.7,
    "pdf.fonttype": 42,
})
PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}

dark, mid_gray = "#374151", "#9ca3af"
blue, red, green, orange = "#2563eb", "#dc2626", "#059669", "#E69F00"
line_gray = "#7b8794"

ART, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

# ---- data, from the artifacts rather than transcribed ----
sens_path = ART / "p5p6_results" / "p5_sensitivity.json"
payload = sens_path.read_bytes()
print(f"  p5_sensitivity.json sha256[:16] = {hashlib.sha256(payload).hexdigest()[:16]}")
rows = json.loads(payload)["rows"]

ulp = collections.defaultdict(collections.Counter)
for r in rows:
    m = re.search(r"max ulp distance (\d+)", r["checks"]["real_scale_tolerance"]["detail"])
    if m:
        ulp[r["fault"]][int(m.group(1))] += 1

FAULTS = [("F1_scale_in_bf16", "F1  scale precision"),
          ("F2_double_rounding", "F2  double rounding"),
          ("F3_scale_order", "F3  scale order"),
          ("F4_truncate_output", "F4  output truncation"),
          ("F5_fused_order", "F5  fused order"),
          ("F9_operand_mismatch", "F9  operand mismatch")]

COST = []
for tag, label in [("pow2_nearest", "Qwen3-1.7B"), ("8b", "Qwen3-8B"), ("14b", "Qwen3-14B")]:
    d = json.loads((ART / "p8_results" / f"p8_cost_acc_{tag}.json").read_bytes())
    base = d["base"]["ppl"]
    b = d["bootstrap"]
    COST.append((label, 100 * d["relative_delta"],
                 100 * b["ci90_low"] / base, 100 * b["ci90_high"] / base))

# ============================ Figure 1: overview ============================
# Text is placed by measuring, not by eye: every panel line is checked against the panel
# width and the box is sized from the line count, because the first attempt let three
# panels overrun their frames and each other.
fig, ax = plt.subplots(figsize=(6.5, 2.85), layout="constrained")
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=3 / 72)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_xticks([]); ax.set_yticks([])
ax.axhspan(0.585, 1.0, color="#f4f6f8", zorder=0)

def stage(x, w, title, body, edge=dark, face="white"):
    ax.add_patch(FancyBboxPatch((x, 0.745), w, 0.185,
                                boxstyle="round,pad=0.006,rounding_size=0.010",
                                linewidth=0.9, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, 0.895, title, ha="center", va="center", fontsize=8.6,
            weight="bold", zorder=3)
    for i, ln in enumerate(body):
        ax.text(x + w / 2, 0.835 - i * 0.045, ln, ha="center", va="center",
                fontsize=7.1, color=dark, zorder=3)

stage(0.008, 0.212, "int8 operands",
      ["$A$ ($M{\\times}K$),  $W$ ($N{\\times}K$)", "with their scales"])
stage(0.255, 0.230, "INT32 accumulation",
      ["exact and order-free while", "$16256\\,K < 2^{31}$   (the alibi)"], edge=green, face="#f0faf6")
stage(0.530, 0.230, "epilogue",
      ["$\\times s_a$,  $\\times s_w$,  round to bf16", "implementations legally differ"],
      edge=orange, face="#fdf6e9")
stage(0.805, 0.185, "bf16 output",
      ["what the checks", "actually compare"])

for x0, x1 in [(0.220, 0.255), (0.485, 0.530), (0.760, 0.805)]:
    ax.add_patch(FancyArrowPatch((x0, 0.838), (x1, 0.838), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.0, color=line_gray, zorder=2))

for cx, txt in [(0.110, "F9  operands"), (0.370, "F6, F7  preconditions"),
                (0.645, "F1–F5  epilogue")]:
    ax.add_patch(FancyArrowPatch((cx, 0.673), (cx, 0.740), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.0, color=red, zorder=2))
    ax.text(cx, 0.640, txt, ha="center", va="center", fontsize=7.3, color=red, weight="bold")
ax.text(0.992, 0.962, "nine injected faults, ground truth by construction, 8,232 cells",
        ha="right", va="center", fontsize=7.8, style="italic", color=dark)

PANELS = [
    (0.008, 0.325, red,   "Do the checks fire?   (Sec. IV)", [
        ("F6, F7 preconditions: every cell", 0),
        ("F9 operands: every cell", 0),
        ("F1, F2, F3, F5: never caught", 1),
        ("4,704 cells, both regimes, all severities", 0),
        ("F4: only under pow2 scales", 0),
        ("no epilogue fault exceeds 1 spacing", 1)]),
    (0.348, 0.336, green, "Is pow2 servable?   (Sec. V)", [
        ("the reported +157% was 99.8% the", 0),
        ("probe's weight–scale mismatch", 0),
        ("requantized: CUTLASS = Triton bitwise", 1),
        ("196/196 and 252/252 layers", 0),
        ("8/8 token sequences at 1.7B, 8B, 14B", 0),
        ("PPL point estimates −.28% to +.48%", 1)]),
    (0.700, 0.290, blue,  "What follows", [
        ("a one-spacing tolerance cannot", 0),
        ("certify bitwise epilogue equality", 0),
        ("so do not tighten it — remove", 1),
        ("the freedom it accommodates", 1),
        ("pow2 makes conformance an", 0),
        ("equality instead of a tolerance", 0)]),
]
# Panel body text is fitted, not guessed: each panel's longest line is measured at a trial
# size and the size scaled down until it clears the frame's inner margin. Eyeballing the
# rasterized figure missed two overruns that a bbox measurement found immediately.
_probe_fig = plt.figure(figsize=(6.5, 2.85))
_pr = _probe_fig.canvas.get_renderer()
def fit_size(lines, w_frac, start=7.05, floor=6.0):
    avail = (w_frac - 0.024) * 6.5 * _probe_fig.dpi          # frame width minus 2x inset, px
    size = start
    while size > floor:
        widest = max(_probe_fig.text(0, 0, t, fontsize=size,
                     weight="bold" if b else "normal").get_window_extent(_pr).width
                     for t, b in lines)
        _probe_fig.texts.clear()
        if widest <= avail:
            return round(size, 2)
        size -= 0.05
    return floor
for x, w, accent, title, lines in PANELS:
    fs = fit_size(lines, w)
    ax.add_patch(FancyBboxPatch((x, 0.030), w, 0.495,
                                boxstyle="round,pad=0.006,rounding_size=0.010",
                                linewidth=0.9, edgecolor=accent, facecolor="white", zorder=2))
    ax.text(x + 0.012, 0.482, title, ha="left", va="center", fontsize=8.4,
            weight="bold", color=accent, zorder=3)
    for i, (txt, bold) in enumerate(lines):
        # Panel body text is measured against its own frame by tools/check: the middle
        # panel's longest line overran the box at 7.05pt, so the size is set per panel from
        # the widest line rather than shared.
        ax.text(x + 0.012, 0.415 - i * 0.063, txt, ha="left", va="center", fontsize=fs,
                color=dark, weight="bold" if bold else "normal", zorder=3)

fig.savefig(OUT / "overview.pdf", metadata=PDF_METADATA)
plt.close(fig)
plt.close(_probe_fig)

# ======================= Figure 2: ULP composition =======================
# The claim this figure has to make in one glance: the epilogue faults occupy exactly the
# interval a legal implementation is allowed to occupy, and the operand fault does not.
# So the >=2 segment gets the only hatch and the only annotation, and the dashed rule
# separates the two fault kinds.
fig, ax = plt.subplots(figsize=(3.42, 2.42), layout="constrained")
fig.get_layout_engine().set(w_pad=3 / 72, h_pad=3 / 72)
labels, z, o, hi = [], [], [], []
for key, label in FAULTS:
    c = ulp[key]; n = sum(c.values())
    labels.append(label)
    z.append(100 * c.get(0, 0) / n)
    o.append(100 * c.get(1, 0) / n)
    hi.append(100 * sum(v for k, v in c.items() if k >= 2) / n)
y = np.arange(len(labels))
ax.barh(y, z, color=mid_gray, edgecolor=dark, linewidth=0.45, label="max ULP $=0$")
ax.barh(y, o, left=z, color=orange, edgecolor=dark, linewidth=0.45, label="max ULP $=1$")
ax.barh(y, hi, left=np.array(z) + np.array(o), color=red, edgecolor=dark, linewidth=0.45,
        hatch="///", label="max ULP $\\geq 2$")
for yy, v in zip(y, hi):
    if v > 0:
        ax.text(101, yy, f"{v:.0f}%", va="center", ha="left", fontsize=7.4,
                color=red, weight="bold")
ax.axhline(4.5, color=dark, linewidth=0.85, linestyle=(0, (4, 2)))
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_ylim(5.65, -0.65)
ax.set_xlim(0, 122)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("share of the fault's cells (%)", fontsize=8.5, labelpad=2)
ax.tick_params(axis="both", labelsize=8.0)
ax.grid(axis="x", alpha=0.18, linewidth=0.7)
ax.set_axisbelow(True)
ax.set_title("Every epilogue fault fits\ninside one bf16 spacing", loc="left",
             fontsize=8.9, weight="bold", pad=3)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.055), ncol=1, fontsize=6.8,
          framealpha=0.96, edgecolor=mid_gray, handlelength=1.0, handletextpad=0.4,
          borderpad=0.35, labelspacing=0.3)
fig.savefig(OUT / "ulp_composition.pdf", metadata=PDF_METADATA)
plt.close(fig)

# ========================= Figure 3: accuracy cost =========================
fig, ax = plt.subplots(figsize=(3.42, 2.22), layout="constrained")
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=3 / 72)
ys = np.arange(len(COST))[::-1]
for yy, (label, est, lo, hi_) in zip(ys, COST):
    covers = lo <= 0 <= hi_
    c = mid_gray if covers else red
    ax.plot([lo, hi_], [yy, yy], color=c, linewidth=2.0, solid_capstyle="butt", zorder=2)
    for b in (lo, hi_):
        ax.plot([b, b], [yy - 0.13, yy + 0.13], color=c, linewidth=1.1, zorder=2)
    ax.plot([est], [yy], "o", color=blue, markersize=4.6, markeredgecolor="white",
            markeredgewidth=0.7, zorder=3)
    # The label used to sit to the right of the interval's upper bound, which pushed the
    # widest one past the axes; measuring the text extents caught it. It now sits above the
    # estimate, inside the axes by construction whatever the interval width.
    ax.text(est, yy + 0.26, f"{est:+.2f}%", va="bottom", ha="center", fontsize=7.9,
            color=dark, weight="bold" if not covers else "normal")
ax.axvline(0, color=dark, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
ax.set_yticks(ys, [c[0] for c in COST])
ax.set_ylim(-0.55, len(COST) - 0.25)
ax.set_xlim(-0.95, 1.05)
ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
ax.set_xlabel("perplexity change under pow2 scales (%)", fontsize=8.7, labelpad=2)
ax.tick_params(axis="both", labelsize=8.2)
ax.grid(axis="x", alpha=0.18, linewidth=0.7)
ax.set_axisbelow(True)
ax.set_title("Point estimates −0.28% to +0.48%;\nonly 14B clears zero", loc="left",
             fontsize=8.9, weight="bold", pad=3)
ax.text(0.985, 0.05, "90% paired cluster bootstrap, 256 windows", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.7, style="italic", color=mid_gray)
fig.savefig(OUT / "accuracy_cost.pdf", metadata=PDF_METADATA)
plt.close(fig)

print("  wrote overview.pdf, ulp_composition.pdf, accuracy_cost.pdf")
for f in sorted(OUT.glob("*.pdf")):
    print(f"    {f.name:24s} {hashlib.sha256(f.read_bytes()).hexdigest()[:16]}")
