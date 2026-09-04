"""P4 statistics v3 — fixes per Codex review #1.5 R3. No GPU.

CLI: python3 w4_p4_stats_v3.py tf_A.json tf_B.json out.json

Changes vs v2 (all semantics fixed pre-re-analysis, recorded in A-5):
- strict (prompt_id,pos) join with missing/duplicate diagnostics (metrics_v2).
- ROC-AUC unchanged (tie-halved Mann-Whitney was correct).
- PR metric renamed and reimplemented as grouped-tie average precision;
  None when no positives.
- cluster bootstrap CI (n_prompts<16) labelled exploratory small-cluster;
  naive per-position binomial CI reported ONLY as an error contrast;
  LOPO output carries prompt IDs and is labelled influence-diagnostic.
- calibration: duplicate quantile edges merged; bins below minimum support
  merged with neighbor; Beta(1,1) smoothing on bin probabilities; eval
  fallbacks to NEAREST bin and are counted; per-bin cal_n/eval_n/flips
  reported; Brier compared against constant-prevalence baseline; Brier
  prompt-bootstrap CI labelled exploratory.
"""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_v2 import strict_pair_join, average_precision_grouped, mann_whitney_auc

MIN_BIN_N = 20


def build_bins(cal):
    ms = sorted(m for _, m, _, _ in cal)
    edges = sorted(set(ms[int(len(ms) * i / 10)] for i in range(1, 10)))
    def binof(m):
        b = 0
        for e in edges:
            if m > e:
                b += 1
        return b
    # merge low-support bins with left neighbor
    from collections import defaultdict
    cnt = defaultdict(int)
    for _, m, _, _ in cal:
        cnt[binof(m)] += 1
    keep = []
    for i in range(len(edges) + 1):
        keep.append(cnt.get(i, 0))
    # build merged mapping
    merged, cur = {}, 0
    acc = 0
    for i in range(len(keep)):
        acc += keep[i]
        merged[i] = cur
        if acc >= MIN_BIN_N:
            cur += 1
            acc = 0
    return edges, lambda m: merged[binof(m)]


def main():
    C = json.load(open(sys.argv[1]))
    T = json.load(open(sys.argv[2]))
    pairs, diag = strict_pair_join(C, T)
    prompts = sorted(set(p for p, *_ in pairs))
    n = len(pairs)
    flips = sum(1 for r in pairs if r[2])

    auc = mann_whitney_auc(pairs)
    ap = average_precision_grouped(pairs)

    rng = random.Random(20260813)
    boots = []
    for _ in range(5000):
        pick = [rng.choice(prompts) for _ in prompts]
        sub = [r for p in pick for r in pairs if r[0] == p]
        a = mann_whitney_auc(sub)
        if a is not None:
            boots.append(a)
    boots.sort()
    lopo = [{"left_out_prompt": p,
             "auc": mann_whitney_auc([r for r in pairs if r[0] != p])}
            for p in prompts]
    # naive per-position CI — error contrast only (positions are clustered)
    import math
    fr = flips / n
    se = math.sqrt(fr * (1 - fr) / n)

    cal = [r for r in pairs if r[0] % 2 == 0]
    ev = [r for r in pairs if r[0] % 2 == 1]
    edges, binof = build_bins(cal)
    from collections import defaultdict
    calbin = defaultdict(lambda: [0, 0])
    for _, m, f, _ in cal:
        b = calbin[binof(m)]; b[0] += f; b[1] += 1
    prob = {k: (v[0] + 1) / (v[1] + 2) for k, v in calbin.items()}   # Beta(1,1)
    known = sorted(prob)
    prev = (sum(f for _, _, f, _ in cal) + 1) / (len(cal) + 2)
    fallbacks = 0
    evbin = defaultdict(lambda: [0, 0, None])
    brier = base_brier = 0.0
    for _, m, f, _ in ev:
        b = binof(m)
        if b in prob:
            p = prob[b]
        else:
            fallbacks += 1
            b_near = min(known, key=lambda k: abs(k - b))
            p = prob[b_near]
        brier += (p - f) ** 2
        base_brier += (prev - f) ** 2
        e = evbin[b]; e[0] += f; e[1] += 1; e[2] = p
    curve = [{"bin": b, "pred": e[2], "obs": e[0] / e[1], "eval_n": e[1],
              "eval_flips": e[0], "cal_n": calbin.get(b, [0, 0])[1],
              "cal_flips": calbin.get(b, [0, 0])[0]}
             for b, e in sorted(evbin.items())]
    # Brier prompt-bootstrap (exploratory)
    bb = []
    ev_prompts = sorted(set(p for p, *_ in ev))
    for _ in range(2000):
        pick = [rng.choice(ev_prompts) for _ in ev_prompts]
        sub = [r for p in pick for r in ev if r[0] == p]
        s = 0.0
        for _, m, f, _ in sub:
            b = binof(m)
            p = prob.get(b) or prob[min(known, key=lambda k: abs(k - b))]
            s += (p - f) ** 2
        bb.append(s / len(sub))
    bb.sort()

    out = {
        "join_diagnostics": diag,
        "n_prompts": len(prompts), "n_positions": n, "n_flips": flips,
        "flip_rate": fr,
        "roc_auc": auc,
        "roc_auc_cluster_ci90": [boots[int(len(boots) * .05)],
                                 boots[int(len(boots) * .95)]],
        "cluster_ci_status": ("exploratory small-cluster interval"
                              if len(prompts) < 16 else "primary"),
        "naive_positionwise_flip_se_CONTRAST_ONLY": se,
        "lopo_auc_influence_diagnostic": lopo,
        "average_precision_grouped_ties": ap,
        "brier_heldout": brier / len(ev) if ev else None,
        "brier_constant_prevalence_baseline": base_brier / len(ev) if ev else None,
        "brier_prompt_bootstrap_ci90_EXPLORATORY": [bb[int(len(bb) * .05)],
                                                    bb[int(len(bb) * .95)]] if bb else None,
        "calibration_fallback_count": fallbacks,
        "calibration_curve_heldout": curve,
        "smoothing": "Beta(1,1) on bin probabilities; bins merged to >= %d cal positions" % MIN_BIN_N,
    }
    json.dump(out, open(sys.argv[3], "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("join_diagnostics", "roc_auc",
        "roc_auc_cluster_ci90", "average_precision_grouped_ties",
        "brier_heldout", "brier_constant_prevalence_baseline",
        "calibration_fallback_count")}, indent=1, default=str))
    print("P4STATS_V3_OK")


if __name__ == "__main__":
    main()
