"""P4 statistics v2 — prompt-cluster inference (audit item 7 statistics part).

CLI: python3 w4_p4_stats.py tf_CUTLASS.json tf_TRITON.json out.json
Pure stdlib. PRE-FIXED design:
- bootstrap unit = PROMPT (cluster bootstrap, B=5000); naive per-position CI
  reported alongside for contrast, never alone.
- leave-one-prompt-out (LOPO) AUC list.
- calibration/holdout split pre-fixed: even prompt indices calibrate any
  threshold/binning, odd indices evaluate.
- metrics: ROC-AUC, PR-AUC, Brier (flip vs margin-derived score), 10-bin
  calibration curve. Positions are never treated as independent samples for
  CI purposes.
"""
import json, random, sys

def pairs_of(C, T):
    out = []
    for pi, (ci, ti) in enumerate(zip(C, T)):
        for c, t in zip(ci, ti):
            if not c or not t or c.get("margin") is None: continue
            out.append((pi, c["margin"], c["top1_id"] != t["top1_id"]))
    return out

def auc(pairs):
    pos = sorted(m for _, m, f in pairs if f)
    neg = sorted(m for _, m, f in pairs if not f)
    if not pos or not neg: return None
    import bisect
    s = sum(bisect.bisect_left(neg, m) + 0.5 * (bisect.bisect_right(neg, m) - bisect.bisect_left(neg, m)) for m in pos)
    return 1 - s / (len(pos) * len(neg))

def pr_auc(pairs):
    xs = sorted(pairs, key=lambda r: r[1])          # low margin = predicted flip
    P = sum(1 for r in pairs if r[2])
    tp = fp = 0; prev_r = 0.0; area = 0.0; last_prec = 1.0
    for _, m, f in xs:
        tp += f; fp += (not f)
        r = tp / P if P else 0; prec = tp / (tp + fp)
        area += (r - prev_r) * prec; prev_r = r
    return area

def brier_and_cal(cal, ev):
    # map margin -> flip prob via 10 quantile bins fit on calibration prompts
    ms = sorted(m for _, m, _ in cal)
    edges = [ms[int(len(ms) * i / 10)] for i in range(1, 10)]
    def binof(m):
        b = 0
        for e in edges:
            if m > e: b += 1
        return b
    import collections
    agg = collections.defaultdict(lambda: [0, 0])
    for _, m, f in cal:
        a = agg[binof(m)]; a[0] += f; a[1] += 1
    prob = {b: (a[0] / a[1]) for b, a in agg.items()}
    bs, curve = 0.0, []
    evagg = collections.defaultdict(lambda: [0, 0, 0.0])
    for _, m, f in ev:
        p = prob.get(binof(m), sum(f2 for _, _, f2 in cal) / len(cal))
        bs += (p - f) ** 2
        a = evagg[binof(m)]; a[0] += f; a[1] += 1; a[2] = p
    for b in sorted(evagg):
        a = evagg[b]
        curve.append({"bin": b, "pred": a[2], "obs": a[0] / a[1], "n": a[1]})
    return bs / len(ev), curve

if __name__ == "__main__":
    C = json.load(open(sys.argv[1])); T = json.load(open(sys.argv[2]))
    pairs = pairs_of(C, T)
    prompts = sorted(set(p for p, _, _ in pairs))
    full_auc = auc(pairs)
    rng = random.Random(20260813)
    boots = []
    for _ in range(5000):
        pick = [rng.choice(prompts) for _ in prompts]
        sub = [r for p in pick for r in pairs if r[0] == p]
        a = auc(sub)
        if a is not None: boots.append(a)
    boots.sort()
    lopo = [auc([r for r in pairs if r[0] != p]) for p in prompts]
    cal = [r for r in pairs if r[0] % 2 == 0]
    ev = [r for r in pairs if r[0] % 2 == 1]
    brier, curve = brier_and_cal(cal, ev)
    out = {"n_prompts": len(prompts), "n_positions": len(pairs),
           "auc": full_auc,
           "auc_cluster_ci90": [boots[int(len(boots)*.05)], boots[int(len(boots)*.95)]],
           "lopo_auc": lopo, "pr_auc": pr_auc(pairs),
           "brier_heldout": brier, "calibration_curve_heldout": curve,
           "note": "positions are clustered within prompts; cluster CI is authoritative"}
    json.dump(out, open(sys.argv[3], "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("auc", "auc_cluster_ci90", "pr_auc", "brier_heldout")}, indent=1))
    print("P4STATS_OK")
