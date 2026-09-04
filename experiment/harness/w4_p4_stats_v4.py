"""P4 statistics v4 — fixes per Codex review #2 (P0/P2). v3 file frozen.

CLI: python3 w4_p4_stats_v4.py tf_A.json tf_B.json out.json [--isotonic]

vs v3:
- strict_pair_join_v3 (duplicate keys fail closed; prompt_sha identity when
  records carry it; legacy records labelled identity-unverified).
- calibration bin merger fixed: a trailing occupied group below MIN_BIN_N is
  merged back into its left neighbor; if total calibration < MIN_BIN_N a
  single bin with small-sample status is returned; asserts every occupied
  bin >= MIN_BIN_N (unless single-bin status).
- bin metadata (margin ranges, cal_n, median margin) returned and used for
  fallback: eval positions falling in unseen bins map to the bin with the
  nearest MARGIN boundary, and fallbacks are counted.
- diagnostics from the v3 join reported verbatim.
- cluster_ci_status: descriptive; the 16-prompt threshold is reported as an
  analysis-freedom choice (A-7), never as blanket "primary".
- optional --isotonic: decreasing-isotonic (PAVA) sensitivity analysis,
  pre-specified in A-8 BEFORE first execution; calibration prompts only,
  weighted by position counts; reported alongside, never replacing, the
  binned-Beta main method.
"""
import argparse, json, math, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_v3 import (strict_pair_join_v3, manifest_aware_join,
                        manifest_aware_join_v7,
                        average_precision_grouped, mann_whitney_auc)

MIN_BIN_N = 20


def build_bins_v4(cal):
    """Quantile bins with dedup + full support guarantee."""
    ms = sorted(m for _, m, _, _ in cal)
    if len(ms) < MIN_BIN_N:
        return {"edges": [], "n_bins": 1, "status": "small-sample-single-bin",
                "binof": lambda m: 0,
                "meta": [{"bin": 0, "lo": ms[0] if ms else None,
                          "hi": ms[-1] if ms else None, "cal_n": len(ms)}]}
    edges = sorted(set(ms[int(len(ms) * i / 10)] for i in range(1, 10)))
    def rawbin(m):
        b = 0
        for e in edges:
            if m > e:
                b += 1
        return b
    from collections import Counter
    cnt = Counter(rawbin(m) for _, m, _, _ in cal)
    nraw = len(edges) + 1
    # left-to-right accumulate; then merge a deficient tail into left neighbor
    groups, acc, cur = {}, 0, 0
    for i in range(nraw):
        groups[i] = cur
        acc += cnt.get(i, 0)
        if acc >= MIN_BIN_N:
            cur += 1
            acc = 0
    if acc > 0 and cur > 0:          # trailing group under threshold
        for i in range(nraw):
            if groups[i] == cur:
                groups[i] = cur - 1
    # merge any ZERO-count group (e.g. empty trailing raw bins) into its left
    # neighbor, then renumber densely (review #3: empty meta bins are the
    # root cause of the isotonic pseudo-observation bug)
    from collections import Counter as _C
    gcnt = _C()
    for _, m, _, _ in cal:
        gcnt[groups[rawbin(m)]] += 1
    for i in range(nraw):
        g = groups[i]
        while g > 0 and gcnt.get(g, 0) == 0:
            g -= 1
        groups[i] = g
    remap = {g: j for j, g in enumerate(sorted(set(groups.values())))}
    for i in range(nraw):
        groups[i] = remap[groups[i]]
    n_bins = max(groups.values()) + 1
    # metadata per merged bin
    meta = []
    for b in range(n_bins):
        raws = [i for i in range(nraw) if groups[i] == b]
        mm = [m for _, m, _, _ in cal if groups[rawbin(m)] == b]
        meta.append({"bin": b, "cal_n": len(mm),
                     "lo": min(mm) if mm else None, "hi": max(mm) if mm else None,
                     "median_margin": sorted(mm)[len(mm)//2] if mm else None})
    occ = [x["cal_n"] for x in meta if x["cal_n"] > 0]
    assert all(n >= MIN_BIN_N for n in occ), f"bin support violation: {occ}"
    return {"edges": edges, "n_bins": n_bins, "status": "ok",
            "binof": lambda m: groups[rawbin(m)], "meta": meta}


def pava_decreasing(xs, ys, ws):
    """Weighted PAVA for a DECREASING fit of ys against ascending xs."""
    ys = [-y for y in ys]                       # increasing on negated ys
    blocks = [[y, w, [i]] for i, (y, w) in enumerate(zip(ys, ws))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            y2, w2, i2 = out.pop()
            y1, w1, i1 = out.pop()
            w = w1 + w2
            out.append([(y1 * w1 + y2 * w2) / w, w, i1 + i2])
    fit = [0.0] * len(ys)
    for y, w, idx in out:
        for i in idx:
            fit[i] = -y
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tf_a"); ap.add_argument("tf_b"); ap.add_argument("out")
    ap.add_argument("--isotonic", action="store_true")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="analyze even when the v7 identity contract fails; "
                         "output schema records allow_unverified_used")
    args = ap.parse_args()
    C = json.load(open(args.tf_a)); T = json.load(open(args.tf_b))
    both_env = all(isinstance(x, dict) and "manifest" in x for x in (C, T))
    any_env = any(isinstance(x, dict) and "manifest" in x for x in (C, T))
    if both_env:
        pairs, diag = manifest_aware_join_v7(C, T, allow_unverified=args.allow_unverified)
    elif any_env:
        pairs, diag = [], {"error": "mixed envelope/legacy inputs",
                           "identity_verified": False}
    else:
        pairs, diag = strict_pair_join_v3(C, T)
    if "error" in diag:
        json.dump({"join_diagnostics": diag, "status": "join-failed"},
                  open(args.out, "w"), indent=2)
        print("JOIN_FAILED", diag["error"]); sys.exit(1)
    prompts = sorted(set(p for p, *_ in pairs))
    n = len(pairs); flips = sum(1 for r in pairs if r[2])
    if n == 0 or flips == 0 or flips == n or len(prompts) < 2:
        json.dump({"join_diagnostics": diag, "status": "degenerate-data",
                   "n_positions": n, "n_flips": flips,
                   "n_prompts": len(prompts)}, open(args.out, "w"), indent=2)
        print("DEGENERATE_DATA"); sys.exit(0)

    by_prompt = {}
    for r in pairs:
        by_prompt.setdefault(r[0], []).append(r)

    auc = mann_whitney_auc(pairs)
    ap_ = average_precision_grouped(pairs)
    rng = random.Random(20260813)
    def cluster_boot(stat, source_prompts, B):
        vals = []
        for _ in range(B):
            sub = [r for p2 in (rng.choice(source_prompts) for _ in source_prompts)
                   for r in by_prompt[p2]]
            v = stat(sub)
            if v is not None:
                vals.append(v)
        vals.sort()
        if not vals:
            return None
        return [vals[int(len(vals) * .05)], vals[int(len(vals) * .95)]]
    auc_ci = cluster_boot(mann_whitney_auc, prompts, 5000)
    lopo = [{"left_out_prompt": p,
             "auc": mann_whitney_auc([r for p2 in prompts if p2 != p for r in by_prompt[p2]])}
            for p in prompts]
    fr = flips / n
    se = math.sqrt(fr * (1 - fr) / n)

    cal = [r for r in pairs if r[0] % 2 == 0]
    ev = [r for r in pairs if r[0] % 2 == 1]
    if not cal or not ev:
        json.dump({"join_diagnostics": diag,
                   "status": ("empty-calibration-split" if not cal
                              else "empty-evaluation-split"),
                   "n_positions": n, "n_prompts": len(prompts)},
                  open(args.out, "w"), indent=2)
        print("EMPTY_SPLIT"); sys.exit(0)
    bins = build_bins_v4(cal)
    binof = bins["binof"]
    from collections import defaultdict
    calbin = defaultdict(lambda: [0, 0])
    for _, m, f, _ in cal:
        b = calbin[binof(m)]; b[0] += f; b[1] += 1
    prob = {k: (v[0] + 1) / (v[1] + 2) for k, v in calbin.items()}
    known_meta = [x for x in bins["meta"] if x["bin"] in prob]
    prev = (sum(f for _, _, f, _ in cal) + 1) / (len(cal) + 2)

    def prob_of(m):
        b = binof(m)
        if b in prob:
            return prob[b], False
        # nearest by MARGIN boundary distance
        best = min(known_meta,
                   key=lambda x: min(abs(m - (x["lo"] if x["lo"] is not None else m)),
                                     abs(m - (x["hi"] if x["hi"] is not None else m))))
        return prob[best["bin"]], True

    fallbacks = 0
    evbin = defaultdict(lambda: [0, 0, None])
    brier = base_brier = 0.0
    for _, m, f, _ in ev:
        pgm, fb = prob_of(m)
        fallbacks += fb
        brier += (pgm - f) ** 2
        base_brier += (prev - f) ** 2
        e = evbin[binof(m)]; e[0] += f; e[1] += 1; e[2] = pgm
    curve = [{"bin": b, "pred": e[2], "obs": e[0] / e[1], "eval_n": e[1],
              "eval_flips": e[0], "cal_n": calbin.get(b, [0, 0])[1],
              "cal_flips": calbin.get(b, [0, 0])[0]}
             for b, e in sorted(evbin.items())]
    # conditional evaluation-prompt bootstrap for Brier (calibration fixed;
    # labelled conditional — see A-5; restored per review #3 P0-4)
    ev_prompts = sorted(set(p for p, *_ in ev))
    def brier_stat(sub):
        if not sub:
            return None
        s = 0.0
        for _, m, f, _ in sub:
            pgm, _ = prob_of(m)
            s += (pgm - f) ** 2
        return s / len(sub)
    ev_by_prompt = {}
    for r in ev:
        ev_by_prompt.setdefault(r[0], []).append(r)
    bb = []
    if len(ev_prompts) >= 2:
        for _ in range(2000):
            sub = [r for p2 in (rng.choice(ev_prompts) for _ in ev_prompts)
                   for r in ev_by_prompt[p2]]
            v = brier_stat(sub)
            if v is not None:
                bb.append(v)
        bb.sort()
    brier_ci = ([bb[int(len(bb) * .05)], bb[int(len(bb) * .95)]] if bb else None)

    iso = None
    if args.isotonic and bins["status"] == "ok":
        occ = [x for x in bins["meta"] if x["cal_n"] > 0]     # cal_n=0 excluded
        xs = [x["median_margin"] for x in occ]
        ys = [prob[x["bin"]] for x in occ]
        ws = [x["cal_n"] for x in occ]
        fit = pava_decreasing(xs, ys, ws)
        iso_prob = {occ[i]["bin"]: fit[i] for i in range(len(fit))}
        iso_meta = {occ[i]["bin"]: occ[i] for i in range(len(fit))}
        iso_fallbacks = 0
        def iso_prob_of(m):
            nonlocal iso_fallbacks
            b = binof(m)
            if b in iso_prob:
                return iso_prob[b]
            iso_fallbacks += 1
            best = min(iso_meta.values(),
                       key=lambda x: min(abs(m - (x["lo"] if x["lo"] is not None else m)),
                                         abs(m - (x["hi"] if x["hi"] is not None else m))))
            return iso_prob[best["bin"]]
        ib = 0.0
        iso_evbin = {}
        for _, m, f, _ in ev:
            pgm = iso_prob_of(m)
            ib += (pgm - f) ** 2
            e = iso_evbin.setdefault(binof(m), [0, 0, None]); e[0] += f; e[1] += 1; e[2] = pgm
        iso = {"method": "decreasing PAVA on occupied calibration bins (A-8, post-data sensitivity)",
               "brier_heldout": ib / len(ev) if ev else None,
               "fallback_count": iso_fallbacks,
               "heldout_curve": [{"bin": b, "iso_pred": e[2], "obs": e[0] / e[1],
                                  "eval_n": e[1], "eval_flips": e[0]}
                                 for b, e in sorted(iso_evbin.items())],
               "bin_fits": [{"bin": occ[i]["bin"], "raw": ys[i], "iso": fit[i],
                             "cal_n": occ[i]["cal_n"]} for i in range(len(fit))],
               "role": "sensitivity only; binned-Beta remains the main method"}

    out = {
        "schema": "p4-stats-v7",
        "join_diagnostics": diag,
        "n_prompts": len(prompts), "n_positions": n, "n_flips": flips,
        "flip_rate": fr,
        "roc_auc": auc,
        "roc_auc_cluster_ci90": auc_ci,
        "cluster_ci_note": ("prompt-cluster bootstrap; interpretation depends on "
                            "cluster count (%d prompts); the 16-prompt reporting "
                            "threshold is an analysis-freedom choice recorded in A-7"
                            % len(prompts)),
        "naive_positionwise_flip_se_CONTRAST_ONLY": se,
        "lopo_auc_influence_diagnostic": lopo,
        "average_precision_grouped_ties": ap_,
        "brier_heldout": brier / len(ev) if ev else None,
        "brier_constant_prevalence_baseline": base_brier / len(ev) if ev else None,
        "brier_prompt_bootstrap_ci90_CONDITIONAL": brier_ci,
        "calibration_bins": {"status": bins["status"], "meta": bins["meta"]},
        "calibration_fallback_count": fallbacks,
        "calibration_curve_heldout": curve,
        "isotonic_sensitivity": iso,
        "smoothing": "Beta(1,1); occupied merged bins >= %d cal positions (asserted)" % MIN_BIN_N,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("roc_auc", "average_precision_grouped_ties",
        "brier_heldout", "brier_constant_prevalence_baseline",
        "calibration_fallback_count")}, indent=1, default=str))
    print("join identity_verified:", diag["identity_verified"])
    print("P4STATS_V4_OK")


if __name__ == "__main__":
    main()
