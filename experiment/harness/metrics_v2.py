"""Shared analysis-semantics v2 — fixes per Codex review #1.5 (2026-08-13).

Semantics fixed BEFORE any re-analysis (recorded in pre-registration A-5):
- bitwise divergence and numerical ULP distance are SEPARATE quantities;
  n_differ is bit-pattern inequality (catches -0/+0), ULP histograms are
  computed only where both sides are finite, and the numerical mapping
  deliberately collapses signed zero (distance(-0,+0)=0) — documented.
- non-finite elements are counted (nan_any / inf_any / finite_mismatch),
  never fed into ULP histograms.
- near-zero region (|ref| < 0.01 * rms_ref, reference arm = first argument)
  is excluded from relative-error stats but gets its own sub-report
  (bitwise count, finite ULP histogram, absolute-error quantiles).
- abs/RMS reported as max AND p50/p99 AND rmse/rms (never max alone).
- average precision uses grouped ties (threshold groups), None when no
  positives.
- pair alignment is a strict (prompt_id, pos) dictionary join; missing /
  duplicate / misordered entries are counted and reported, never silently
  zipped away.
"""
from __future__ import annotations


def bf16_bitwise_differ(a, b):
    import torch
    return a.view(torch.int16) != b.view(torch.int16)


def bf16_ulp_distance_finite(a, b):
    """Numerical ULP distance on the finite-and-finite subset only.
    Returns (dist_int32_tensor_on_subset, finite_mask). Signed zero collapses
    to distance 0 by design (documented)."""
    import torch
    fin = torch.isfinite(a.float()) & torch.isfinite(b.float())
    ia = a[fin].view(torch.int16).to(torch.int32)
    ib = b[fin].view(torch.int16).to(torch.int32)
    oa = torch.where(ia >= 0, ia + 2**15, 2**15 - (ia & 0x7FFF))
    ob = torch.where(ib >= 0, ib + 2**15, 2**15 - (ib & 0x7FFF))
    return (oa - ob).abs(), fin


def divergence_report(ref, other):
    """Full v2 comparison of two bf16 tensors; ref is the reference arm."""
    import torch
    bd = bf16_bitwise_differ(ref, other)
    ud, fin = bf16_ulp_distance_finite(ref, other)
    rf, of = ref.float(), other.float()
    nan_any = int((torch.isnan(rf) | torch.isnan(of)).sum())
    inf_any = int((torch.isinf(rf) | torch.isinf(of)).sum())
    finite_mismatch = int((torch.isfinite(rf) != torch.isfinite(of)).sum())
    rms = rf.pow(2).mean().sqrt()
    near0 = rf.abs() < 0.01 * rms
    d = (rf - of).abs()

    def q(t, p):
        return float(t.flatten().kthvalue(max(1, int(t.numel() * p))).values) if t.numel() else None

    main = ~near0 & torch.isfinite(rf) & torch.isfinite(of)
    ud_all = torch.zeros_like(bd, dtype=torch.int32)
    ud_all[fin] = ud
    def hist(mask):
        u = ud_all[mask & fin]
        return {"0": int((u == 0).sum()), "1": int((u == 1).sum()),
                "2": int((u == 2).sum()), "gt2": int((u > 2).sum())}
    rel_main = (d / rf.abs().clamp(min=1e-30))[main & bd]
    return {
        "n_total": int(bd.numel()),
        "n_bitwise_differ": int(bd.sum()),
        "bitwise_identical": bool(bd.sum() == 0),
        "nan_any": nan_any, "inf_any": inf_any,
        "finite_mismatch": finite_mismatch,
        "ulp_hist_main": hist(main),
        "max_ulp_distance_finite": int(ud.max()) if ud.numel() else 0,
        "abs_over_rms": {"max": float((d / rms).max()),
                         "p50": q(d / rms, 0.5), "p99": q(d / rms, 0.99),
                         "rmse_over_rms": float(d.pow(2).mean().sqrt() / rms)},
        "rel_of_differing_main": {"p50": q(rel_main, 0.5) if rel_main.numel() else None,
                                  "max": float(rel_main.max()) if rel_main.numel() else None,
                                  "n": int(rel_main.numel())},
        "near_zero": {"n": int(near0.sum()),
                      "n_bitwise_differ": int((bd & near0).sum()),
                      "ulp_hist": hist(near0),
                      "abs_err_max": float(d[near0].max()) if int(near0.sum()) else None,
                      "abs_err_p99": q(d[near0], 0.99) if int(near0.sum()) else None},
        "note": "numerical ULP collapses signed zero; bitwise count does not",
    }


def strict_pair_join(C, T):
    """(prompt_id,pos)-keyed join of two teacher-forcing record lists.
    Returns (pairs, diagnostics). pairs: list of (prompt_id, margin, flip,
    chosen_lp_delta). Missing/duplicate/misaligned entries are counted."""
    def index(recs):
        m, dup = {}, 0
        for pi, seq in enumerate(recs):
            for s in seq or []:
                if s is None:
                    continue
                k = (pi, s["pos"])
                if k in m:
                    dup += 1
                m[k] = s
        return m, dup
    a, dup_a = index(C)
    b, dup_b = index(T)
    keys = sorted(set(a) & set(b))
    pairs = []
    for k in keys:
        sa, sb = a[k], b[k]
        if sa.get("margin") is None:
            continue
        delta = (abs(sa["chosen_lp"] - sb["chosen_lp"])
                 if sa.get("chosen_lp") is not None and sb.get("chosen_lp") is not None
                 else None)
        pairs.append((k[0], sa["margin"], sa["top1_id"] != sb["top1_id"], delta))
    diag = {"n_joined": len(keys), "only_in_a": len(set(a) - set(b)),
            "only_in_b": len(set(b) - set(a)), "dup_a": dup_a, "dup_b": dup_b,
            "n_prompts_a": len(C), "n_prompts_b": len(T)}
    return pairs, diag


def average_precision_grouped(pairs):
    """Non-interpolated AP with grouped ties. Score = -margin (low margin
    predicts flip). Returns None if there are no positives."""
    P = sum(1 for r in pairs if r[2])
    if P == 0:
        return None
    from itertools import groupby
    xs = sorted(pairs, key=lambda r: r[1])           # ascending margin
    tp = fp = 0
    ap = 0.0
    for _, grp in groupby(xs, key=lambda r: r[1]):   # one threshold per group
        g = list(grp)
        gtp = sum(1 for r in g if r[2])
        gfp = len(g) - gtp
        tp += gtp; fp += gfp
        if gtp:
            ap += (tp / (tp + fp)) * (gtp / P)
    return ap


def mann_whitney_auc(pairs):
    import bisect
    pos = sorted(m for _, m, f, _ in pairs if f)
    neg = sorted(m for _, m, f, _ in pairs if not f)
    if not pos or not neg:
        return None
    s = sum(bisect.bisect_left(neg, m)
            + 0.5 * (bisect.bisect_right(neg, m) - bisect.bisect_left(neg, m))
            for m in pos)
    return 1 - s / (len(pos) * len(neg))
