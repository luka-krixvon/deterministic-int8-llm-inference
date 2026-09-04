"""metrics v3 — fixes per Codex review #2 (P0). Supersedes metrics_v2 for all
new analyses; metrics_v2.py is frozen for lineage.

Changes vs v2:
- input validation: both tensors must be bf16, same shape, same device.
- reference RMS uses FINITE reference elements only; absolute-error and
  RMSE/RMS stats use finite/finite pairs only.
- zero/absent finite reference -> explicit status + None fields; a pure
  absolute-error summary is always provided; NO JSON NaN ever leaves this
  module (json_safe guard on every float).
- non-finite classification is EXCLUSIVE (both_finite / ref_only_nonfinite /
  other_only_nonfinite / both_nonfinite); overlapping nan_any/inf_any kept
  with an explicit note.
- near-zero mask restricted to finite reference; undefined when RMS invalid.
- strict join: duplicate keys FAIL CLOSED (key invalidated + counted);
  supports v3 teacher-forcing records carrying prompt_sha for identity
  verification; legacy list records are joined with identity_verified=False.
- diagnostics: n_common_keys / n_pairs_used / n_skipped_missing_margin.
"""
from __future__ import annotations
import math


def _js(x):
    """json-safe float: non-finite -> None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _validate(a, b):
    import torch
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(f"expected bf16 pair, got {a.dtype}/{b.dtype}")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.device != b.device:
        raise ValueError(f"device mismatch {a.device} vs {b.device}")


def bf16_bitwise_differ(a, b):
    import torch
    _validate(a, b)
    return a.view(torch.int16) != b.view(torch.int16)


def bf16_ulp_distance_finite(a, b):
    import torch
    _validate(a, b)
    fin = torch.isfinite(a.float()) & torch.isfinite(b.float())
    ia = a[fin].view(torch.int16).to(torch.int32)
    ib = b[fin].view(torch.int16).to(torch.int32)
    oa = torch.where(ia >= 0, ia + 2**15, 2**15 - (ia & 0x7FFF))
    ob = torch.where(ib >= 0, ib + 2**15, 2**15 - (ib & 0x7FFF))
    return (oa - ob).abs(), fin


def divergence_report(ref, other):
    import torch
    _validate(ref, other)
    bd = bf16_bitwise_differ(ref, other)
    ud, fin_pair = bf16_ulp_distance_finite(ref, other)
    rf, of = ref.float(), other.float()
    fr, fo = torch.isfinite(rf), torch.isfinite(of)
    cls = {  # exclusive classification
        "both_finite": int((fr & fo).sum()),
        "ref_only_nonfinite": int((~fr & fo).sum()),
        "other_only_nonfinite": int((fr & ~fo).sum()),
        "both_nonfinite": int((~fr & ~fo).sum()),
    }
    n_fin_ref = int(fr.sum())
    status = "ok"
    rms = None
    if n_fin_ref == 0:
        status = "no-finite-reference"
    else:
        # float64: squares/differences of extreme finite bf16 overflow fp32
        rms = float(ref.double()[fr].pow(2).mean().sqrt())
        if rms == 0.0:
            status = "zero-reference-rms"
        elif not math.isfinite(rms):
            status = "nonfinite-rms-computation"
            rms = None

    d64 = (ref.double() - other.double()).abs()
    d_pair = d64[fin_pair]                      # finite/finite only, fp64

    def q(t, p):
        return _js(t.flatten().kthvalue(max(1, int(t.numel() * p))).values) if t.numel() else None

    abs_summary = {"max": _js(d_pair.max()) if d_pair.numel() else None,
                   "p50": q(d_pair, 0.5), "p99": q(d_pair, 0.99),
                   "n_pairs": int(fin_pair.sum())}

    ud_all = torch.zeros_like(bd, dtype=torch.int32)
    ud_all[fin_pair] = ud
    def hist(mask):
        u = ud_all[mask & fin_pair]
        return {"0": int((u == 0).sum()), "1": int((u == 1).sum()),
                "2": int((u == 2).sum()), "gt2": int((u > 2).sum())}

    if status == "ok":
        near0 = fr & (rf.abs() < 0.01 * rms)
        main = fin_pair & ~near0
        d_main = d64[main]
        rel_main = (d64 / ref.double().abs().clamp(min=1e-300))[main & bd]
        # normalized summary over ALL finite/finite pairs (review #3 P0-1)
        abs_over_rms = {"max": _js(d_pair.max() / rms) if d_pair.numel() else None,
                        "p50": _js(q(d_pair, 0.5) / rms) if d_pair.numel() and q(d_pair, 0.5) is not None else None,
                        "p99": _js(q(d_pair, 0.99) / rms) if d_pair.numel() and q(d_pair, 0.99) is not None else None,
                        "rmse_over_rms": _js(d_pair.pow(2).mean().sqrt() / rms) if d_pair.numel() else None}
        abs_over_rms_main_non_nearzero = {
            "max": _js(d_main.max() / rms) if d_main.numel() else None,
            "rmse_over_rms": _js(d_main.pow(2).mean().sqrt() / rms) if d_main.numel() else None}
        nz_pair = d64[near0 & fin_pair]
        near_zero = {"n": int(near0.sum()),
                     "n_bitwise_differ": int((bd & near0).sum()),
                     "ulp_hist": hist(near0),
                     "abs_err_max": _js(nz_pair.max()) if nz_pair.numel() else None,
                     "abs_err_p99": q(nz_pair, 0.99)}
        ulp_hist_main = hist(main)
        rel_summary = {"p50": q(rel_main, 0.5) if rel_main.numel() else None,
                       "max": _js(rel_main.max()) if rel_main.numel() else None,
                       "n": int(rel_main.numel())}
    else:
        near_zero = None
        abs_over_rms = {"max": None, "p50": None, "p99": None, "rmse_over_rms": None}
        abs_over_rms_main_non_nearzero = {"max": None, "rmse_over_rms": None}
        ulp_hist_main = hist(fin_pair)     # all finite pairs, no near-zero split
        rel_summary = {"p50": None, "max": None, "n": 0}

    return {
        "status": status,
        "n_total": int(bd.numel()),
        "n_bitwise_differ": int(bd.sum()),
        "bitwise_identical": bool(bd.sum() == 0),
        "finiteness_exclusive": cls,
        "nan_any": int((torch.isnan(rf) | torch.isnan(of)).sum()),
        "inf_any": int((torch.isinf(rf) | torch.isinf(of)).sum()),
        "counts_note": "nan_any/inf_any overlap by position; finiteness_exclusive is the partition",
        "rms_finite_ref": _js(rms),
        "abs_err_finite_pairs": abs_summary,
        "ulp_hist_main": ulp_hist_main,
        "max_ulp_distance_finite": int(ud.max()) if ud.numel() else 0,
        "abs_over_rms": abs_over_rms,
        "abs_over_rms_main_non_nearzero": abs_over_rms_main_non_nearzero,
        "rel_of_differing_main": rel_summary,
        "near_zero": near_zero,
        "note": "numerical ULP collapses signed zero; bitwise count does not",
    }


def strict_pair_join_v3(C, T):
    """v3 join. Records may be (new) {"prompt_sha":…, "steps":[…]} or
    (legacy) plain step lists. Duplicate (prompt,pos) keys FAIL CLOSED:
    the key is invalidated and counted, never used."""
    def norm(recs):
        if isinstance(recs, dict) and "records" in recs:
            recs = recs["records"]
        out = []
        for r in recs:
            if isinstance(r, dict) and "steps" in r:
                out.append((r.get("prompt_sha"), r["steps"]))
            else:
                out.append((None, r))
        return out
    Cn, Tn = norm(C), norm(T)
    has_sha = [s is not None for s, _ in Cn] + [s is not None for s, _ in Tn]
    if any(has_sha) and not all(has_sha):
        return [], {"error": "mixed-schema: some records carry prompt_sha and some do not",
                    "identity_verified": False}
    identity_verified = all(has_sha) and bool(has_sha)
    if identity_verified:
        sa = [s for s, _ in Cn]; sb = [s for s, _ in Tn]
        if sa != sb:
            return [], {"error": "prompt_sha sequence mismatch",
                        "identity_verified": False,
                        "sha_mismatch_positions": [i for i, (x, y) in enumerate(zip(sa, sb)) if x != y]}
    def index(recs):
        m, bad = {}, set()
        for pi, (_, seq) in enumerate(recs):
            for s in seq or []:
                if s is None:
                    continue
                k = (pi, s["pos"])
                if k in m:
                    bad.add(k)
                m[k] = s
        for k in bad:
            m.pop(k, None)          # fail closed
        return m, len(bad)
    a, dup_a = index(Cn)
    b, dup_b = index(Tn)
    keys = sorted(set(a) & set(b))
    pairs, skipped = [], 0
    for k in keys:
        sa_, sb_ = a[k], b[k]
        if sa_.get("margin") is None:
            skipped += 1
            continue
        delta = (abs(sa_["chosen_lp"] - sb_["chosen_lp"])
                 if sa_.get("chosen_lp") is not None and sb_.get("chosen_lp") is not None else None)
        pairs.append((k[0], sa_["margin"], sa_["top1_id"] != sb_["top1_id"], delta))
    diag = {"n_common_keys": len(keys), "n_pairs_used": len(pairs),
            "n_skipped_missing_margin": skipped,
            "duplicate_keys_invalidated_a": dup_a,
            "duplicate_keys_invalidated_b": dup_b,
            "only_in_a": len(set(a) - set(b)), "only_in_b": len(set(b) - set(a)),
            "n_prompts_a": len(Cn), "n_prompts_b": len(Tn),
            "identity_verified": identity_verified,
            "identity_note": ("prompt_sha verified per sequence" if identity_verified
                              else "LEGACY records: index-based identity only; "
                                   "prompt identity NOT verified")}
    return pairs, diag


# re-export unchanged, correct implementations from v2
from metrics_v2 import average_precision_grouped, mann_whitney_auc  # noqa: E402,F401


def manifest_aware_join(A, B):
    """v6 identity contract (Codex review #4 P0). BOTH arms must be tf_v5
    envelopes {"manifest","records"}; envelope/legacy mixtures fail closed.
    Verifies, separately: record-level sha sequence (FULL sha256), prompt-list
    manifest sha, rails sha, and model identity via checkpoint digest
    (local paths are diagnostic only). identity_verified = all-of."""
    def unwrap(x, arm):
        if not (isinstance(x, dict) and "manifest" in x and "records" in x):
            return None, None, f"{arm}-arm is not a manifest envelope"
        return x["manifest"], x["records"], None
    ma, ra, e1 = unwrap(A, "A")
    mb, rb, e2 = unwrap(B, "B")
    if e1 or e2:
        return [], {"error": "envelope-required: " + (e1 or e2),
                    "identity_verified": False}
    def counts_ok(m, r):
        return (m.get("requested_prompts") is not None
                and m.get("requested_prompts") == m.get("actual_prompts") == len(r))
    flags = {
        "record_counts_verified": bool(counts_ok(ma, ra) and counts_ok(mb, rb)
                                       and len(ra) == len(rb)),
        "prompt_manifest_verified": bool(ma.get("prompt_list_sha256")
                                         and ma.get("prompt_list_sha256") == mb.get("prompt_list_sha256")),
        "rails_verified": bool(ma.get("rails_sha256")
                               and ma.get("rails_sha256") == mb.get("rails_sha256")),
        "model_identity_verified": bool(ma.get("checkpoint_digest")
                                        and ma.get("checkpoint_digest") == mb.get("checkpoint_digest")),
    }
    sa = [r.get("prompt_sha") for r in ra]
    sb = [r.get("prompt_sha") for r in rb]
    full = all(isinstance(s, str) and len(s) == 64 for s in sa + sb)
    flags["record_identity_verified"] = bool(full and sa == sb)
    pairs, diag = strict_pair_join_v3({"records": ra}, {"records": rb})
    if "error" in diag:
        diag.update(flags)
        diag["identity_verified"] = False
        return [], diag
    diag.update(flags)
    diag["identity_verified"] = all(flags.values())
    diag["identity_note"] = "manifest-aware v6 contract: all-of five verifications"
    diag["model_paths_diagnostic"] = [ma.get("model"), mb.get("model")]
    return pairs, diag


import re as _re
_SHA256_RE = _re.compile(r"^[0-9a-f]{64}$")


def is_sha256(s):
    return isinstance(s, str) and bool(_SHA256_RE.match(s))


# fields that constitute IDENTITY (must be equal AND valid sha where noted)
_V7_IDENTITY = [
    ("weights_verified", "checkpoint_digest", True),
    ("quant_config_verified", "quantization_config_sha256", True),
    ("tokenizer_verified", "tokenizer_files_sha256", True),
    ("parent_revision_verified", "parent_revision", False),
    ("runtime_verified", "vllm_version", False),
    ("prompt_manifest_verified", "prompt_list_sha256", True),
    ("rails_verified", "rails_sha256", True),
]
# fields ALLOWED (and expected) to differ between arms — the declared treatment
_V7_TREATMENT_FIELDS = {"arm", "kernel_class", "kernel_selection_env",
                        "kernel_log_sha256", "kernel_log_lines",
                        "command_sha256", "model"}
_V7_TREATMENT_EVIDENCE = ("arm", "kernel_class", "kernel_log_sha256")


def manifest_aware_join_v7(A, B, allow_unverified=False):
    """v7 identity/treatment contract (Codex review #5 P0-3).

    Identity: weights, quantization config, tokenizer, parent revision,
    runtime, prompt manifest, rails, record sha sequence — each must be
    present, equal across arms, and (for digest fields) a 64-hex sha256.
    Treatment: arms MAY differ only in the declared treatment fields; both
    arms must carry kernel-selection evidence (arm + kernel_class +
    kernel_log_sha256), else treatment_identity_verified=False.
    Default is FAIL CLOSED: any required flag False -> no pairs returned,
    unless allow_unverified=True (callers must surface that in their schema).
    """
    def unwrap(x, arm):
        if not (isinstance(x, dict) and "manifest" in x and "records" in x):
            return None, None, f"{arm}-arm is not a manifest envelope"
        return x["manifest"], x["records"], None
    ma, ra, e1 = unwrap(A, "A")
    mb, rb, e2 = unwrap(B, "B")
    if e1 or e2:
        return [], {"error": "envelope-required: " + (e1 or e2),
                    "identity_verified": False, "schema": "join-v7"}
    flags = {}
    for flag, field, needs_sha in _V7_IDENTITY:
        va, vb = ma.get(field), mb.get(field)
        ok = va is not None and va == vb
        if needs_sha:
            ok = ok and is_sha256(va)
        flags[flag] = bool(ok)
    def counts_ok(m, r):
        return (m.get("requested_prompts") is not None
                and m.get("requested_prompts") == m.get("actual_prompts") == len(r))
    flags["record_counts_verified"] = bool(counts_ok(ma, ra) and counts_ok(mb, rb)
                                           and len(ra) == len(rb))
    sa = [r.get("prompt_sha") for r in ra]
    sb = [r.get("prompt_sha") for r in rb]
    flags["record_sequence_verified"] = bool(
        all(is_sha256(s) for s in sa + sb) and sa == sb)
    # treatment: evidence present on both arms + no undeclared differences
    ev_ok = all(ma.get(k) for k in _V7_TREATMENT_EVIDENCE) and             all(mb.get(k) for k in _V7_TREATMENT_EVIDENCE)
    undeclared = [k for k in set(ma) | set(mb)
                  if k not in _V7_TREATMENT_FIELDS
                  and not any(k == f for _, f, _ in _V7_IDENTITY)
                  and k not in ("requested_prompts", "actual_prompts", "generator")
                  and ma.get(k) != mb.get(k)]
    flags["treatment_identity_verified"] = bool(ev_ok and not undeclared)
    identity = all(flags.values())
    diag = dict(flags)
    diag["identity_verified"] = identity
    diag["schema"] = "join-v7"
    diag["undeclared_differences"] = undeclared
    diag["treatment_declared"] = {"A": {k: ma.get(k) for k in ("arm", "kernel_class")},
                                  "B": {k: mb.get(k) for k in ("arm", "kernel_class")}}
    diag["model_paths_diagnostic"] = [ma.get("model"), mb.get("model")]
    if not identity and not allow_unverified:
        diag["error"] = "identity-contract-failed (pass allow_unverified to analyze anyway)"
        return [], diag
    pairs, jd = strict_pair_join_v3({"records": ra}, {"records": rb})
    if "error" in jd:
        jd.update(diag); jd["identity_verified"] = False
        return [], jd
    jd.update(diag)
    jd["allow_unverified_used"] = bool(not identity and allow_unverified)
    return pairs, jd
