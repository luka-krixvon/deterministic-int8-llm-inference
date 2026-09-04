"""The seven conformance checks of Section X, as executable predicates.

The paper states these checks in a table and the study's scripts instantiate them
in scattered form. Measuring their sensitivity (pre-registration A-10, P5) requires
them as one callable suite with a fixed contract, which is what this module is.

Contract. A check receives two arms and returns a Verdict. `fired` means the check
reports a violation. `applicable` is False when the check's precondition does not
hold for the observation at hand -- an inapplicable check is not a pass and must not
be counted as one. Checks 1-5 are exact: `fired` is a bitwise or integer fact.
Checks 6-7 are tolerance-based and carry their threshold in the Verdict so that a
report cannot silently move it.

Nothing here touches a GPU. The arms are observations, however they were produced,
so the same suite runs against real kernels and against the fault injector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

# Three-tier int8 product bound, as stated in Section III. 16129 is what symmetric
# [-127,127] quantization actually reaches, 16256 admits a -128 activation, 16384 is
# the unrestricted (-128)^2. The default is the conservative middle tier the paper
# uses in Equation (1); a caller measuring the unrestricted case passes 16384.
PRODUCT_BOUND_SYMMETRIC = 16129
PRODUCT_BOUND_WITH_MIN_ACT = 16256
PRODUCT_BOUND_UNRESTRICTED = 16384

INT32_LIMIT = 2 ** 31
FP32_EXACT_INT_LIMIT = 2 ** 24


@dataclass
class Arm:
    """One implementation's observation of one layer.

    a_q, w_q, s_a, s_w are the operands as that arm received them -- not as they
    were intended. Check 1 exists because those can differ.

    acc is the INT32 accumulator if the arm exposes it, else None; a check that
    needs it reports applicable=False rather than guessing.
    """

    name: str
    a_q: torch.Tensor          # int8, (M, K)
    w_q: torch.Tensor          # int8, (N, K)
    s_a: torch.Tensor          # float32, (M, 1) per-token
    s_w: torch.Tensor          # float32, (1, N) per-channel
    out: torch.Tensor          # the arm's output, typically bfloat16, (M, N)
    acc: Optional[torch.Tensor] = None   # int32 or int64, (M, N)


@dataclass
class Verdict:
    check: str
    fired: bool
    applicable: bool
    exact: bool
    detail: str
    threshold: Optional[float] = None
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "fired": self.fired,
            "applicable": self.applicable,
            "exact": self.exact,
            "detail": self.detail,
            "threshold": self.threshold,
            "evidence": self.evidence,
        }


def _bitwise_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Bitwise equality, so that -0.0 != 0.0 and NaN payloads are distinguished.

    Plain == would call two signed zeros equal and two NaNs unequal, both of which
    hide exactly the kind of epilogue difference this suite is looking for.
    """
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    xb = x.detach().cpu().contiguous().view(torch.uint8)
    yb = y.detach().cpu().contiguous().view(torch.uint8)
    return bool(torch.equal(xb, yb))


def _to_ordered_int(t: torch.Tensor) -> torch.Tensor:
    """Map a float tensor to a monotonically ordered integer, for ULP distance.

    This is the same mapping as ``metrics_v3.bf16_ulp_distance_finite`` (lines
    54-55), written the same way on purpose: the published max-ULP-distance result
    depends on it, and a second implementation that disagreed would make the two
    numbers incomparable.

    The two signed zeros both land on the midpoint, so they are numerically zero
    apart while remaining bitwise unequal. That separation is deliberate -- check 5
    is bitwise and catches signed zeros, check 6 is numerical and should not report
    a spacing where there is no value difference.
    """
    if t.dtype in (torch.bfloat16, torch.float16):
        bits = t.detach().cpu().contiguous().view(torch.int16).to(torch.int64)
        half, mask = 1 << 15, 0x7FFF
    elif t.dtype == torch.float32:
        bits = t.detach().cpu().contiguous().view(torch.int32).to(torch.int64)
        half, mask = 1 << 31, 0x7FFFFFFF
    else:
        raise TypeError(f"ULP distance undefined for {t.dtype}")
    return torch.where(bits >= 0, bits + half, half - (bits & mask))


# ---------------------------------------------------------------- checks 1-5, exact

def check1_shared_operands(a: Arm, b: Arm) -> Verdict:
    """Do both arms hold bitwise-identical int8 operands and scales?

    Every later check interprets a difference as coming from the implementation.
    That reading is only available once this one passes, so a violation here means
    the rest of the suite reports nothing about implementations.
    """
    parts = {
        "a_q": _bitwise_equal(a.a_q, b.a_q),
        "w_q": _bitwise_equal(a.w_q, b.w_q),
        "s_a": _bitwise_equal(a.s_a, b.s_a),
        "s_w": _bitwise_equal(a.s_w, b.s_w),
    }
    bad = [k for k, ok in parts.items() if not ok]
    return Verdict(
        check="shared_operands",
        fired=bool(bad),
        applicable=True,
        exact=True,
        detail=("operands identical" if not bad else f"differ in {', '.join(bad)}"),
        evidence=parts,
    )


def check2_int32_no_overflow(K: int, product_bound: int = PRODUCT_BOUND_WITH_MIN_ACT) -> Verdict:
    """Can the INT32 accumulator wrap at this reduction depth?

    A violation is not a tolerance matter: accumulation may wrap, the integer alibi
    does not apply at all, and no localization is licensed for that layer.
    """
    worst = product_bound * int(K)
    fired = not (worst < INT32_LIMIT)
    return Verdict(
        check="int32_no_overflow",
        fired=fired,
        applicable=True,
        exact=True,
        detail=f"{product_bound} * K={K} = {worst} {'>=' if fired else '<'} 2^31",
        evidence={"K": int(K), "product_bound": product_bound, "worst_case": worst,
                  "max_K_admissible": (INT32_LIMIT - 1) // product_bound},
    )


def check3_lossless_fp32_entry(acc_ref: torch.Tensor) -> Verdict:
    """Does every accumulator reach the epilogue exactly representable in float32?

    Above 2^24 the universal no-loss guarantee is gone, but individual values may
    still be exact (2^24+1 rounds, 2^24+2 does not), so a violation means judge per
    value or widen the tolerance -- not discard the layer.
    """
    m = int(acc_ref.abs().max().item()) if acc_ref.numel() else 0
    fired = m > FP32_EXACT_INT_LIMIT
    headroom = (math.log2(FP32_EXACT_INT_LIMIT / m) if m > 0 else float("inf"))
    return Verdict(
        check="lossless_fp32_entry",
        fired=fired,
        applicable=True,
        exact=True,
        detail=f"max|acc| = {m} {'>' if fired else '<='} 2^24",
        evidence={"max_abs_acc": m, "headroom_bits": headroom},
    )


def check4_exact_accumulator(a: Arm, b: Arm, acc_ref: Optional[torch.Tensor] = None) -> Verdict:
    """Do both arms produce the reference INT32 accumulator exactly?

    Under checks 1 and 2 this cannot fail for arithmetic reasons, so a violation
    points at an integer-path defect rather than at rounding.
    """
    if a.acc is None or b.acc is None:
        return Verdict(
            check="exact_accumulator", fired=False, applicable=False, exact=True,
            detail="accumulator not exposed by at least one arm",
            evidence={"a_exposed": a.acc is not None, "b_exposed": b.acc is not None},
        )
    ab = a.acc.detach().cpu().to(torch.int64)
    bb = b.acc.detach().cpu().to(torch.int64)
    arms_agree = bool(torch.equal(ab, bb))
    ev = {"arms_agree": arms_agree}
    ref_ok = None
    if acc_ref is not None:
        rr = acc_ref.detach().cpu().to(torch.int64)
        ref_ok = bool(torch.equal(ab, rr)) and bool(torch.equal(bb, rr))
        ev["matches_reference"] = ref_ok
        ev["max_abs_diff_vs_ref"] = int((ab - rr).abs().max().item()) if ab.numel() else 0
    fired = (not arms_agree) or (ref_ok is False)
    return Verdict(
        check="exact_accumulator", fired=fired, applicable=True, exact=True,
        detail=("accumulators identical" if not fired else "accumulator mismatch"),
        evidence=ev,
    )


def check5_pow2_scale_identity(a: Arm, b: Arm) -> Verdict:
    """Under power-of-two scales, are the two outputs bitwise identical?

    Applicable only when every scale actually is a power of two; the identity
    rnd(x)*2^k = rnd(x*2^k) is what removes the epilogue's freedom, and it does not
    hold in the subnormal range, which is checked rather than assumed.
    """
    def all_pow2(t: torch.Tensor) -> bool:
        f = t.detach().cpu().to(torch.float64).flatten()
        if f.numel() == 0:
            return False
        if not bool(torch.isfinite(f).all()) or bool((f <= 0).any()):
            return False
        return bool(torch.equal(torch.log2(f), torch.round(torch.log2(f))))

    scales_pow2 = all(all_pow2(t) for t in (a.s_a, a.s_w, b.s_a, b.s_w))
    if not scales_pow2:
        return Verdict(
            check="pow2_scale_identity", fired=False, applicable=False, exact=True,
            detail="not all scales are positive powers of two",
            evidence={"scales_pow2": False},
        )
    finite = bool(torch.isfinite(a.out.to(torch.float32)).all() and
                  torch.isfinite(b.out.to(torch.float32)).all())
    subnormal = _has_subnormal(a.out) or _has_subnormal(b.out)
    eq = _bitwise_equal(a.out, b.out)
    return Verdict(
        check="pow2_scale_identity", fired=not eq, applicable=True, exact=True,
        detail=("outputs bitwise identical" if eq else "outputs differ under pow2 scales"),
        evidence={"scales_pow2": True, "bitwise_equal": eq, "all_finite": finite,
                  "subnormal_present": subnormal},
    )


def _has_subnormal(t: torch.Tensor) -> bool:
    f = t.detach().cpu().to(torch.float32)
    tiny = torch.finfo(t.dtype if t.dtype != torch.bfloat16 else torch.bfloat16).tiny
    nz = f != 0
    return bool((nz & (f.abs() < float(tiny))).any())


# ------------------------------------------------------- checks 6-7, tolerance-based

def check6_real_scale_tolerance(a: Arm, b: Arm, max_ulp: int = 1) -> Verdict:
    """Under the checkpoint's real scales, do the outputs stay within max_ulp?

    Exceeding the tolerance suggests reduced intermediate precision, fusion, or
    another departure from single-rounding semantics -- not mere double rounding.
    The threshold travels in the Verdict so a report cannot move it silently.
    """
    if a.out.shape != b.out.shape or a.out.dtype != b.out.dtype:
        return Verdict(
            check="real_scale_tolerance", fired=True, applicable=True, exact=False,
            detail="shape or dtype mismatch", threshold=float(max_ulp),
            evidence={"a": [list(a.out.shape), str(a.out.dtype)],
                      "b": [list(b.out.shape), str(b.out.dtype)]},
        )
    fa = a.out.detach().cpu().to(torch.float32)
    fb = b.out.detach().cpu().to(torch.float32)
    finite = torch.isfinite(fa) & torch.isfinite(fb)
    if not bool(finite.any()):
        return Verdict(
            check="real_scale_tolerance", fired=False, applicable=False, exact=False,
            detail="no finite/finite pair to compare", threshold=float(max_ulp),
            evidence={"n_finite_pairs": 0},
        )
    ia = _to_ordered_int(a.out)[finite.flatten().reshape(a.out.shape)]
    ib = _to_ordered_int(b.out)[finite.flatten().reshape(b.out.shape)]
    ulp = (ia - ib).abs()
    worst = int(ulp.max().item())
    n_over = int((ulp > max_ulp).sum().item())
    bitwise = _bitwise_equal(a.out, b.out)
    return Verdict(
        check="real_scale_tolerance", fired=worst > max_ulp, applicable=True, exact=False,
        detail=f"max ulp distance {worst} (tolerance {max_ulp})",
        threshold=float(max_ulp),
        evidence={"max_ulp": worst, "n_over_tolerance": n_over,
                  "n_finite_pairs": int(finite.sum().item()),
                  "n_differing": int((ulp > 0).sum().item()),
                  "bitwise_equal": bitwise},
    )


def check7_token_level_risk(margins: torch.Tensor, flips: torch.Tensor,
                            max_flip_rate: float = 0.05) -> Verdict:
    """Is token-level disagreement risk ranked and bounded on this workload?

    This check is tolerance-based by construction and bounds nothing on its own:
    it reports the observed flip rate and whether the margin ranks flips at all.
    A low rate on one workload does not transfer to another.
    """
    if margins.numel() == 0 or flips.numel() != margins.numel():
        return Verdict(
            check="token_level_risk", fired=False, applicable=False, exact=False,
            detail="no positions, or margin/flip length mismatch",
            threshold=max_flip_rate,
            evidence={"n_margins": int(margins.numel()), "n_flips": int(flips.numel())},
        )
    f = flips.detach().cpu().to(torch.bool)
    m = margins.detach().cpu().to(torch.float64)
    rate = float(f.to(torch.float64).mean().item())
    n_pos, n_neg = int(f.sum().item()), int((~f).sum().item())
    if n_pos == 0 or n_neg == 0:
        auc = None
        ranks_flips = None
    else:
        # AUC as the probability a flipped position has a smaller margin than an
        # unflipped one, ties counted as half, computed from ranks.
        order = torch.argsort(m)
        ranks = torch.empty_like(m)
        ranks[order] = torch.arange(1, m.numel() + 1, dtype=torch.float64)
        # average ranks within ties, so ties contribute 0.5 rather than an accident
        uniq, inv = torch.unique(m, return_inverse=True)
        for i in range(uniq.numel()):
            sel = inv == i
            if int(sel.sum().item()) > 1:
                ranks[sel] = ranks[sel].mean()
        r_pos = ranks[f].sum().item()
        auc = (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        auc = 1.0 - auc          # small margin should predict a flip
        ranks_flips = auc > 0.5
    return Verdict(
        check="token_level_risk", fired=rate > max_flip_rate, applicable=True, exact=False,
        detail=f"flip rate {rate:.6f} (tolerance {max_flip_rate}); margin AUC {auc}",
        threshold=max_flip_rate,
        evidence={"flip_rate": rate, "n_flips": n_pos, "n_positions": int(f.numel()),
                  "margin_auc": auc, "margin_ranks_flips": ranks_flips},
    )


CHECK_ORDER = [
    "shared_operands",
    "int32_no_overflow",
    "lossless_fp32_entry",
    "exact_accumulator",
    "pow2_scale_identity",
    "real_scale_tolerance",
    "token_level_risk",
]
EXACT_CHECKS = set(CHECK_ORDER[:5])


def run_suite(a: Arm, b: Arm, K: int, acc_ref: Optional[torch.Tensor] = None,
              product_bound: int = PRODUCT_BOUND_WITH_MIN_ACT, max_ulp: int = 1,
              margins: Optional[torch.Tensor] = None,
              flips: Optional[torch.Tensor] = None,
              max_flip_rate: float = 0.05) -> dict:
    """Run all seven checks and return {check_name: verdict dict}.

    Checks whose inputs were not supplied report applicable=False. That is
    deliberate: a suite that silently skipped them would let a report count seven
    checks when it ran five.
    """
    vs = [
        check1_shared_operands(a, b),
        check2_int32_no_overflow(K, product_bound),
        (check3_lossless_fp32_entry(acc_ref) if acc_ref is not None else
         Verdict("lossless_fp32_entry", False, False, True, "no reference accumulator supplied")),
        check4_exact_accumulator(a, b, acc_ref),
        check5_pow2_scale_identity(a, b),
        check6_real_scale_tolerance(a, b, max_ulp),
        (check7_token_level_risk(margins, flips, max_flip_rate) if
         (margins is not None and flips is not None) else
         Verdict("token_level_risk", False, False, False, "no margins/flips supplied",
                 threshold=max_flip_rate)),
    ]
    out = {v.check: v.as_dict() for v in vs}
    out["_summary"] = {
        "n_checks": len(CHECK_ORDER),
        "n_applicable": sum(1 for v in vs if v.applicable),
        "n_fired": sum(1 for v in vs if v.applicable and v.fired),
        "fired": [v.check for v in vs if v.applicable and v.fired],
        "inapplicable": [v.check for v in vs if not v.applicable],
    }
    return out
