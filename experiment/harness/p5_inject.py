"""Reference scaled-INT8 GEMM and the nine-fault catalogue for the positive control.

The reference and the faults live in one module on purpose: every fault is defined
as a deviation from the reference in the same file, so the two cannot drift apart
under later edits.

What this measures, and what it does not. The checks in p5_checks take an
observation pair and decide; their sensitivity to a given perturbation is a property
of check and perturbation, not of which code produced it. So injecting known faults
into a reference implementation measures the checks honestly. It does not measure
their sensitivity to real kernel bugs in the wild, whose forms we do not get to
choose. That limitation belongs in the report.

Severity is a coverage ladder -- one element, one percent, all elements -- not a
magnitude ladder. Faults F1 to F5 are categorical: a bf16 multiply, a double
rounding, a reassociation. None has a magnitude knob; its effect is whatever the
arithmetic yields. What is controllable is how many output elements it touches,
which is also the quantity a reader wants: how small a fault does each check still
catch. See pre-registration A-10.3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

BF16 = torch.bfloat16
FP32 = torch.float32

SEVERITIES = ("one_element", "one_percent", "all_elements")


# ------------------------------------------------------------------ the reference

# float64 holds every partial sum of an int8 dot product exactly while the depth stays
# inside 2^53 / 16384, which is 549,755,813,888 -- far beyond any layer here. Where that
# holds, fp64 matmul is exact and about forty times faster than int64 on CPU (measured
# 0.03s against 1.23s at M=64, K=6144, N=2048, bit-identical). This is also the path
# p1_predictions.py already uses for its exact emulation. The guard is checked rather
# than assumed, and int64 is used where it does not hold.
FP64_EXACT_MAX_TERMS = (2 ** 53) // 16384


def exact_accumulator(a_q: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    """The ground-truth accumulator, exact and wide enough to show an INT32 wrap.

    Returned as int64, not int32, so that the reference can represent what an INT32
    accumulator would have wrapped to -- otherwise F6 could not be distinguished from
    a correct result.
    """
    K = a_q.shape[1]
    if K <= FP64_EXACT_MAX_TERMS:
        return (a_q.to(torch.float64) @ w_q.to(torch.float64).T).to(torch.int64)
    return (a_q.to(torch.int64) @ w_q.to(torch.int64).T)


def wrap_int32(acc64: torch.Tensor) -> torch.Tensor:
    """Wrap an int64 accumulator into INT32 two's-complement, as hardware would."""
    return ((acc64 + 2 ** 31) % 2 ** 32 - 2 ** 31).to(torch.int64)


def reference_epilogue(acc: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor,
                       out_dtype: torch.dtype = BF16) -> torch.Tensor:
    """The canonical epilogue: everything in fp32, rounded to the output once.

    Association is fixed as (acc * s_a) * s_w. That choice is arbitrary among the
    correct orders, which is exactly why F3 exists -- it applies a different one, so
    the checks get asked whether they notice.
    """
    return ((acc.to(FP32) * s_a.to(FP32)) * s_w.to(FP32)).to(out_dtype)


def reference_arm(a_q, w_q, s_a, s_w, out_dtype: torch.dtype = BF16):
    """Return (out, acc_int32_view, acc_exact_int64) for the correct implementation."""
    acc64 = exact_accumulator(a_q, w_q)
    acc32 = wrap_int32(acc64)
    return reference_epilogue(acc32, s_a, s_w, out_dtype), acc32, acc64


# ------------------------------------------------------------------- masking helper

def _coverage_mask(shape, severity: str, generator: torch.Generator) -> torch.Tensor:
    """Boolean mask selecting which output elements the fault touches."""
    n = 1
    for d in shape:
        n *= d
    flat = torch.zeros(n, dtype=torch.bool)
    if severity == "all_elements":
        flat[:] = True
    elif severity == "one_element":
        flat[torch.randint(n, (1,), generator=generator)] = True
    elif severity == "one_percent":
        k = max(1, n // 100)
        idx = torch.randperm(n, generator=generator)[:k]
        flat[idx] = True
    else:
        raise ValueError(f"unknown severity {severity!r}")
    return flat.reshape(shape)


def _blend(correct: torch.Tensor, faulted: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Take the faulted value where mask is set, the correct value elsewhere."""
    return torch.where(mask, faulted, correct)


# --------------------------------------------------------------- the nine faults
#
# Each fault returns a dict with the faulted arm's out / acc / operands. Operands are
# returned because F9 changes them; every other fault returns them untouched, and the
# self-tests assert that.

@dataclass
class Injected:
    out: torch.Tensor
    acc: torch.Tensor
    a_q: torch.Tensor
    w_q: torch.Tensor
    s_a: torch.Tensor
    s_w: torch.Tensor
    touched: int
    note: str
    n_output_differing: int = -1   # filled by inject(); see observability below


# Observability. A fault can be injected and still leave the output bit-identical,
# because bfloat16 carries eight mantissa bits and absorbs any fp32 perturbation
# below its rounding step. When that happens, no check can be blamed for staying
# silent, and a sensitivity table that did not separate the two cases would be
# uninterpretable. So inject() records how many output elements actually changed,
# and a P5 report must treat n_output_differing == 0 as "fault not observable at
# output precision", never as "check insensitive". See pre-registration A-10.3.


def f1_scale_in_bf16(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """Scales rounded to bfloat16 before the multiply: reduced intermediate precision."""
    out_c, acc32, _ = reference_arm(a_q, w_q, s_a, s_w)
    bad = ((acc32.to(FP32) * s_a.to(BF16).to(FP32)) * s_w.to(BF16).to(FP32)).to(BF16)
    m = _coverage_mask(out_c.shape, severity, gen)
    return Injected(_blend(out_c, bad, m), acc32, a_q, w_q, s_a, s_w, int(m.sum()),
                    "scales cast to bf16 before multiplying")


def f2_double_rounding(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """fp32 -> bf16 -> fp32 -> bf16: an extra rounding between the two scales."""
    out_c, acc32, _ = reference_arm(a_q, w_q, s_a, s_w)
    half = (acc32.to(FP32) * s_a.to(FP32)).to(BF16).to(FP32)
    bad = (half * s_w.to(FP32)).to(BF16)
    m = _coverage_mask(out_c.shape, severity, gen)
    return Injected(_blend(out_c, bad, m), acc32, a_q, w_q, s_a, s_w, int(m.sum()),
                    "rounded to bf16 between the two scale multiplies")


def f3_scale_order(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """acc * (s_a * s_w) instead of (acc * s_a) * s_w: a different association."""
    out_c, acc32, _ = reference_arm(a_q, w_q, s_a, s_w)
    bad = (acc32.to(FP32) * (s_a.to(FP32) * s_w.to(FP32))).to(BF16)
    m = _coverage_mask(out_c.shape, severity, gen)
    return Injected(_blend(out_c, bad, m), acc32, a_q, w_q, s_a, s_w, int(m.sum()),
                    "scale product formed first, then applied")


def f4_truncate_output(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """Output cast truncates toward zero instead of round-to-nearest-even."""
    out_c, acc32, _ = reference_arm(a_q, w_q, s_a, s_w)
    v = (acc32.to(FP32) * s_a.to(FP32)) * s_w.to(FP32)
    bits = v.view(torch.int32)
    # bf16 is the top 16 bits of fp32; dropping the low 16 truncates toward zero
    bad = ((bits >> 16) << 16).view(FP32).to(BF16)
    m = _coverage_mask(out_c.shape, severity, gen)
    return Injected(_blend(out_c, bad, m), acc32, a_q, w_q, s_a, s_w, int(m.sum()),
                    "output truncated instead of round-to-nearest-even")


def f5_fused_order(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """Scale applied inside the reduction rather than after it.

    A fused epilogue that folds the per-column scale into the accumulation sees a
    float sum instead of an integer one, so order matters again.
    """
    out_c, acc32, _ = reference_arm(a_q, w_q, s_a, s_w)
    prod = a_q.to(FP32).unsqueeze(1) * w_q.to(FP32).unsqueeze(0)        # (M,N,K)
    scaled = prod * s_w.to(FP32).reshape(1, -1, 1)
    bad = (scaled.sum(dim=2) * s_a.to(FP32)).to(BF16)
    m = _coverage_mask(out_c.shape, severity, gen)
    return Injected(_blend(out_c, bad, m), acc32, a_q, w_q, s_a, s_w, int(m.sum()),
                    "per-column scale folded into the reduction")


def f6_int32_overflow(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """Operands whose dot product exceeds INT32: the alibi's precondition violated.

    Both arms receive these operands, so check 1 must stay silent and check 2 must
    fire. The arm reports the wrapped accumulator, which is what hardware would give.
    """
    M, K = a_q.shape
    N = w_q.shape[0]
    # K columns of +-127 * +-127 sum to 16129*K; make that exceed 2^31
    need_K = (2 ** 31) // 16129 + 1
    a2 = torch.full((M, need_K), 127, dtype=torch.int8)
    w2 = torch.full((N, need_K), 127, dtype=torch.int8)
    acc64 = exact_accumulator(a2, w2)
    acc32 = wrap_int32(acc64)
    out = reference_epilogue(acc32, s_a, s_w)
    return Injected(out, acc32, a2, w2, s_a, s_w, out.numel(),
                    f"K={need_K} so 16129*K exceeds 2^31; accumulator wraps")


def f7_above_2p24(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """Accumulator above 2^24 but inside INT32: lossless fp32 entry violated."""
    M, K = a_q.shape
    N = w_q.shape[0]
    need_K = 2 ** 24 // 16129 + 2          # over 2^24, far below 2^31
    a2 = torch.full((M, need_K), 127, dtype=torch.int8)
    w2 = torch.full((N, need_K), 127, dtype=torch.int8)
    acc64 = exact_accumulator(a2, w2)
    acc32 = wrap_int32(acc64)
    out = reference_epilogue(acc32, s_a, s_w)
    return Injected(out, acc32, a2, w2, s_a, s_w, out.numel(),
                    f"K={need_K} puts max|acc| above 2^24 but below 2^31")


def f8_null(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """A semantics-preserving rewrite: multiply by one, add zero, reorder nothing.

    The false-positive control. Any check that fires here is reporting a difference
    that does not exist, and that is the finding.
    """
    acc64 = exact_accumulator(a_q, w_q)
    acc32 = wrap_int32(acc64)
    one = torch.ones((), dtype=FP32)
    out = (((acc32.to(FP32) * one) * s_a.to(FP32)) * s_w.to(FP32) + 0.0).to(BF16)
    return Injected(out, acc32, a_q, w_q, s_a, s_w, 0,
                    "multiply by 1.0 and add 0.0; output must be bit-identical")


def f9_operand_mismatch(a_q, w_q, s_a, s_w, severity, gen) -> Injected:
    """The second arm receives different operands. Check 1 must fire (A-10.2).

    Severity here IS a magnitude ladder, because an operand difference has one:
    one int8 step, one float32 ulp of a scale, or a wholly different activation.
    """
    a2, s2 = a_q.clone(), s_a.clone()
    if severity == "one_element":
        i = int(torch.randint(a2.shape[0], (1,), generator=gen))
        j = int(torch.randint(a2.shape[1], (1,), generator=gen))
        a2[i, j] = torch.tensor(int(a2[i, j]) - 1 if int(a2[i, j]) > -128 else -127,
                                dtype=torch.int8)
        note = f"activation element ({i},{j}) shifted by one int8 step"
    elif severity == "one_percent":
        w32 = s_w.to(FP32).clone()
        k = int(torch.randint(w32.numel(), (1,), generator=gen))
        flat = w32.flatten()
        flat[k] = torch.tensor(float(torch.nextafter(flat[k], torch.tensor(float("inf")))))
        s_w = flat.reshape(s_w.shape)
        note = f"weight scale index {k} moved by one float32 ulp"
    elif severity == "all_elements":
        a2 = torch.randint(-127, 128, a_q.shape, generator=gen, dtype=torch.int8)
        note = "activations requantized from a different seed"
    else:
        raise ValueError(severity)
    acc64 = exact_accumulator(a2, w_q)
    acc32 = wrap_int32(acc64)
    out = reference_epilogue(acc32, s2, s_w)
    return Injected(out, acc32, a2, w_q, s2, s_w, -1, note)


FAULTS: dict[str, Callable] = {
    "F1_scale_in_bf16": f1_scale_in_bf16,
    "F2_double_rounding": f2_double_rounding,
    "F3_scale_order": f3_scale_order,
    "F4_truncate_output": f4_truncate_output,
    "F5_fused_order": f5_fused_order,
    "F6_int32_overflow": f6_int32_overflow,
    "F7_above_2p24": f7_above_2p24,
    "F8_null": f8_null,
    "F9_operand_mismatch": f9_operand_mismatch,
}

# F6, F7 and F8 are binary or semantics-preserving: a coverage ladder is meaningless
# for them, so they run once at "all_elements".
BINARY_FAULTS = {"F6_int32_overflow", "F7_above_2p24", "F8_null"}


def severities_for(fault: str) -> tuple[str, ...]:
    return ("all_elements",) if fault in BINARY_FAULTS else SEVERITIES


def inject(fault: str, a_q, w_q, s_a, s_w, severity: str = "all_elements",
           seed: int = 0) -> Injected:
    if fault not in FAULTS:
        raise KeyError(f"unknown fault {fault!r}; catalogue is closed (A-10.2)")
    if severity not in severities_for(fault):
        raise ValueError(f"{fault} does not take severity {severity!r}")
    gen = torch.Generator().manual_seed(seed)
    inj = FAULTS[fault](a_q, w_q, s_a, s_w, severity, gen)
    ref_out, _, _ = reference_arm(a_q, w_q, s_a, s_w)
    if inj.out.shape == ref_out.shape and inj.out.dtype == ref_out.dtype:
        same_dtype_bits = torch.int16 if inj.out.dtype in (BF16, torch.float16) else torch.int32
        inj.n_output_differing = int(
            (inj.out.contiguous().view(same_dtype_bits)
             != ref_out.contiguous().view(same_dtype_bits)).sum())
    else:
        inj.n_output_differing = -1     # different shape: F6/F7 change K, not comparable
    return inj


def observability_survey(a_q, w_q, s_a, s_w, seed: int = 0) -> dict:
    """For every fault and severity, how many output elements actually changed.

    Run this before a P5 measurement and report it alongside the sensitivity table.
    A fault with zero differing elements tests nothing about the checks.
    """
    rows = {}
    for name in FAULTS:
        for sev in severities_for(name):
            inj = inject(name, a_q, w_q, s_a, s_w, sev, seed)
            rows[f"{name}/{sev}"] = {
                "n_output_differing": inj.n_output_differing,
                "elements_touched": inj.touched,
                "observable": (inj.n_output_differing > 0) if inj.n_output_differing >= 0 else None,
                "note": inj.note,
            }
    return rows
