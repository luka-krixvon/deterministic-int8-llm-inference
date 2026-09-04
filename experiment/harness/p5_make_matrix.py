"""Generate the P5 prediction matrix: 77 cells, written before the 196-layer measurement.

Every cell carries a prediction and the reasoning behind it, so each is falsifiable in
both directions: predicted fire that stays silent is a false negative, predicted
silence that fires is a false positive. Several cells deliberately predict a MISS --
those are the ones worth pre-registering, because a suite that only predicted its own
successes would measure nothing.

Run: python3 p5_make_matrix.py > ../artifacts/p5_prediction_matrix.json
"""
import json

FIRE, SILENT, NA = "should_fire", "should_not_fire", "not_applicable"

CHECKS = ["shared_operands", "int32_no_overflow", "lossless_fp32_entry",
          "exact_accumulator", "pow2_scale_identity", "real_scale_tolerance",
          "token_level_risk"]

# Pairing rule per fault, fixed here so the runner cannot choose it later.
PAIRING = {
    "epilogue": ("Arm A is the reference on the original operands; arm B is the injected "
                 "epilogue on those same operands."),
    "precondition": ("Both arms are the correct implementation on the precondition-violating "
                     "operands, so A and B agree by construction. These faults do not create "
                     "an A/B difference; they create a situation checks 2 and 3 must report. "
                     "acc_ref is always the int64 exact accumulator, so a wrapped arm differs "
                     "from it."),
    "operand": ("Arm A is the reference on the original operands; arm B is the reference on "
                "the modified operands. This is the only fault where the arms legitimately "
                "hold different operands."),
    "null": ("Arm A is the reference; arm B is a semantics-preserving rewrite of it. Any "
             "check that fires is reporting a difference that does not exist."),
}

# Every cell: (prediction, reason). Written before any P5 measurement.
M = {}

M["F1_scale_in_bf16"] = dict(kind="epilogue", observable_real_scales=True,
  observable_pow2_scales=False,
  note=("Under power-of-two scales this fault is a no-op: a power of two is exactly "
        "representable in bfloat16, so casting the scale changes nothing. Its "
        "observability is therefore regime-dependent, and the pow2 column below says "
        "silent for that reason rather than because the check is weak."),
  cells={
    "shared_operands":      (SILENT, "operands untouched; asserted by the injector self-tests"),
    "int32_no_overflow":    (SILENT, "K unchanged and inside the bound"),
    "lossless_fp32_entry":  (SILENT, "accumulator untouched"),
    "exact_accumulator":    (SILENT, "the fault is entirely after the accumulator"),
    "pow2_scale_identity":  (SILENT, "no-op under pow2 scales, so outputs stay bit-identical"),
    "real_scale_tolerance": (SILENT, "PREDICTED MISS. Rounding a scale to bf16 moves it by at "
                            "most 2^-9 relative, about half an output spacing, so differences "
                            "should land at 1 ulp and sit inside a tolerance of 1. If measurement "
                            "confirms this, check 6 has a false negative for reduced scale "
                            "precision at its stated tolerance, which is a finding about the "
                            "tolerance and not about the implementation."),
    "token_level_risk":     (NA, "token-level; P5 is layer-level and supplies no margins"),
  })

M["F2_double_rounding"] = dict(kind="epilogue", observable_real_scales=True,
  observable_pow2_scales=False,
  note=("Under pow2 scales the intermediate is a power of two times an exact integer, "
        "representable in bf16 when the accumulator is, so the extra rounding is a no-op."),
  cells={
    "shared_operands":      (SILENT, "operands untouched"),
    "int32_no_overflow":    (SILENT, "K unchanged"),
    "lossless_fp32_entry":  (SILENT, "accumulator untouched"),
    "exact_accumulator":    (SILENT, "fault is after the accumulator"),
    "pow2_scale_identity":  (SILENT, "no-op under pow2 scales"),
    "real_scale_tolerance": (SILENT, "PREDICTED MISS at all_elements. An extra bf16 rounding "
                            "costs at most half a spacing at that step, so the compounded error "
                            "should stay at 1 ulp. Predicting a miss here is the point: if it "
                            "does exceed 1 ulp, the tolerance is tighter than this reasoning."),
    "token_level_risk":     (NA, "token-level"),
  })

M["F3_scale_order"] = dict(kind="epilogue", observable_real_scales=False,
  observable_pow2_scales=False,
  note=("Measured unobservable in all three probed regimes: fp32 reassociation perturbs by "
        "about 2^-23 while bf16 rounds at 2^-8, roughly fifteen binades below resolution. "
        "Zero differing elements is arithmetic, not a broken injector. Every cell below is "
        "therefore silent BY CONSTRUCTION and must be excluded from detection-rate "
        "denominators; crediting check 6 with a pass here would be counting an unobservable "
        "fault as a detected one."),
  cells={
    "shared_operands":      (SILENT, "operands untouched"),
    "int32_no_overflow":    (SILENT, "K unchanged"),
    "lossless_fp32_entry":  (SILENT, "accumulator untouched"),
    "exact_accumulator":    (SILENT, "fault is after the accumulator"),
    "pow2_scale_identity":  (SILENT, "under pow2 scales both associations are exact"),
    "real_scale_tolerance": (SILENT, "fault produces no differing output element; excluded "
                            "from the denominator, not credited as a detection"),
    "token_level_risk":     (NA, "token-level"),
  })

M["F4_truncate_output"] = dict(kind="epilogue", observable_real_scales=True,
  observable_pow2_scales=True,
  note=("Observable in BOTH regimes, contrary to the original note, which claimed the "
        "pow2 product is exactly representable and the fault a no-op there. It is exact "
        "in fp32 but not in bf16, so the output cast still rounds. See A-10.6."),
  cells={
    "shared_operands":      (SILENT, "operands untouched"),
    "int32_no_overflow":    (SILENT, "K unchanged"),
    "lossless_fp32_entry":  (SILENT, "accumulator untouched"),
    "exact_accumulator":    (SILENT, "fault is after the accumulator"),
    "pow2_scale_identity":  (FIRE, "CORRECTED POST-DATA (A-10.6). Originally predicted "
                            "silent, on the reasoning that a power-of-two scale makes the "
                            "product exactly representable so truncation and "
                            "round-to-nearest agree. That confused exact in fp32 with exact "
                            "in bf16: the accumulator needs seventeen mantissa bits and "
                            "bf16 carries eight, so the cast still rounds and the two modes "
                            "still differ. A two-layer smoke run fired this cell. The "
                            "correction favours the study, so it is flagged: power-of-two "
                            "scales immunise the epilogue against intermediate precision "
                            "and rounding order -- F1 and F2 are no-ops here, which is "
                            "Equation (2) working -- but not against a wrong rounding mode, "
                            "and check 5 detects that."),
    "real_scale_tolerance": (SILENT, "PREDICTED MISS. Truncation instead of round-to-nearest "
                            "costs at most one spacing, so differences should be exactly 1 ulp "
                            "and inside the tolerance. Of the three predicted misses this is "
                            "the most consequential: a wrong rounding mode is a real defect that "
                            "a tolerance of one spacing cannot see."),
    "token_level_risk":     (NA, "token-level"),
  })

M["F5_fused_order"] = dict(kind="epilogue", observable_real_scales=None,
  observable_pow2_scales=False,
  note=("Marginal: 0, 2 and 0 differing elements across the three probed regimes. Whether "
        "any cell is testable depends on the regime actually run, so observability is "
        "recorded per run and cells with zero differing elements are excluded from "
        "denominators exactly as for F3."),
  cells={
    "shared_operands":      (SILENT, "operands untouched"),
    "int32_no_overflow":    (SILENT, "K unchanged"),
    "lossless_fp32_entry":  (SILENT, "accumulator untouched"),
    "exact_accumulator":    (SILENT, "the injected arm reports the same integer accumulator; "
                            "the fusion changes when the scale is applied, not the integer sum"),
    "pow2_scale_identity":  (SILENT, "under pow2 scales the folded multiply is exact"),
    "real_scale_tolerance": (SILENT, "PREDICTED MISS where observable at all: a float reduction "
                            "of scaled products differs from an integer reduction by well under "
                            "an output spacing at these magnitudes"),
    "token_level_risk":     (NA, "token-level"),
  })

M["F6_int32_overflow"] = dict(kind="precondition", observable_real_scales=True,
  observable_pow2_scales=True,
  note=("K is raised until 16129*K exceeds 2^31, so the accumulator wraps. Both arms are "
        "correct implementations on those operands, so A equals B; what must fire is the "
        "precondition machinery, not an A/B comparison."),
  cells={
    "shared_operands":      (SILENT, "both arms hold the same overflow-inducing operands"),
    "int32_no_overflow":    (FIRE, "this is the check's purpose: 16129*K exceeds 2^31 at the "
                            "constructed K, so it must report the layer as unlicensed"),
    "lossless_fp32_entry":  (FIRE, "the exact accumulator is far above 2^24, so lossless fp32 "
                            "entry fails as well; two checks firing on one fault is expected "
                            "and is not double counting"),
    "exact_accumulator":    (FIRE, "the arms report the wrapped INT32 value while acc_ref is "
                            "the int64 exact value, so this check catches the wrap "
                            "independently of check 2"),
    "pow2_scale_identity":  (SILENT, "A equals B, so outputs are bit-identical even though the "
                            "accumulator is wrong; PREDICTED MISS in the sense that a bitwise "
                            "output check cannot see a shared wrap"),
    "real_scale_tolerance": (SILENT, "same reason: identical arms, zero ulp distance, while the "
                            "computed values are wrong. This is the clearest demonstration that "
                            "output agreement is not correctness"),
    "token_level_risk":     (NA, "token-level"),
  })

M["F7_above_2p24"] = dict(kind="precondition", observable_real_scales=True,
  observable_pow2_scales=True,
  note="K raised so max|acc| exceeds 2^24 while staying below 2^31: no wrap, but the "
       "universal lossless-fp32-entry guarantee is gone.",
  cells={
    "shared_operands":      (SILENT, "both arms hold the same operands"),
    "int32_no_overflow":    (SILENT, "K is chosen to stay inside the INT32 bound, so this "
                            "check must NOT fire; a fire here would mean the two precondition "
                            "checks are not separable"),
    "lossless_fp32_entry":  (FIRE, "max|acc| exceeds 2^24, which is exactly what it tests"),
    "exact_accumulator":    (SILENT, "no wrap, so the arms match the exact reference"),
    "pow2_scale_identity":  (SILENT, "A equals B"),
    "real_scale_tolerance": (SILENT, "A equals B"),
    "token_level_risk":     (NA, "token-level"),
  })

M["F8_null"] = dict(kind="null", observable_real_scales=False, observable_pow2_scales=False,
  note="The false-positive control. Injector self-tests already assert the output is "
       "bit-identical to the reference, so every cell must be silent. Any fire is a "
       "false positive and is reported as such.",
  cells={c: (SILENT, "semantics-preserving rewrite; a fire is a false positive")
         for c in CHECKS[:6]} | {"token_level_risk": (NA, "token-level")})

M["F9_operand_mismatch"] = dict(kind="operand", observable_real_scales=True,
  observable_pow2_scales=True,
  note=("Added by A-10.2 because none of the original eight faults made the arms' "
        "operands differ, leaving check 1's detection rate untestable. Split per severity "
        "by A-10.6: the three rungs change different things, so one prediction cannot "
        "cover them. Under any rung, the other checks' verdicts say nothing about their "
        "own sensitivity -- once operands differ their premise is gone."),
  cells_by_severity={
    "one_element": {
      "shared_operands":      (FIRE, "one int8 activation element shifted by one step; the "
                              "check's purpose is to see exactly this"),
      "int32_no_overflow":    (SILENT, "K unchanged"),
      "lossless_fp32_entry":  (SILENT, "magnitudes stay inside 2^24"),
      "exact_accumulator":    (FIRE, "an int8 element changed, so that row's accumulator "
                              "changes across all N columns; fires as a consequence of the "
                              "operand difference, NOT as evidence of its own sensitivity"),
      "pow2_scale_identity":  (FIRE, "bitwise, so any single differing output element fires. "
                              "The per-element change is about a quarter of a spacing, so "
                              "most elements will round the same way; across 512xN elements "
                              "at least one is expected to differ. Observability is recorded "
                              "per row in case that expectation fails."),
      "real_scale_tolerance": (FIRE, "CORRECTED POST-DATA (A-10.7). Originally predicted "
                              "silent on a quarter-spacing argument that looked only at the "
                              "absolute change. Shifting one int8 activation element moves "
                              "every accumulator entry in that row by up to |w| = 127, and "
                              "entries driven near zero by cancellation see that as an "
                              "enormous relative change: measured 366 to 384 ulp. Check 6 "
                              "detects operand differences far better than the matrix "
                              "originally allowed."),
      "token_level_risk":     (NA, "token-level; P5 is layer-level"),
    },
    "one_percent": {
      "shared_operands":      (FIRE, "one weight scale moved by one float32 ulp; the check "
                              "compares scales as well as int8 tensors"),
      "int32_no_overflow":    (SILENT, "K unchanged"),
      "lossless_fp32_entry":  (SILENT, "magnitudes unchanged"),
      "exact_accumulator":    (SILENT, "CORRECTED POST-DATA (A-10.6). Originally predicted "
                              "fire, on the reasoning that different operands give different "
                              "accumulators. At this rung the difference is in a scale and "
                              "the int8 tensors are untouched, so the accumulator is "
                              "identical and silence is correct. The original prediction was "
                              "right for the other two rungs, which is why the cell had to "
                              "be split."),
      "pow2_scale_identity":  (SILENT, "in the pow2 regime a one-ulp fp32 move almost "
                              "certainly rounds to the same power of two, leaving the "
                              "epilogue input unchanged"),
      "real_scale_tolerance": (SILENT, "PREDICTED MISS, and measured as such in the smoke "
                              "run at exactly 1 ulp: an instance of the detection floor, "
                              "not a defect in the check"),
      "token_level_risk":     (NA, "token-level; P5 is layer-level"),
    },
    "all_elements": {
      "shared_operands":      (FIRE, "activations requantised from a different seed"),
      "int32_no_overflow":    (SILENT, "K unchanged"),
      "lossless_fp32_entry":  (SILENT, "magnitudes stay in the same range"),
      "exact_accumulator":    (FIRE, "wholly different activations give a wholly different "
                              "accumulator; again a consequence, not self-evidence"),
      "pow2_scale_identity":  (FIRE, "outputs differ far beyond bitwise equality"),
      "real_scale_tolerance": (FIRE, "differences are orders of magnitude past one spacing, "
                              "so this is the one rung where check 6 has real power against "
                              "an operand fault"),
      "token_level_risk":     (NA, "token-level; P5 is layer-level"),
    },
  })


def build():
    out = {
        "written": "2026-08-14",
        "status": "PRE-REGISTERED. Written and hash-pinned before any P5 measurement.",
        "governance": ("Pre-registration A-10 fixes the catalogue and the requirement that every "
                       "cell carry a written prediction; A-10.2 adds F9; A-10.3 replaces the "
                       "magnitude ladder with a coverage ladder and requires that a fault with "
                       "zero differing output elements be reported as unobservable rather than "
                       "as a check insensitivity; A-10.4 discloses what the matrix predicts; "
                       "A-10.6 is a POST-DATA correction of two cells after a two-layer smoke "
                       "run, and splits F9 per severity. Cells carrying corrected_post_data are "
                       "the ones a reader should scrutinise: both the original reasoning and "
                       "the correction are stated in the reason field, and the P5 report must "
                       "give detection rates under the original and the corrected matrix."),
        "scope": ("P5 measures the sensitivity of checks 1 to 6. Check 7 is token-level and "
                  "tolerance-based; a layer-level injection supplies no margins or flips, so it "
                  "is marked not_applicable in all nine rows and its sensitivity is NOT measured "
                  "by P5. That gap is stated here rather than hidden behind a count of seven."),
        "reading_rule": ("A predicted should_fire that stays silent is a false negative. A "
                         "predicted should_not_fire that fires is a false positive. A cell whose "
                         "fault produced zero differing output elements in the run is excluded "
                         "from both denominators and reported separately."),
        "predicted_misses": ("Cells marked PREDICTED MISS expect a real defect to pass a check. "
                             "They are pre-registered deliberately: F1, F2, F4 and F5 against "
                             "check 6, and F6 against checks 5 and 6. If measurement contradicts "
                             "them, the reasoning stated in the cell is what was wrong."),
        "pairing_rules": PAIRING,
        "checks": CHECKS,
        "faults": {},
    }
    n_fire = n_silent = n_na = n_corrected = 0

    def _mk(cellspec):
        nonlocal n_fire, n_silent, n_na, n_corrected
        d = {}
        for c in CHECKS:
            pred, why = cellspec[c]
            corrected = "CORRECTED POST-DATA" in why
            d[c] = {"prediction": pred, "reason": why,
                    "predicted_miss": "PREDICTED MISS" in why,
                    "corrected_post_data": corrected}
            n_fire += pred == FIRE
            n_silent += pred == SILENT
            n_na += pred == NA
            n_corrected += corrected
        return d

    for fault, spec in M.items():
        if "cells_by_severity" in spec:
            cells = None
            by_sev = {sev: _mk(cs) for sev, cs in spec["cells_by_severity"].items()}
        else:
            cells = _mk(spec["cells"])
            by_sev = None
        out["faults"][fault] = {
            "kind": spec["kind"],
            "pairing": PAIRING[spec["kind"]],
            "observable_real_scales": spec["observable_real_scales"],
            "observable_pow2_scales": spec["observable_pow2_scales"],
            "note": spec["note"],
            "cells": cells,
            "cells_by_severity": by_sev,
        }
    def _all_cells():
        for f in out["faults"].values():
            if f["cells"]:
                yield from f["cells"].values()
            for sev in (f["cells_by_severity"] or {}).values():
                yield from sev.values()
    out["cell_counts"] = {"total": n_fire + n_silent + n_na, "should_fire": n_fire,
                          "should_not_fire": n_silent, "not_applicable": n_na,
                          "predicted_miss": sum(1 for c in _all_cells() if c["predicted_miss"]),
                          "corrected_post_data": n_corrected}
    assert out["cell_counts"]["total"] == 77, out["cell_counts"]
    return out


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
