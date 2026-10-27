"""Tests for p5_rescore: the committed cells re-scored against both prediction matrices.

These pin the numbers experiment/CORRECTIONS.md reports, so that a later edit to the
scoring rules or to either matrix shows up here rather than in a reader's re-computation.
Against the re-pinned matrix the re-score must equal the stored run exactly; against the
hash-pinned pre-data matrix it must give the 1,086 / 886 comparison, every violation sitting
in one of the four predictions the smoke run changed.
"""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p5_rescore as RS

P = F = 0
def ck(n, c):
    global P, F
    if c: P += 1; print(f"PASS {n}")
    else: F += 1; print(f"FAIL {n}")

ART = os.path.join(HERE, "..", "artifacts")
ROWS = json.load(open(os.path.join(ART, "p5p6_results", "p5_sensitivity.json")))["rows"]
def load(name):
    raw = open(os.path.join(ART, name), "rb").read()
    return hashlib.sha256(raw).hexdigest()[:16], json.loads(raw)

sha77, M77 = load("p5_prediction_matrix.json")
sha63, M63 = load("p5_prediction_matrix_A-10.1_pinned.json")
ck("re-pinned matrix is the A-10.5 pin", sha77 == "718e04cd93eb2653")
ck("pre-data matrix is the A-10.1 pin", sha63 == "8f6ada4492c688b0")

# ---------------------------------------------------------------- re-pinned: the stored run
r77 = RS.rescore(ROWS, M77)
ck("8,232 rows", r77["n_rows"] == 8232)
ck("re-score equals the stored classifications", r77["mismatches_vs_stored"] == 0)
ck("no violations under the re-pinned matrix", r77["violating_pairs"] == 0)
t = r77["totals"]
ck("38,039 scored pairs", r77["scored_pairs"] == 38039)
ck("5,198 hits / 32,841 silences", (t["hit"], t["correct_silence"]) == (5198, 32841))
ck("7,041 excluded / 12,544 not applicable",
   (t["excluded_fault_unobservable"], t["not_applicable"]) == (7041, 12544))

# ---------------------------------------------------------------- pinned pre-data matrix
r63 = RS.rescore(ROWS, M63)
pc = r63["per_check"]
ck("same 38,039 scored pairs", r63["scored_pairs"] == 38039)
ck("1,086 violating pairs", r63["violating_pairs"] == 1086)
ck("886 cells with a violation", r63["violating_cells"] == 886)
ck("accumulator: 1,176 hits, 392 misses",
   (pc["exact_accumulator"]["hit"], pc["exact_accumulator"]["false_negative"]) == (1176, 392))
ck("power-of-two identity: 494 false positives over 886 scored silences",
   (pc["pow2_scale_identity"]["false_positive"],
    pc["pow2_scale_identity"]["false_positive"] + pc["pow2_scale_identity"]["correct_silence"])
   == (494, 886))
ck("tolerance: 784 hits, 200 misses, no false positive",
   (pc["real_scale_tolerance"]["hit"], pc["real_scale_tolerance"]["false_negative"],
    pc["real_scale_tolerance"]["false_positive"]) == (784, 200, 0))

# Every violation must sit in one of the four changed predictions; otherwise the account in
# CORRECTIONS.md is incomplete.
CHANGED = {("F4_truncate_output", None, "pow2_scale_identity"),
           ("F9_operand_mismatch", "one_percent", "exact_accumulator"),
           ("F9_operand_mismatch", "one_percent", "pow2_scale_identity"),
           ("F9_operand_mismatch", "one_percent", "real_scale_tolerance")}
outside = sum((v["fault"], v["severity"] if v["fault"] == "F9_operand_mismatch" else None,
                v["check"]) not in CHANGED for v in r63["violations"])
ck("every pre-data violation is at a changed prediction", outside == 0)

# ---------------------------------------------------------------- arXiv v1's 'original matrix'
# v1's figures came from the re-pinned matrix with its three flagged cells set back.
import copy
MV1 = copy.deepcopy(M77); flipped = 0
for spec in MV1["faults"].values():
    groups = ([spec["cells"]] if spec["cells"] is not None
              else list((spec.get("cells_by_severity") or {}).values()))
    for cells in groups:
        for c in cells.values():
            if c.get("corrected_post_data"):
                c["prediction"] = ("should_not_fire" if c["prediction"] == "should_fire"
                                   else "should_fire")
                flipped += 1
rv1 = RS.rescore(ROWS, MV1); pv = rv1["per_check"]
ea, p2, tl = pv["exact_accumulator"], pv["pow2_scale_identity"], pv["real_scale_tolerance"]
ck("three flagged cells set back", flipped == 3)
ck("v1: 1,278 violating cells", rv1["violating_cells"] == 1278)
ck("v1: accumulator 75.0%", (ea["hit"], ea["false_negative"]) == (1176, 392))
ck("v1: power-of-two false-positive rate 55.76%",
   (p2["false_positive"], p2["correct_silence"]) == (494, 392))
ck("v1: tolerance false-positive rate 11.39%",
   (tl["false_positive"], tl["correct_silence"]) == (392, 3049))

# ---------------------------------------------------------------- selection guard
bad = {"faults": {"F9_operand_mismatch": {"cells": None, "cells_by_severity": {}}}}
try:
    RS.cells_for(bad, "F9_operand_mismatch", "one_element"); raised = False
except RuntimeError:
    raised = True
ck("missing per-severity prediction raises rather than guessing", raised)

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
