"""Re-score the committed P5 cells against a prediction matrix, on a CPU.

p5_runner.py scores each cell while it runs, against the matrix it is given: the re-pinned
77-cell matrix (A-10.5). The pre-registration requires the same cells to be reported against
the pre-data matrix too, which is how post-data changes to the predictions are counted; the
hash-pinned one is A-10.1's, whereas arXiv:2609.00363v1 used a never-pinned reconstruction
(see experiment/CORRECTIONS.md).
This script does that from the committed rows alone, so it needs neither the activation
captures nor a checkpoint, and not torch either: the runner's classification rule (score())
is restated below rather than imported, and the check that re-scoring against the re-pinned
matrix reproduces every stored classification is what ties the two together.

Usage:
  python3 p5_rescore.py <p5_sensitivity.json> <prediction_matrix.json> [--json OUT]

Against artifacts/p5_prediction_matrix.json it must reproduce every stored classification
(0 mismatches, 0 violations). Against artifacts/p5_prediction_matrix_A-10.1_pinned.json it
reports 1,086 violating check-cell pairs in 886 cells; experiment/CORRECTIONS.md explains
why that, and not the figure in arXiv:2609.00363v1, is the comparison the protocol asks for.
"""
from __future__ import annotations

import argparse
import hashlib
import json

FIRE, NA = "should_fire", "not_applicable"
OUTPUT_CHECKS = ("pow2_scale_identity", "real_scale_tolerance")
VIOLATIONS = ("false_negative", "false_positive")
SCORED = ("hit", "false_negative", "false_positive", "correct_silence")
CLASSES = SCORED + ("excluded_fault_unobservable", "not_applicable")


def classify(fired: bool, applicable: bool, prediction: str, check: str,
             observable: bool | None) -> str:
    """p5_runner.score() for one check, in the same order of tests."""
    if prediction == NA or not applicable:
        return "not_applicable"
    if observable is False and check in OUTPUT_CHECKS:
        return "excluded_fault_unobservable"     # A-10.3: nothing to see
    if prediction == FIRE:
        return "hit" if fired else "false_negative"
    return "false_positive" if fired else "correct_silence"


def cells_for(matrix: dict, fault: str, severity: str) -> dict:
    """The predictions for one cell, chosen the way p5_runner.main() chooses them."""
    spec = matrix["faults"][fault]
    cells = spec["cells"]
    if cells is None:
        by_sev = spec.get("cells_by_severity") or {}
        if severity not in by_sev:
            raise RuntimeError(f"{fault} has no prediction for severity {severity!r}; the "
                               f"matrix and the fault catalogue disagree")
        cells = by_sev[severity]
    return cells


def rescore(rows: list, matrix: dict) -> dict:
    """Classify every check of every row against `matrix`. Only the stored verdicts (fired,
    applicable) are reused; the stored classification is what the re-score is checked
    against."""
    checks = matrix["checks"]
    per_check = {c: {k: 0 for k in CLASSES} for c in checks}
    mismatches = violating_cells = 0
    violations = []
    for i, r in enumerate(rows):
        cells = cells_for(matrix, r["fault"], r["severity"])
        bad = False
        for c in checks:
            v = r["checks"][c]
            cls = classify(v["fired"], v["applicable"], cells[c]["prediction"], c,
                           r["fault_observable"])
            per_check[c][cls] += 1
            mismatches += cls != v["classification"]
            if cls in VIOLATIONS:
                bad = True
                violations.append({"row": i, "fault": r["fault"], "severity": r["severity"],
                                   "regime": r["regime"], "check": c, "class": cls})
        violating_cells += bad
    totals = {k: sum(per_check[c][k] for c in checks) for k in CLASSES}
    return {"n_rows": len(rows),
            "scored_pairs": sum(totals[k] for k in SCORED),
            "violating_pairs": sum(totals[k] for k in VIOLATIONS),
            "violating_cells": violating_cells,
            "totals": totals,
            "per_check": per_check,
            "mismatches_vs_stored": mismatches,
            "violations": violations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sensitivity")
    ap.add_argument("matrix")
    ap.add_argument("--json", help="write the full result here")
    a = ap.parse_args()
    raw = open(a.matrix, "rb").read()
    res = rescore(json.load(open(a.sensitivity))["rows"], json.loads(raw))
    res["matrix_sha256"] = hashlib.sha256(raw).hexdigest()
    print(f"matrix sha256[:16] {res['matrix_sha256'][:16]}   rows {res['n_rows']}")
    print(f"scored pairs {res['scored_pairs']}   violating pairs {res['violating_pairs']}   "
          f"cells with a violation {res['violating_cells']}")
    print(f"classifications differing from the stored run: {res['mismatches_vs_stored']}")
    for chk, s in res["per_check"].items():
        print(f"  {chk:24s} hit {s['hit']:5d}  FN {s['false_negative']:4d}  "
              f"FP {s['false_positive']:4d}  silence {s['correct_silence']:6d}  "
              f"excluded {s['excluded_fault_unobservable']:5d}  NA {s['not_applicable']:5d}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
