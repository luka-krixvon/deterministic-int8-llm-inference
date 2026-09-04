"""P5 runner: drive the nine faults through the seven checks over every layer.

This is where the remaining decisions live, which is why it is pinned before it runs
rather than written at execution time: which layers, how an observation is built, how
the two arms are paired per fault kind, and how a verdict is scored against the
prediction matrix.

Data path. Activations come from the committed per-layer captures, which hold
``{"q": int8 (M,K), "scale": fp32 (M,1)}`` -- the activation side only. Weights and
their per-channel scales come from the checkpoint's safetensors, the same way
w3_perlayer.py obtains them. So the operands are the ones the study measured, not
synthetic stand-ins.

Two scale regimes, because the checks divide along them. Check 5 is bitwise and applies
only under powers of two; check 6 is a tolerance and applies under the checkpoint's real
scales. Running only one regime would leave one of them untested. The pow2 regime uses
the same transform make_probe_pow2.py applies: exp2(round(log2(s))).

Scoring. Each cell is compared against p5_prediction_matrix.json. A predicted fire that
stays silent is a false negative; a predicted silence that fires is a false positive. A
cell whose fault changed no output element is excluded from both denominators and
counted separately -- never credited as a detection and never blamed on the check
(pre-registration A-10.3).

Usage (venv, needs the captures and the checkpoint):
  python3 p5_runner.py --capture ../../perlayer_capture \\
                       --model ../../models/qwen3-1.7b-int8-w8a8 \\
                       --matrix ../artifacts/p5_prediction_matrix.json \\
                       --out ../artifacts/p5_sensitivity.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch

from p5_checks import Arm, run_suite, CHECK_ORDER, PRODUCT_BOUND_WITH_MIN_ACT
from p5_inject import (FAULTS, SEVERITIES, inject, reference_arm, exact_accumulator,
                       severities_for)

# Pinned. The capture holds 512 activation rows; all of them are used, which fp64
# accumulation makes affordable (about a quarter second per layer against ten seconds
# for int64).
M_TILE = 512

# F5 folds the per-column scale into the reduction, so it materialises an (M,N,K)
# intermediate: 25 GB at real layer shapes. It runs on this sub-tile instead, recorded
# per row so no report can claim F5 covered the layer.
F5_TILE = (32, 32)

SCALE_REGIMES = ("real", "pow2")
MAX_ULP = 1                     # v1's tolerance, unchanged
MAX_FLIP_RATE = 0.05            # v1's tolerance, unchanged
FIRE, SILENT, NA = "should_fire", "should_not_fire", "not_applicable"


def load_layer_names(capture_dir: str):
    with open(os.path.join(capture_dir, "_layers.json")) as fh:
        return json.load(fh)


def open_checkpoint(model_dir: str):
    """Return a dict of tensor-name -> loader over the checkpoint's shards."""
    from safetensors.torch import safe_open
    handles, index = [], {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        h = safe_open(shard, framework="pt")
        handles.append(h)
        for k in h.keys():
            index[k] = h
    return index, handles


def build_observation(capture_dir: str, index, layer: str, regime: str):
    """Assemble one layer's operands in one scale regime.

    Returns (a_q, w_q, s_a, s_w, meta). s_w is transposed to (1,N) to match the check
    suite's convention. Scale dtypes travel in meta because the checkpoint stores
    weight_scale in bfloat16, which makes F1 only half a fault on a real layer.
    """
    cap = torch.load(os.path.join(capture_dir, layer + ".pt"),
                     map_location="cpu", weights_only=False)
    a_q = cap["q"][:M_TILE].contiguous()
    s_a = cap["scale"][:M_TILE].contiguous().to(torch.float32)
    w_q = index[layer + ".weight"].get_tensor(layer + ".weight")
    w_s_raw = index[layer + ".weight_scale"].get_tensor(layer + ".weight_scale")
    s_w = w_s_raw.reshape(1, -1).to(torch.float32)
    if regime == "pow2":
        # Same transform make_probe_pow2.py applies to the checkpoint.
        s_w = torch.exp2(torch.round(torch.log2(s_w)))
        s_a = torch.exp2(torch.round(torch.log2(s_a)))
    meta = {
        "M": int(a_q.shape[0]), "K": int(a_q.shape[1]), "N": int(w_q.shape[0]),
        "s_a_source_dtype": str(cap["scale"].dtype),
        "s_w_source_dtype": str(w_s_raw.dtype),
        "s_w_already_bf16": w_s_raw.dtype == torch.bfloat16,
        "regime": regime,
    }
    return a_q, w_q, s_a, s_w, meta


def arms_for(fault: str, kind: str, a_q, w_q, s_a, s_w, severity: str, seed: int):
    """Build the (A, B, acc_ref, injected) tuple per the matrix's pairing rule.

    epilogue    A is the reference on the original operands, B is the injected epilogue.
    precondition Both arms are the correct implementation on the violating operands, so
                A equals B; these faults do not create an A/B difference, they create a
                situation checks 2 and 3 must report. acc_ref stays the exact value, so
                a wrapped arm differs from it.
    operand     A is the reference on the original operands, B on the modified ones.
    null        A is the reference, B a semantics-preserving rewrite.
    """
    inj = inject(fault, a_q, w_q, s_a, s_w, severity, seed)
    if kind == "precondition":
        # Rebuild the reference on the fault's own operands so the arms match.
        out_a, acc_a, acc_exact = reference_arm(inj.a_q, inj.w_q, inj.s_a, inj.s_w)
        A = Arm("reference", inj.a_q, inj.w_q, inj.s_a, inj.s_w, out_a, acc_a)
        B = Arm("reference-same", inj.a_q, inj.w_q, inj.s_a, inj.s_w, out_a.clone(),
                acc_a.clone())
        return A, B, acc_exact, inj, int(inj.a_q.shape[1])
    out_a, acc_a, acc_exact = reference_arm(a_q, w_q, s_a, s_w)
    A = Arm("reference", a_q, w_q, s_a, s_w, out_a, acc_a)
    B = Arm(fault, inj.a_q, inj.w_q, inj.s_a, inj.s_w, inj.out, inj.acc)
    if kind == "operand":
        # acc_ref must describe arm A's operands; arm B legitimately differs.
        acc_exact = exact_accumulator(a_q, w_q)
    return A, B, acc_exact, inj, int(a_q.shape[1])


def score(verdicts: dict, predictions: dict, observable: bool | None) -> dict:
    """Classify each check's verdict against its prediction."""
    out = {}
    for chk in CHECK_ORDER:
        v = verdicts[chk]
        pred = predictions[chk]["prediction"]
        if pred == NA or not v["applicable"]:
            cls = "not_applicable"
        elif observable is False and chk in ("pow2_scale_identity", "real_scale_tolerance"):
            # The fault changed no output element, so an output-comparison check has
            # nothing to see. Excluded from both denominators (A-10.3).
            cls = "excluded_fault_unobservable"
        elif pred == FIRE:
            cls = "hit" if v["fired"] else "false_negative"
        else:
            cls = "false_positive" if v["fired"] else "correct_silence"
        out[chk] = {"prediction": pred, "fired": v["fired"],
                    "applicable": v["applicable"], "classification": cls,
                    "detail": v["detail"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--seed", type=int, default=20260815)
    a = ap.parse_args()

    matrix = json.load(open(a.matrix))
    names = load_layer_names(a.capture)
    if a.layers:
        names = names[:a.layers]
    index, handles = open_checkpoint(a.model)

    rows, t0 = [], time.perf_counter()
    for li, layer in enumerate(names):
        for regime in SCALE_REGIMES:
            a_q, w_q, s_a, s_w, meta = build_observation(a.capture, index, layer, regime)
            for fault, spec in matrix["faults"].items():
                kind = spec["kind"]
                for sev in severities_for(fault):
                    if fault == "F5_fused_order":
                        m, n = F5_TILE
                        aq, wq = a_q[:m], w_q[:n]
                        sa, sw = s_a[:m], s_w[:, :n]
                        tile = {"M": m, "N": n, "note": "F5 sub-tiled; see A-10.5"}
                    else:
                        aq, wq, sa, sw, tile = a_q, w_q, s_a, s_w, None
                    # F9 carries per-severity predictions (A-10.6); every other fault
                    # has one cell set. Selecting here rather than in score() keeps the
                    # matrix the single source of truth for what was predicted.
                    cells = spec["cells"]
                    if cells is None:
                        by_sev = spec.get("cells_by_severity") or {}
                        if sev not in by_sev:
                            raise RuntimeError(
                                f"{fault} has no prediction for severity {sev!r}; the "
                                f"matrix and the fault catalogue disagree")
                        cells = by_sev[sev]
                    A, B, acc_ref, inj, K = arms_for(fault, kind, aq, wq, sa, sw,
                                                     sev, a.seed + li)
                    v = run_suite(A, B, K=K, acc_ref=acc_ref,
                                  product_bound=PRODUCT_BOUND_WITH_MIN_ACT,
                                  max_ulp=MAX_ULP, max_flip_rate=MAX_FLIP_RATE)
                    obs = (inj.n_output_differing > 0
                           if inj.n_output_differing >= 0 else None)
                    rows.append({
                        "layer": layer, "layer_index": li, "regime": regime,
                        "fault": fault, "severity": sev, "kind": kind,
                        "K": K, "shape": {k: meta[k] for k in ("M", "K", "N")},
                        "f5_tile": tile,
                        "n_output_differing": inj.n_output_differing,
                        "fault_observable": obs,
                        "elements_touched": inj.touched,
                        "scale_dtypes": {k: meta[k] for k in
                                         ("s_a_source_dtype", "s_w_source_dtype",
                                          "s_w_already_bf16")},
                        "checks": score(v, cells, obs),
                    })
        if (li + 1) % 20 == 0:
            print(f"  {li+1}/{len(names)} layers, {len(rows)} rows, "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    agg = aggregate(rows)
    out = {
        "schema": "p5-sensitivity-v1",
        "governance": ("Pre-registration A-10 with A-10.2 (F9), A-10.3 (coverage ladder "
                       "and the observability rule) and A-10.4 (what the matrix predicts). "
                       "Pinned in A-10.5 before this ran."),
        "matrix_cell_counts": matrix["cell_counts"],
        "pinned": {"M_TILE": M_TILE, "F5_TILE": list(F5_TILE),
                   "scale_regimes": list(SCALE_REGIMES), "max_ulp": MAX_ULP,
                   "max_flip_rate": MAX_FLIP_RATE,
                   "product_bound": PRODUCT_BOUND_WITH_MIN_ACT, "seed": a.seed},
        "scope": matrix["scope"],
        "reading_rule": matrix["reading_rule"],
        "n_layers": len(names), "n_rows": len(rows),
        "model": a.model, "capture": a.capture,
        "elapsed_seconds": time.perf_counter() - t0,
        "aggregate": agg,
        "rows": rows,
    }
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
    for chk in CHECK_ORDER:
        s = agg["per_check"][chk]
        print(f"{chk:24s} hit {s['hit']:5d} FN {s['false_negative']:5d} "
              f"FP {s['false_positive']:5d} silence {s['correct_silence']:6d} "
              f"excluded {s['excluded_fault_unobservable']:5d} NA {s['not_applicable']:6d}")
    print(f"\n{len(rows)} rows over {len(names)} layers in "
          f"{out['elapsed_seconds']:.0f}s -> {a.out}")


def aggregate(rows):
    per_check = {c: {k: 0 for k in ("hit", "false_negative", "false_positive",
                                    "correct_silence", "excluded_fault_unobservable",
                                    "not_applicable")} for c in CHECK_ORDER}
    per_fault = {}
    for r in rows:
        pf = per_fault.setdefault(r["fault"], {"rows": 0, "observable_rows": 0,
                                               "hit": 0, "false_negative": 0,
                                               "false_positive": 0})
        pf["rows"] += 1
        pf["observable_rows"] += 1 if r["fault_observable"] else 0
        for chk, c in r["checks"].items():
            per_check[chk][c["classification"]] += 1
            if c["classification"] in ("hit", "false_negative", "false_positive"):
                pf[c["classification"]] += 1
    for c, s in per_check.items():  # noqa: B020 - shadowing is local and intended
        den = s["hit"] + s["false_negative"]
        s["detection_rate"] = (s["hit"] / den) if den else None
        s["false_negative_rate"] = (s["false_negative"] / den) if den else None
        neg = s["correct_silence"] + s["false_positive"]
        s["false_positive_rate"] = (s["false_positive"] / neg) if neg else None
        s["denominator_note"] = ("detection over predicted-fire cells only; "
                                 "excluded_fault_unobservable is in neither denominator")
    return {"per_check": per_check, "per_fault": per_fault}


if __name__ == "__main__":
    main()
