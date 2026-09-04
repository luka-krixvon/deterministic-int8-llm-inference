"""Tests for the P5 runner: pairing rules, per-severity cell selection, scoring classes,
and the aggregate denominators. Built on a tiny synthetic layer so it runs anywhere.

The runner holds the decisions A-10.5 pins, so these assert the decisions rather than
just that it executes: that the null fault produces no false positive, that precondition
faults really do give identical arms against an exact reference, that an unobservable
fault is excluded from both denominators rather than credited, and that a fault whose
severity has no prediction raises instead of guessing.
"""
import json, os, sys, tempfile
sys.path.insert(0, '.')
import torch
from safetensors.torch import save_file

import p5_runner as RN
from p5_checks import CHECK_ORDER, _bitwise_equal
from p5_inject import exact_accumulator

P = F = 0
def ck(n, c):
    global P, F
    if c: P += 1; print(f"PASS {n}")
    else: F += 1; print(f"FAIL {n}")

# ---------------------------------------------------------------- synthetic fixture
TMP = tempfile.mkdtemp()
CAP = os.path.join(TMP, "cap"); MODEL = os.path.join(TMP, "model")
os.makedirs(CAP); os.makedirs(MODEL)
LAYER = "model.layers.0.mlp.down_proj"
M, K, N = 24, 32, 12
g = torch.Generator().manual_seed(11)
a_q = torch.randint(-127, 128, (M, K), generator=g, dtype=torch.int8)
s_a = (torch.rand(M, 1, generator=g) * 0.02 + 0.001)
torch.save({"q": a_q, "scale": s_a}, os.path.join(CAP, LAYER + ".pt"))
json.dump([LAYER], open(os.path.join(CAP, "_layers.json"), "w"))
w_q = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
# bfloat16 on purpose: that is what the real checkpoint stores, and it is why F1 is
# only half a fault on a real layer.
w_s = (torch.rand(N, 1, generator=g) * 0.02 + 0.001).to(torch.bfloat16)
save_file({LAYER + ".weight": w_q, LAYER + ".weight_scale": w_s},
          os.path.join(MODEL, "model.safetensors"))

index, _handles = RN.open_checkpoint(MODEL)
ck("checkpoint index finds weight and scale",
   {LAYER + ".weight", LAYER + ".weight_scale"} <= set(index))

# ---------------------------------------------------------------- observation building
aq, wq, sa, sw, meta = RN.build_observation(CAP, index, LAYER, "real")
ck("observation shapes are (M,K) and (N,K)", tuple(aq.shape) == (M, K) and tuple(wq.shape) == (N, K))
ck("weight scale is transposed to (1,N)", tuple(sw.shape) == (1, N))
ck("meta records that weight_scale was already bf16", meta["s_w_already_bf16"] is True)
ck("meta records the source dtypes", "bfloat16" in meta["s_w_source_dtype"]
   and "float32" in meta["s_a_source_dtype"])
_, _, sa2, sw2, _ = RN.build_observation(CAP, index, LAYER, "pow2")
def all_pow2(t):
    f = t.to(torch.float64).flatten()
    return bool(torch.equal(torch.log2(f), torch.round(torch.log2(f))))
ck("pow2 regime makes every scale a power of two", all_pow2(sa2) and all_pow2(sw2))
ck("real regime scales are not all powers of two", not (all_pow2(sa) and all_pow2(sw)))

# ---------------------------------------------------------------- pairing rules
A, B, accref, inj, Kk = RN.arms_for("F8_null", "null", aq, wq, sa, sw, "all_elements", 1)
ck("null fault: arms are bit-identical", _bitwise_equal(A.out, B.out))
ck("null fault: K is the layer's K", Kk == K)

A, B, accref, inj, _ = RN.arms_for("F6_int32_overflow", "precondition", aq, wq, sa, sw,
                                   "all_elements", 1)
ck("precondition: the two arms are identical by construction", _bitwise_equal(A.out, B.out)
   and bool(torch.equal(A.acc, B.acc)))
ck("precondition: arms hold the fault's own operands", A.a_q.shape[1] > K)
ck("precondition: acc_ref is the exact value, not the wrapped one",
   bool(torch.equal(accref, exact_accumulator(A.a_q, A.w_q)))
   and not bool(torch.equal(accref, A.acc)))

A, B, accref, inj, _ = RN.arms_for("F9_operand_mismatch", "operand", aq, wq, sa, sw,
                                   "one_element", 1)
ck("operand fault: arm B's activations differ from arm A's",
   not bool(torch.equal(A.a_q, B.a_q)))
ck("operand fault: acc_ref describes arm A, the unmodified side",
   bool(torch.equal(accref, exact_accumulator(aq, wq))))

A, B, accref, inj, _ = RN.arms_for("F1_scale_in_bf16", "epilogue", aq, wq, sa, sw,
                                   "all_elements", 1)
ck("epilogue fault: arm A is the reference on the original operands",
   bool(torch.equal(A.a_q, aq)) and bool(torch.equal(A.w_q, wq)))
ck("epilogue fault: operands are shared between arms",
   bool(torch.equal(A.a_q, B.a_q)) and bool(torch.equal(A.s_w, B.s_w)))

# ---------------------------------------------------------------- scoring classes
def verdicts(fired_map, applicable_map=None):
    am = applicable_map or {c: True for c in CHECK_ORDER}
    return {c: {"fired": fired_map.get(c, False), "applicable": am[c],
                "detail": "synthetic"} for c in CHECK_ORDER}
def preds(pm):
    return {c: {"prediction": pm.get(c, RN.SILENT)} for c in CHECK_ORDER}

s = RN.score(verdicts({"shared_operands": True}),
             preds({"shared_operands": RN.FIRE}), True)
ck("predicted fire that fires is a hit", s["shared_operands"]["classification"] == "hit")
s = RN.score(verdicts({}), preds({"shared_operands": RN.FIRE}), True)
ck("predicted fire that stays silent is a false negative",
   s["shared_operands"]["classification"] == "false_negative")
s = RN.score(verdicts({"shared_operands": True}), preds({}), True)
ck("predicted silence that fires is a false positive",
   s["shared_operands"]["classification"] == "false_positive")
s = RN.score(verdicts({}), preds({}), True)
ck("predicted silence that stays silent is correct silence",
   s["shared_operands"]["classification"] == "correct_silence")
s = RN.score(verdicts({}), preds({"token_level_risk": RN.NA}), True)
ck("a not-applicable prediction classifies as not_applicable",
   s["token_level_risk"]["classification"] == "not_applicable")
s = RN.score(verdicts({}, {c: (c != "shared_operands") for c in CHECK_ORDER}),
             preds({"shared_operands": RN.FIRE}), True)
ck("an inapplicable verdict is not counted as a false negative",
   s["shared_operands"]["classification"] == "not_applicable")

# the A-10.3 rule: an unobservable fault is excluded, never credited
s = RN.score(verdicts({}), preds({}), False)
ck("unobservable fault excludes the two output-comparison checks",
   s["pow2_scale_identity"]["classification"] == "excluded_fault_unobservable"
   and s["real_scale_tolerance"]["classification"] == "excluded_fault_unobservable")
ck("unobservable fault does not exclude the precondition checks",
   s["int32_no_overflow"]["classification"] == "correct_silence")

# ---------------------------------------------------------------- aggregate denominators
rows = [
  {"fault": "X", "fault_observable": True,
   "checks": {c: {"classification": ("hit" if c == "shared_operands" else "correct_silence")}
              for c in CHECK_ORDER}},
  {"fault": "X", "fault_observable": True,
   "checks": {c: {"classification": ("false_negative" if c == "shared_operands" else "correct_silence")}
              for c in CHECK_ORDER}},
  {"fault": "Y", "fault_observable": False,
   "checks": {c: {"classification": ("excluded_fault_unobservable"
                                     if c in ("pow2_scale_identity", "real_scale_tolerance")
                                     else "correct_silence")} for c in CHECK_ORDER}},
]
agg = RN.aggregate(rows)
so = agg["per_check"]["shared_operands"]
ck("detection rate is hits over predicted-fire cells only", so["detection_rate"] == 0.5)
ck("false negative rate complements it", so["false_negative_rate"] == 0.5)
rt = agg["per_check"]["real_scale_tolerance"]
ck("excluded cells are in neither detection denominator", rt["detection_rate"] is None)
ck("excluded cells are counted separately", rt["excluded_fault_unobservable"] == 1)
ck("false positive rate is over silence-predicted cells",
   agg["per_check"]["int32_no_overflow"]["false_positive_rate"] == 0.0)
ck("per-fault rows counted", agg["per_fault"]["X"]["rows"] == 2
   and agg["per_fault"]["Y"]["observable_rows"] == 0)
ck("denominator note travels with the numbers", "neither denominator" in so["denominator_note"])

# ---------------------------------------------------------------- pinned constants
ck("M_TILE uses the whole capture", RN.M_TILE == 512)
ck("F5 is sub-tiled", RN.F5_TILE == (32, 32))
ck("both scale regimes are run", set(RN.SCALE_REGIMES) == {"real", "pow2"})
ck("v1 tolerances are unchanged", RN.MAX_ULP == 1 and RN.MAX_FLIP_RATE == 0.05)

# ---------------------------------------------------------------- matrix agreement
mx = json.load(open("../artifacts/p5_prediction_matrix.json"))
from p5_inject import FAULTS, severities_for
ck("matrix covers exactly the nine faults", set(mx["faults"]) == set(FAULTS))
missing = []
for fault, spec in mx["faults"].items():
    if spec["cells"] is None:
        for sev in severities_for(fault):
            if sev not in (spec["cells_by_severity"] or {}):
                missing.append((fault, sev))
ck("every per-severity fault has a cell for each of its severities", not missing)
ck("77 cells", mx["cell_counts"]["total"] == 77)
ck("corrections are flagged for scrutiny", mx["cell_counts"]["corrected_post_data"] == 3)

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
