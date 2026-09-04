"""Tests for the P6 scripts' pure logic: window derivation and disjointness live in
tests run against the real corpus (see the record in run_logs); here we cover the
summary statistics and the comparison paths, which decide how a cost gets reported."""
import json, os, sys, tempfile, statistics
sys.path.insert(0, '.')
from p6_throughput import _iqr, _summarise, BATCH_ISL_GRID, OSL, REPEATS, WARMUP
from p6_accuracy import bootstrap_delta, ppl_from_rows

P = F = 0
def ck(n, c):
    global P, F
    if c: P += 1; print(f"PASS {n}")
    else: F += 1; print(f"FAIL {n}")

# --- summary statistics: median must be an observed value, spread must be reported
s = _summarise([10.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0], "x")
ck("median is an observed value", s["median"] in [10.0,1.0,3.0,2.0,5.0,4.0,6.0])
ck("median is the middle, not the mean", s["median"] == 4.0)
ck("min and max reported", s["min"] == 1.0 and s["max"] == 10.0)
ck("full series retained for inspection", len(s["series"]) == 7)
ck("IQR present when n>=4", s["spread"] is not None and s["spread"]["iqr"] > 0)
ck("IQR absent when n<4", _summarise([1.0,2.0], "y")["spread"] is None)
# an outlier must not move the median the way it moves a mean
base = [5.0]*6 + [5.0]
out  = [5.0]*6 + [500.0]
ck("outlier does not move the median",
   _summarise(base,"a")["median"] == _summarise(out,"b")["median"])
ck("outlier does move the mean (why we report median)",
   statistics.mean(base) != statistics.mean(out))

# --- pinned grid is fixed and odd-repeat
ck("grid has six points", len(BATCH_ISL_GRID) == 6)
ck("grid covers batch 1,4,16", {b for b,_ in BATCH_ISL_GRID} == {1,4,16})
ck("grid covers short and long ISL", 128 in {i for _,i in BATCH_ISL_GRID} and 2048 in {i for _,i in BATCH_ISL_GRID})
ck("repeats odd so median is observed", REPEATS % 2 == 1)
ck("warmup nonzero", WARMUP > 0)
ck("osl > 1 so decode is measurable", OSL > 1)

# --- comparison path: grid mismatch must be refused, not silently zipped
def arm(tps_prefill, tps_decode, grid=None):
    grid = grid or BATCH_ISL_GRID
    return {"grid": [{"batch": b, "isl": i,
                      "prefill_tokens_per_s": {"median": tps_prefill},
                      "decode_tokens_per_s": {"median": tps_decode}} for b, i in grid],
            "identity": {"checkpoint_digest": "deadbeef"}}

import p6_throughput as T
def run_compare(a_obj, b_obj):
    d = tempfile.mkdtemp()
    pa, pb, po = (os.path.join(d, n) for n in ("a.json","b.json","o.json"))
    json.dump(a_obj, open(pa,"w")); json.dump(b_obj, open(pb,"w"))
    sys.argv = ["p6_throughput.py", "--compare", pa, pb, "--out", po]
    T.main()
    return json.load(open(po))

r = run_compare(arm(1000.0, 100.0), arm(900.0, 95.0))
ck("relative change is negative when pow2 is slower",
   all(x["prefill_relative_change"] < 0 for x in r["rows"]))
ck("relative change magnitude correct (10% slower prefill)",
   abs(r["rows"][0]["prefill_relative_change"] + 0.10) < 1e-12)
ck("decode change reported separately",
   abs(r["rows"][0]["decode_relative_change"] + 0.05) < 1e-12)
ck("comparison keeps both identity blocks", set(r["identity"]) == {"base","pow2"})
ck("commitment text travels with the result", "whatever it says" in r["commitment"])

try:
    run_compare(arm(1000.0,100.0), arm(1000.0,100.0, grid=[(1,128)]))
    ck("grid mismatch refused", False)
except RuntimeError as e:
    ck("grid mismatch refused", "different grids" in str(e))

# missing decode data must yield None, not a crash or a fabricated zero
a2 = arm(1000.0, 100.0); b2 = arm(900.0, 95.0)
for row in b2["grid"]: row["decode_tokens_per_s"] = None
r2 = run_compare(a2, b2)
ck("absent decode data -> None, not fabricated",
   r2["rows"][0]["decode_relative_change"] is None and r2["rows"][0]["decode_tps_pow2"] is None)

# --- accuracy: a zero-cost intervention must show a CI that includes zero
rows = [{"window": i, "nll_sum": 100.0 + (i % 7), "n_tokens": 319} for i in range(64)]
bs = bootstrap_delta(rows, [dict(r) for r in rows], n_boot=2000)
ck("identical arms -> zero delta and CI containing zero",
   bs["ci90_low"] <= 0.0 <= bs["ci90_high"] and abs(bs["median"]) < 1e-9)
p, s_, n_ = ppl_from_rows(rows)
ck("ppl uses token-weighted mean nll", abs(p - __import__("math").exp(s_/n_)) < 1e-12)

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
