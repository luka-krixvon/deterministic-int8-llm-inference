"""Regression tests v4 — Codex review #3 counterexamples. CPU+GPU-optional.
Run: .venv/bin/python3 tests_metrics_v4.py"""
import json, math, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from metrics_v3 import divergence_report, strict_pair_join_v3

fails = []
def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond: fails.append(name)

# 1. a large absolute error on near-zero elements must show up in the global abs_over_rms
a = torch.tensor([100.0, 0.0], dtype=torch.bfloat16)
b = torch.tensor([100.0, 100.0], dtype=torch.bfloat16)
r = divergence_report(a, b)
check("near-zero large abs error visible in global abs_over_rms",
      r["abs_over_rms"]["max"] is not None and r["abs_over_rms"]["max"] > 1.0)
check("main-only normalized summary kept under its own name",
      "abs_over_rms_main_non_nearzero" in r)

# 2. extreme finite bf16 values: the fp64 computation must not overflow
big = torch.tensor([3.39e38], dtype=torch.bfloat16)
r = divergence_report(big, -big)
check("extreme finite bf16: status ok, rms finite",
      r["status"] == "ok" and r["rms_finite_ref"] is not None)
check("extreme finite bf16: abs diff finite in fp64",
      r["abs_err_finite_pairs"]["max"] is not None)

# 3. device guard (only run when a GPU is present)
if torch.cuda.is_available():
    try:
        divergence_report(torch.zeros(2, dtype=torch.bfloat16),
                          torch.zeros(2, dtype=torch.bfloat16).cuda())
        check("device guard raises", False)
    except ValueError:
        check("device guard raises", True)
else:
    print("SKIP device guard (no GPU)")

# 4/5. the alpha boundary optimum must be found
from w3_p2_refit_v4 import fit, identifiable
f = fit([512, 1024, 2048, 4096], [0.4, 0.3, 0.2, 0.1], "M_gensat")
check("decreasing p: alpha=0 boundary optimum SUCCEEDS",
      f.get("status") == "ok" and abs(f["alpha"]) < 1e-6 and f.get("alpha_at_bound"))
f2 = fit([512, 1024, 2048, 4096], [1e-6, 1e-4, 1e-2, 0.9], "M_pow")
check("steep p: fit succeeds within bounds",
      f2.get("status") == "ok" and 0 <= f2["alpha"] <= 1.5)

# 6. schema guards
check("negative K flagged", identifiable([-512, 1024, 2048], [0.1, 0.2, 0.3], [10]*3) == "non-positive-K")
check("p out of range flagged", identifiable([512, 1024, 2048], [0.1, 1.2, 0.3], [10]*3) == "p-out-of-range")
check("count inconsistency flagged",
      identifiable([512, 1024, 2048], [0.1, 0.2, 0.3], [10]*3, nds=[1, 11, 2]) == "count-inconsistent")
check("incomplete K grid per seed flagged",
      identifiable([512, 1024, 2048, 512, 1024], [0.1, 0.2, 0.3, 0.1, 0.2], [10]*5,
                   seeds_ks={0: [512, 1024, 2048], 1: [512, 1024]}) == "incomplete-K-grid-per-seed")

# 7. mixed schema join fail closed
mixed = [{"prompt_sha": "aaa", "steps": [{"pos": 1, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]},
         [{"pos": 1, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]]
_, diag = strict_pair_join_v3(mixed, mixed)
check("mixed schema fail closed", "error" in diag and "mixed-schema" in diag["error"])

# 8. the tf_v4 envelope is accepted
env = {"manifest": {"requested_prompts": 1}, "records": [
    {"prompt_sha": "aaa", "steps": [{"pos": 1, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]}]}
pairs, diag = strict_pair_join_v3(env, env)
check("tf_v4 envelope accepted with identity verified",
      diag["identity_verified"] is True and diag["n_pairs_used"] == 1)

# 9. P4 driver: single-class data degrades gracefully (subprocess)
tmp = tempfile.mkdtemp()
one_class = [[{"pos": i, "margin": 1.0 + i, "top1_id": 1, "chosen_lp": -0.1} for i in range(30)]
             for _ in range(4)]
pa = os.path.join(tmp, "a.json"); json.dump(one_class, open(pa, "w"))
out = os.path.join(tmp, "o.json")
rc = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "w4_p4_stats_v4.py"), pa, pa, out], capture_output=True, text=True)
check("single-class driver exits gracefully (DEGENERATE_DATA)",
      rc.returncode == 0 and "DEGENERATE_DATA" in rc.stdout)

# 10. empty-group merging: the review counterexample (25*m0 + 73*m1) must not produce a meta bin with cal_n=0
from w4_p4_stats_v4 import build_bins_v4, MIN_BIN_N
cal = [(0, 0.0, False, None)] * 25 + [(0, 1.0, False, None)] * 73
bins = build_bins_v4(cal)
check("no empty meta bins after zero-group merge",
      all(x["cal_n"] > 0 for x in bins["meta"]))
check("all bins >= MIN_BIN_N", all(x["cal_n"] >= MIN_BIN_N for x in bins["meta"]))

# 11/12. the restored fields are present (quick driver run on the older 8-prompt data)
here = os.path.dirname(os.path.abspath(__file__))
cands = [here, os.path.join(here, "..", "artifacts")]
art = next((c for c in cands if os.path.exists(os.path.join(c, "tf_CUTLASS.json"))), cands[0])
tfC = os.path.join(art, "tf_CUTLASS.json"); tfT = os.path.join(art, "tf_TRITON.json")
if os.path.exists(tfC):
    out2 = os.path.join(tmp, "p4.json")
    rc = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "w4_p4_stats_v4.py"), tfC, tfT, out2, "--isotonic"],
                       capture_output=True, text=True)
    d = json.load(open(out2))
    check("Brier prompt-bootstrap CI restored",
          d.get("brier_prompt_bootstrap_ci90_CONDITIONAL") is not None)
    check("cal_flips present in curve",
          all("cal_flips" in row for row in d["calibration_curve_heldout"]))
    check("isotonic heldout curve + fallback count present",
          d["isotonic_sensitivity"] is not None
          and "heldout_curve" in d["isotonic_sensitivity"]
          and "fallback_count" in d["isotonic_sensitivity"])
else:
    print("SKIP driver field checks (artifacts not present)")

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
