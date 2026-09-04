"""Regression tests v6 — Codex review #5 counterexamples. CPU only."""
import copy, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_v3 import manifest_aware_join_v7, is_sha256

fails = []
def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond: fails.append(name)

H = lambda c: c * 64
def env(arm="CUTLASS", **kw):
    m = {"requested_prompts": 1, "actual_prompts": 1,
         "prompt_list_sha256": H("a"), "rails_sha256": H("b"),
         "checkpoint_digest": H("c"), "quantization_config_sha256": H("d"),
         "tokenizer_files_sha256": H("e"), "parent_revision": "rev123",
         "vllm_version": "0.27.1", "model": "/models/x", "generator": "tf_v6.py",
         "arm": arm, "kernel_class": f"{arm}Int8ScaledMMLinearKernel",
         "kernel_selection_env": "" if arm == "CUTLASS" else "CutlassInt8ScaledMMLinearKernel",
         "kernel_log_sha256": H("f"), "command_sha256": H("0")}
    m.update(kw)
    return {"manifest": m,
            "records": [{"prompt_sha": H("9"),
                         "steps": [{"pos": 1, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1},
                                   {"pos": 2, "margin": 2.0, "top1_id": 2, "chosen_lp": -0.2}]}]}

# is_sha256
check("is_sha256 accepts 64 lowercase hex", is_sha256(H("a")))
check("is_sha256 rejects uppercase/short/non-hex",
      not is_sha256(H("A")) and not is_sha256("a" * 16) and not is_sha256(H("z")))

# baseline: a legal treatment pair (arm/kernel/env/log differ, everything else same) -> verified
A = env("CUTLASS"); B = env("TRITON", kernel_log_sha256=H("1"), command_sha256=H("2"),
                            model="/models/x")
pairs, d = manifest_aware_join_v7(A, B)
check("valid treatment pair: identity_verified True, pairs returned",
      d["identity_verified"] is True and len(pairs) == 2)

# same weights, different quant config -> fail closed
B2 = env("TRITON", quantization_config_sha256=H("7"))
pairs, d = manifest_aware_join_v7(A, B2)
check("same weights, different quant config: FAIL closed, no pairs",
      d["identity_verified"] is False and d["quant_config_verified"] is False and pairs == [])

# different tokenizer / parent / runtime -> the corresponding flag fails
for field, val, flag in [("tokenizer_files_sha256", H("8"), "tokenizer_verified"),
                         ("parent_revision", "otherrev", "parent_revision_verified"),
                         ("vllm_version", "0.99.0", "runtime_verified")]:
    Bx = env("TRITON", **{field: val})
    pairs, d = manifest_aware_join_v7(A, Bx)
    check(f"different {field}: {flag} False and fail closed",
          d[flag] is False and d["identity_verified"] is False and pairs == [])

# non-hex or wrong-length digest -> fail, even when both arms are equal
for bad in ("z" * 64, "not-a-sha-but-equal", "x", "A" * 64):
    Ax = env("CUTLASS", checkpoint_digest=bad); Bx = env("TRITON", checkpoint_digest=bad)
    pairs, d = manifest_aware_join_v7(Ax, Bx)
    check(f"invalid digest format ({bad[:8]}…): weights_verified False",
          d["weights_verified"] is False and d["identity_verified"] is False)

# an undeclared field difference (not the treatment) -> treatment_identity_verified False
Bu = env("TRITON"); Bu["manifest"]["extra_setting"] = "different"
pairs, d = manifest_aware_join_v7(A, Bu)
check("undeclared field difference: treatment identity False",
      d["treatment_identity_verified"] is False and "extra_setting" in d["undeclared_differences"])

# missing kernel-selection evidence -> treatment unverified
Bn = env("TRITON", kernel_log_sha256=None)
pairs, d = manifest_aware_join_v7(A, Bn)
check("missing kernel evidence: treatment identity False",
      d["treatment_identity_verified"] is False)

# --allow-unverified: return the pairs, with the schema flagging it
Bq = env("TRITON", quantization_config_sha256=H("7"))
pairs, d = manifest_aware_join_v7(A, Bq, allow_unverified=True)
check("allow_unverified: pairs returned and flagged",
      len(pairs) == 2 and d["allow_unverified_used"] is True and d["identity_verified"] is False)

# kernel_log_lines differing between arms is the declared treatment evidence -> still verified
Ak = env("CUTLASS"); Ak["manifest"]["kernel_log_lines"] = ["Selected CutlassInt8..."]
Bk = env("TRITON", kernel_log_sha256=H("1")); Bk["manifest"]["kernel_log_lines"] = ["Selected TritonInt8..."]
pairs, d = manifest_aware_join_v7(Ak, Bk)
check("differing kernel_log_lines are declared treatment, still verified",
      d["identity_verified"] is True)

# ---- P1-2 genuine empty-split counterexample (4 prompts, margin=None hollows one side) ----
tmp = tempfile.mkdtemp()
def prompt_steps(has_margin, flip_seed):
    return [{"pos": i, "margin": (float(1 + i % 5) if has_margin else None),
             "top1_id": (2 if (i + flip_seed) % 3 == 0 else 1), "chosen_lp": -0.1}
            for i in range(30)]
def build(only):  # only='odd' -> every even prompt gets margin=None -> cal is empty
    A, B = [], []
    for pid in range(4):
        keep = (pid % 2 == 1) if only == "odd" else (pid % 2 == 0)
        A.append(prompt_steps(keep, 0))
        B.append(prompt_steps(keep, 1))
    return A, B
drv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w4_p4_stats_v4.py")
for only, want in (("odd", "empty-calibration-split"), ("even", "empty-evaluation-split")):
    Aj, Bj = build(only)
    pa = os.path.join(tmp, f"a_{only}.json"); json.dump(Aj, open(pa, "w"))
    pb = os.path.join(tmp, f"b_{only}.json"); json.dump(Bj, open(pb, "w"))
    out = os.path.join(tmp, f"o_{only}.json")
    rc = subprocess.run([sys.executable, drv, pa, pb, out], capture_output=True, text=True)
    st = json.load(open(out)).get("status") if rc.returncode == 0 and os.path.exists(out) else None
    check(f"only-{only}-prompt pairs -> status == {want} (got {st})", st == want)

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
