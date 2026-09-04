"""Regression tests v5 — Codex review #4 counterexamples. CPU only."""
import copy, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_v3 import manifest_aware_join
from w3_p2_refit_v4 import ci_metadata

fails = []
def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond: fails.append(name)

FULL = "a" * 64
def envelope(**kw):
    m = {"requested_prompts": 1, "actual_prompts": 1,
         "prompt_list_sha256": "P" * 64, "rails_sha256": "R" * 64,
         "checkpoint_digest": "M" * 64, "model": "/models/x"}
    m.update(kw)
    return {"manifest": m,
            "records": [{"prompt_sha": FULL,
                         "steps": [{"pos": 1, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1},
                                   {"pos": 2, "margin": 2.0, "top1_id": 2, "chosen_lp": -0.2}]}]}

# baseline: fully consistent -> every flag True
A = envelope(); B = envelope()
pairs, d = manifest_aware_join(A, B)
check("identical envelopes: identity_verified True",
      d["identity_verified"] is True and all(
          d[k] for k in ("record_identity_verified", "prompt_manifest_verified",
                         "rails_verified", "model_identity_verified", "record_counts_verified")))

# P0 counterexample: identical records but differing manifest fields -> overall must be False
for field, val, flag in [("requested_prompts", 99, "record_counts_verified"),
                         ("prompt_list_sha256", "Q" * 64, "prompt_manifest_verified"),
                         ("rails_sha256", "S" * 64, "rails_verified"),
                         ("checkpoint_digest", "N" * 64, "model_identity_verified")]:
    B2 = envelope(**{field: val})
    pairs, d = manifest_aware_join(A, B2)
    check(f"manifest mismatch ({field}): overall False, {flag} False",
          d["identity_verified"] is False and d[flag] is False)

# mixed envelope/legacy -> fail closed
legacy = [[{"pos": 1, "margin": 1.0, "top1_id": 1, "chosen_lp": -0.1}]]
pairs, d = manifest_aware_join(A, legacy)
check("envelope/legacy mixed: fail closed", "error" in d and d["identity_verified"] is False)

# truncated sha (16 hex) -> record_identity_verified must be False
A16 = envelope(); A16["records"][0]["prompt_sha"] = "a" * 16
B16 = copy.deepcopy(A16)
pairs, d = manifest_aware_join(A16, B16)
check("truncated 16-hex prompt_sha: record identity NOT verified",
      d["record_identity_verified"] is False and d["identity_verified"] is False)

# differing model path but identical digest -> still verified (the path is diagnostic only)
Bp = envelope(model="/models/other-path")
pairs, d = manifest_aware_join(A, Bp)
check("model path differs but digest equal: still verified (path diagnostic)",
      d["identity_verified"] is True)

# CI metadata, dynamic role
m5, m20 = ci_metadata(5), ci_metadata(20)
check("5-seed role exploratory / 20-seed role primary",
      m5["role"] == "exploratory" and m20["role"] == "primary")
check("conditional is separate boolean, both True",
      m5["conditional"] is True and m20["conditional"] is True)
check("no EXPLORATORY in stable keys", "EXPLORATORY" not in json.dumps(list(m20.keys())))

# P4 driver: all-odd prompt IDs -> empty-calibration-split status (subprocess)
tmp = tempfile.mkdtemp()
def mkrec(pid_count, margin_flip):
    # in the legacy format the prompt index comes from position, so producing an odd-only ev set needs >=2 prompts
    return [[{"pos": i, "margin": float(1 + (i % 7)), "top1_id": 1 if (i + j) % 3 else 2,
              "chosen_lp": -0.1} for i in range(30)] for j in range(pid_count)]
# two prompts: index 0 (even, cal) and 1 (odd, ev) -> normal; a single prompt at index 0 -> ev empty
one = mkrec(1, None)
other = json.loads(json.dumps(one))
for s in other[0]:
    s["top1_id"] = 99 if s["pos"] % 2 else s["top1_id"]   # manufacture both flip classes
pa = os.path.join(tmp, "a.json"); json.dump(one, open(pa, "w"))
pb = os.path.join(tmp, "b.json"); json.dump(other, open(pb, "w"))
out = os.path.join(tmp, "o.json")
rc = subprocess.run([sys.executable,
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "w4_p4_stats_v4.py"),
                     pa, pb, out], capture_output=True, text=True)
ok = False
if rc.returncode == 0 and os.path.exists(out):
    dd = json.load(open(out))
    ok = dd.get("status") in ("empty-evaluation-split", "degenerate-data")
check("single-even-prompt driver: explicit empty/degenerate status", ok)

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
