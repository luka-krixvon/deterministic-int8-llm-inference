"""Tests for P6 stage A: window derivation, the disjointness guard, and the content
hash that stage B verifies. These are the tests behind the pre-registered claim that
the evaluation windows do not touch the 64 the study used.

Needs the wikitext-2 cache and the GPT-2 tokenizer, so it runs in the quantization
venv. Set CALIB_PROMPTS if calib_prompts.json is not at the default relative path.
"""
import json, os, sys, tempfile
sys.path.insert(0, '.')
from p6_windows import (derive_windows, assert_disjoint_from_study, content_hash,
                        build, load_verified, N_STUDY_WINDOWS, N_EVAL_WINDOWS,
                        WINDOW, SCHEMA)

P = F = 0
def ck(n, c):
    global P, F
    if c: P += 1; print(f"PASS {n}")
    else: F += 1; print(f"FAIL {n}")

CAL = os.environ.get('CALIB_PROMPTS', '../artifacts/calib_prompts.json')

# --- the load-bearing assertion: the derivation reproduces what the study used
study, sspans, n_ids = derive_windows(0, N_STUDY_WINDOWS)
committed = json.load(open(CAL))
ck("derivation reproduces calib_prompts.json exactly", study == committed)
ck("64 study windows", len(study) == 64 and len(sspans) == 64)
ck("study spans are contiguous non-overlapping 320-token",
   sspans[0] == [0, 320] and sspans[-1] == [63*320, 64*320])
print(f"  corpus tokens {n_ids}, windows available {n_ids//WINDOW}")

ev, espans, _ = derive_windows(N_STUDY_WINDOWS, N_EVAL_WINDOWS)
ck("256 eval windows derived", len(ev) == 256)
ck("eval starts right after the study set", espans[0] == [64*320, 65*320])
ck("no eval span intersects a study span",
   all(b <= c or d <= a for a, b in espans for c, d in sspans))
ck("eval texts and study texts are disjoint sets", not (set(ev) & set(study)))

d = assert_disjoint_from_study(espans, ev, CAL)
ck("guard passes on the real eval set", d["token_span_overlaps"] == 0)
ck("guard records the calib file hash", len(d["calib_prompts_sha256"]) == 64)

# --- the guard must fire, in both of its two independent ways
try:
    assert_disjoint_from_study(sspans, study, CAL); ck("guard fires on span overlap", False)
except RuntimeError as e: ck("guard fires on span overlap", "overlap" in str(e).lower())

tf = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
json.dump(committed[:-1] + ["tampered"], tf); tf.close()
try:
    assert_disjoint_from_study(espans, ev, tf.name); ck("guard fires on derivation drift", False)
except RuntimeError as e: ck("guard fires on derivation drift", "do not match" in str(e))
os.unlink(tf.name)

# --- content hash: over the windows only, stable, and order-sensitive
h1 = content_hash(ev, espans)
ck("content hash is 64 hex", len(h1) == 64)
ck("content hash is stable across calls", h1 == content_hash(ev, espans))
ck("content hash changes if a text changes",
   h1 != content_hash(ev[:-1] + [ev[-1] + " "], espans))
ck("content hash changes if a span changes",
   h1 != content_hash(ev, espans[:-1] + [[0, 320]]))
ck("content hash changes if window order changes",
   h1 != content_hash(list(reversed(ev)), espans))

# --- the file stage B reads
doc = build(CAL, 8)
p = tempfile.mktemp(suffix='.json')
json.dump(doc, open(p, 'w'), ensure_ascii=False)
ck("built file carries schema, hash, verdict",
   doc["schema"] == SCHEMA and len(doc["content_sha256"]) == 64
   and doc["disjointness"]["study_rederived_matches_committed"] is True)
ck("built file records the derivation environment", "transformers" in doc["derived_with"])
v = load_verified(p)
ck("load_verified accepts an untouched file", v["content_sha256"] == doc["content_sha256"])

# metadata may be edited without breaking verification; windows may not
d2 = dict(doc); d2["n_windows"] = doc["n_windows"]; d2["corpus_tokens"] = 1
p2 = tempfile.mktemp(suffix='.json'); json.dump(d2, open(p2, 'w'), ensure_ascii=False)
ck("verification is over the windows, not the metadata", load_verified(p2) is not None)

for label, mutate in (
    ("a text edited", lambda x: x.update(texts=x["texts"][:-1] + [x["texts"][-1] + "!"])),
    ("a span edited", lambda x: x.update(spans=x["spans"][:-1] + [[0, 320]])),
    ("windows reordered", lambda x: x.update(texts=list(reversed(x["texts"])))),
):
    bad = json.loads(json.dumps(doc)); mutate(bad)
    pb = tempfile.mktemp(suffix='.json'); json.dump(bad, open(pb, 'w'), ensure_ascii=False)
    try:
        load_verified(pb); ck(f"load_verified refuses: {label}", False)
    except RuntimeError as e: ck(f"load_verified refuses: {label}", "hash mismatch" in str(e))

bad = json.loads(json.dumps(doc)); bad["schema"] = "something-else"
pb = tempfile.mktemp(suffix='.json'); json.dump(bad, open(pb, 'w'), ensure_ascii=False)
try:
    load_verified(pb); ck("load_verified refuses a wrong schema", False)
except RuntimeError as e: ck("load_verified refuses a wrong schema", "schema" in str(e))

bad = json.loads(json.dumps(doc))
bad["disjointness"]["study_rederived_matches_committed"] = False
pb = tempfile.mktemp(suffix='.json'); json.dump(bad, open(pb, 'w'), ensure_ascii=False)
try:
    load_verified(pb); ck("load_verified refuses a failing verdict", False)
except RuntimeError as e: ck("load_verified refuses a failing verdict", "verdict" in str(e))

# --- stage B must not need datasets: check the module-level imports stay light
# Check for real import statements, not the word appearing in prose: the docstrings
# discuss the datasets package precisely because the split exists because of it.
import re as _re
def _module_level_imports(src, first_def):
    head = src[:src.index(first_def)]
    return _re.findall(r'^\s*(?:import|from)\s+([A-Za-z0-9_.]+)', head, _re.M)
ck("stage A defers datasets and transformers to inside functions",
   not {'datasets', 'transformers'} & set(_module_level_imports(
       open('p6_windows.py').read(), 'def derive_windows')))
ck("stage B defers vllm to inside window_nll",
   'vllm' not in set(_module_level_imports(open('p6_accuracy.py').read(), 'def window_nll')))
ck("stage B does import p6_windows at module level (needs load_verified)",
   'p6_windows' in set(_module_level_imports(open('p6_accuracy.py').read(), 'def window_nll')))

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
