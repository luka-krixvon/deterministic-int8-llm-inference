"""P6 stage A: derive the evaluation windows, prove they are disjoint, emit them.

Runs in the quantization venv, which is the environment that has `datasets` and the
one make_prompts.py itself ran in. The pinned vLLM container has no `datasets`, so
the derivation cannot happen there, and installing it would mean the identity
contract records an image digest that never ran. Hence the split.

What this stage emits is a window file carrying the texts, their token spans, the
dataset provenance, the disjointness verdict, and a content hash over the windows.
Stage B (p6_accuracy.py, in the container) verifies that hash before scoring, so the
disjointness argument is established once, by the environment able to establish it,
and afterwards travels as a checkable artifact rather than as a step repeated per arm.

Usage (venv):
  python3 p6_windows.py --calib ../artifacts/calib_prompts.json \
                        --out ../artifacts/p6_eval_windows.json
"""
from __future__ import annotations

import argparse
import hashlib
import json

WINDOW = 320
N_STUDY_WINDOWS = 64          # what make_prompts.py took; do not change
N_EVAL_WINDOWS = 256          # pinned: 4x the study set, all disjoint from it
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SCHEMA = "p6-eval-windows-v1"


def derive_windows(n_skip: int, n_take: int):
    """Return (texts, token_spans, corpus_tokens) for windows [n_skip, n_skip+n_take).

    Same walk as make_prompts.py, parameterised by where to start. Spans come back so
    disjointness can be checked on token indices rather than on decoded strings, which
    could collide by coincidence.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                      revision=DATASET_REVISION)
    text = "".join(ds["text"])
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok(text, add_special_tokens=False)["input_ids"]

    texts, spans = [], []
    i = idx = 0
    while idx < n_skip + n_take and i + WINDOW <= len(ids):
        if idx >= n_skip:
            texts.append(tok.decode(ids[i:i + WINDOW]))
            spans.append([i, i + WINDOW])
        i += WINDOW
        idx += 1
    if len(texts) < n_take:
        raise RuntimeError(f"corpus yields only {len(texts)} windows after skipping "
                           f"{n_skip}; need {n_take}")
    return texts, spans, len(ids)


def assert_disjoint_from_study(eval_spans, eval_texts, calib_path: str) -> dict:
    """Refuse to proceed unless the evaluation windows miss the study's 64 entirely.

    Two independent guards. Token spans must not intersect. And the decoded texts must
    not appear in the committed prompt file, which catches the case where the
    derivation drifted and produced study text from a different index.

    The first thing checked is neither of those: it is whether rederiving windows 0..63
    reproduces the committed calib_prompts.json. If it does not, the derivation or the
    dataset has changed and the disjointness claim cannot be made at all, so the run is
    refused rather than reported with a caveat.
    """
    study_texts, study_spans, _ = derive_windows(0, N_STUDY_WINDOWS)
    with open(calib_path) as fh:
        committed = json.load(fh)
    if study_texts != committed:
        raise RuntimeError(
            "rederived study windows do not match the committed calib_prompts.json; "
            "the derivation or the dataset revision has changed, so the disjointness "
            "claim cannot be made. Refusing to proceed.")
    overlaps = [[a, b] for a, b in eval_spans
                for c, d in study_spans if not (b <= c or d <= a)]
    if overlaps:
        raise RuntimeError(f"evaluation windows overlap study windows: {overlaps[:3]}")
    shared = set(committed) & set(eval_texts)
    if shared:
        raise RuntimeError(f"{len(shared)} evaluation texts appear in calib_prompts.json")
    return {
        "study_windows": N_STUDY_WINDOWS,
        "study_rederived_matches_committed": True,
        "token_span_overlaps": 0,
        "text_collisions": 0,
        "calib_prompts_sha256": hashlib.sha256(
            open(calib_path, "rb").read()).hexdigest(),
    }


def content_hash(texts, spans) -> str:
    """Hash over the windows themselves, independent of the surrounding metadata.

    Stage B recomputes this from the texts and spans it is about to score, so a file
    whose metadata was edited but whose windows were not still verifies, and a file
    whose windows were touched does not. Canonical JSON with sorted separators, so the
    value does not depend on how the file happened to be written.
    """
    payload = json.dumps({"texts": texts, "spans": spans}, ensure_ascii=False,
                         separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(calib_path: str, n_windows: int = N_EVAL_WINDOWS) -> dict:
    texts, spans, corpus_tokens = derive_windows(N_STUDY_WINDOWS, n_windows)
    disjointness = assert_disjoint_from_study(spans, texts, calib_path)
    import transformers
    return {
        "schema": SCHEMA,
        "content_sha256": content_hash(texts, spans),
        "n_windows": len(texts),
        "windows_skipped": N_STUDY_WINDOWS,
        "window_tokens": WINDOW,
        "corpus_tokens": corpus_tokens,
        "windows_available": corpus_tokens // WINDOW,
        "dataset": {"name": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
                    "split": "test", "revision": DATASET_REVISION, "tokenizer": "gpt2"},
        "derived_with": {"transformers": transformers.__version__,
                         "note": ("Derived in the quantization venv, the environment "
                                  "make_prompts.py used. The pinned vLLM container has "
                                  "no datasets package and a different transformers "
                                  "version, which is why stage A and stage B are "
                                  "separate processes.")},
        "disjointness": disjointness,
        "spans": spans,
        "texts": texts,
    }


def load_verified(path: str) -> dict:
    """Read a window file and verify its content hash. Used by stage B.

    Raises rather than warning. A stage that scored windows it could not verify would
    make the disjointness commitment unfalsifiable.
    """
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected schema {doc.get('schema')!r}, want {SCHEMA!r}")
    actual = content_hash(doc["texts"], doc["spans"])
    if actual != doc.get("content_sha256"):
        raise RuntimeError(
            "window file content hash mismatch: the texts or spans were modified after "
            f"derivation (recorded {doc.get('content_sha256')}, recomputed {actual}). "
            "Refusing to score.")
    if len(doc["texts"]) != len(doc["spans"]) != doc["n_windows"]:
        if not (len(doc["texts"]) == len(doc["spans"]) == doc["n_windows"]):
            raise RuntimeError("window file is internally inconsistent in length")
    if not doc.get("disjointness", {}).get("study_rederived_matches_committed"):
        raise RuntimeError("window file does not carry a passing disjointness verdict")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="../artifacts/calib_prompts.json")
    ap.add_argument("--n-windows", type=int, default=N_EVAL_WINDOWS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doc = build(a.calib, a.n_windows)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"{a.out}: {doc['n_windows']} windows, spans "
          f"{doc['spans'][0]}..{doc['spans'][-1]}, content_sha256 "
          f"{doc['content_sha256'][:16]}, {doc['windows_available']} available in corpus")


if __name__ == "__main__":
    main()
