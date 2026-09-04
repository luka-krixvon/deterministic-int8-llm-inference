"""Quantize Qwen3-14B to INT8 W8A8, the same recipe shape quant_qwen8b.py uses.

One deliberate difference from quant_qwen.py and quant_qwen8b.py: the parent revision is
passed explicitly. Those two call snapshot_download(MODEL) with no revision and record
whatever they resolved, which the decision log notes as a defect -- a rebuild there resolves
main and overwrites the manifest that recorded the pin. Here the revision pinned in A-12's
second scope amendment is passed in, so the checkpoint's parent cannot drift.

Produces the base arm for the 14B replication. The pow2 arm is then built by p8_requant.py
using this checkpoint as its structure reference, which is the same flow the 8B replication
used and keeps 'the arms differ only in the scale constraint' resting on A-12.1's
byte-for-byte gate rather than on a structure I inferred from the parent.
"""
import json
import os

from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a9"          # pinned in A-12's second scope amendment
OUT = "models/qwen3-14b-int8-w8a8"

local = snapshot_download(MODEL, revision=REVISION)
resolved = os.path.basename(os.path.realpath(local))
json.dump({"model": MODEL, "revision_requested": REVISION, "revision_resolved": resolved,
           "note": "revision passed explicitly, unlike quant_qwen{,8b}.py"},
          open("models/parent14b_manifest.json", "w"), indent=2)
print(f"parent {MODEL}@{REVISION} -> {local}", flush=True)

oneshot(model=local, dataset="open_platypus",
        recipe=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
        max_seq_length=2048, num_calibration_samples=512,
        output_dir=OUT)
print("INT8_14B_done", flush=True)
