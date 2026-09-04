"""integer-alibi W1: produce pinned quantized checkpoints of Qwen3-1.7B.
FP8-DYNAMIC first (no calibration), then INT8-W8A8 (channelwise weights +
dynamic per-token activations, 512 calibration samples). Revision recorded."""
import json, os
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL = "Qwen/Qwen3-1.7B"
local = snapshot_download(MODEL)
rev = os.path.basename(os.path.realpath(os.path.join(local, "..", "..", "snapshots"))) if False else None
# record the resolved commit hash from the cache path
import re
m = re.search(r"snapshots/([0-9a-f]{40})", os.path.realpath(local))
manifest = {"model": MODEL, "revision": m.group(1) if m else "unknown"}
json.dump(manifest, open("models/parent_manifest.json", "w"), indent=2)
print("parent revision:", manifest["revision"], flush=True)

print("=== FP8_DYNAMIC ===", flush=True)
oneshot(
    model=local,
    recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
    output_dir="models/qwen3-1.7b-fp8-dynamic",
)
print("FP8 done", flush=True)

print("=== INT8 W8A8 (channelwise W, dynamic per-token A) ===", flush=True)
oneshot(
    model=local,
    dataset="open_platypus",
    recipe=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    max_seq_length=2048,
    num_calibration_samples=512,
    output_dir="models/qwen3-1.7b-int8-w8a8",
)
print("INT8 done", flush=True)
