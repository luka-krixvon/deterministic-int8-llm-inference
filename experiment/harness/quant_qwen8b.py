import json, os, re
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
MODEL = "Qwen/Qwen3-8B"
local = snapshot_download(MODEL)
m = re.search(r"snapshots/([0-9a-f]{40})", os.path.realpath(local))
json.dump({"model": MODEL, "revision": m.group(1) if m else "unknown"},
          open("models/parent8b_manifest.json", "w"), indent=2)
print("parent revision:", m.group(1) if m else "?", flush=True)
oneshot(model=local, dataset="open_platypus",
        recipe=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
        max_seq_length=2048, num_calibration_samples=512,
        output_dir="models/qwen3-8b-int8-w8a8")
print("INT8_8B_done", flush=True)
