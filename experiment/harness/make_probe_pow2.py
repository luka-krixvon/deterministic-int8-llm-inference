"""Probe level 1: copy the INT8-W8A8 checkpoint and rewrite every
weight_scale to the nearest power of two (log-space rounding). Nothing else
changes. Quality is irrelevant — this is a conformance probe (prereg A-2)."""
import glob, hashlib, json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
n_scales = 0
for shard in glob.glob(os.path.join(dst, "*.safetensors")):
    t = load_file(shard)
    changed = False
    for k in list(t):
        if k.endswith("weight_scale"):
            s = t[k].float()
            t[k] = torch.exp2(torch.round(torch.log2(s))).to(t[k].dtype)
            n_scales += s.numel()
            changed = True
    if changed:
        save_file(t, shard)
h = hashlib.sha256()
for shard in sorted(glob.glob(os.path.join(dst, "*.safetensors"))):
    h.update(open(shard, "rb").read())
print(f"rewrote {n_scales} scale values; probe sha256[:16] = {h.hexdigest()[:16]}")
print("PROBE_BUILD_OK")
