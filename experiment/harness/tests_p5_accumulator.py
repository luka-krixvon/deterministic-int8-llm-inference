"""The fp64 accumulator path must be bit-identical to int64 on real layer shapes.

Runs against the committed per-layer captures and the rebuilt checkpoint, so it checks
the shapes P5 will actually see rather than synthetic ones. Needs the VM's capture
directory and models; set P5_CAPTURE and P5_MODEL to relocate.
"""
import glob, sys, torch
sys.path.insert(0,'.')
from p5_inject import exact_accumulator, FP64_EXACT_MAX_TERMS
from safetensors.torch import safe_open
P=F=0
def ck(n,c):
    global P,F
    if c: P+=1; print(f"PASS {n}")
    else: F+=1; print(f"FAIL {n}")
ck("fp64 term bound is far above any real K", FP64_EXACT_MAX_TERMS > 500_000_000)
import os
CAP=os.environ.get("P5_CAPTURE","/home/ubuntu/integer_alibi/perlayer_capture")
MODEL=os.environ.get("P5_MODEL","/home/ubuntu/integer_alibi/models/qwen3-1.7b-int8-w8a8")
d=torch.load(os.path.join(CAP,"model.layers.0.mlp.down_proj.pt"),
             map_location="cpu", weights_only=False)
f=sorted(glob.glob(os.path.join(MODEL,"*.safetensors")))[0]
with safe_open(f, framework="pt") as sf:
    W=sf.get_tensor("model.layers.0.mlp.down_proj.weight")
for M in (16,64,128):
    a=d["q"][:M]
    got=exact_accumulator(a,W)
    want=(a.to(torch.int64) @ W.to(torch.int64).T)
    ck(f"fp64 path bit-identical to int64 at M={M}", bool(torch.equal(got,want)))
    ck(f"result dtype is int64 at M={M}", got.dtype==torch.int64)
# extreme worst case: all -128, deep K -> still exact
a=torch.full((8,6144),-128,dtype=torch.int8); w=torch.full((8,6144),-128,dtype=torch.int8)
ck("all -128 worst case still bit-identical",
   bool(torch.equal(exact_accumulator(a,w), a.to(torch.int64)@w.to(torch.int64).T)))
ck("worst case magnitude is what we expect (16384*K)",
   int(exact_accumulator(a,w).abs().max())==16384*6144)
print(f"\n{P} passed, {F} failed"); sys.exit(1 if F else 0)
