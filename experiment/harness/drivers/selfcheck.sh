#!/bin/bash
cd ~/integer_alibi
exec > logs/selfcheck.log 2>&1
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
cp run_arm.py run_arm2.py
sed -i 's/tokens_{label}/tokens_{label}2/' run_arm2.py
# rerun each arm once (fresh container, fresh process) against the first round
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w \
  -v $PWD/models:/models:ro -e HF_HOME=/w/.hf --entrypoint python3 $PIN /w/run_arm2.py CUTLASS
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w \
  -v $PWD/models:/models:ro -e HF_HOME=/w/.hf \
  -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel \
  --entrypoint python3 $PIN /w/run_arm2.py TRITON
python3 - <<'PY'
import json
for arm in ("CUTLASS","TRITON"):
    a=json.load(open(f"tokens_{arm}.json")); b=json.load(open(f"tokens_{arm}2.json"))
    ident=sum(1 for x,y in zip(a,b) if x==y)
    print(f"SELFCHECK {arm}: {ident}/8 identical across two cold runs")
PY
echo SELFCHECK_DONE
