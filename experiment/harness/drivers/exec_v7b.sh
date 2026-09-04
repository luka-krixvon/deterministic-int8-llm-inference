#!/bin/bash
cd ~/integer_alibi
exec > logs/exec_v7b.log 2>&1
set -x
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
PREV=$(python3 -c "import json;print(json.load(open('models/parent_manifest.json'))['revision'])")
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e IMAGE_DIGEST=$PIN --entrypoint python3 $PIN /w/tf_v6.py --model /models/qwen3-1.7b-int8-w8a8 --rails /w/tokens64_CUTLASS.json --out /w/tf64v6b_CUTLASS.json --n-prompts 64 --parent-revision $PREV --arm CUTLASS
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e IMAGE_DIGEST=$PIN -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/tf_v6.py --model /models/qwen3-1.7b-int8-w8a8 --rails /w/tokens64_CUTLASS.json --out /w/tf64v6b_TRITON.json --n-prompts 64 --parent-revision $PREV --arm TRITON
./.venv/bin/python3 w4_p4_stats_v4.py tf64v6b_CUTLASS.json tf64v6b_TRITON.json p4_stats64_v7b_verified.json --isotonic
echo EXEC_V7B_DONE
