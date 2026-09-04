#!/bin/bash
set -u
cd ~/integer_alibi
M=~/integer_alibi/models
IMG=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
TRITON_ENV=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel
GPUFRAC=0.85
exec >> logs/p8_14b_2026-08-15.log 2>&1
run_c () { docker run --rm --gpus '"device=0"' --shm-size=4g -v ~/integer_alibi:/w -w /w \
  -v "$M":/models:ro -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e IMAGE_DIGEST="$IMG" -e CUDA_VISIBLE_DEVICES=0 "$@"; }
echo; echo "===== 14B E2E RETRY (gpu_memory_utilization=$GPUFRAC) $(date -u +%FT%TZ) ====="
for ARM in qwen3-14b-int8-w8a8 qwen3-14b-int8-pow2_nearest; do
  D=p8_e2e/$ARM; rm -rf "$D"; mkdir -p "$D"
  echo; echo "########## $ARM ##########"
  run_c --entrypoint python3 "$IMG" /w/run_arm_mem.py /models/"$ARM" /w/"$D"/tokens_CUTLASS.json $GPUFRAC \
    && echo CUTLASS_OK || echo CUTLASS_FAIL=$?
  run_c -e VLLM_DISABLED_KERNELS="$TRITON_ENV" --entrypoint python3 "$IMG" \
    /w/run_arm_mem.py /models/"$ARM" /w/"$D"/tokens_TRITON.json $GPUFRAC \
    && echo TRITON_OK || echo TRITON_FAIL=$?
  ( cd "$D" && python3 ~/integer_alibi/compare.py ) && echo "CMP_${ARM}_OK" || echo "CMP_${ARM}_FAIL=$?"
done
echo "===== 14B E2E RETRY DONE $(date -u +%FT%TZ) ====="
