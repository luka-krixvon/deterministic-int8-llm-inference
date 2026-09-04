#!/bin/bash
# P8 end-to-end: token sequences from CUTLASS vs Triton for each arm, using v1's pinned
# run_arm_generic.py and compare.py unchanged. compare.py reads relative filenames, so each
# arm runs in its own directory rather than the scripts being edited.
set -u
cd ~/integer_alibi
IMG=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
TRITON_ENV=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel
exec > logs/p8_e2e_2026-08-15.log 2>&1
echo "===== E2E START $(date -u +%FT%TZ) ====="
for ARM in qwen3-1.7b-int8-w8a8 qwen3-1.7b-int8-pow2_nearest qwen3-1.7b-int8-pow2_ceil; do
  D=p8_e2e/$ARM; mkdir -p "$D"
  echo; echo "########## $ARM ##########"
  docker run --rm --gpus '"device=0"' --shm-size=2g -v "$PWD":/w -w /w \
    -v "$PWD"/models:/models:ro -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf \
    -e HF_HUB_OFFLINE=1 -e IMAGE_DIGEST="$IMG" -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint python3 "$IMG" /w/run_arm_generic.py /models/"$ARM" /w/"$D"/tokens_CUTLASS.json \
    && echo "CUTLASS_OK" || echo "CUTLASS_FAIL=$?"
  docker run --rm --gpus '"device=0"' --shm-size=2g -v "$PWD":/w -w /w \
    -v "$PWD"/models:/models:ro -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf \
    -e HF_HUB_OFFLINE=1 -e IMAGE_DIGEST="$IMG" -e CUDA_VISIBLE_DEVICES=0 \
    -e VLLM_DISABLED_KERNELS="$TRITON_ENV" \
    --entrypoint python3 "$IMG" /w/run_arm_generic.py /models/"$ARM" /w/"$D"/tokens_TRITON.json \
    && echo "TRITON_OK" || echo "TRITON_FAIL=$?"
  echo "--- compare ($ARM) ---"
  ( cd "$D" && python3 ~/integer_alibi/compare.py ) && echo "CMP_${ARM}_OK" || echo "CMP_${ARM}_FAIL=$?"
done
echo; echo "===== E2E DONE $(date -u +%FT%TZ) ====="
