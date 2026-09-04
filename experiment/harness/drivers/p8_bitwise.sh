#!/bin/bash
# P8 bitwise: per-layer CUTLASS vs Triton on each arm, using v1's pinned w3_perlayer.py
# unchanged. For the pow2 arms the checkpoint's own scales ARE powers of two, so the
# tool's "real" regime is the per-channel pow2 test -- the case v1's layer-level run did
# not cover, since its "pow2" branch overrides every scale to a uniform 2^-9 / 2^-8.
set -u
cd ~/integer_alibi
IMG=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
exec > logs/p8_bitwise_2026-08-15.log 2>&1
echo "===== BITWISE START $(date -u +%FT%TZ) ====="
for ARM in qwen3-1.7b-int8-w8a8 qwen3-1.7b-int8-pow2_nearest qwen3-1.7b-int8-pow2_ceil; do
  echo; echo "########## $ARM ##########"
  docker run --rm --gpus '"device=0"' --shm-size=4g -v "$PWD":/w -w /w \
    -v "$PWD"/models:/models:ro -v ~/.cache/huggingface:/hf:ro \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e IMAGE_DIGEST="$IMG" \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint python3 "$IMG" /w/w3_perlayer.py --stage verdict \
      --checkpoint /models/"$ARM" \
      --capture-dir /w/perlayer_capture \
      --p1 /w/p1_predictions_qwen3-1.7b.json \
      --out /w/p5/out/p8_bitwise_"$ARM".json \
    && echo "BW_${ARM}_OK" || echo "BW_${ARM}_FAIL=$?"
done
echo; echo "===== BITWISE DONE $(date -u +%FT%TZ) ====="
