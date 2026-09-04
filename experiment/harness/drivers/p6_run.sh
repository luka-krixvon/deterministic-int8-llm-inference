#!/bin/bash
# P6: pow2 intervention cost. Two environments, by necessity.
#
# Stage A runs in the quantization venv because it needs `datasets`, which the pinned
# vLLM container does not have and must not be given -- installing it would mean the
# identity contract records an image digest that never ran. Stage A emits a window file
# with a content hash over the windows themselves.
#
# Every stage that touches a model runs inside the pinned container, which is where all
# v1 measurements ran. Stage B verifies the window file's hash before scoring and
# refuses if it does not match.
#
# The container gets the existing HF cache read-only and is put offline. Without that it
# would be free to fetch a tokenizer or config from the Hub mid-measurement, which would
# both write into the run directory and quietly introduce an unpinned input.
set -u
cd ~/integer_alibi
IMG=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
VENV=~/integer_alibi/.venv/bin/python
P5=~/integer_alibi/p5
OUT=$P5/out
mkdir -p "$OUT" logs
exec > logs/p6_exec_$(date -u +%Y-%m-%d).log 2>&1

echo "=== P6 START $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "image: $IMG"

run_in_container() {
  docker run --rm --gpus '"device=0"' --shm-size=4g \
    -v "$P5":/w -w /w \
    -v ~/integer_alibi/models:/models:ro \
    -v ~/.cache/huggingface:/hf:ro \
    -e HF_HOME=/hf \
    -e HF_HUB_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    -e IMAGE_DIGEST="$IMG" \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint python3 "$IMG" "$@"
}

echo "--- stage A: derive windows (venv, needs datasets) ---"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 $VENV "$P5"/p6_windows.py \
  --calib ~/integer_alibi/calib_prompts.json \
  --out "$OUT"/p6_eval_windows.json && echo WINDOWS_OK || { echo WINDOWS_FAIL=$?; exit 1; }
sha256sum "$OUT"/p6_eval_windows.json

echo "--- stage B: accuracy, base arm (container) ---"
run_in_container /w/p6_accuracy.py --model /models/qwen3-1.7b-int8-w8a8 \
  --windows /w/out/p6_eval_windows.json --out /w/out/p6_acc_base.json \
  && echo ACC_BASE_OK || echo ACC_BASE_FAIL=$?

echo "--- stage B: accuracy, pow2 arm (container) ---"
run_in_container /w/p6_accuracy.py --model /models/qwen3-1.7b-int8-w8a8-pow2 \
  --windows /w/out/p6_eval_windows.json --out /w/out/p6_acc_pow2.json \
  && echo ACC_POW2_OK || echo ACC_POW2_FAIL=$?

echo "--- accuracy compare ---"
run_in_container /w/p6_accuracy.py --compare /w/out/p6_acc_base.json /w/out/p6_acc_pow2.json \
  --out /w/out/p6_accuracy_cost.json && echo ACC_CMP_OK || echo ACC_CMP_FAIL=$?

echo "--- throughput, base arm (container) ---"
run_in_container /w/p6_throughput.py --model /models/qwen3-1.7b-int8-w8a8 \
  --out /w/out/p6_tp_base.json && echo TP_BASE_OK || echo TP_BASE_FAIL=$?

echo "--- throughput, pow2 arm (container) ---"
run_in_container /w/p6_throughput.py --model /models/qwen3-1.7b-int8-w8a8-pow2 \
  --out /w/out/p6_tp_pow2.json && echo TP_POW2_OK || echo TP_POW2_FAIL=$?

echo "--- throughput compare ---"
run_in_container /w/p6_throughput.py --compare /w/out/p6_tp_base.json /w/out/p6_tp_pow2.json \
  --out /w/out/p6_throughput_cost.json && echo TP_CMP_OK || echo TP_CMP_FAIL=$?

echo "=== P6 DONE $(date -u +%FT%TZ) ==="
