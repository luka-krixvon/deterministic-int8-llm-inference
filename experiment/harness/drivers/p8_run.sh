#!/bin/bash
# P8 execution under A-12.1. Contains invocations only: every program it calls is already
# pinned (p8_requant.py by A-12.1; p6_windows/p6_accuracy/p6_throughput by A-10.5, unchanged).
set -u
cd ~/integer_alibi/p5
V=~/integer_alibi/.venv/bin/python
M=~/integer_alibi/models
PAR=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/ | head -1)
IMG=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
mkdir -p out
exec > ~/integer_alibi/logs/p8_exec_2026-08-15.log 2>&1

ctx () { echo "--- context ($1) ---"; uptime; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; df -h / | tail -1; }
run_c () {
  docker run --rm --gpus '"device=0"' --shm-size=4g -v ~/integer_alibi/p5:/w -w /w \
    -v "$M":/models:ro -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e IMAGE_DIGEST="$IMG" \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint python3 "$IMG" "$@"
}

echo "===== P8 START $(date -u +%FT%TZ) ====="; ctx before
echo "parent: $PAR"

for RULE in pow2_nearest pow2_ceil; do
  echo; echo "########## build $RULE ##########"
  $V p8_requant.py --parent "$PAR" --base "$M"/qwen3-1.7b-int8-w8a8 \
     --rule "$RULE" --out "$M"/qwen3-1.7b-int8-"$RULE" \
     --report out/p8_report_"$RULE".json && echo "BUILD_${RULE}_OK" || echo "BUILD_${RULE}_FAIL=$?"
  df -h / | tail -1
done

for RULE in pow2_nearest pow2_ceil; do
  echo; echo "########## accuracy $RULE ##########"
  run_c /w/p6_accuracy.py --model /models/qwen3-1.7b-int8-"$RULE" \
     --windows /w/out/p6_eval_windows.json --out /w/out/p8_acc_"$RULE".json \
     && echo "ACC_${RULE}_OK" || echo "ACC_${RULE}_FAIL=$?"
done

for RULE in pow2_nearest pow2_ceil; do
  echo; echo "########## throughput $RULE ##########"
  run_c /w/p6_throughput.py --model /models/qwen3-1.7b-int8-"$RULE" \
     --out /w/out/p8_tp_"$RULE".json && echo "TP_${RULE}_OK" || echo "TP_${RULE}_FAIL=$?"
done

echo; echo "########## compares against base ##########"
for RULE in pow2_nearest pow2_ceil; do
  run_c /w/p6_accuracy.py --compare /w/out/p6_acc_base.json /w/out/p8_acc_"$RULE".json \
     --out /w/out/p8_cost_acc_"$RULE".json && echo "CMP_ACC_${RULE}_OK" || echo "CMP_ACC_${RULE}_FAIL=$?"
  run_c /w/p6_throughput.py --compare /w/out/p6_tp_base.json /w/out/p8_tp_"$RULE".json \
     --out /w/out/p8_cost_tp_"$RULE".json && echo "CMP_TP_${RULE}_OK" || echo "CMP_TP_${RULE}_FAIL=$?"
done

echo; ctx after; echo "===== P8 DONE $(date -u +%FT%TZ) ====="; ls -l out/ | tail -14
