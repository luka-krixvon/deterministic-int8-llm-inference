#!/bin/bash
# Resume the 14B run from the pow2 build. The first attempt failed on disk: llmcompressor
# wrote base as a single 16 GiB shard, not the ~14 GiB I sized the guard from, so a guard of
# exactly 16 GiB passed with zero margin and the write filled the disk. The guard is now
# derived from the actual base size plus a 4 GiB margin instead of a constant.
set -u
cd ~/integer_alibi
V=~/integer_alibi/.venv/bin/python
M=~/integer_alibi/models
OUT=~/integer_alibi/p5/out
IMG=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
TRITON_ENV=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel
exec >> logs/p8_14b_2026-08-15.log 2>&1
avail () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
mark () { echo "@@@ $1 $(date -u +%FT%TZ) avail=$(avail)G"; }
run_c () { docker run --rm --gpus '"device=0"' --shm-size=4g -v ~/integer_alibi:/w -w /w \
  -v "$M":/models:ro -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 -e IMAGE_DIGEST="$IMG" -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e CUDA_VISIBLE_DEVICES=0 "$@"; }

echo; echo "===== 14B RESUME $(date -u +%FT%TZ) ====="
mark "reclaim-v7"
docker rmi integer-alibi-test:v7 2>&1 | tail -2
docker builder prune -af 2>&1 | tail -1
rm -rf "$M"/qwen3-14b-int8-pow2_nearest
echo "RECLAIM_V7_OK"; df -h / | tail -1

BASEG=$(du -sBG "$M"/qwen3-14b-int8-w8a8 | cut -f1 | tr -dc 0-9)
NEED=$((BASEG + 4))
mark "build-pow2 (base=${BASEG}G, need>=${NEED}G)"
a=$(avail); [ "$a" -ge "$NEED" ] || { echo "!!! ABORT need ${NEED}G have ${a}G"; echo ABORTED_DISK; exit 1; }
PAR=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/ | head -1)
$V p5/p8_requant.py --parent "$PAR" --base "$M"/qwen3-14b-int8-w8a8 --rule pow2_nearest \
  --out "$M"/qwen3-14b-int8-pow2_nearest --report "$OUT"/p8_report_14b_pow2_nearest.json \
  && echo BUILD_POW2_OK || { echo "BUILD_POW2_FAIL=$?"; exit 1; }
df -h / | tail -1

mark "drop-parent"; rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B
echo PARENT_DROPPED; df -h / | tail -1

for ARM in qwen3-14b-int8-w8a8 qwen3-14b-int8-pow2_nearest; do
  mark "acc-$ARM"
  run_c --entrypoint python3 "$IMG" /w/p5/p6_accuracy.py --model /models/"$ARM" \
    --windows /w/p5/out/p6_eval_windows.json --out /w/p5/out/p8_acc_14b_"$ARM".json \
    && echo "ACC_${ARM}_OK" || echo "ACC_${ARM}_FAIL=$?"
done
run_c --entrypoint python3 "$IMG" /w/p5/p6_accuracy.py --compare \
  /w/p5/out/p8_acc_14b_qwen3-14b-int8-w8a8.json \
  /w/p5/out/p8_acc_14b_qwen3-14b-int8-pow2_nearest.json \
  --out /w/p5/out/p8_cost_acc_14b.json && echo CMP_ACC_14B_OK || echo CMP_ACC_14B_FAIL=$?

for ARM in qwen3-14b-int8-w8a8 qwen3-14b-int8-pow2_nearest; do
  D=p8_e2e/$ARM; mkdir -p "$D"; mark "e2e-$ARM"
  run_c --entrypoint python3 "$IMG" /w/run_arm_generic.py /models/"$ARM" /w/"$D"/tokens_CUTLASS.json \
    && echo CUTLASS_OK || echo CUTLASS_FAIL=$?
  run_c -e VLLM_DISABLED_KERNELS="$TRITON_ENV" --entrypoint python3 "$IMG" \
    /w/run_arm_generic.py /models/"$ARM" /w/"$D"/tokens_TRITON.json \
    && echo TRITON_OK || echo TRITON_FAIL=$?
  ( cd "$D" && python3 ~/integer_alibi/compare.py ) && echo "CMP_${ARM}_OK" || echo "CMP_${ARM}_FAIL=$?"
done

mark "capture-attempt"
$V w3_perlayer.py --stage capture --checkpoint "$M"/qwen3-14b-int8-w8a8 \
  --capture-dir ~/integer_alibi/perlayer_capture_14b --prompts ~/integer_alibi/calib_prompts.json \
  && echo CAPTURE_14B_OK || echo "CAPTURE_14B_FAIL=$? (recorded; layer-level not obtained at 14B)"
if [ -f ~/integer_alibi/perlayer_capture_14b/_layers.json ]; then
  mark "p1-14b"
  $V p1_predictions.py --checkpoint "$M"/qwen3-14b-int8-w8a8 \
    --prompts-file ~/integer_alibi/calib_prompts.json --out "$OUT"/p1_predictions_qwen3-14b.json \
    && echo P1_14B_OK || echo "P1_14B_FAIL=$?"
  for ARM in qwen3-14b-int8-w8a8 qwen3-14b-int8-pow2_nearest; do
    mark "bitwise-$ARM"
    run_c --entrypoint python3 "$IMG" /w/w3_perlayer.py --stage verdict \
      --checkpoint /models/"$ARM" --capture-dir /w/perlayer_capture_14b \
      --p1 /w/p5/out/p1_predictions_qwen3-14b.json \
      --out /w/p5/out/p8_bitwise_14b_"$ARM".json && echo "BW_${ARM}_OK" || echo "BW_${ARM}_FAIL=$?"
  done
else
  echo "LAYER_LEVEL_SKIPPED: no 14B capture produced"
fi
mark done; df -h / | tail -1; echo "===== 14B RESUME DONE $(date -u +%FT%TZ) ====="
