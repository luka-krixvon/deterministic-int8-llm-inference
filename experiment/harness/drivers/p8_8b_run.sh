#!/bin/bash
# P8 8B replication under the A-12 scope amendment. Invocations only; every program is
# already pinned (p8_requant.py by A-12.1; p6_windows/p6_accuracy by A-10.5; w3_perlayer.py,
# run_arm_generic.py and compare.py are v1's, unchanged).
set -u
cd ~/integer_alibi
V=~/integer_alibi/.venv/bin/python
M=~/integer_alibi/models
PAR8=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
IMG=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
TRITON_ENV=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel
mkdir -p p5/out p8_e2e
exec > logs/p8_8b_2026-08-15.log 2>&1
run_c () {
  docker run --rm --gpus '"device=0"' --shm-size=4g -v ~/integer_alibi:/w -w /w \
    -v "$M":/models:ro -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e IMAGE_DIGEST="$IMG" \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CUDA_VISIBLE_DEVICES=0 "$@"
}
echo "===== 8B START $(date -u +%FT%TZ) ====="; df -h / | tail -1; echo "parent: $PAR8"

echo; echo "########## build 8B pow2_nearest ##########"
$V p5/p8_requant.py --parent "$PAR8" --base "$M"/qwen3-8b-int8-w8a8 \
  --rule pow2_nearest --out "$M"/qwen3-8b-int8-pow2_nearest \
  --report p5/out/p8_report_8b_pow2_nearest.json && echo BUILD_8B_OK || echo BUILD_8B_FAIL=$?
df -h / | tail -1

for ARM in qwen3-8b-int8-w8a8 qwen3-8b-int8-pow2_nearest; do
  echo; echo "########## accuracy $ARM ##########"
  run_c --entrypoint python3 "$IMG" /w/p5/p6_accuracy.py --model /models/"$ARM" \
    --windows /w/p5/out/p6_eval_windows.json --out /w/p5/out/p8_acc_8b_"$ARM".json \
    && echo "ACC_${ARM}_OK" || echo "ACC_${ARM}_FAIL=$?"
done
run_c --entrypoint python3 "$IMG" /w/p5/p6_accuracy.py --compare \
  /w/p5/out/p8_acc_8b_qwen3-8b-int8-w8a8.json /w/p5/out/p8_acc_8b_qwen3-8b-int8-pow2_nearest.json \
  --out /w/p5/out/p8_cost_acc_8b.json && echo CMP_ACC_8B_OK || echo CMP_ACC_8B_FAIL=$?

for ARM in qwen3-8b-int8-w8a8 qwen3-8b-int8-pow2_nearest; do
  echo; echo "########## bitwise per-layer $ARM ##########"
  run_c --entrypoint python3 "$IMG" /w/w3_perlayer.py --stage verdict \
    --checkpoint /models/"$ARM" --capture-dir /w/perlayer_capture_8b \
    --p1 /w/p1_predictions_qwen3-8b.json --out /w/p5/out/p8_bitwise_8b_"$ARM".json \
    && echo "BW_${ARM}_OK" || echo "BW_${ARM}_FAIL=$?"
done

for ARM in qwen3-8b-int8-w8a8 qwen3-8b-int8-pow2_nearest; do
  D=p8_e2e/$ARM; mkdir -p "$D"
  echo; echo "########## e2e $ARM ##########"
  run_c --entrypoint python3 "$IMG" /w/run_arm_generic.py /models/"$ARM" /w/"$D"/tokens_CUTLASS.json \
    && echo CUTLASS_OK || echo CUTLASS_FAIL=$?
  run_c -e VLLM_DISABLED_KERNELS="$TRITON_ENV" --entrypoint python3 "$IMG" \
    /w/run_arm_generic.py /models/"$ARM" /w/"$D"/tokens_TRITON.json \
    && echo TRITON_OK || echo TRITON_FAIL=$?
  ( cd "$D" && python3 ~/integer_alibi/compare.py ) && echo "CMP_${ARM}_OK" || echo "CMP_${ARM}_FAIL=$?"
done
echo; df -h / | tail -1; echo "===== 8B DONE $(date -u +%FT%TZ) ====="
