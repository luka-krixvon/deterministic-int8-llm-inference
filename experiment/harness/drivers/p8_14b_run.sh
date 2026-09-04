#!/bin/bash
# 14B replication, unattended, under A-12's second scope amendment and its pre-run note.
# Invocations only. Disk follows the declared plan with a hard guard before the second build.
set -u
cd ~/integer_alibi
V=~/integer_alibi/.venv/bin/python
M=~/integer_alibi/models
OUT=~/integer_alibi/p5/out
IMG=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
TRITON_ENV=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel
mkdir -p "$OUT" p8_e2e
exec > logs/p8_14b_2026-08-15.log 2>&1

avail () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
guard () { a=$(avail); if [ "$a" -lt "$1" ]; then echo "!!! ABORT: need ${1}G free, have ${a}G"; echo "ABORTED_DISK"; exit 1; fi; }
mark () { echo "@@@ $1 $(date -u +%FT%TZ) avail=$(avail)G"; }
run_c () { docker run --rm --gpus '"device=0"' --shm-size=4g -v ~/integer_alibi:/w -w /w \
  -v "$M":/models:ro -v ~/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 -e IMAGE_DIGEST="$IMG" -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e CUDA_VISIBLE_DEVICES=0 "$@"; }

echo "===== 14B FULL RUN START $(date -u +%FT%TZ) ====="; df -h / | tail -1

# ---- wait for the parent download that is already in flight
mark "wait-download"
while pgrep -u ubuntu -f "\.venv/bin/python" | grep -qv "$$" && \
      [ "$(du -sm ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B 2>/dev/null | cut -f1)" -lt 27000 ]; do
  sleep 60
done
sleep 30
SZ=$(du -sm ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B | cut -f1)
echo "parent size ${SZ}M"
[ "$SZ" -ge 27000 ] || { echo "!!! parent incomplete (${SZ}M < 27000M)"; echo "ABORTED_DOWNLOAD"; exit 1; }
echo "DOWNLOAD_OK"

# ---- declared extra reclamation (A-12 second-amendment pre-run note)
mark "reclaim"
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B \
       ~/integer_alibi/perlayer_capture_8b \
       ~/integer_alibi/models/qwen3-1.7b-int8-w8a8
echo "RECLAIM_OK"; df -h / | tail -1

# ---- base arm, by llmcompressor, revision passed explicitly
mark "build-base"; guard 16
$V quant_qwen14b.py && echo BUILD_BASE_OK || { echo "BUILD_BASE_FAIL=$?"; exit 1; }
$V - <<'PY'
import hashlib, glob, json, os
d = os.path.expanduser("~/integer_alibi/models/qwen3-14b-int8-w8a8")
h = hashlib.sha256()
for s in sorted(glob.glob(d + "/*.safetensors")):
    with open(s, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
json.dump({"arm": "base-14b", "checkpoint_digest": h.hexdigest(),
           "shards": len(glob.glob(d + "/*.safetensors"))},
          open(os.path.expanduser("~/integer_alibi/p5/out/p8_digest_14b_base.json"), "w"), indent=2)
print("BASE_DIGEST", h.hexdigest()[:16], flush=True)
PY
df -h / | tail -1

# ---- pow2 arm, from the parent with base as structure reference (same flow as 8B)
mark "build-pow2"; guard 16
PAR=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/ | head -1)
$V p5/p8_requant.py --parent "$PAR" --base "$M"/qwen3-14b-int8-w8a8 --rule pow2_nearest \
  --out "$M"/qwen3-14b-int8-pow2_nearest --report "$OUT"/p8_report_14b_pow2_nearest.json \
  && echo BUILD_POW2_OK || { echo "BUILD_POW2_FAIL=$?"; exit 1; }
df -h / | tail -1

# ---- parent no longer needed by either build
mark "drop-parent"
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B
echo "PARENT_DROPPED"; df -h / | tail -1

# ---- accuracy, both arms, on the same 256 windows
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

# ---- end-to-end, both arms
for ARM in qwen3-14b-int8-w8a8 qwen3-14b-int8-pow2_nearest; do
  D=p8_e2e/$ARM; mkdir -p "$D"; mark "e2e-$ARM"
  run_c --entrypoint python3 "$IMG" /w/run_arm_generic.py /models/"$ARM" /w/"$D"/tokens_CUTLASS.json \
    && echo CUTLASS_OK || echo CUTLASS_FAIL=$?
  run_c -e VLLM_DISABLED_KERNELS="$TRITON_ENV" --entrypoint python3 "$IMG" \
    /w/run_arm_generic.py /models/"$ARM" /w/"$D"/tokens_TRITON.json \
    && echo TRITON_OK || echo TRITON_FAIL=$?
  ( cd "$D" && python3 ~/integer_alibi/compare.py ) && echo "CMP_${ARM}_OK" || echo "CMP_${ARM}_FAIL=$?"
done

# ---- layer-level: attempt, record the outcome either way (pre-run note)
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
      --out /w/p5/out/p8_bitwise_14b_"$ARM".json \
      && echo "BW_${ARM}_OK" || echo "BW_${ARM}_FAIL=$?"
  done
else
  echo "LAYER_LEVEL_SKIPPED: no 14B capture produced"
fi

mark "done"; df -h / | tail -1
echo "===== 14B FULL RUN DONE $(date -u +%FT%TZ) ====="
grep -E "^RESULT: [0-9]|^P1a:|^P1b:" logs/p8_14b_2026-08-15.log | tail -20
