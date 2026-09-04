#!/bin/bash
cd ~/integer_alibi
exec > logs/w3_perlayer.log 2>&1
set -x
./.venv/bin/python3 w3_perlayer.py --stage capture \
  --checkpoint models/qwen3-1.7b-int8-w8a8 --prompts calib_prompts.json \
  --capture-dir perlayer_capture
grep -q CAPTURE_OK logs/w3_perlayer.log || { echo CAPTURE_FAILED; exit 1; }
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w \
  -v $PWD/models:/models:ro --entrypoint python3 $PIN /w/w3_perlayer.py \
  --stage verdict --checkpoint /models/qwen3-1.7b-int8-w8a8 \
  --capture-dir /w/perlayer_capture --p1 /w/p1_predictions_qwen3-1.7b.json \
  --out /w/w3_perlayer_verdict.json
echo W3_PERLAYER_DONE
