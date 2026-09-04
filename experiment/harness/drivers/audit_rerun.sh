#!/bin/bash
cd ~/integer_alibi
exec > logs/audit_rerun.log 2>&1
set -x
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
V () { docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e IMAGE_DIGEST=$PIN "$@"; }
V --entrypoint python3 $PIN /w/test_numeric_boundaries.py
V --entrypoint bash $PIN -c "pip -q install scipy 2>&1|tail -1; python3 /w/w3_p2_fp8_v2.py --seeds 5 --out /w/w3_p2_fp8_v2.json"
V --entrypoint python3 $PIN /w/w3_perlayer.py --stage verdict --checkpoint /models/qwen3-1.7b-int8-w8a8 --capture-dir /w/perlayer_capture --p1 /w/p1_predictions_qwen3-1.7b.json --out /w/w3_perlayer_verdict_v2.json
python3 w4_p4_stats.py tf_CUTLASS.json tf_TRITON.json p4_stats_v2.json
echo AUDIT_RERUN_DONE
