#!/bin/bash
cd ~/integer_alibi
exec > logs/exec_v5.log 2>&1
set -x
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
# authoritative test run, with the full environment recorded
{
  echo "command: tests_metrics_v3.py + tests_metrics_v4.py"
  echo "git_sha: 81ca595"
  echo "platform: $(uname -a)"
  echo "device: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  ./.venv/bin/python3 tests_metrics_v3.py; echo "EXIT_v3=$?"
  ./.venv/bin/python3 tests_metrics_v4.py; echo "EXIT_v4=$?"
  ./.venv/bin/python3 -c 'import torch,numpy,scipy,sys; print("torch",torch.__version__,"numpy",numpy.__version__,"scipy",scipy.__version__,"python",sys.version.split()[0])'
} > test_run_v5_record.txt 2>&1
# reanalysis v5
./.venv/bin/python3 w3_p2_refit_v4.py w3_p2_raw_20seeds.json w3_p2_fits_v5_20seeds.json
./.venv/bin/python3 w4_p4_stats_v4.py tf64_CUTLASS.json tf64_TRITON.json p4_stats64_v5.json --isotonic
./.venv/bin/python3 w4_p4_stats_v4.py tf_CUTLASS.json tf_TRITON.json p4_stats8_v5.json --isotonic
# GPU: tf_v4 identity-verified rerun (64 prompts, both arms)
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro --entrypoint python3 $PIN /w/tf_v4.py --model /models/qwen3-1.7b-int8-w8a8 --rails /w/tokens64_CUTLASS.json --out /w/tf64v4_CUTLASS.json --n-prompts 64
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/tf_v4.py --model /models/qwen3-1.7b-int8-w8a8 --rails /w/tokens64_CUTLASS.json --out /w/tf64v4_TRITON.json --n-prompts 64
./.venv/bin/python3 w4_p4_stats_v4.py tf64v4_CUTLASS.json tf64v4_TRITON.json p4_stats64_v5_verified.json --isotonic
echo EXEC_V5_DONE
