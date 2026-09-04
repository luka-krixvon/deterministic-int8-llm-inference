#!/bin/bash
cd ~/integer_alibi
exec > logs/exec_v7.log 2>&1
set -x
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
# 1) clean-env: build and run recorded separately
{
  echo "build_command: docker build -f Dockerfile.test -t integer-alibi-test:v7 ."
  echo "git_sha: fee5864"
  docker build -f Dockerfile.test -t integer-alibi-test:v7 . 
  echo "EXIT_build=$?"
  echo "image_id: $(docker inspect --format '{{.Id}}' integer-alibi-test:v7)"
} > clean_env_build_record.txt 2>&1
{
  echo "run_command: docker run --rm integer-alibi-test:v7"
  docker run --rm integer-alibi-test:v7
  echo "EXIT_run=$?"
  echo '--- per-suite PASS/SKIP counts (CPU container) ---'
} > clean_env_run_record.txt 2>&1
grep -c '^PASS' clean_env_run_record.txt | xargs -I{} echo "TOTAL_PASS={}" >> clean_env_run_record.txt
grep -c '^SKIP' clean_env_run_record.txt | xargs -I{} echo "TOTAL_SKIP={}" >> clean_env_run_record.txt
docker save integer-alibi-test:v7 | sha256sum | cut -d' ' -f1 | xargs -I{} echo "image_tar_sha256={}" >> clean_env_build_record.txt
# 2) E2 genuine first run (files scp-ed this time; on failure, classify and record)
{
  echo "command: docker run … --entrypoint python3 $PIN /w/w5_first_divergence.py --model /models/qwen3-1.7b-int8-w8a8-pow2 --arm CUTLASS --prompt-idx 0 --out /w/fd_CUTLASS_0.json"
  echo "git_sha: fee5864"; echo "image: $PIN"
  ls -la w5_first_divergence.py
  docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e IMAGE_DIGEST=$PIN --entrypoint python3 $PIN /w/w5_first_divergence.py --model /models/qwen3-1.7b-int8-w8a8-pow2 --arm CUTLASS --prompt-idx 0 --out /w/fd_CUTLASS_0.json
  echo "EXIT_e2=$?"
} > e2_real_first_run_record.txt 2>&1
# 3) GPU: tf_v6 both arms (treatment provenance)
PREV=$(python3 -c "import json;print(json.load(open('models/parent_manifest.json'))['revision'])")
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e IMAGE_DIGEST=$PIN --entrypoint python3 $PIN /w/tf_v6.py --model /models/qwen3-1.7b-int8-w8a8 --rails /w/tokens64_CUTLASS.json --out /w/tf64v6_CUTLASS.json --n-prompts 64 --parent-revision $PREV --arm CUTLASS
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e IMAGE_DIGEST=$PIN -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/tf_v6.py --model /models/qwen3-1.7b-int8-w8a8 --rails /w/tokens64_CUTLASS.json --out /w/tf64v6_TRITON.json --n-prompts 64 --parent-revision $PREV --arm TRITON
# 4) P4 v7 (verified artifact under the fail-closed contract)
./.venv/bin/python3 w4_p4_stats_v4.py tf64v6_CUTLASS.json tf64v6_TRITON.json p4_stats64_v7_verified.json --isotonic
echo EXEC_V7_DONE
