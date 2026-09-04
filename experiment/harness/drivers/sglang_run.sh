#!/bin/bash
cd ~/integer_alibi
exec > logs/sglang_arm.log 2>&1
IMG=lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155
docker run --rm --gpus '"device=0"' --shm-size=4g -v $PWD:/w -w /w \
  -v $PWD/models:/models:ro -e HF_HOME=/w/.hf --entrypoint python3 $IMG /w/run_sglang_arm.py 1
docker run --rm --gpus '"device=0"' --shm-size=4g -v $PWD:/w -w /w \
  -v $PWD/models:/models:ro -e HF_HOME=/w/.hf --entrypoint python3 $IMG /w/run_sglang_arm.py 2
echo SGLANG_DONE
