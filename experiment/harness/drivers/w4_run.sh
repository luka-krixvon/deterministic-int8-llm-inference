#!/bin/bash
cd ~/integer_alibi
exec > logs/w4_tf.log 2>&1
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w \
  -v $PWD/models:/models:ro -e HF_HOME=/w/.hf --entrypoint python3 $PIN /w/w4_teacher_forcing.py CUTLASS
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w \
  -v $PWD/models:/models:ro -e HF_HOME=/w/.hf \
  -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel \
  --entrypoint python3 $PIN /w/w4_teacher_forcing.py TRITON
echo W4_TF_DONE
