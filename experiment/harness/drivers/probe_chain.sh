#!/bin/bash
cd ~/integer_alibi
exec > logs/probe_chain.log 2>&1
set -x
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
SGL=lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155
M=/models/qwen3-1.7b-int8-w8a8-pow2
V () { docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e HF_HOME=/w/.hf "$@"; }
# E1: two runs per arm
V --entrypoint python3 $PIN /w/run_arm_generic.py $M /w/probe_CUTLASS_1.json
V --entrypoint python3 $PIN /w/run_arm_generic.py $M /w/probe_CUTLASS_2.json
V -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/run_arm_generic.py $M /w/probe_TRITON_1.json
V -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/run_arm_generic.py $M /w/probe_TRITON_2.json
# E2: same-arm rail sanity (the CUTLASS probe replays its own probe rail)
V --entrypoint python3 $PIN /w/tf_generic.py $M /w/probe_CUTLASS_1.json /w/probe_tf_CUTLASS.json
# E3: SGLang on probe
V --entrypoint python3 $SGL /w/sglang_generic.py $M /w/probe_sglang.json
echo PROBE_CHAIN_DONE
