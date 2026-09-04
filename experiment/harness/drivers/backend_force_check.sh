#!/bin/bash
cd ~/integer_alibi
exec > logs/backend_force.log 2>&1
set -x
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
run_arm () {  # $1=label $2=extra_env
  docker run --rm --gpus '"device=0"' --shm-size=2g \
    -v $PWD:/w -w /w -v $PWD/models:/models:ro -e HF_HOME=/w/.hf $2 \
    -e VLLM_LOGGING_LEVEL=DEBUG --entrypoint bash $PIN -c \
    "python3 -c \"
from vllm import LLM, SamplingParams
llm = LLM(model='/models/qwen3-1.7b-int8-w8a8', enforce_eager=True,
          gpu_memory_utilization=0.5, max_model_len=2048)
out = llm.generate(['The capital of France is'], SamplingParams(max_tokens=8, temperature=0))
print('GEN_$1:', out[0].outputs[0].text.strip()[:40])
print('TOKENS_$1:', list(out[0].outputs[0].token_ids))
\" 2>&1 | grep -E 'GEN_|TOKENS_|[Ss]elected|[Kk]ernel|[Ff]allback|scaled_mm|W8A8|Int8' | head -25"
}
echo "=== ARM A: default (expect CutlassScaledMM) ==="
run_arm A ""
echo "=== ARM B: cutlass disabled (expect Triton path) ==="
run_arm B "-e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel"
echo "BACKEND_FORCE_DONE"
