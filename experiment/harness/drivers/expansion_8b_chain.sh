#!/bin/bash
cd ~/integer_alibi
exec > logs/expansion_8b.log 2>&1
set -x
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
V () { docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e HF_HOME=/w/.hf "$@"; }
# ===== Part A: 1.7B expansion, 64 prompts x 256 tokens =====
M=/models/qwen3-1.7b-int8-w8a8
V --entrypoint python3 $PIN /w/run_arm_v2.py --model $M --out /w/tokens64_CUTLASS.json
V -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/run_arm_v2.py --model $M --out /w/tokens64_TRITON.json
V --entrypoint python3 $PIN /w/tf_v2.py --model $M --rails /w/tokens64_CUTLASS.json --out /w/tf64_CUTLASS.json
V -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/tf_v2.py --model $M --rails /w/tokens64_CUTLASS.json --out /w/tf64_TRITON.json
python3 w4_p4_stats.py tf64_CUTLASS.json tf64_TRITON.json p4_stats_64.json
echo PART_A_DONE
# ===== Part B: Qwen3-8B =====
./.venv/bin/python3 quant_qwen8b.py
grep -q INT8_8B_done logs/expansion_8b.log || { echo QUANT8B_FAILED; exit 1; }
./.venv/bin/python3 p1_predictions.py --checkpoint models/qwen3-8b-int8-w8a8 --prompts-file calib_prompts.json --out p1_predictions_qwen3-8b.json --max-prompts 32
./.venv/bin/python3 w3_perlayer.py --stage capture --checkpoint models/qwen3-8b-int8-w8a8 --prompts calib_prompts.json --capture-dir perlayer_capture_8b
V --entrypoint python3 $PIN /w/w3_perlayer.py --stage verdict --checkpoint /models/qwen3-8b-int8-w8a8 --capture-dir /w/perlayer_capture_8b --p1 /w/p1_predictions_qwen3-8b.json --out /w/w3_perlayer_verdict_8b.json
./.venv/bin/python3 make_probe_pow2.py models/qwen3-8b-int8-w8a8 models/qwen3-8b-int8-w8a8-pow2
M8=/models/qwen3-8b-int8-w8a8-pow2
V --entrypoint python3 $PIN /w/run_arm_v2.py --model $M8 --out /w/probe8b_CUTLASS.json --n-prompts 16 --max-tokens 64
V -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/run_arm_v2.py --model $M8 --out /w/probe8b_TRITON.json --n-prompts 16 --max-tokens 64
# raw-scale 8B control (the precondition control for E1)
V --entrypoint python3 $PIN /w/run_arm_v2.py --model /models/qwen3-8b-int8-w8a8 --out /w/raw8b_CUTLASS.json --n-prompts 16 --max-tokens 64
V -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/run_arm_v2.py --model /models/qwen3-8b-int8-w8a8 --out /w/raw8b_TRITON.json --n-prompts 16 --max-tokens 64
echo EXPANSION_8B_DONE
