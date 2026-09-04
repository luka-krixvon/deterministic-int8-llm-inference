"""Token dump for one kernel arm, with the engine memory budget parameterised.

A copy of v1's run_arm_generic.py in every respect that touches the measurement -- same
prompts, same count, same sampling, same output shape -- differing only in that
gpu_memory_utilization is an argument instead of the hardcoded 0.5. That constant was sized
for 1.7B and 8B; a 14B INT8 checkpoint is 16 GiB and leaves no room for a KV cache under it,
which is why 14B's end-to-end run failed with vLLM's own suggestion to raise it.

v1's script is left untouched. The condition difference -- 14B at 0.85 where the smaller
models ran at 0.5 -- is disclosed in the pre-registration rather than hidden by editing the
original in place.
"""
import json
import sys

from vllm import LLM, SamplingParams

if __name__ == "__main__":
    model, out = sys.argv[1], sys.argv[2]
    gpu_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    prompts = json.load(open("/w/calib_prompts.json"))[:8]
    llm = LLM(model=model, enforce_eager=True, gpu_memory_utilization=gpu_frac,
              max_model_len=2048)
    outs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0,
                                                ignore_eos=True))
    json.dump([list(o.outputs[0].token_ids) for o in outs], open(out, "w"))
    print(f"ARM_OK {out} gpu_memory_utilization={gpu_frac}")
