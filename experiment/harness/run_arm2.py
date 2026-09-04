import json, sys
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    label = sys.argv[1]
    prompts = json.load(open("/w/calib_prompts.json"))[:8]
    llm = LLM(model="/models/qwen3-1.7b-int8-w8a8", enforce_eager=True,
              gpu_memory_utilization=0.5, max_model_len=2048)
    outs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0,
                                                ignore_eos=True))
    json.dump([list(o.outputs[0].token_ids) for o in outs],
              open(f"/w/tokens_{label}2.json", "w"))
    print(f"ARM_{label}_DONE")
