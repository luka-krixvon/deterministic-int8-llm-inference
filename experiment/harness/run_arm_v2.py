import argparse, json
from vllm import LLM, SamplingParams
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    prompts = json.load(open("/w/calib_prompts.json"))[:a.n_prompts]
    llm = LLM(model=a.model, enforce_eager=True, gpu_memory_utilization=0.6,
              max_model_len=2048)
    outs = llm.generate(prompts, SamplingParams(max_tokens=a.max_tokens,
                                                temperature=0, ignore_eos=True))
    json.dump([list(o.outputs[0].token_ids) for o in outs], open(a.out, "w"))
    print("ARMV2_OK", a.out)
