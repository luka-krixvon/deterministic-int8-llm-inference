import json, sys
from vllm import LLM, SamplingParams, TokensPrompt
if __name__ == "__main__":
    model, rails_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    prompts = json.load(open("/w/calib_prompts.json"))[:8]
    gen = json.load(open(rails_path))
    llm = LLM(model=model, enforce_eager=True, gpu_memory_utilization=0.5,
              max_model_len=2048)
    tok = llm.get_tokenizer()
    rails = []
    for p, g in zip(prompts, gen):
        ids = tok(p, add_special_tokens=False)["input_ids"]
        rails.append({"plen": len(ids), "ids": ids + list(g)})
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"]) for r in rails], sp)
    recs = []
    for r, o in zip(rails, outs):
        pls = o.prompt_logprobs or []
        seq = []
        for pos in range(r["plen"], len(r["ids"])):
            lp = pls[pos] if pos < len(pls) else None
            if lp is None:
                seq.append(None); continue
            ranked = sorted(((float(v.logprob), int(t)) for t, v in lp.items()),
                            reverse=True)
            seq.append({"pos": pos, "rail_id": r["ids"][pos],
                        "top1_id": ranked[0][1] if ranked else None})
        recs.append(seq)
    json.dump(recs, open(out, "w"))
    print(f"TF_OK {out}")
