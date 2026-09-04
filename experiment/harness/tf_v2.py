import argparse, json
from vllm import LLM, SamplingParams, TokensPrompt
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--rails", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-prompts", type=int, default=64)
    a = ap.parse_args()
    prompts = json.load(open("/w/calib_prompts.json"))[:a.n_prompts]
    gen = json.load(open(a.rails))[:a.n_prompts]
    llm = LLM(model=a.model, enforce_eager=True, gpu_memory_utilization=0.6,
              max_model_len=2048)
    tok = llm.get_tokenizer()
    rails = []
    for p, g in zip(prompts, gen):
        ids = tok(p, add_special_tokens=False)["input_ids"]
        rails.append({"plen": len(ids), "ids": ids + list(g)})
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=2)
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"]) for r in rails], sp)
    recs = []
    for r, o in zip(rails, outs):
        pls = o.prompt_logprobs or []
        seq = []
        for pos in range(r["plen"], len(r["ids"])):
            lp = pls[pos] if pos < len(pls) else None
            if lp is None:
                seq.append(None); continue
            ranked = sorted(((float(v.logprob), int(t)) for t, v in lp.items()), reverse=True)
            chosen = lp.get(r["ids"][pos])
            seq.append({"pos": pos,
                        "chosen_lp": float(chosen.logprob) if chosen else None,
                        "top1_id": ranked[0][1] if ranked else None,
                        "margin": (ranked[0][0] - ranked[1][0]) if len(ranked) > 1 else None})
        recs.append(seq)
    json.dump(recs, open(a.out, "w"))
    print("TFV2_OK", a.out)
