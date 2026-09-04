import argparse, json
from vllm import LLM, SamplingParams, TokensPrompt
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--rails", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-prompts", type=int, default=64)
    a = ap.parse_args()
    all_prompts = json.load(open("/w/calib_prompts.json"))
    all_gen = json.load(open(a.rails))
    if len(all_prompts) < a.n_prompts or len(all_gen) < a.n_prompts:
        raise SystemExit(f"REQUESTED {a.n_prompts} but prompts={len(all_prompts)} rails={len(all_gen)}")
    prompts = all_prompts[:a.n_prompts]
    gen = all_gen[:a.n_prompts]
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
        import hashlib
        recs.append({"prompt_sha": hashlib.sha256(prompts[len(recs)].encode()).hexdigest()[:16],
                     "steps": seq})
    if len(recs) != a.n_prompts:
        raise SystemExit(f"OUTPUT {len(recs)} != requested {a.n_prompts}")
    import hashlib
    manifest = {"requested_prompts": a.n_prompts, "actual_prompts": len(recs),
                "prompt_list_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
                "rails_sha256": hashlib.sha256(open(a.rails, "rb").read()).hexdigest(),
                "model": a.model}
    json.dump({"manifest": manifest, "records": recs}, open(a.out, "w"))
    print("TFV4_OK", a.out)
