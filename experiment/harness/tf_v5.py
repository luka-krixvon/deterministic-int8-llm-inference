import argparse, glob, hashlib, json
from vllm import LLM, SamplingParams, TokensPrompt

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def checkpoint_digest(model_dir):
    h = hashlib.sha256()
    for shard in sorted(glob.glob(model_dir + "/*.safetensors")):
        with open(shard, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--rails", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--parent-revision", default=None)
    a = ap.parse_args()
    all_prompts = json.load(open("/w/calib_prompts.json"))
    all_gen = json.load(open(a.rails))
    if len(all_prompts) < a.n_prompts or len(all_gen) < a.n_prompts:
        raise SystemExit(f"REQUESTED {a.n_prompts} but prompts={len(all_prompts)} rails={len(all_gen)}")
    prompts = all_prompts[:a.n_prompts]; gen = all_gen[:a.n_prompts]

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
    for i, (r, o) in enumerate(zip(rails, outs)):
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
        recs.append({"prompt_sha": hashlib.sha256(prompts[i].encode()).hexdigest(),  # FULL sha
                     "steps": seq})
    if len(recs) != a.n_prompts:
        raise SystemExit(f"OUTPUT {len(recs)} != requested {a.n_prompts}")
    try:
        qcfg = json.load(open(a.model + "/config.json")).get(
            "quantization_config") or json.load(open(a.model + "/config.json")).get(
            "compression_config")
    except Exception:
        qcfg = None
    tok_files = sorted(glob.glob(a.model + "/tokenizer*") + glob.glob(a.model + "/vocab*"))
    manifest = {
        "requested_prompts": a.n_prompts, "actual_prompts": len(recs),
        "prompt_list_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
        "rails_sha256": sha_file(a.rails),
        "checkpoint_digest": checkpoint_digest(a.model),
        "quantization_config_sha256": (hashlib.sha256(
            json.dumps(qcfg, sort_keys=True).encode()).hexdigest() if qcfg else None),
        "tokenizer_files_sha256": hashlib.sha256(
            "".join(sha_file(f) for f in tok_files).encode()).hexdigest() if tok_files else None,
        "parent_revision": a.parent_revision,
        "model": a.model,          # diagnostic only, not identity
        "generator": "tf_v5.py",
    }
    import vllm
    manifest["vllm_version"] = vllm.__version__
    json.dump({"manifest": manifest, "records": recs}, open(a.out, "w"))
    print("TFV5_OK", a.out)
