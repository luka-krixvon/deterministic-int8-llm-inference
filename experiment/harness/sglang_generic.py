import json, sys
if __name__ == "__main__":
    model, out = sys.argv[1], sys.argv[2]
    import sglang as sgl
    prompts = json.load(open("/w/calib_prompts.json"))[:8]
    llm = sgl.Engine(model_path=model, mem_fraction_static=0.5,
                     context_length=2048, disable_cuda_graph=True)
    outs = llm.generate(prompts, {"temperature": 0, "max_new_tokens": 64,
                                  "ignore_eos": True})
    recs = []
    for o in outs:
        rec = {"text": o.get("text")}
        mi = o.get("meta_info", {})
        for k in ("output_ids", "completion_tokens_ids", "output_token_ids"):
            if k in o: rec["ids"] = o[k]
            elif k in mi: rec["ids"] = mi[k]
        recs.append(rec)
    json.dump(recs, open(out, "w"))
    print(f"SGLANG_OK {out}")
