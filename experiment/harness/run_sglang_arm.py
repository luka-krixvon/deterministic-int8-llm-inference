"""Third implementation arm: SGLang on the same INT8-W8A8 checkpoint.
Greedy 64 tokens on the same 8 pinned prompts; dumps output token ids
(and text as fallback) for offline comparison against the vLLM arms."""
import json, sys

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "1"
    import sglang as sgl
    prompts = json.load(open("/w/calib_prompts.json"))[:8]
    llm = sgl.Engine(model_path="/models/qwen3-1.7b-int8-w8a8",
                     mem_fraction_static=0.5, context_length=2048,
                     disable_cuda_graph=True)
    outs = llm.generate(prompts, {"temperature": 0, "max_new_tokens": 64,
                                  "ignore_eos": True})
    recs = []
    for o in outs:
        rec = {"text": o.get("text") if isinstance(o, dict) else str(o)}
        if isinstance(o, dict):
            mi = o.get("meta_info", {})
            for k in ("output_ids", "completion_tokens_ids", "output_token_ids"):
                if k in o: rec["ids"] = o[k]
                elif k in mi: rec["ids"] = mi[k]
        recs.append(rec)
    json.dump(recs, open(f"/w/sglang_out_{tag}.json", "w"))
    print(f"SGLANG_ARM_{tag}_OK")
