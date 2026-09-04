"""W4/P4 stage 1: teacher-forced per-step logit divergence between kernel arms.

Replays IDENTICAL token sequences (prompt + the CUTLASS arm's greedy
continuation, from tokens_CUTLASS.json) through BOTH arms using
prompt_logprobs, so every position is scored under the same context —
no cascade. For each position we record, per arm:
  - logprob of the actually-chosen token
  - top-1/top-2 logprobs and the margin between them
Cross-arm deltas of these quantities are the P4 transfer function's input:
flip probability at a position should be ~ P(margin < cross-arm delta).

Run inside the pinned vLLM container (real __main__ file, spawn-safe):
  python3 /w/w4_teacher_forcing.py CUTLASS
  python3 /w/w4_teacher_forcing.py TRITON   (with VLLM_DISABLED_KERNELS set)
Then compare the two JSON outputs offline.
"""
import json, sys
from vllm import LLM, SamplingParams, TokensPrompt

if __name__ == "__main__":
    label = sys.argv[1]
    prompts = json.load(open("/w/calib_prompts.json"))[:8]
    gen = json.load(open("/w/tokens_CUTLASS.json"))       # reference rails

    llm = LLM(model="/models/qwen3-1.7b-int8-w8a8", enforce_eager=True,
              gpu_memory_utilization=0.5, max_model_len=2048)
    tok = llm.get_tokenizer()

    rails = []
    for p, g in zip(prompts, gen):
        ids = tok(p, add_special_tokens=False)["input_ids"]
        rails.append({"prompt_len": len(ids), "ids": ids + list(g)})

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=2)
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"]) for r in rails], sp)

    recs = []
    for r, o in zip(rails, outs):
        pls = o.prompt_logprobs or []
        seq = []
        for pos in range(r["prompt_len"], len(r["ids"])):
            lp = pls[pos] if pos < len(pls) else None
            if lp is None:
                seq.append(None); continue
            chosen_id = r["ids"][pos]
            entry = lp.get(chosen_id)
            chosen_lp = float(entry.logprob) if entry is not None else None
            ranked = sorted(((float(v.logprob), int(t)) for t, v in lp.items()),
                            reverse=True)
            top1 = ranked[0] if ranked else (None, None)
            top2 = ranked[1] if len(ranked) > 1 else (None, None)
            seq.append({
                "pos": pos, "chosen_id": chosen_id, "chosen_lp": chosen_lp,
                "top1_id": top1[1], "top1_lp": top1[0],
                "top2_lp": top2[0],
                "margin": (top1[0] - top2[0])
                          if top1[0] is not None and top2[0] is not None else None,
            })
        recs.append(seq)

    json.dump(recs, open(f"/w/tf_{label}.json", "w"))
    print(f"TF_{label}_OK")
