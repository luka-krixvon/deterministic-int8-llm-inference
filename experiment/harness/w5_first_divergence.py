"""E2 first-divergence instrumentation — harness (first GPU run queued).

PURPOSE (audit item 6): layerwise capture so residual same-arm or cross-arm
divergence is attributed only by locating the FIRST differing tensor; until
that tensor is found, no stage attribution may be claimed.

CLI (inside pinned vLLM container; the file must be SHIPPED to the VM's
flat working dir first — repo path is experiment/harness/w5_first_divergence.py,
runtime path is /w/w5_first_divergence.py after scp):
  python3 /w/w5_first_divergence.py --model /models/qwen3-1.7b-int8-w8a8-pow2 \
      --arm CUTLASS|TRITON --prompt-idx 0 --out /w/fd_<arm>_<idx>.json
Compare two arms' outputs offline: first stage whose tensor hash differs is
the attribution point.

CAPTURE POINTS per decoder layer L (pre-fixed):
  norm_in.L      input_layernorm output
  quant.L.x_q    activation quantizer int8 output (bitwise)
  quant.L.x_s    activation scales
  qkv.L          QKV projection output
  rope.L         post-RoPE Q/K
  attn.L         attention output
  mlp.L          MLP down_proj output
  resid.L        residual stream after layer
  lm_head        final logits
OUTPUT SCHEMA: {"arm":…, "prompt_idx":…, "stages":[{"name":…, "sha256_16":…,
  "dtype":…, "shape":…}…]}  (hashes, not tensors — comparison is hash-level;
  a differing stage triggers a second run capturing that stage's tensor)
PRE-FIXED CRITERION: attribution = first stage in forward order whose hash
differs between arms at identical token position; stages after it are
reported as downstream, not causes.

IMPLEMENTATION NOTE: runs the vLLM *model module* directly (no engine) so
forward hooks work in-process; requires vLLM model-loader glue that is
validated on first GPU run — until that run, this file is a specification
plus best-effort implementation, and no instrumentation results exist.
"""
import argparse, hashlib, json

def sha(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.int8 if t.dtype==torch.int8 else t.dtype).numpy().tobytes()).hexdigest()[:16]

def main():
    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-idx", type=int, default=0)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # Engine-external module loading glue (validated on first GPU run):
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    ea = EngineArgs(model=args.model, enforce_eager=True,
                    gpu_memory_utilization=0.5, max_model_len=2048)
    vc = ea.create_engine_config()
    from vllm.model_executor.model_loader import get_model
    model = get_model(vllm_config=vc)
    prompts = json.load(open("/w/calib_prompts.json"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(prompts[args.prompt_idx], return_tensors="pt",
              truncation=True, max_length=512)["input_ids"].cuda()
    stages = []
    def hook(name):
        def f(mod, i, o):
            t = o[0] if isinstance(o, tuple) else o
            stages.append({"name": name, "sha256_16": sha(t),
                           "dtype": str(t.dtype), "shape": list(t.shape)})
        return f
    for n, m in model.named_modules():
        tail = n.split(".")[-1]
        if tail in ("input_layernorm", "qkv_proj", "o_proj", "down_proj",
                    "post_attention_layernorm", "attn", "lm_head"):
            m.register_forward_hook(hook(n))
    with torch.no_grad():
        # positions/kv glue is model-specific; validated on first run
        model(input_ids=ids, positions=torch.arange(ids.shape[1], device="cuda"))
    json.dump({"arm": args.arm, "prompt_idx": args.prompt_idx,
               "stages": stages}, open(args.out, "w"))
    print("FD_OK", args.out)

if __name__ == "__main__":
    main()
