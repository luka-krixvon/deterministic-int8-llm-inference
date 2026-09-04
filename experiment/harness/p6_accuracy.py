"""P6 stage B: perplexity of the pow2 probe against the base INT8 checkpoint.

Runs inside the pinned vLLM container, which is where every v1 measurement ran. It
does not derive windows -- the container has no `datasets` package, and installing one
would mean the identity contract records an image digest that never ran. Windows come
from a file that p6_windows.py produced in the venv, and this stage verifies that
file's content hash before scoring. If the hash does not match, it refuses; a stage
that scored windows it could not verify would make the disjointness commitment
unfalsifiable.

Pre-registration A-10 commits to reporting the result whatever it says, and forbids
restating the pow2 intervention as diagnostic-only to avoid publishing a cost.

Per-window negative log-likelihood is summed in float64. At bf16 output precision the
two arms differ by at most an output spacing per token, so a float32 accumulation over
320 tokens x 256 windows could bury the real difference under summation order -- the
confusion this study exists to avoid.

Usage (in the container, via drivers/p6_run.sh):
  python3 p6_accuracy.py --model /models/qwen3-1.7b-int8-w8a8 \
                         --windows p6_eval_windows.json --out p6_acc_base.json
  python3 p6_accuracy.py --compare p6_acc_base.json p6_acc_pow2.json --out p6_cost.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import statistics

from p6_windows import load_verified


def window_nll(model_dir: str, texts, max_model_len: int = 512, gpu_frac: float = 0.85):
    """Per-window summed NLL and token count, teacher-forced, accumulated in float64.

    Uses vLLM prompt_logprobs so the numbers come from the runtime the study measured,
    not from a second inference stack whose kernels differ.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_frac, enforce_eager=True,
              disable_log_stats=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate(texts, sp)

    rows = []
    for i, o in enumerate(outs):
        lps = o.prompt_logprobs or []
        total = 0.0                      # python float is IEEE double
        n = 0
        for pos, entry in enumerate(lps):
            if entry is None:            # the first token has no conditional
                continue
            tok_id = o.prompt_token_ids[pos]
            lp = entry[tok_id]
            total += float(lp.logprob if hasattr(lp, "logprob") else lp)
            n += 1
        rows.append({"window": i, "nll_sum": -total, "n_tokens": n})

    import vllm
    return rows, {
        "runtime": "vllm",
        "vllm_version": vllm.__version__,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "enforce_eager": True,
        "prompt_logprobs": 0,
    }


def ppl_from_rows(rows):
    s = sum(r["nll_sum"] for r in rows)
    n = sum(r["n_tokens"] for r in rows)
    return (math.exp(s / n) if n else float("nan")), s, n


def bootstrap_delta(base_rows, pow2_rows, n_boot: int = 10000, seed: int = 20260814):
    """Cluster bootstrap over windows for the PPL difference (pow2 minus base).

    Windows are the clusters, matched pairwise: a resample takes the same window index
    from both arms, so the two arms see identical resamples and the interval is about
    the difference rather than about sampling windows twice over.
    """
    import random
    if len(base_rows) != len(pow2_rows):
        raise ValueError("arms have different window counts")
    rnd = random.Random(seed)
    n = len(base_rows)
    deltas = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        bs = sum(base_rows[i]["nll_sum"] for i in idx)
        bn = sum(base_rows[i]["n_tokens"] for i in idx)
        ps = sum(pow2_rows[i]["nll_sum"] for i in idx)
        pn = sum(pow2_rows[i]["n_tokens"] for i in idx)
        if bn and pn:
            deltas.append(math.exp(ps / pn) - math.exp(bs / bn))
    deltas.sort()
    return {"n_boot": len(deltas),
            "ci90_low": deltas[int(0.05 * len(deltas))],
            "ci90_high": deltas[int(0.95 * len(deltas)) - 1],
            "median": statistics.median(deltas), "seed": seed,
            "cluster": "window", "paired": True}


def checkpoint_digest(model_dir: str) -> str:
    """Same scope as tf_v6.checkpoint_digest, so P6 records identity the same way."""
    h = hashlib.sha256()
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(shard, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--windows", help="window file from p6_windows.py (stage A)")
    ap.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "POW2_JSON"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.compare:
        base = json.load(open(a.compare[0]))
        pow2 = json.load(open(a.compare[1]))
        if base["windows_content_sha256"] != pow2["windows_content_sha256"]:
            raise RuntimeError("arms scored different window sets; comparison refused")
        pb, sb, nb = ppl_from_rows(base["rows"])
        pp, sp_, np_ = ppl_from_rows(pow2["rows"])
        out = {
            "schema": "p6-accuracy-compare-v2",
            "commitment": ("Pre-registration A-10 forbids withholding this result or "
                           "restating the pow2 intervention as diagnostic-only to avoid "
                           "publishing a cost. Whatever the delta is, it is reported."),
            "n_windows": len(base["rows"]),
            "windows_content_sha256": base["windows_content_sha256"],
            "windows_disjointness": base["windows_disjointness"],
            "base": {"model": base["model"], "checkpoint_digest": base["checkpoint_digest"],
                     "ppl": pb, "nll_sum": sb, "n_tokens": nb},
            "pow2": {"model": pow2["model"], "checkpoint_digest": pow2["checkpoint_digest"],
                     "ppl": pp, "nll_sum": sp_, "n_tokens": np_},
            "delta_ppl_pow2_minus_base": pp - pb,
            "relative_delta": ((pp - pb) / pb if pb else None),
            "bootstrap": bootstrap_delta(base["rows"], pow2["rows"]),
            "nll_accumulated_in": "float64",
        }
        json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
        b = out["bootstrap"]
        print(f"base ppl {pb:.6f} | pow2 ppl {pp:.6f} | delta {pp - pb:+.6f} "
              f"| ci90 [{b['ci90_low']:+.6f}, {b['ci90_high']:+.6f}]")
        return

    if not (a.model and a.windows):
        ap.error("--model and --windows are both required unless --compare is given")

    win = load_verified(a.windows)          # raises if the hash or verdict is wrong
    rows, runtime = window_nll(a.model, win["texts"])
    ppl, s, n = ppl_from_rows(rows)
    out = {
        "schema": "p6-accuracy-v2",
        "model": a.model,
        "checkpoint_digest": checkpoint_digest(a.model),
        "windows_file": os.path.basename(a.windows),
        "windows_content_sha256": win["content_sha256"],
        "windows_disjointness": win["disjointness"],
        "windows_dataset": win["dataset"],
        "windows_derived_with": win["derived_with"],
        "n_windows": win["n_windows"],
        "rows": rows,
        "ppl": ppl, "nll_sum": s, "n_tokens": n,
        "nll_accumulated_in": "float64",
        "runtime": runtime,
        "env": {k: os.environ.get(k) for k in
                ("VLLM_DISABLED_KERNELS", "CUDA_VISIBLE_DEVICES",
                 "VLLM_ENABLE_V1_MULTIPROCESSING", "IMAGE_DIGEST")},
    }
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"{a.model}: ppl {ppl:.6f} over {len(rows)} windows, {n} scored tokens, "
          f"windows {win['content_sha256'][:16]}")


if __name__ == "__main__":
    main()
