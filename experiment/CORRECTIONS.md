# Corrections to paper 3 (arXiv:2609.00363v1)

These were found after publication, while preparing a revised version of the paper (not
yet public), which incorporates them. None changes a measured value in this repository; all concern how the
paper reported or scoped those values. The pinned artifacts are not edited, because their
hashes are part of the record; where a file's own fields are affected, this note is the
correction.

## 1. The comparison against the original prediction matrix

The pre-registration requires the sensitivity results to be reported against both the
original and the corrected prediction matrix, so a reader can judge whether the post-data
changes favoured the study. Version 1 reported, for the original matrix, 75.0% detection for
the accumulator check, false-positive rates of 55.76% (power-of-two check) and 11.39%
(tolerance check), and 1,278 of the 8,232 cells flipped from error to agreement by the
corrections.

Those figures were computed against a matrix that was never pinned: the re-pinned 77-cell
matrix with its three `corrected_post_data` cells set back to the predictions their recorded
reasoning calls original. Setting those three cells back and re-scoring reproduces version
1's figures exactly (1,278 cells; 75.0%, 55.76% and 11.39%); `tests_p5_rescore.py` checks
this. The matrix that was hash-pinned before any data (amendment A-10.1,
SHA-256 prefix `8f6ada4492c688b0`) is now included as
`artifacts/p5_prediction_matrix_A-10.1_pinned.json`. Against it, the committed cells give:

| check | pinned pre-data matrix (A-10.1) | re-pinned 77-cell matrix (A-10.5) |
|---|---|---|
| exact accumulator | 1,176 of 1,568 predicted fires held (75.0%); 392 stayed silent | all held |
| power-of-two identity | 494 of 886 scored predicted silences fired (55.76%) | all held |
| real-scale tolerance | 784 of 984 predicted fires held (79.7%); no false positive | all held |
| all checks | 1,086 of 38,039 scored check–cell pairs violated, in 886 of 8,232 cells | 0 violated |

These are prediction-adherence figures: they count predictions that held, not faults that
were detected. Reproduce them on a CPU with

```bash
python3 experiment/harness/p5_rescore.py experiment/artifacts/p5p6_results/p5_sensitivity.json \
        experiment/artifacts/p5_prediction_matrix_A-10.1_pinned.json
python3 experiment/harness/tests_p5_rescore.py
```

## 2. Which predictions changed after the smoke run

Compared cell by cell with the pinned matrix (whose single F9 row applies to all three of
F9's magnitudes), four predictions in the re-pinned matrix differ. All four were set after the
two-layer smoke run had shown their outcome. F9's rungs carry other names elsewhere: in the
matrices, `p5_sensitivity.json` and `tests_p5_rescore.py` they are stored under the
coverage-ladder severity keys, so the scale-ulp rung (one weight scale moved by one float32
ulp) is `one_percent` (a misnomer; version 1 calls it the "one-percent rung" and its table row
"F9, one pct."), the one-int8-step rung is `one_element`, and the redrawn rung is
`all_elements`.

| cell | pinned | re-pinned | effect on the scores |
|---|---|---|---|
| F4 × power-of-two identity (5) | silent | fire | removes 494 false positives |
| F9 scale-ulp rung × exact accumulator (4) | fire | silent | removes 392 misses |
| F9 scale-ulp rung × real-scale tolerance (6) | fire | silent, as a sixth predicted miss | removes 200 misses |
| F9 scale-ulp rung × power-of-two identity (5) | fire | silent | none: check (5) applies in no scale-ulp cell |

Three of the changes favour the suite and one affects no score. Version 1 described three
corrected cells instead; so does the pre-registration's own record (A-10.5, known limitation
3, `preregistration.md` L363), which this count supersedes.

The `corrected_post_data` flags in `artifacts/p5_prediction_matrix.json` (count 3, repeated as
`matrix_cell_counts` in `p5p6_results/p5_sensitivity.json`) mark a different set of three:
F4 × (5), the scale-ulp rung × (4), and the one-int8-step rung × (6). The last one is unchanged
against the pinned matrix. Its "original" silent prediction belonged to a draft written after
the F9 split and never pinned (amendment A-10.7 records it). The two scale-ulp changes at (5)
and (6) carry no flag.

Version 1 also said the original reasoning is preserved verbatim in the artifact. It is not:
each flagged cell's reason restates, in paraphrase rather than quotation, the prediction and
reasoning it calls original, and for the one-int8-step cell that original is the unpinned
draft rather than the pinned matrix.

Two more points about the re-pinned file:

- It keeps the original's status line, "Written and hash-pinned before any P5 measurement",
  although it was re-pinned after the smoke run (A-10.5).
- Its stated reason for the scale-ulp rung × (5) is that a one-ulp move "almost certainly
  rounds to the same power of two". The data contradict this. The injector does not re-snap
  the moved scale, so the scale is no longer a power of two and check (5) applies in none of
  the 392 scale-ulp cells. Under power-of-two scales the move changed an output in all 196
  cells, by one bfloat16 spacing.

## 3. Scope statements

- **The reproduction gate ran at 1.7B only.** The byte-exact gate
  (`p8_results/p8_report_minmax.json`) was run against the 1.7B base only. The 8B and 14B
  power-of-two arms reuse the gated pipeline, so at those sizes "the arms differ only in the
  scale rule" is inferred, not audited. The docstring of `harness/quant_qwen14b.py` (L9–L12)
  describes the 14B arms as resting on A-12.1's gate; read it with this correction.
- **Per-layer bitwise agreement was measured at 1.7B and 8B only.** Under requantized
  power-of-two scales this agreement was measured at 1.7B and 8B; 14B was compared end to end
  only.
- **The two "power-of-two" regimes differ.** The fault-injection regime snaps both the weight
  and the activation scales. The requantized checkpoints snap only the weight scales. That is
  enough to make the two epilogue orderings agree, but not enough for check (5) as specified,
  which applies only when every scale is a power of two.
- **Not every operand is a captured one.** The precondition faults F6 and F7 use constant
  all-127 int8 operands with the layer's M, N and scales but at constructed reduction depths
  (K = 133,145 for F6, K = 1,042 for F7), not captured operands. In F9's redrawn rung, the
  second arm's activations are random.
- **The epilogue result is measured, not guaranteed.** Every measured epilogue cell stayed
  within one bfloat16 spacing; this is not a structural guarantee. F3 is a legal
  re-association, so its silence is correct rather than a miss. The pre-registration's
  A-10.7 point 4 (`preregistration.md` L309–L311) states the one-spacing tolerance's
  blindness as a structural certainty; that statement is superseded here.

## 4. A condition difference that was not disclosed

The 14B end-to-end runs used vLLM's `gpu_memory_utilization` of 0.85
(`harness/run_arm_mem.py`, `harness/drivers/p8_14b_e2e_retry.sh`). The 1.7B and 8B end-to-end
runs used 0.5 (`harness/run_arm_generic.py`), which at 14B left no room for the KV cache; the
perplexity runs used 0.85 at all three sizes. The pre-registration requires this difference
to be disclosed, and version 1 did not disclose it. Both arms at a given size share the
setting, and only the within-size verdict (0/8 versus 8/8 identical sequences) is compared
across sizes.

## 5. Smaller statements

- **The 64 tokens are generated tokens.** "Prompts of 64 tokens" means 64 tokens generated
  from each prompt. The prompts are the first eight of 64 consecutive 320-token windows of
  WikiText-2, cut with the GPT-2 tokenizer. The perplexity windows are the 256 windows that
  follow them.
- **The 64 tokens span 63 decode steps.** The 64 generated tokens cover the prefill pass and
  63 decode steps, not 64.
- **The 6.42% mismatch is for one layer.** The first reconstruction, which divided in
  float32, mismatched 6.42% of the int8 weights in `layers.0.mlp.down_proj` (and 6.89% in
  `layers.5.self_attn.q_proj`, the other layer checked). It is not a figure for the whole
  checkpoint.
- **The captured layers have K = 2,048 and 6,144.** Version 1 said the captured operands
  were replayed "at K from 1,042 to 12,288". Qwen3-1.7B's linear layers have reduction depth
  2,048 or 6,144; 1,042 is F7's constructed depth, and no layer has 12,288.
- **F1's bound is 1/257, not 2^-9.** Version 1 repeated the amendment's statement that F1
  moves a scale by at most 2^-9 relative. Round-to-nearest-even to bfloat16 has relative
  error up to 2^-8/(1+2^-8) = 1/257 (257/256 rounds to 1). The measured displacements are
  unaffected.
- **17 to 21 bits, and |w| up to 128.** The quoted matrix reasons say the accumulator needs
  seventeen mantissa bits and that one activation step moves an accumulator entry by up to
  |w| = 127. Across the 196 layers the maximum |acc| lies between 2^16 and 2^21, so 17 to 21
  bits are needed, and the weights reach -128, so the shift is up to 128. The conclusions are
  unchanged.
- **The reproduction-failure record describes a superseded route.**
  `artifacts/p8_requant_reproduction_failure.json` records the first, float32-division version
  of `p8_requant.py` (SHA-256 prefix `b6ab48f6f2dc136e`). Its `route_chosen` and
  `status_of_p8_requant_py` fields were superseded the same day: the version pinned by A-12.1
  (`010eda7602eb6ee9`) divides in bfloat16, reproduces the 1.7B base byte for byte
  (`p8_results/p8_report_minmax.json`) and built every P8 arm.
- **The ceiling arm is not shown to cost more.** Its point estimate is higher (+0.54% versus
  +0.32%), but the two 90% intervals overlap (+0.18% to +0.92% and −0.06% to +0.71%), so
  the data do not establish an ordering of the two arms.
