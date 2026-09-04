# Pre-Registration: English Summary of the Amendments Cited by the Conformance Paper

The authoritative record is `preregistration.md`, which is append-only and written in the
author's working language (Traditional Chinese). This file summarizes, in English, the
amendments the conformance paper relies on: A-10 and its sub-amendments (the sensitivity
experiment), A-11 (the second kernel pair, closed as a negative report), and A-12 (the
power-of-two treatment). It is a reading aid, not a substitute: where the two disagree, the
Chinese original governs, and nothing here restates a commitment the original does not
contain.

Amendment classes used throughout: **class A** means recorded *before* the measurement it
governs began; **PIN** means plan, code, and tests were committed with their SHA-256 digests
and no results; **post-data** means recorded after seeing data, which is a protocol deviation
that must be reported as such. Amendment numbers are not chronological — they are issued in
the order commitments were made, and a later number can precede an earlier one in time. The
ordering is visible in each amendment's date.

---

## A-10 (class A, 2026-08-14, before any measurement in this round)

Adds two experiments, both closing gaps that the companion study itself disclosed:
**P5**, a positive control measuring the sensitivity and miss rate of the companion's seven
conformance checks, and **P6**, the cost of the power-of-two intervention in accuracy and
throughput. The companion's own claims do not change with this round's outcome: if P5 fails,
the seven checks are downgraded to "a proposal whose sensitivity was never validated" rather
than the companion being retroactively edited.

**Fault catalogue, fixed at eight items, closed to additions or deletions after execution.**
F1 scales multiplied in bfloat16 rather than float32; F2 double rounding
(fp32→bf16→fp32→bf16); F3 the two scales' multiplication order exchanged; F4 the output cast
changed to truncation rather than round-to-nearest-even; F5 the epilogue's fusion order
changed; F6 forced accumulator overflow (via a $K$ exceeding the bound); F7 accumulator
forced above $2^{24}$ on float32 entry without overflowing INT32; and **F8, a null fault** —
a semantically equivalent rewrite, serving as a false-positive control, **predicted to be
judged a violation by no check**.

**Injection points, with the reason.** Faults are injected in both the epilogue and the
accumulator. Injecting only in the epilogue would be insufficient to exercise checks 2 and
3, which assert the alibi's *preconditions*: those must be tested by cases that deliberately
violate the preconditions, or their false-negative rate cannot be known.

**Prediction matrix, pinned by this amendment and public before execution.** For every
fault × check pair, "should fire / should not fire / not applicable" is written in advance
with its reason, into `artifacts/p5_prediction_matrix.json`. Every cell is falsifiable: a
predicted fire that stays silent is a miss, a predicted silence that fires is a false
positive.

**No pass threshold.** The output of the experiment is the sensitivity numbers, not a
verdict of pass or fail.

**Null commitments added.** (1) If a check misses faults it was predicted to catch, report
that its sensitivity is insufficient — do not redefine the check and re-measure, reporting
only the second run. (2) If the power-of-two intervention carries a real cost, report it —
do not restate power-of-two as "diagnostic only, deployment not recommended" to avoid
publishing the number, since the diagnostic claim was already established and this round's
question is the deployment cost. (3) If F8 is judged a violation by any check, that is direct
evidence of a false positive and must be reported alongside the detection rates.

## A-10.2 (class A, 2026-08-14, before any P5 execution)

Filling in the matrix exposed a structural defect in the eight-fault catalogue: **check 1
(shared operands) had no case that should fire, so its detection rate could not be
measured.** F1–F5 and F8 perturb inside the epilogue with identical operands in both arms;
F6 and F7 feed both arms the same (deliberately overflowing) operands. Nothing made the
operands differ, and check 1 is the check that compares them. Reporting "check 1 raised no
false alarm under eight faults" would have been reporting *not applicable* as *passed*.

**F9, operand mismatch, is therefore added**, at three magnitudes: one int8 element differing
by one, one per-channel scale differing by one float32 ulp, and the whole activation tensor
requantized under a different seed. Check 1 is predicted to fire in all three. The report must
mark explicitly that what the *other* checks do under F9 says nothing about their detection
ability, since once check 1 fails the other checks' conclusions do not stand — a dependency
recorded in check 1's docstring. The catalogue becomes **nine items and is closed from this
amendment onward**.

Also recorded here, as an implementation self-correction rather than a change of commitment:
the first version of the ULP ordered mapping in `p5_checks.py` mapped $\pm 0$ to 65536 apart
rather than 0, caught by regression test. Review confirmed the companion's implementation was
correct, so the reported maximum ULP distance of 1 is unaffected; `p5_checks.py` was changed
to be textually identical to it, with a test comparing both mappings so they agree by
construction rather than by coincidence.

## A-10.3 (class A, 2026-08-14, before any P5 execution)

**The severity ladder becomes a coverage ladder.** A-10 specified three magnitudes per fault.
Implementation confirmed that F1–F5 are *categorical, not scalar*: multiplying in bfloat16,
double rounding, reassociation, truncation, fusion order — none has a magnitude knob, the
effect being simply what the arithmetic yields. What can be controlled is **how many output
elements the fault touches**, so the ladder becomes `one_element` / `one_percent` /
`all_elements`. This is also closer to what a reader wants to know: how small a fault each
check can still catch. F6/F7 and F8 are binary; F9's ladder is genuinely magnitudinal and
is unchanged.

**"Check insensitive" and "fault unobservable at output precision" must be distinguished.**
bfloat16 carries eight mantissa bits, so any float32 perturbation below its rounding step is
absorbed. A pre-execution observability survey found F3 (scale reassociation) produced **zero**
differing elements in all three regimes and F5 nearly zero; that is an arithmetic fact, not
an injector defect — float32 reassociation has relative error near $2^{-23}$ against a
bfloat16 rounding step of $2^{-8}$, roughly fifteen binades below visibility. `inject()`
therefore always records `n_output_differing`, and **the report must read
`n_output_differing == 0` as "the fault is unobservable at output precision," never as "the
check is insensitive."** The rule is written into the injector's module comment and into a
regression test, so that anyone later "fixing" F3's unobservability is noticed rather than
silently changing the sensitivity table's meaning.

**A limitation of the paper, not merely of the harness, follows.** Check 6 (real-scale
tolerance) has a **detection floor**: passing it does not establish epilogue equivalence,
only that any difference is below bfloat16 resolution plus one spacing. Two kernels may use
different scale association or fusion and still produce byte-identical output. This is listed
here as a limitation the round must state in writing.

## A-10.4 (class A, 2026-08-14, mandatory disclosure once the matrix was written)

The matrix was written and committed before any measurement. Completing it surfaced a
structural fact that had to be disclosed in advance: **every predicted fire came from
F6/F7 (precondition violations) and F9 (operand mismatch); not one cell predicted that any
check would catch any of the five epilogue faults F1–F5.**

The reason is arithmetic. F1 changes a scale by at most $2^{-9}$ relative, about half an
output spacing; F2 rounds at most half a spacing per step; F4 at most one spacing; F3 and F5
were measured below bfloat16 resolution. All four therefore land at $\leq 1$ ulp, and check
6's tolerance *is* 1 ulp. Check 5 is a bitwise check but applies only under power-of-two
scales, where F1/F2/F4 are no-ops.

**The pre-registered statement that follows:** P5 is expected to show that this suite can
detect precondition violations and operand mismatch, **but has no demonstrated detection
power against epilogue differences that stay within one output spacing.** This is consistent
with the companion rather than contradicting it — the companion measured two real kernels
differing by $\leq 1$ ulp, so real kernels' epilogue differences fall exactly in the interval
this suite cannot judge a violation. Check 6's tolerance was itself set from the value the
companion observed, so it has no demonstrated power against epilogue defects; the companion
did not point this out.

**If the measurement overturns this expectation** (some epilogue fault does fire check 6),
this amendment's reasoning is what was wrong, and the measurement replaces it. **If the
measurement confirms it, the round must rewrite the suite's positioning: from "deciding
whether two implementations are interchangeable" to "deciding whether the alibi's
preconditions hold, whether operands are shared, and whether differences exceed one
spacing"** — the last being a descriptive threshold, not a defect detector. That rewrite is
committed here in advance, not left to a reviewer to demand.

**Check 7 is also disclosed as untested by P5**: it is a token-level tolerance check, and P5
is a layer-level injection supplying no margins and no flips, so all its rows are marked
`not_applicable` and its sensitivity is **outside this round's measurement**. The matrix's
`scope` field records this, so that a count of seven checks is not read as seven checks
tested.

## A-10.5 (PIN, 2026-08-15; supersedes A-10.1; no measurement started)

A-10.1's pin had two defects, both found on attempting to execute: no P5 runner existed, and
the accuracy script derived evaluation windows and ran inference in one process, which did not
match the machine's actual environment split. The first P6 run therefore failed entirely and
is recorded (`artifacts/p6_first_run_record.txt`); it produced no result files, so A-10.1 was
never used for any measurement and is superseded outright. **The commit carrying this
amendment contains plan, code, and tests only — no measurement results**, with the digests of
nine pinned artifacts and 196 passing tests listed.

## A-10.6 and A-10.7 (post-data, 2026-08-15; a smoke run overturned four predictions)

A two-layer smoke run contradicted the stated reasoning of three cells, which were corrected.
These are **post-data changes and therefore protocol deviations**, flagged
`corrected_post_data` in the matrix with the original reasoning preserved verbatim. All three
moved in the suite's favour, together flipping 1,278 of the 8,232 cells from error to
agreement, so results are reported under both the original and the corrected matrix. The
three cells and their original reasoning are quoted in the paper.

## A-11 (class A, 2026-08-14) and its closure as a negative report

Adds **P7: a second kernel pair** — a new treatment variable, not a gap being filled, since
all of the companion's conclusions come from one pair. Three hard criteria are pinned
*before* any candidate was chosen: **C1** two paths inside one engine build, switchable by
configuration (a cross-engine comparison explicitly does not count, since it binds kernel,
cache, graph execution, and scheduler together); **C2** both arms provably consuming
bit-identical quantized operands, with "they should be the same" not accepted; **C3** both
arms' kernel selection capturable from execution logs and passing the existing identity
contract. **If no candidate satisfies all three, the amendment closes as a negative report**
naming which candidates failed which criterion, and the existing scope wording is kept.
Substituting a cross-engine comparison and declaring P7 done is forbidden — the clause exists
to stop the criteria being loosened later to fit whatever could be run.

**Closure (2026-08-14, no measurement performed).** Ten engines were surveyed and **zero
candidates satisfied C1, C2, and C3 together**; P7 was not executed. Notable individual
findings: the first-priority candidate's two apparently distinct INT8 paths call the same
kernel symbol, differing only in a reshape, so they are an empty pair rather than a pair (C1
fails); a third INT8 implementation does exist in the pinned build and satisfies C1 and C3
but does not share the activation quantization path (C2 fails); the cleanest candidate on all
three criteria is gated on a different accelerator vendor and a compute capability the
measurement hardware does not have. The temptation explicitly refused: FP8 backend selectors
have exactly the mechanism C1 asks for and would satisfy C2, but C1 pins *scaled-INT8*, and a
null commitment forbids adjusting criteria after seeing results — using an FP8 pair as P7
would be criteria loosening.

## A-12 (class A, 2026-08-15; new treatment variable: power-of-two scales at quantization time)

**P8: quantize with power-of-two scales rather than rounding an existing checkpoint's scales
after the fact.** P6 measured the after-the-fact probe at $+157.4\%$ perplexity, leaving
"can power-of-two serve as a deployable determinism mechanism" open. P8 tests it.

**Timing and the provenance of the hypothesis are stated rather than hidden**: this amendment
was written *after* the P6 result and its core hypothesis is derived from that data, so the
hypothesis is **not blind**. Class A here means P8's own measurement had not begun, not that
the hypothesis was uninfluenced by existing data.

**Core hypothesis:** the $+157\%$ comes mainly from clipping rather than from the power-of-two
constraint. The probe used the *nearest* power of two, which can make a scale *smaller*,
pushing $w/s$ outside $[-127,127]$ and clipping. A *ceiling* rule never shrinks a scale and
so cannot clip. Prediction: the ceiling arm's cost will be much smaller than $+157\%$; if the
measurement overturns this, clipping is not the main cause and the constraint itself is
expensive — report that.

**Arms:** `base`; `pow2-nearest` (the existing probe, retained as a control);
`pow2-ceil`, with weights **requantized** under the constrained scale rather than only the
scale field rewritten; and `pow2-search`, executed **only if** the ceiling arm's cost is still
unacceptable, with that condition defined here in advance as a relative perplexity increase
above $5\%$. The threshold routes execution and is **not** a deployability criterion.

**Other pre-registered predictions.** Cross-kernel bitwise identity must hold, and this is an
arithmetic claim, not an empirical one: the commutation holds for *any* power of two
regardless of how it was chosen. **If it does not hold, the commutation has a gap in this
implementation that we do not understand, and that matters more than the accuracy result and
must be chased first.** Throughput is predicted unchanged, since only scale values change;
this prediction's value is as a consistency check — a significant difference should first
raise suspicion of the measurement, not of the mechanism. The ceiling arm's scales are on
average larger, so coarser resolution is its cost source, unrelated to clipping; its
perplexity cost is predicted positive but far below $157\%$, with **no specific value
predicted**.

**No deployability threshold.** Report the perplexity difference with its interval and the
throughput change, and **explicitly refuse to declare a universal "acceptable" bound** —
acceptability depends on the workload. P8 supplies the price, not a permission.

**Null commitment added.** All arms are reported whichever is better. Reporting only
`pow2-search` because its numbers look better is forbidden; `ceil` is the primary arm and
`search` a conditional supplement, with those roles fixed here.

## Later A-12 addenda (class A, each before the measurement it governs)

Scope extensions to 8B and then to 14B replication, each recorded before that size was
measured, together with in-flight notes: a disk guard that was set from a wrong assumption
about shard layout and had to be corrected, and the 14B end-to-end run's failure and its
handling, including the condition difference (a higher GPU memory utilization at 14B) that
the paper discloses because it changes batch composition, which is known to affect numerics.
