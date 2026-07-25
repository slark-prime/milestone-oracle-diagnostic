# E6: does the roadmap track the student's own reasoning? (AC-Q2 faithfulness)

No new sampling — reads the released unaided C1 traces (4 models x 50 families x 8 rollouts) and asks,
with the same deterministic grader, whether each milestone's gold result is verifiably present in the
student's own trace. Script: `scripts/experiments/milestone_trace_alignment.py`; confound controls added
after an internal faithfulness review.

## Controls (the second one is the one that matters)

- **Control A** — a *different* family's golds checked against this trace. Weak: it does not hold the
  gold values fixed, and it varies 3.0-10.3% across models.
- **Control B** — *this* family's golds checked against a *different* problem's trace. This holds the
  gold values fixed and isolates the trace-specific signal from "how easy is this value to hit in any
  math trace". Stable at 3.1-5.1%.
- **Restatement check** — is the gold already in the problem statement (so that a model merely
  restating the problem would score a hit)? Only 5.5% of golds are.

| Model | Own golds / own trace | Ctrl A | **Ctrl B** | Ratio vs B | Statement-novel golds only |
|---|---:|---:|---:|---:|---:|
| MilestoneRL-2K | 17.2% | 8.5% | **4.1%** | **4.2x** | 12.5% |
| Qwen-base | 13.6% | 3.0% | **5.1%** | 2.7x | 9.4% |
| DeepSeek-V3.1 | 15.2% | 10.3% | **3.1%** | **5.0x** | 11.3% |
| Llama-70B | 14.5% | 4.1% | **4.5%** | 3.2x | 9.9% |

The effect is trace-specific (2.7-5.0x over Control B), survives removal of statement-present golds
(9.4-12.5% vs 3.1-5.1%), and is prefix-shaped (positions 0-1 ~0.15, falling to 0.00-0.06 at 2-3).

## Stage-0-conditional attainment (the strongest signal)

An *independent* measurement predicts what shows up in spontaneous reasoning:

| Model | Attained if Stage-0 solvable | if not |
|---|---:|---:|
| MilestoneRL-2K | 18.6% | 1.4% |
| DeepSeek-V3.1 | 17.7% | 2.8% |
| Qwen-base | 14.6% | 0.0% |
| Llama-70B | 14.3% | 17.2% (no effect) |

## Limits stated in the rebuttal

Conservative containment test (rates are lower bounds); positions 2+ have small denominators (roadmaps
average 2.2 non-INTEGRATE milestones); Llama-70B shows no Stage-0-conditional effect and is reported,
not excluded. A residual confound we cannot fully remove: Stage-0-solvable milestones tend to be the
easier ones, and easier results are intrinsically more likely to appear, so this supports "the roadmap's
difficulty structure tracks the student's" more strongly than it supports any claim about internal
computation. The claim we make in the rebuttal is the former.

---

## The decisive test (added after the faithfulness review)

**Question.** Does the student's unaided trace track *our* roadmap specifically, or would any valid
decomposition of the same problem score the same?

**Unblocking it.** The comparison needs an independent decomposition whose golds are gradable by the
same containment test. The second teacher's golds were compound (median 55 chars vs GPT-5.4's 7), so a
first attempt gave a meaningless 0.0-0.3% for the alternative. Adding a few-shot answer-atomicity block
to the teacher prompt (examples of good/bad golds, plus "if a milestone produces several results, split
it") raised the atomic-gold rate from **31% to 86%** and the median length from **55 to 21** chars, with
12/12 packets valid. `scripts/experiments/atomic_gold_fewshot.py`. Note this is an *additive* suffix;
`decomposer/common/prompts.py` is unchanged, so the submission's protocol is untouched.

**Result** (same traces, same 12 problems; ours = GPT-5.4 submission roadmap, alt = independent
decomposition of the same problem):

| Model | Raw ratio | Length-matched <=25 | Length-matched <=15 |
|---|---:|---:|---:|
| MilestoneRL-2K | 1.68x | 1.13x | 0.79x |
| Qwen-base | 1.15x | 0.75x | 0.59x |
| DeepSeek-V3.1 | 1.46x | 0.96x | 1.04x |
| Llama-70B | 0.98x | 0.68x | 0.34x |

The raw ratios favour our roadmap only because our golds are shorter (median 15 vs 21) and short strings
are easier to hit. **At matched gold length the advantage disappears and often reverses.**

**Interpretation.** The tautology objection is confirmed: the 2.7-5.0x effect over the cross-problem
control reflects "these are quantities of *this problem*", not "this roadmap mirrors *this model's*
trajectory". What the milestones capture is problem structure. That is the correct property for this
instrument — a roadmap fitted to one model's path could not support cross-model fingerprint comparison —
and it is corroborated by two independent teachers aligning with the traces equally well. The claim we
make is therefore: milestones are steps models do produce on these problems, the structure is
teacher-independent, and its difficulty profile tracks the student's Stage-0 profile. We do not claim
recovery of internal computation.

## Why the negative result is the predicted one (construction evidence)

The teacher's inputs are ORIGINAL_PROBLEM and PRIVATE_SOLUTION_REFERENCE only; it never sees any
evaluated model's output. `scripts/run_typed_pipeline.py:95` takes NuminaMath's `solution` field and
**drops any problem without one** (`or not sol` in the filter), so every family in the 354-set was
compiled from a problem plus its human reference solution. Model-independence is therefore
architectural, and the length-matched result (no advantage over an independent decomposition) is what
the construction predicts rather than a surprise to be explained away.

Bookkeeping caveat for the appendix: the INTEGRATE-recovery re-decomposition and the second-corpus suite
pass `reference_solution=""` (the diagnostic files do not retain the reference solution), so those
packets are problem+gold-answer driven. This does not affect model-independence — neither sees model
output — but the appendix should state which packets had a reference solution available.
