# Composition-Gap Neutral Rater Audit

Rubric: `docs/cross_model_audit_prompt.md`, with the binding rules applied conservatively.
Canonical-answer artifacts dominate, then non-standalone milestones, then incomplete decompositions versus genuine composition.

| model | n | genuine | decomp incomplete | verifier artifact | not standalone | beyond capability | genuine fraction | low conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-base | 173 | 0 | 130 | 43 | 0 | 0 | 0.0000 | 0 |
| OutcomeRL-2K | 155 | 0 | 116 | 39 | 0 | 0 | 0.0000 | 0 |
| MilestoneRL-2K | 167 | 0 | 129 | 37 | 1 | 0 | 0.0000 | 0 |
| gpt-oss-20b | 120 | 2 | 85 | 33 | 0 | 0 | 0.0167 | 0 |
| Llama-70B | 164 | 1 | 124 | 38 | 1 | 0 | 0.0061 | 0 |
| DeepSeek-V3.1 | 127 | 4 | 85 | 37 | 1 | 0 | 0.0315 | 0 |

## Main Risks And Boundary Cases

- The audit used one label per repeated `pid` across models for structural consistency; model-specific capability judgments were not needed for the observed cases.
- Many families have clean-looking but incomplete decompositions: the milestones establish reductions, bounds, or intermediate quantities but omit a nontrivial final calculation.
- Verifier artifacts include garbled LaTeX, concatenated multi-part answers, bare symbols, prompt translation contamination, and figure-dependent prompts without self-contained text.
- A few entries were marked genuine when milestones already supplied the canonical final value or only a direct union/reporting step remained.
- Borderline canonical mismatches were marked medium confidence when the answer was parseable but appeared to answer only part of a multi-part parent.

