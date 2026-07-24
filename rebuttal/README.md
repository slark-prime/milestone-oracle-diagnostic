# Response-period verification data (NeurIPS 2026 E&D #2216)

New experiments run during the author-response period. Every number quoted in the responses
is reproducible from the files below. Protocols are byte-identical to the paper's pipelines
unless stated in the response text.

| Folder | Experiment | Used in |
|---|---|---|
| `e1_composition_probe/` | C3 stress probe of the 18 author-audited families, raw outputs, K=8, two models (`qwen3_8b_c3.jsonl`, `gpt_oss_20b_c3.jsonl`), plus Stage-0 re-check on the failing families and hand-graded summaries | global response (AC-Q3), i227 Q1 |
| `e2_teacher_consistency/` | Second-teacher (Inkling) packets for 60 sampled families (`inkling_packets.jsonl`), C2/C3 re-probe on the same student (`inkling_c2c3_gpt_oss_20b.jsonl`), structural comparison | global (AC-Q2), i227 Q5, FWVb W1, Qf6B W3 |
| `e3_multi_anchor/` | Multi-anchor re-filtering with the 16K Stage-0 source: full count/rank/tau tables | global (AC-Q1), i227 Q3, Qf6B W2 |
| `e4_frontier_judge/` | gpt-5.5 paired verifier audit, n=997 (`paired_gpt_5_5_v2.jsonl`, summary) | FWVb W2 |
| `audit_full_panel/` | Rubric-based residual labels for all six panel models (906 family-labels) with rubric summary; see i227 Q4 response for the disclosed limitation of its decomposition-vs-composition axis | i227 Q1/Q4 |
| `scripts/` | The exact scripts that produced the above | all |

Grading conventions: recovery threshold >=1/8 unless stated; strict symbolic cascade as in the
paper; E1 additionally hand-graded against independently verified answers (summaries included).
