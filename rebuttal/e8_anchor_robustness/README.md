# E8: Anchor robustness — third-anchor screen yields

Question: how much does the anchor-failure screen depend on the anchor model?
The same rule (fail all four direct attempts, k=4, 16K tokens, same prompts and
verifier) is applied to MATH500 (500 problems) and AIME 2024/2025 (60 problems)
with different anchor models.

## Results

| Anchor | MATH500 anchor-failed | AIME anchor-failed |
|---|---:|---:|
| Qwen3-8B (suite anchor; e7) | 36/500 (7%) | 38/60 (63%) |
| gpt-oss-20b (this experiment) | 5/500 (1%) | 10/60 (17%) |

Candidate anchors ruled out by the 30-problem pilot (`data/anchor_pilot/`):
- Llama-3.2-3B: follows the \boxed{} output format in only 28/60 rollouts and
  solves 0/60 — a screen built on it measures format compliance, not ability.
- Nemotron-3-Nano-30B-A3B: solves 54/56 rollouts on MATH500 — near-zero yield.

## Files
- `data/anchor_pilot/*.jsonl` — 30 problems x k=2 per candidate model
  (boxed-format flag, verifier accept, response tail).
- `data/anchor_screens/gpt-oss-20b/{math500,aime}/c1_screen.jsonl` — full
  screens. The math500 file contains duplicate keys from an interrupted first
  launch; analyses keep the first occurrence of each (pid, rollout) key, i.e.
  exactly 4 attempts per problem (see `scripts/anchor_intersection.py`).
- `data/anchor_screens/gpt-oss-20b/intersection_report.json` — recovery of the
  e7-probed families that the second anchor also fails (n=4 MATH500 / n=9 AIME).
  We do not read a contrast from these subsets: they are too small, and the
  doubly-failed families are largely beyond the 8B student under every
  condition, consistent with CAPABILITY-GAP dominance on harder problems.
- `scripts/` — pilot, screen, and intersection analysis, runnable against e7.
