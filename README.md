# Milestone-Oracle Diagnostic

A judge-free diagnostic for math-reasoning failure-mode analysis, accompanying the NeurIPS 2026 Evaluations & Datasets Track submission *Diagnosing Math-Reasoning Failure Structure with Milestone Oracles*.

For each parent math problem, a teacher compiles a frozen **milestone family**: a set of typed, symbolically checkable intermediate sub-goals, each with a self-contained description and a gold answer. A student model is then probed under three help levels — no help (`C1`), milestone roadmap (`C2`), and roadmap plus gold milestone answers (`C3`) — together with a Stage 0 milestone-test pass/fail check. Crossing the smallest help level that solves the parent (the *parent-probe outcome*) with the milestone test yields the reasoning-gap taxonomy: `roadmap`, `milestone-execution`, `composition`, `missing-milestone`, or `capability` gap.

All grading is deterministic symbolic verification — no LLM judge.

## What is in this release

```
data/
  diagnostic_354_families.jsonl       Main diagnostic set (354 milestone families).
  diagnostic_731_compiled.jsonl       Pre-restriction 731-family compiled set (provenance).
  diagnostic_held_out_32.jsonl        32-family held-out slice (disjoint compilation).
  oracle_panel_16k/                   Per-(model, condition, family) probe outcomes
    qwen3_8b_pre_rl.jsonl             at max_tokens=16384, K=8.
    base_2k_step_180.jsonl            (OutcomeRL-2K)
    mile_2k_step_180.jsonl            (MilestoneRL-2K)
    gpt_oss_20b.jsonl
    llama_3_3_70b_instruct.jsonl
    deepseek_v3_1.jsonl
  audit_consensus.jsonl               100-family unrecovered audit (consensus rubric).
  audit_rater_a.jsonl                 Per-rater audit labels.
  audit_rater_b.jsonl
  oracle_leak_safe_repair.jsonl       Oracle outcomes after leak-safe decomposition repair.
  repaired_decomp_leak_safe.jsonl     Regenerated leak-safe milestone families.
  repaired_verifier.jsonl             Verifier-repaired families (canonical-answer fixes).
prompts/
  oracle/                             Verbatim student-facing oracle prompts.
    c1_direct.txt                     C1: parent only.
    c2_descriptions.txt               C2: parent + milestone roadmap.
    c3_gold.txt                       C3: parent + roadmap + gold milestone answers.
    c2_random.txt                     Corruption control: roadmap from a different family.
    c2_generic.txt                    Corruption control: generic decomposition prompt.
    c3_mismatched.txt                 Corruption control: correct roadmap, wrong answers.
    student_system.txt                System-side boxed-answer instruction.
  teacher_prompt.py                   Teacher-side family-compilation prompt.
code/
  decomposer/                         Family compiler, Stage 0 runner, oracle pipeline.
    common/                           Schemas, data types, LLM client wrappers.
    pipeline/                         Core diagnostic loop and training-set construction.
    teacher/                          Teacher-side family compilation.
    verifier/                         Strict symbolic verifier cascade (boxed extract,
                                       schema parse, sympy equivalence, math-verify).
  scripts/                            Figure / table / bootstrap / audit scripts. These
                                       are the ones we ran to generate the paper figures
                                       and tables; many use repository-relative paths
                                       (data/logs/rl/..., docs/latex/...) and are
                                       reproducibility specifications rather than
                                       drop-in commands. See "Reproducing the paper
                                       numbers" below for the verified entry points.
requirements.txt                      pip dependency list (Python >=3.10).
environment.yml                       Conda environment spec (Python 3.11).
croissant.json                        Croissant metadata (Croissant 1.0).
LICENSE                               CC BY 4.0 for the dataset; MIT for the code.
README.md                             This file.
```

## Schema (one record from `diagnostic_354_families.jsonl`)

```json
{
  "pid": "cabf0726-b738-4eca-ae15-e4b02f2d7506",
  "parent_prompt": "Solve the following problem...",
  "parent_answer": "\\frac{49}{50}",
  "parent_note": "ACCEPT IF equivalent final answer is present.",
  "milestones": [
    {
      "prompt": "You are solving one milestone from a larger problem...",
      "answer": "(6,8)",
      "note": "ACCEPT IF equivalent final answer is present.",
      "type": "MODEL",
      "pre_train_phat": 0.75
    },
    ...
  ]
}
```

`pre_train_phat` is the per-milestone solve rate of pre-RL Qwen3-8B-Base under `K=8` isolation rollouts.

## Schema (one record from `oracle_panel_16k/<model>.jsonl`)

```json
{
  "model": "qwen3_8b_pre_rl",
  "condition": "C1_direct",
  "pid": "cabf0726-...",
  "n_correct": 0,
  "k": 8
}
```

Each model file has 354 rows × 6 conditions = 2124 records. Conditions:
- `C1_direct` — parent only.
- `C2_descriptions` — parent + milestone roadmap.
- `C3_gold_answers` — parent + roadmap + gold milestone answers.
- `C2_random` — corruption control: roadmap from a different family.
- `C2_generic` — corruption control: generic decomposition prompt.
- `C3_mismatched` — corruption control: correct roadmap, wrong milestone answers.

## Quickstart: parent-probe outcome from the panel files

```python
import json
from collections import defaultdict

panel = defaultdict(dict)
with open("data/oracle_panel_16k/mile_2k_step_180.jsonl") as f:
    for line in f:
        r = json.loads(line)
        panel[r["pid"]][r["condition"]] = r["n_correct"]

def parent_probe_outcome(p, threshold=1):
    if p.get("C1_direct", 0)        >= threshold: return "Direct"
    if p.get("C2_descriptions", 0)  >= threshold: return "Roadmap-Needed"
    if p.get("C3_gold_answers", 0)  >= threshold: return "Answers-Needed"
    return "Unrecovered"

from collections import Counter
print(Counter(parent_probe_outcome(p) for p in panel.values()))
# Counter({'Unrecovered': 189, 'Direct': 96, 'Roadmap-Needed': 47, 'Answers-Needed': 22})
```

## Quickstart: reproducing a panel (run the verifier yourself)

```python
import sys
sys.path.insert(0, "code")
from decomposer.verifier.verifier import verify_answer

candidate = r"The answer is \boxed{\frac{49}{50}}."
gold      = r"\frac{49}{50}"

result = verify_answer(candidate, gold)
print(result["label"])   # ACCEPT
print(result["reason"])  # boxed match (math_reward)
```

The verifier is a strict symbolic cascade — boxed extraction, schema-aware parse, sympy equivalence, and `math-verify`. There is no LLM-judge fallback; the cascade abstains by emitting `NOT_ACCEPT` when none of the symbolic stages can resolve equivalence.

To run the full diagnostic on your own model, see `code/decomposer/pipeline/core_loop.py` — it expects a callable that accepts a prompt and returns up to `K` rollouts.

## Installation

```bash
# Option A: pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate milestone-oracle
```

Python 3.10+ is required. Both quickstart snippets above were verified under Python 3.11 with `mlcroissant` 1.1.0 and `math-verify` 0.5+.

## Reproducing the paper numbers

We have verified the following entry points end-to-end on this release:

1. **Parent-probe outcomes per model.** Run the first quickstart snippet against any of the six panel files in `data/oracle_panel_16k/`. For `mile_2k_step_180.jsonl`, the output is `{Direct: 96, Roadmap-Needed: 47, Answers-Needed: 22, Unrecovered: 189}` (matches paper Table 12).
2. **Strict symbolic verifier.** Run the second quickstart snippet. Returns `ACCEPT` with reason `boxed match (math_reward)`.
3. **Croissant validation.**
   ```python
   import mlcroissant
   ds = mlcroissant.Dataset(jsonld="croissant.json")
   assert len(list(ds.records("milestone_family"))) == 354
   assert len(list(ds.records("oracle_panel_record"))) == 12_744
   ```

The figure / table reproduction scripts in `code/scripts/` are the exact ones we used to generate the paper figures. They use repository-relative paths (`data/logs/rl/...`) that mirror our development repository layout. To make them drop-in runnable in this release directory, run the one-time setup once:

```bash
bash setup_release_paths.sh
```

This creates a `data/logs/rl/` directory of symlinks pointing at the released files (and a `code/data` -> `../data` symlink so the `parent.parent`-relative `REPO` constant resolves correctly).

**Scripts that reproduce the core 16K analyses from the released outputs** (run after `setup_release_paths.sh`):

| Script | Paper artifact |
|---|---|
| `build_bottleneck_lattice.py` | Figure 4, Table 12 (reasoning-gap taxonomy) |
| `build_two_arm_redistribution.py` | Figure 5 (longitudinal) |
| `build_audit_triage_figure.py` | Figure 6 (audit triage) |
| `audit_bounds.py` | Section 4.5 lower/upper-bound bookkeeping |
| `build_4k_vs_16k_comparison.py` | Appendix L (decoding-budget audit) |
| `build_trajectory_tomography.py` | Figure 8 (training trajectory) |
| `build_accuracy_vs_recoverability_scatter.py` | Figure 3(b) inputs |
| `bootstrap_fingerprint_cis.py` | Appendix F (fingerprint CIs) |
| `analyze_format_audit.py` | Appendix L (format audit) |

These nine scripts cover the headline figures and tables in the main text and the audit/budget appendices. The exact panel numbers behind every figure are also recoverable directly from the JSONL files using the parent-probe snippet at the top of this section.

**Scripts that are reproducibility specifications, not drop-in commands** — they depend on intermediate artifacts not packaged here (`manifest.jsonl`, `panel_fingerprints_16k.json`, `oracle_canonical.jsonl`, the 4K `oracle_panel/`, the 150-problem held-out source pool): `build_oracle_specificity.py`, `build_panel_figures_16k.py`, `build_specificity_decoupling.py`, `compute_bootstrap_cis.py`, `build_held_out_families.py`, `analyze_milestone_types.py`. The corresponding figures and tables in the paper are precomputed; the scripts document how they were built so reviewers can audit the procedure even though they cannot regenerate the outputs from this release alone.

## Preparing a clean copy for submission

If you are uploading this directory as supplementary material (e.g. on OpenReview), run:

```bash
bash prepare_submission.sh
```

This produces `../milestone-oracle-diagnostic-submission.tar.gz` and a matching `.zip`, with `.git/`, `.DS_Store`, `__pycache__/`, and `.pyc` files excluded. Either archive is safe for double-blind upload (the in-tree `.git/config` would otherwise expose the GitHub remote).

## License

- **Dataset** (everything under `data/` and `prompts/`): CC BY 4.0. You may share and adapt with attribution.
- **Code** (everything under `code/`): MIT.
- **Source problems**: derived from NuminaMath-1.5-RL-Verifiable. Please respect the upstream license terms.

## Citation

```
@inproceedings{anonymous2026milestoneoracles,
  title  = {Diagnosing Math-Reasoning Failure Structure with Milestone Oracles},
  author = {Anonymous},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track (under review)},
  year = {2026}
}
```

## Limitations

- The diagnostic set is anchored on Qwen3-8B-Base screening (a fixed slice rather than a model-neutral sample of all math problems).
- Milestone families are compiled by a single teacher (GPT-5.4 with thinking mode); a small Claude-Sonnet pilot reproduces the qualitative pattern but is underpowered.
- Symbolic verification is the ground truth, so problems requiring proof writing, drawing, or other non-symbolic outputs are out of scope.
- Source problems come from public benchmarks (NuminaMath, MATH); standard public-benchmark contamination caveats apply.
