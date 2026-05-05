"""Bootstrap uncertainty for cross-model fingerprints.

Family-level bootstrap (1000 resamples of 354 families with replacement).
For each resample, recompute fingerprint counts per model; report:

(a) 95% CI for any-probe count per model (as absolute count out of 354)
(b) Pairwise ordering frequency: for each (model_a, model_b), fraction of
    resamples where a's any-probe count > b's.

Output:
  - data/logs/rl/bootstrap_fingerprint_cis.json
  - docs/latex/tables/bootstrap_cis.tex
  - docs/latex/tables/bootstrap_ordering.tex
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PANEL_DIR = REPO / "data/logs/rl/oracle_panel_16k"
OUT_JSON = REPO / "data/logs/rl/bootstrap_fingerprint_cis.json"
OUT_TEX_CI = REPO / "docs/latex/tables/bootstrap_cis.tex"
OUT_TEX_ORD = REPO / "docs/latex/tables/bootstrap_ordering.tex"

MODELS = [
    ("qwen3_8b_pre_rl", "pre-RL"),
    ("base_2k_step_180", "Base-2k"),
    ("mile_2k_step_180", "Mile-2k"),
    ("gpt_oss_20b", "gpt-oss-20b"),
    ("llama_3_3_70b_instruct", "Llama-70B"),
    ("deepseek_v3_1", "DeepSeek-V3.1"),
]

N_BOOTSTRAP = 1000


def state_from_counts(c1, c2, c3):
    if c1 >= 1: return "direct"
    if c2 >= 1: return "desc_only"
    if c3 >= 1: return "gold_only"
    return "unrecovered"


def load_model(slug):
    fn = PANEL_DIR / f"{slug}.jsonl"
    by_pid = defaultdict(dict)
    for line in open(fn):
        r = json.loads(line)
        by_pid[r["pid"]][r["condition"]] = r["n_correct"]
    # For each pid, compute state
    states = {}
    for pid, conds in by_pid.items():
        states[pid] = state_from_counts(
            conds.get("C1_direct", 0),
            conds.get("C2_descriptions", 0),
            conds.get("C3_gold_answers", 0),
        )
    return states


def main():
    # Load per-model state for each pid
    model_states = {slug: load_model(slug) for slug, _ in MODELS}
    all_pids = sorted(set(next(iter(model_states.values())).keys()))
    n = len(all_pids)
    print(f"n = {n} families")

    # Bootstrap
    random.seed(42)
    np.random.seed(42)

    ceiling_samples = {slug: [] for slug, _ in MODELS}
    direct_samples = {slug: [] for slug, _ in MODELS}
    pairwise_wins = defaultdict(lambda: defaultdict(int))

    for b in range(N_BOOTSTRAP):
        sample_pids = [all_pids[i] for i in np.random.randint(0, n, size=n)]
        # For each model, count states in this resample
        model_counts = {}
        for slug, _ in MODELS:
            states = model_states[slug]
            direct = sum(1 for p in sample_pids if states[p] == "direct")
            desc = sum(1 for p in sample_pids if states[p] == "desc_only")
            gold = sum(1 for p in sample_pids if states[p] == "gold_only")
            ceiling = direct + desc + gold
            model_counts[slug] = {"direct": direct, "ceiling": ceiling}
            ceiling_samples[slug].append(ceiling)
            direct_samples[slug].append(direct)
        # Pairwise ordering
        for i, (slug_a, _) in enumerate(MODELS):
            for slug_b, _ in MODELS[i+1:]:
                if model_counts[slug_a]["ceiling"] > model_counts[slug_b]["ceiling"]:
                    pairwise_wins[slug_a][slug_b] += 1
                elif model_counts[slug_b]["ceiling"] > model_counts[slug_a]["ceiling"]:
                    pairwise_wins[slug_b][slug_a] += 1

    # CIs
    results = {}
    print(f"\n{'model':<20s}  {'ceiling CI':>20s}  {'direct CI':>18s}")
    print("-" * 70)
    for slug, label in MODELS:
        ci_lo, ci_hi = np.percentile(ceiling_samples[slug], [2.5, 97.5])
        d_lo, d_hi = np.percentile(direct_samples[slug], [2.5, 97.5])
        ci_mean = np.mean(ceiling_samples[slug])
        d_mean = np.mean(direct_samples[slug])
        results[slug] = {
            "label": label,
            "ceiling_mean": float(ci_mean),
            "ceiling_ci_lo": float(ci_lo),
            "ceiling_ci_hi": float(ci_hi),
            "direct_mean": float(d_mean),
            "direct_ci_lo": float(d_lo),
            "direct_ci_hi": float(d_hi),
        }
        print(f"{label:<20s}  {ci_mean:>6.1f} [{ci_lo:>5.1f}, {ci_hi:>5.1f}]  {d_mean:>5.1f} [{d_lo:>4.1f}, {d_hi:>4.1f}]")

    # Pairwise ordering frequency
    print(f"\nPairwise ordering frequency (fraction of bootstrap resamples where row > col on ceiling):")
    print(f"  {'':<14}" + "".join(f"{label:>12s}" for _, label in MODELS))
    for slug_a, label_a in MODELS:
        row = [label_a]
        for slug_b, _ in MODELS:
            if slug_a == slug_b:
                row.append("-")
            else:
                w = pairwise_wins[slug_a].get(slug_b, 0) / N_BOOTSTRAP
                row.append(f"{w:.3f}")
        print(f"  {row[0]:<14}" + "".join(f"{v:>12}" for v in row[1:]))

    results["pairwise_wins"] = {a: dict(b) for a, b in pairwise_wins.items()}

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {OUT_JSON}")

    # LaTeX: per-model CI table
    with open(OUT_TEX_CI, "w") as f:
        f.write(r"""\begin{table}[h]
\centering
\footnotesize
\caption{Bootstrap 95\% CIs on fingerprint counts (1000 resamples of 354 families with replacement). \textsc{direct} and any-probe counts with central tendency and $[2.5, 97.5]$ percentile interval.}
\label{tab:bootstrap_cis}
\vspace{4pt}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{\textsc{direct}} & \textbf{rec. ceiling} & \textbf{Ceiling rank} \\
 & (mean [95\% CI]) & (mean [95\% CI]) & (95\% CI range) \\
\midrule
""")
        # Determine rank range per model
        ceilings_all = {slug: sorted(ceiling_samples[slug]) for slug, _ in MODELS}
        for slug, label in MODELS:
            r = results[slug]
            # For each resample, compute this model's rank
            ranks = []
            for b_idx in range(N_BOOTSTRAP):
                c = ceiling_samples[slug][b_idx]
                others = [ceiling_samples[s][b_idx] for s, _ in MODELS if s != slug]
                # Rank = 1 + number of models with higher ceiling
                rank = 1 + sum(1 for o in others if o > c)
                ranks.append(rank)
            r_lo, r_hi = np.percentile(ranks, [2.5, 97.5])
            r_mode = max(set(ranks), key=ranks.count)
            results[slug]["ceiling_rank_mean"] = float(np.mean(ranks))
            results[slug]["ceiling_rank_ci_lo"] = int(r_lo)
            results[slug]["ceiling_rank_ci_hi"] = int(r_hi)
            f.write(f"{label} & {r['direct_mean']:.1f} [{r['direct_ci_lo']:.0f}, {r['direct_ci_hi']:.0f}] & {r['ceiling_mean']:.1f} [{r['ceiling_ci_lo']:.0f}, {r['ceiling_ci_hi']:.0f}] & {r_mode} [{int(r_lo)}, {int(r_hi)}] \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"Wrote {OUT_TEX_CI}")

    # LaTeX: pairwise ordering table
    with open(OUT_TEX_ORD, "w") as f:
        f.write(r"""\begin{table}[h]
\centering
\scriptsize
\caption{Pairwise ordering frequency on any-probe count: fraction of 1000 bootstrap resamples (family-level) where the row model's ceiling exceeds the column model's. Values near 0.5 indicate unstable order; values near 1.0 indicate stable dominance.}
\label{tab:bootstrap_ordering}
\vspace{4pt}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" + "c" * len(MODELS) + r"""}
\toprule
& """ + " & ".join(label for _, label in MODELS) + r""" \\
\midrule
""")
        for slug_a, label_a in MODELS:
            cells = [label_a]
            for slug_b, _ in MODELS:
                if slug_a == slug_b:
                    cells.append("---")
                else:
                    w = pairwise_wins[slug_a].get(slug_b, 0) / N_BOOTSTRAP
                    cells.append(f"{w:.2f}")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"Wrote {OUT_TEX_ORD}")

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
