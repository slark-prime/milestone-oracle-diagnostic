"""Per-milestone-type breakdown of recoverability.

Question: which milestone types dominate DESC-ONLY and GOLD-ONLY recovered
families? Does C2/C3 lift depend on the type composition of the milestones?

Two analyses:

A) Family-level: for each state (DIRECT / DESC-ONLY / GOLD-ONLY / UNRECOVERED),
   what's the type composition of milestones across families in that state?

B) Model-level: for each of 6 panel models, report the fraction of families
   with >= 1 type-X milestone that were recovered via C2-correct (among those
   unsolved at C1).
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PANEL_DIR = REPO / "data/logs/rl/oracle_panel"
FAM_FILE = REPO / "data/logs/rl/diagnostic_multi_families_repaired.jsonl"
OUT_JSON = REPO / "data/logs/rl/milestone_type_breakdown.json"
OUT_TEX = REPO / "docs/latex/tables/milestone_type_breakdown.tex"

MODELS = [
    ("qwen3_8b_pre_rl", "pre-RL"),
    ("base_2k_step_180", "Base-2k"),
    ("mile_2k_step_180", "Mile-2k"),
    ("gpt_oss_20b", "gpt-oss-20b"),
    ("llama_3_3_70b_instruct", "Llama-70B"),
    ("deepseek_v3_1", "DeepSeek-V3.1"),
]

TYPES = ["KEY_MOVE", "MODEL", "COMPUTE", "NORMALIZE"]
# LEMMA/SANITY/WARMUP collapse to "OTHER" (too few to split)


def load_state(slug):
    fn = PANEL_DIR / f"{slug}.jsonl"
    by_pid = defaultdict(dict)
    for line in open(fn):
        r = json.loads(line)
        by_pid[r["pid"]][r["condition"]] = r["n_correct"]
    states = {}
    for pid, conds in by_pid.items():
        c1 = conds.get("C1_direct", 0)
        c2 = conds.get("C2_descriptions", 0)
        c3 = conds.get("C3_gold_answers", 0)
        if c1 >= 1: s = "direct"
        elif c2 >= 1: s = "desc_only"
        elif c3 >= 1: s = "gold_only"
        else: s = "unrecovered"
        states[pid] = s
    return states


def load_families():
    families = {}
    for line in open(FAM_FILE):
        r = json.loads(line)
        types = [m.get("type", "OTHER") for m in r["milestones"]]
        # Collapse minor types
        types = [t if t in TYPES else "OTHER" for t in types]
        families[r["pid"]] = {"types": types, "n_ms": len(types)}
    return families


def has_type(family_types, t):
    return t in family_types


def main():
    families = load_families()
    model_states = {slug: load_state(slug) for slug, _ in MODELS}

    # --- Analysis A: type-composition by Mile-2k state (most interpretable)
    print("=" * 85)
    print("A) Mile-2k family states vs. milestone-type composition")
    print("=" * 85)

    mile_states = model_states["mile_2k_step_180"]
    # For each state, count how many families contain at least one of each type
    state_counts = defaultdict(lambda: defaultdict(int))
    state_totals = Counter()
    for pid, state in mile_states.items():
        state_totals[state] += 1
        for t in set(families[pid]["types"]):
            state_counts[state][t] += 1

    print(f"{'State':<14s} {'n':>5s}  " + "  ".join(f"{t:>9s}" for t in TYPES + ["OTHER"]))
    print("-" * 85)
    for state in ["direct", "desc_only", "gold_only", "unrecovered"]:
        row = [f"{state:<14s}", f"{state_totals[state]:>5d}"]
        for t in TYPES + ["OTHER"]:
            frac = state_counts[state][t] / state_totals[state] if state_totals[state] else 0
            row.append(f"{100*frac:>7.0f}%")
        print("  ".join(row))
    print("\nCell = percent of families in this state that contain >= 1 milestone of this type.")

    # --- Analysis B: does containing a type-X milestone help C2 recovery?
    # Restrict to C1-unsolved families: what % get recovered by C2 as a function
    # of type composition?
    print("\n" + "=" * 85)
    print("B) Among C1-unsolved families: C2-correct recovery rate by milestone type present")
    print("=" * 85)

    bb_rows = []
    for slug, label in MODELS:
        states = model_states[slug]
        unsolved_c1 = [pid for pid, s in states.items() if s != "direct"]
        row = {"model": label, "n_unsolved": len(unsolved_c1)}
        for t in TYPES + ["OTHER"]:
            with_t = [pid for pid in unsolved_c1 if has_type(families[pid]["types"], t)]
            recovered_c2 = [pid for pid in with_t if states[pid] == "desc_only"]
            row[t] = {
                "n_with_type": len(with_t),
                "n_desc_only": len(recovered_c2),
                "rate": len(recovered_c2) / len(with_t) if with_t else 0,
            }
        bb_rows.append(row)

    print(f"\n{'model':<18s}  {'n@C1=0':>7s}  " + "  ".join(f"{t:>13s}" for t in TYPES + ["OTHER"]))
    print("-" * 105)
    for r in bb_rows:
        print(f"{r['model']:<18s}  {r['n_unsolved']:>7d}  " +
              "  ".join(f"{100*r[t]['rate']:>6.0f}% ({r[t]['n_desc_only']:>3d}/{r[t]['n_with_type']:>3d})" for t in TYPES + ["OTHER"]))

    # Save + write LaTeX
    OUT_JSON.write_text(json.dumps({"A": {"state_counts": {s: dict(c) for s, c in state_counts.items()}, "state_totals": dict(state_totals)}, "B": bb_rows}, indent=2, default=int))

    # LaTeX table: analysis A
    with open(OUT_TEX, "w") as f:
        f.write(r"""\begin{table}[h]
\centering
\footnotesize
\caption{Milestone-type composition of families in each Mile-2k state. Each cell is the percentage of families in that state whose milestone set contains at least one milestone of the given type (rows are not mutually exclusive; a family can contribute to multiple columns).}
\label{tab:milestone_types_by_state}
\vspace{4pt}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Mile-2k state} & \textbf{$n$} & \textbf{\textsc{Key\_Move}} & \textbf{\textsc{Model}} & \textbf{\textsc{Compute}} & \textbf{\textsc{Normalize}} & \textbf{Other} \\
\midrule
""")
        for state in ["direct", "desc_only", "gold_only", "unrecovered"]:
            state_label = {"direct": "\\textsc{direct}", "desc_only": "\\textsc{desc-only}",
                          "gold_only": "\\textsc{gold-only}", "unrecovered": "\\textsc{unrecovered}"}[state]
            parts = [state_label, str(state_totals[state])]
            for t in TYPES + ["OTHER"]:
                frac = state_counts[state][t] / state_totals[state] if state_totals[state] else 0
                parts.append(f"{100*frac:.0f}\\%")
            f.write(" & ".join(parts) + " \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"\nWrote {OUT_TEX}")


if __name__ == "__main__":
    main()
