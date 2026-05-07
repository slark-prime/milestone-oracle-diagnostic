"""Figure 4: Reasoning-gap taxonomy (paper §5.3).

Two-panel figure designed for skimming reviewer:
  Left  (~35% width): definition matrix — milestone test (PASS/FAIL) × parent-probe outcome.
  Right (~65% width): per-model stacked bars showing the 6-category distribution.

Headline visual: Composition gap (orange) is the dominant non-Direct category
across every panel model.

Outputs:
  data/logs/rl/bottleneck_lattice.json   (per-model counts)
  docs/latex/figures/bottleneck_lattice.pdf
  docs/latex/tables/bottleneck_lattice.tex
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import figure_style as S

REPO = Path(__file__).resolve().parent.parent
PANEL_DIR = REPO / "data/logs/rl/oracle_panel_16k"
STAGE0_DIR = REPO / "data/logs/rl/stage0_panel_16k"
OUT_JSON = REPO / "data/logs/rl/bottleneck_lattice.json"
OUT_FIG = REPO / "docs/latex/figures/bottleneck_lattice.pdf"
OUT_TEX = REPO / "docs/latex/tables/bottleneck_lattice.tex"

MODELS = [
    ("qwen3_8b_pre_rl",        S.MODEL_DISPLAY["qwen3_8b_pre_rl"]),
    ("base_2k_step_180",       S.MODEL_DISPLAY["base_2k_step_180"]),
    ("mile_2k_step_180",       S.MODEL_DISPLAY["mile_2k_step_180"]),
    ("gpt_oss_20b",            S.MODEL_DISPLAY["gpt_oss_20b"]),
    ("llama_3_3_70b_instruct", S.MODEL_DISPLAY["llama_3_3_70b_instruct"]),
    ("deepseek_v3_1",          S.MODEL_DISPLAY["deepseek_v3_1"]),
]

# Display order (bottom-to-top in stacked bar): Direct first, then gaps grouped
CATEGORIES = [
    ("DIRECT",                "Direct",                   S.COLOR_DIRECT),
    ("ROADMAP_GAP",           "Roadmap gap",              S.COLOR_ROADMAP_GAP),
    ("MILESTONE_EXEC_GAP",    "Milestone-execution gap",  S.COLOR_MS_EXEC_GAP),
    ("COMPOSITION_GAP",          "Composition gap",             S.COLOR_COMPOSITION),
    ("MISSING_MILESTONE_GAP", "Missing-milestone gap",    S.COLOR_MISSING_MS),
    ("CAPABILITY_GAP",        "Capability gap",           S.COLOR_CAPABILITY),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_oracle_states(slug):
    fn = PANEL_DIR / f"{slug}.jsonl"
    solves = defaultdict(set)
    all_pids = set()
    with open(fn) as f:
        for line in f:
            d = json.loads(line)
            all_pids.add(d["pid"])
            if d["n_correct"] >= 1:
                solves[d["condition"]].add(d["pid"])
    states = {}
    for pid in all_pids:
        if pid in solves["C1_direct"]:
            states[pid] = "DIRECT"
        elif pid in solves["C2_descriptions"]:
            states[pid] = "DESC_ONLY"
        elif pid in solves["C3_gold_answers"]:
            states[pid] = "GOLD_ONLY"
        else:
            states[pid] = "UNRECOVERED"
    return states


def load_stage0_validity(slug):
    fn = STAGE0_DIR / f"{slug}.jsonl"
    fam_milestones = defaultdict(list)
    with open(fn) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ms_type") == "INTEGRATE":
                continue
            fam_milestones[d["pid"]].append(d.get("n_correct", 0))
    return {pid: (all(n >= 1 for n in ncs) if ncs else False)
            for pid, ncs in fam_milestones.items()}


def classify(state, valid):
    if state == "DIRECT":
        return "DIRECT"
    if valid:
        if state == "DESC_ONLY":  return "ROADMAP_GAP"
        if state == "GOLD_ONLY":  return "MILESTONE_EXEC_GAP"
        return "COMPOSITION_GAP"
    else:
        if state in ("DESC_ONLY", "GOLD_ONLY"):
            return "MISSING_MILESTONE_GAP"
        return "CAPABILITY_GAP"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def draw_definition_matrix(ax):
    """Left panel: 2x4 definition matrix (PASS/FAIL × Direct/Roadmap-needed/Answers-needed/Unrecovered).

    Cells are filled with the gap-category color, label inside is the gap name.
    Direct cells are merged visually (same color, same label).
    """
    ax.set_xlim(-0.05, 4.05)
    ax.set_ylim(-0.05, 2.65)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # Header column labels — natural-language form of the parent-probe outcome
    col_labels = ["None", "Roadmap", "Roadmap\n+ answers", "No rescue"]
    col_colors = [S.COLOR_DIRECT, S.COLOR_ROADMAP, S.COLOR_ANSWERS, S.COLOR_UNRECOVERED]
    for j, (lbl, c) in enumerate(zip(col_labels, col_colors)):
        ax.text(j + 0.5, 0.18, lbl, ha="center", va="center",
                fontsize=S.SIZE_BODY - 0.5, color=c, fontweight="bold")
    ax.text(2.0, -0.05, "Smallest help needed on parent",
            ha="center", va="bottom", fontsize=S.SIZE_BOX_TITLE,
            color=S.COLOR_TEXT_MUTED, style="italic")

    # Row labels
    ax.text(-0.08, 0.95, "Passes\nmilestone\ntest", ha="right", va="center",
            fontsize=S.SIZE_BODY, color=S.COLOR_TEXT, fontweight="bold")
    ax.text(-0.08, 1.95, "Fails\nmilestone\ntest", ha="right", va="center",
            fontsize=S.SIZE_BODY, color=S.COLOR_TEXT, fontweight="bold")

    # Cell contents — (row, col, label, color, font_color)
    cells = [
        # PASS row
        (1, 0, "Direct",                     S.COLOR_DIRECT,        "white"),
        (1, 1, "Roadmap\ngap",               S.COLOR_ROADMAP_GAP,   "white"),
        (1, 2, "Milestone-\nexecution\ngap", S.COLOR_MS_EXEC_GAP,   "white"),
        (1, 3, "Composition\ngap",              S.COLOR_COMPOSITION,      "white"),
        # FAIL row
        (2, 0, "Direct",                     S.COLOR_DIRECT,        "white"),
        (2, 1, "rare",                       S.COLOR_DIVIDER,       S.COLOR_TEXT_MUTED),
        (2, 2, "Missing-\nmilestone\ngap",   S.COLOR_MISSING_MS,    "white"),
        (2, 3, "Capability\ngap",            S.COLOR_CAPABILITY,    "white"),
    ]
    for row, col, label, fill, fontc in cells:
        ax.add_patch(mpatches.FancyBboxPatch(
            (col + 0.04, row - 0.46), 0.92, 0.92,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=fill, edgecolor="white", linewidth=S.LW_BAR_EDGE))
        ax.text(col + 0.5, row + 0.0, label, ha="center", va="center",
                fontsize=S.SIZE_ANNOTATION, color=fontc, fontweight="bold")


def draw_stacked_bars(ax, results):
    n_models = len(MODELS)
    x = np.arange(n_models)
    bottom = np.zeros(n_models)
    for cat_key, cat_label, cat_color in CATEGORIES:
        vals = np.array([results[slug]["counts"].get(cat_key, 0) for slug, _ in MODELS])
        ax.bar(x, vals, S.BAR_WIDTH, bottom=bottom,
               color=cat_color, edgecolor=S.BAR_EDGE_COLOR, linewidth=S.LW_BAR_EDGE)
        # Inline numbers — only for segments ≥ 18 families (else too cramped)
        for i, v in enumerate(vals):
            if v >= 18:
                # font color: white on dark, dark on light
                font_color = "white" if cat_key in ("DIRECT", "COMPOSITION_GAP",
                                                     "CAPABILITY_GAP",
                                                     "MS_EXEC_GAP",
                                                     "MILESTONE_EXEC_GAP",
                                                     "ROADMAP_GAP",
                                                     "MISSING_MILESTONE_GAP") else S.COLOR_TEXT
                ax.text(x[i], bottom[i] + v / 2, str(int(v)),
                        ha="center", va="center",
                        fontsize=S.SIZE_ANNOTATION, color=font_color, fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in MODELS],
                       fontsize=S.SIZE_TICK, rotation=20, ha="right")
    ax.set_ylabel("Families (out of 354)", fontsize=S.SIZE_BODY)
    ax.set_ylim(0, 360)
    ax.spines["bottom"].set_color(S.COLOR_TEXT)
    ax.spines["bottom"].set_linewidth(S.LW_MAIN)
    ax.spines["left"].set_color(S.COLOR_TEXT)
    ax.spines["left"].set_linewidth(S.LW_MAIN)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=2, color=S.COLOR_TEXT)

    # Compact legend above bars
    handles = [mpatches.Patch(facecolor=c, edgecolor="white", label=lbl)
               for _, lbl, c in CATEGORIES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, fontsize=S.SIZE_LEGEND,
              handlelength=1.4, handletextpad=0.4, columnspacing=1.2)


def main():
    results = {}
    for slug, label in MODELS:
        states = load_oracle_states(slug)
        validity = load_stage0_validity(slug)
        counts = defaultdict(int)
        for pid, s in states.items():
            counts[classify(s, validity.get(pid, False))] += 1
        total = sum(counts.values())
        results[slug] = {"label": label, "n_total": total,
                         "counts": dict(counts),
                         "n_stage0_valid": sum(1 for pid in states if validity.get(pid, False))}
        print(f"{label:<18} n={total} valid={results[slug]['n_stage0_valid']}")
        for cat, _, _ in CATEGORIES:
            c = counts[cat]
            print(f"  {cat:<24} {c:>4} ({100*c/total:5.1f}%)")
        print()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    # Two-panel figure
    fig, (ax_def, ax_bars) = plt.subplots(
        1, 2, figsize=(7.0, 3.4),
        gridspec_kw={"width_ratios": [0.36, 0.64]}
    )
    draw_definition_matrix(ax_def)
    draw_stacked_bars(ax_bars, results)

    fig.tight_layout()
    fig.subplots_adjust(top=0.86)  # leave headroom for legend
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")

    # Latex table — keep, used in appendix
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\small\n")
        f.write("\\caption{Reasoning-gap taxonomy: Stage 0 milestone test (pass/fail) $\\times$ smallest help needed to unlock the parent, per model on the 354-family diagnostic set. Counts are families. Both Stage 0 and parent probing at \\texttt{max\\_tokens=16384}, $K{=}8$.}\n")
        f.write("\\label{tab:bottleneck_lattice}\n\\vspace{4pt}\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Direct} & \\textbf{Roadmap} & \\textbf{MS-Exec} & \\textbf{Composition} & \\textbf{Missing-MS} & \\textbf{Capability} \\\\\n")
        f.write("\\midrule\n")
        for slug, label in MODELS:
            r = results[slug]
            cs = r["counts"]
            f.write(f"{label} & {cs.get('DIRECT',0)} & {cs.get('ROADMAP_GAP',0)} & {cs.get('MILESTONE_EXEC_GAP',0)} & {cs.get('COMPOSITION_GAP',0)} & {cs.get('MISSING_MILESTONE_GAP',0)} & {cs.get('CAPABILITY_GAP',0)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
