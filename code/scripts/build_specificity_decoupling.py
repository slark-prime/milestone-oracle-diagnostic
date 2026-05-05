"""Combined 2-panel Figure: specificity (left) + accuracy-vs-recoverability (right).

Replaces what were previously two separate figures (oracle_specificity and
accuracy_vs_recoverability). One figure, two panels, single caption — saves
about 4 inches of vertical space on the main paper.

Output: docs/latex/figures/specificity_decoupling.pdf
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_style as S

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data/logs/rl/manifest.jsonl"
PANEL_DIR = REPO / "data/logs/rl/oracle_panel_16k"
OUT = REPO / "docs/latex/figures/specificity_decoupling.pdf"


# ---- Specificity (left panel) ----
SPEC_CONDS = [
    ("C1_direct",        "No help",                   "baseline"),
    ("C2_random",        "Random\nroadmap",           "control"),
    ("C2_generic",       "Generic\nhint",             "control"),
    ("C2_descriptions",  "Correct\nroadmap",          "roadmap"),
    ("C3_mismatched",    "Wrong\nmilestone\nanswers", "control"),
    ("C3_gold_answers",  "Correct\nmilestone\nanswers", "answers"),
]
SPEC_ROLE_COLOR = {
    "baseline": S.COLOR_DIRECT,
    "control":  S.COLOR_CONTROL,
    "roadmap":  S.COLOR_ROADMAP,
    "answers":  S.COLOR_ANSWERS,
}


def panel_specificity(ax):
    rows = [json.loads(l) for l in open(MANIFEST) if json.loads(l).get("in_354_multi_milestone")]
    solved = []
    for cond, _, _ in SPEC_CONDS:
        vals = [r.get(f"mile_2k_step_180__{cond}") for r in rows]
        vals = [v for v in vals if v is not None]
        solved.append(sum(1 for v in vals if v >= 1))

    x = np.arange(len(SPEC_CONDS))
    colors = [SPEC_ROLE_COLOR[role] for _, _, role in SPEC_CONDS]
    bars = ax.bar(x, solved, S.BAR_WIDTH, color=colors,
                  edgecolor=S.BAR_EDGE_COLOR, linewidth=S.LW_BAR_EDGE)
    for b, v in zip(bars, solved):
        ax.text(b.get_x() + b.get_width()/2, v + 2, str(v),
                ha="center", va="bottom",
                fontsize=S.SIZE_BODY, color=S.COLOR_TEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in SPEC_CONDS], fontsize=S.SIZE_TICK - 0.5)
    ax.set_ylabel("Families solved (of 354)", fontsize=S.SIZE_BODY)
    ax.set_ylim(0, max(solved) * 1.15)
    ax.set_yticks(np.arange(0, max(solved) * 1.15, 30))
    ax.spines["bottom"].set_color(S.COLOR_TEXT)
    ax.spines["bottom"].set_linewidth(S.LW_MAIN)
    ax.spines["left"].set_color(S.COLOR_TEXT)
    ax.spines["left"].set_linewidth(S.LW_MAIN)
    ax.tick_params(axis="x", length=0, labelcolor=S.COLOR_TEXT)
    ax.tick_params(axis="y", length=3, color=S.COLOR_TEXT, labelcolor=S.COLOR_TEXT)
    ax.set_title("(a) Specificity (MilestoneRL-2K, $K{=}8$)",
                 fontsize=S.SIZE_BOX_TITLE, color=S.COLOR_TEXT, pad=4, loc="left")


# ---- Accuracy vs recoverability (right panel) ----
# Color = lineage:
#   Qwen lineage (3 models, same starting weights, different training stages):
#     light blue -> medium blue -> dark blue gradient
#   External lineages (3 models, each distinct):
#     each gets its own accent color
SCATTER_MODELS = [
    ("qwen3_8b_pre_rl",            "Qwen-base",       "dense", "#93C5FD"),  # light blue (Qwen, pre-RL)
    ("base_2k_step_180",           "OutcomeRL-2K",    "dense", "#3B82F6"),  # medium blue (Qwen, outcome RL)
    ("mile_2k_step_180",           "MilestoneRL-2K",  "dense", "#1E40AF"),  # dark blue (Qwen, milestone RL)
    ("gpt_oss_20b",                "gpt-oss-20b",     "moe",   "#22C55E"),  # green (external, OpenAI)
    ("llama_3_3_70b_instruct",     "Llama-70B",       "dense", "#F59E0B"),  # amber (external, Meta)
    ("deepseek_v3_1",              "DeepSeek-V3.1",   "moe",   "#EC4899"),  # magenta (external, DeepSeek)
]


def load_scatter_metrics(slug):
    fn = PANEL_DIR / f"{slug}.jsonl"
    solves = defaultdict(set)
    n_correct_total = 0
    k_total = 0
    with open(fn) as f:
        for line in f:
            d = json.loads(line)
            if d["condition"] == "C1_direct":
                n_correct_total += d["n_correct"]
                k_total += d["k"]
            if d["n_correct"] >= 1:
                solves[d["condition"]].add(d["pid"])
    direct_acc = n_correct_total / k_total if k_total else 0.0
    direct = solves["C1_direct"]
    roadmap = solves["C2_descriptions"]
    gold = solves["C3_gold_answers"]
    return direct_acc, len(direct | roadmap | gold)


def panel_decoupling(ax):
    pts = []
    for slug, label, arch, color in SCATTER_MODELS:
        acc, ap = load_scatter_metrics(slug)
        pts.append((slug, label, arch, color, acc, ap))

    for slug, label, arch, color, acc, ap in pts:
        marker = "o" if arch == "dense" else "D"
        ax.scatter(acc, ap, s=70, marker=marker,
                   facecolor=color, edgecolor="black",
                   linewidth=0.7, zorder=4)

    LABEL_OFFSETS = {
        "qwen3_8b_pre_rl":        ( 8, -2),
        "base_2k_step_180":       ( 8, 11),
        "mile_2k_step_180":       ( 8,-11),
        "gpt_oss_20b":            (-8, 8),
        "llama_3_3_70b_instruct": ( 8,-2),
        "deepseek_v3_1":          ( 8, 0),
    }
    from matplotlib.transforms import offset_copy
    for slug, label, arch, color, acc, ap in pts:
        dx, dy = LABEL_OFFSETS[slug]
        ha = "right" if dx < 0 else "left"
        if abs(dy) > 6:  # leader line for offset labels
            ax.annotate("", xy=(acc, ap), xytext=(dx, dy),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="-",
                                        color=S.COLOR_TEXT_MUTED,
                                        lw=S.LW_HELPER, shrinkA=2, shrinkB=2),
                        zorder=2)
        trans = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
        ax.text(acc, ap, label, transform=trans,
                fontsize=S.SIZE_TICK, color=color, fontweight="bold",
                ha=ha, va="center")

    # Headline arrow + callout
    deepseek = next(p for p in pts if p[0] == "deepseek_v3_1")
    gptoss = next(p for p in pts if p[0] == "gpt_oss_20b")
    ax.annotate("", xy=(gptoss[4], gptoss[5]), xytext=(deepseek[4], deepseek[5]),
                arrowprops=dict(arrowstyle="->", color=S.COLOR_TEXT_MUTED,
                                lw=0.9, connectionstyle="arc3,rad=-0.22"),
                zorder=2)
    ax.annotate("lower direct,\nmore probe-recoverable",
                xy=((deepseek[4] + gptoss[4]) / 2,
                    (deepseek[5] + gptoss[5]) / 2 + 6),
                xytext=(0.355, 240),
                ha="right", va="top",
                fontsize=S.SIZE_ANNOTATION, color=S.COLOR_TEXT, style="italic",
                arrowprops=dict(arrowstyle="-", color=S.COLOR_TEXT_MUTED,
                                lw=S.LW_HELPER, shrinkA=2, shrinkB=2),
                zorder=2)

    ax.set_xlabel("Direct accuracy (per-rollout, $K{=}8$)",
                  fontsize=S.SIZE_BODY, color=S.COLOR_TEXT)
    ax.set_ylabel("Solved by any probe (of 354)",
                  fontsize=S.SIZE_BODY, color=S.COLOR_TEXT)
    ax.set_xlim(0.00, 0.36)
    ax.set_ylim(105, 245)
    ax.set_xticks(np.arange(0.0, 0.40, 0.05))
    ax.set_xticklabels([f"{x:.0%}" for x in np.arange(0.0, 0.40, 0.05)],
                       fontsize=S.SIZE_TICK, color=S.COLOR_TEXT)
    ax.set_yticks(np.arange(120, 245, 30))
    ax.tick_params(axis="y", labelcolor=S.COLOR_TEXT, color=S.COLOR_TEXT, length=3)
    ax.tick_params(axis="x", labelcolor=S.COLOR_TEXT, color=S.COLOR_TEXT, length=3)
    ax.spines["bottom"].set_color(S.COLOR_TEXT)
    ax.spines["bottom"].set_linewidth(S.LW_MAIN)
    ax.spines["left"].set_color(S.COLOR_TEXT)
    ax.spines["left"].set_linewidth(S.LW_MAIN)
    ax.grid(True, linestyle=":", alpha=0.3, linewidth=S.LW_HELPER)
    ax.set_axisbelow(True)
    ax.set_title("(b) Direct accuracy vs probe recovery (6 panel models)",
                 fontsize=S.SIZE_BOX_TITLE, color=S.COLOR_TEXT, pad=4, loc="left")

    # Compact lineage legend in bottom-right.
    # Color = lineage (Qwen blues / external accents). Marker = arch (circle/diamond).
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=6,
               markerfacecolor="#3B82F6", markeredgecolor="black",
               markeredgewidth=0.4, label="Qwen lineage (3 stages)"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=5,
               markerfacecolor="#cccccc", markeredgecolor="black",
               markeredgewidth=0.5, label="MoE (else dense)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=S.SIZE_ANNOTATION,
              frameon=False, handletextpad=0.3, labelspacing=0.3)


def main():
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(7.0, 2.7),
        gridspec_kw={"width_ratios": [0.45, 0.55]}
    )
    panel_specificity(ax_l)
    panel_decoupling(ax_r)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
