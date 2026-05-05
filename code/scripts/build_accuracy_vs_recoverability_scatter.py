"""Figure 3: Accuracy vs recoverability scatter (paper §5.2).

Single scatter, 6 points (one per panel model).
- x-axis: Direct accuracy (per-rollout mean, K=8)
- y-axis: Families solved by any probe (out of 354)
- Distinct color per model so the eye can match label to point
- Shape encodes architecture (circle dense, diamond MoE)
- Mile/Outcome overlap: handled with leader-line callout on each side
- One callout highlighting the headline "lower direct, more probe-recoverable"
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
PANEL_DIR = REPO / "data/logs/rl/oracle_panel_16k"
OUT = REPO / "docs/latex/figures/accuracy_vs_recoverability.pdf"

# (slug, display, architecture, training-paradigm)
# Colors are derived from the training paradigm so the deck reuses the
# Tier-1 palette: gray = base / no-RL / instruction-tuned,
# blue = outcome-only RL, purple = milestone-augmented or reasoning-trained.
# Marker shape encodes architecture so the 6 models stay individually identifiable.
MODELS = [
    ("qwen3_8b_pre_rl",            "Qwen-base",       "dense", S.COLOR_PARADIGM_BASE),
    ("base_2k_step_180",           "OutcomeRL-2K",    "dense", S.COLOR_PARADIGM_OUTCOME),
    ("mile_2k_step_180",           "MilestoneRL-2K",  "dense", S.COLOR_PARADIGM_MILESTONE),
    ("gpt_oss_20b",                "gpt-oss-20b",     "moe",   S.COLOR_PARADIGM_MILESTONE),
    ("llama_3_3_70b_instruct",     "Llama-70B",       "dense", S.COLOR_PARADIGM_BASE),
    ("deepseek_v3_1",              "DeepSeek-V3.1",   "moe",   S.COLOR_PARADIGM_MILESTONE),
]


def load_metrics(slug: str):
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
    any_probe = len(direct | roadmap | gold)
    return direct_acc, any_probe


def main():
    pts = []
    for slug, label, arch, color in MODELS:
        acc, any_probe = load_metrics(slug)
        pts.append((slug, label, arch, color, acc, any_probe))
        print(f"  {label:<18} acc={acc:.3f}  any_probe={any_probe}")

    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    # Draw points
    for slug, label, arch, color, acc, any_probe in pts:
        marker = "o" if arch == "dense" else "D"
        ax.scatter(acc, any_probe, s=90, marker=marker,
                   facecolor=color, edgecolor="black",
                   linewidth=0.8, zorder=4)

    # ---- Label placement with explicit handling of overlap ----
    #
    # Place labels at safe offsets in plotting points. For Mile vs Outcome
    # (very close), we use leader lines so labels can sit far from each other.
    #
    LABEL_PLACEMENT = {
        # slug : (dx_pt, dy_pt, ha, with_leader)
        "qwen3_8b_pre_rl":        (10,   0,  "left",  False),
        "base_2k_step_180":       (10,  14,  "left",  True),   # nudge up + leader
        "mile_2k_step_180":       (10, -14,  "left",  True),   # nudge down + leader
        "gpt_oss_20b":            (-10,   8, "right", False),  # to the upper-left of point
        "llama_3_3_70b_instruct": (10,   0,  "left",  False),
        "deepseek_v3_1":          (-10,  0,  "right", False),  # to the left of point
    }
    # Render labels — keep within axis by using ax.transData + offset transform
    from matplotlib.transforms import offset_copy
    for slug, label, arch, color, acc, any_probe in pts:
        dx, dy, ha, leader = LABEL_PLACEMENT[slug]
        trans = offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points")
        if leader:
            # short leader from point to label anchor
            ax.annotate("", xy=(acc, any_probe), xytext=(dx, dy),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="-",
                                        color=S.COLOR_TEXT_MUTED,
                                        lw=S.LW_HELPER,
                                        shrinkA=2, shrinkB=2),
                        zorder=2)
        ax.text(acc, any_probe, label, transform=trans,
                fontsize=S.SIZE_BODY, color=color, fontweight="bold",
                ha=ha, va="center")

    # ---- Headline callout (DeepSeek -> gpt-oss arrow + text in clear space) ----
    deepseek = next(p for p in pts if p[0] == "deepseek_v3_1")
    gptoss = next(p for p in pts if p[0] == "gpt_oss_20b")
    # curved arrow from DeepSeek to gpt-oss
    ax.annotate("",
                xy=(gptoss[4], gptoss[5]), xytext=(deepseek[4], deepseek[5]),
                arrowprops=dict(arrowstyle="->", color=S.COLOR_TEXT_MUTED,
                                lw=0.9, connectionstyle="arc3,rad=-0.22"),
                zorder=2)
    # Annotation text in the empty area to the RIGHT of the curve (reader's
    # right). Leader line goes to the midpoint of the DeepSeek→gpt-oss curve.
    text_x, text_y = 0.355, 207
    arrow_mid_x = (deepseek[4] + gptoss[4]) / 2 - 0.005
    arrow_mid_y = (deepseek[5] + gptoss[5]) / 2 + 14
    ax.annotate("lower direct accuracy,\nmore probe-recoverable",
                xy=(arrow_mid_x, arrow_mid_y),
                xytext=(text_x, text_y),
                ha="right", va="center",
                fontsize=S.SIZE_ANNOTATION, color=S.COLOR_TEXT, style="italic",
                arrowprops=dict(arrowstyle="-", color=S.COLOR_TEXT_MUTED,
                                lw=S.LW_HELPER, shrinkA=0, shrinkB=2),
                zorder=2)

    # ---- Axes ----
    ax.set_xlabel("Direct accuracy (per-rollout, $K{=}8$)",
                  fontsize=S.SIZE_BODY, color=S.COLOR_TEXT)
    ax.set_ylabel("Families solved by any probe (of 354)",
                  fontsize=S.SIZE_BODY, color=S.COLOR_TEXT)
    ax.set_xlim(0.00, 0.36)
    ax.set_ylim(105, 245)
    ax.set_xticks(np.arange(0.0, 0.40, 0.05))
    ax.set_xticklabels([f"{x:.0%}" for x in np.arange(0.0, 0.40, 0.05)],
                       fontsize=S.SIZE_TICK, color=S.COLOR_TEXT)
    ax.set_yticks(np.arange(120, 245, 30))
    ax.tick_params(axis="y", labelcolor=S.COLOR_TEXT, color=S.COLOR_TEXT, length=3)
    ax.tick_params(axis="x", labelcolor=S.COLOR_TEXT, color=S.COLOR_TEXT, length=3)

    # Two-tier legend: color = training paradigm; marker = architecture
    from matplotlib.lines import Line2D
    paradigm_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor=S.COLOR_PARADIGM_BASE, markeredgecolor="black",
               markeredgewidth=0.6, label="base / no RL"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor=S.COLOR_PARADIGM_OUTCOME, markeredgecolor="black",
               markeredgewidth=0.6, label="outcome-only RL"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor=S.COLOR_PARADIGM_MILESTONE, markeredgecolor="black",
               markeredgewidth=0.6, label="reasoning / milestone RL"),
    ]
    arch_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=0.8, label="dense"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=7,
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=0.8, label="MoE"),
    ]
    leg_paradigm = ax.legend(handles=paradigm_handles, loc="lower right",
                             title="Training paradigm",
                             fontsize=S.SIZE_LEGEND, title_fontsize=S.SIZE_LEGEND,
                             frameon=False, handletextpad=0.4, alignment="left")
    ax.add_artist(leg_paradigm)
    ax.legend(handles=arch_handles, loc="lower right",
              bbox_to_anchor=(1.0, 0.20),  # above the paradigm legend
              title="Architecture",
              fontsize=S.SIZE_LEGEND, title_fontsize=S.SIZE_LEGEND,
              frameon=False, handletextpad=0.4, alignment="left")

    ax.spines["bottom"].set_color(S.COLOR_TEXT)
    ax.spines["bottom"].set_linewidth(S.LW_MAIN)
    ax.spines["left"].set_color(S.COLOR_TEXT)
    ax.spines["left"].set_linewidth(S.LW_MAIN)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=S.LW_HELPER)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
