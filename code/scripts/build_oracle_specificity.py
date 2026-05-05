"""Figure 2: Oracle specificity bar chart (paper §5.1).

Designed for skimming reviewer:
- Single bar chart, 6 bars
- Natural-language x-axis (No help / Random roadmap / ... / Correct milestone answers)
- Color groups: gray (no help), light red (controls), blue (correct roadmap), purple (correct answers)
- Number on top of each bar = families solved
- No p-values, no legend (colors carry the meaning)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_style as S

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data/logs/rl/manifest.jsonl"
OUT = REPO / "docs/latex/figures/oracle_specificity.pdf"

CONDS = [
    # (manifest field suffix, x-tick label, role)
    ("C1_direct",        "No help",                   "baseline"),
    ("C2_random",        "Random\nroadmap",           "control"),
    ("C2_generic",       "Generic\nhint",             "control"),
    ("C2_descriptions",  "Correct\nroadmap",          "roadmap"),
    ("C3_mismatched",    "Wrong\nmilestone\nanswers", "control"),
    ("C3_gold_answers",  "Correct\nmilestone\nanswers", "answers"),
]

ROLE_COLOR = {
    "baseline": S.COLOR_DIRECT,
    "control":  S.COLOR_CONTROL,
    "roadmap":  S.COLOR_ROADMAP,
    "answers":  S.COLOR_ANSWERS,
}


def main():
    rows = [json.loads(l) for l in open(MANIFEST) if json.loads(l).get("in_354_multi_milestone")]
    K = 8

    solved = []
    for cond, _, _ in CONDS:
        vals = [r.get(f"mile_2k_step_180__{cond}") for r in rows]
        vals = [v for v in vals if v is not None]
        solved.append(sum(1 for v in vals if v >= 1))

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    x = np.arange(len(CONDS))
    colors = [ROLE_COLOR[role] for _, _, role in CONDS]

    bars = ax.bar(x, solved, S.BAR_WIDTH, color=colors,
                  edgecolor=S.BAR_EDGE_COLOR, linewidth=S.LW_BAR_EDGE)

    # Numbers on top of each bar
    for b, v in zip(bars, solved):
        ax.text(b.get_x() + b.get_width()/2, v + 2,
                str(v), ha="center", va="bottom",
                fontsize=S.SIZE_BODY, color=S.COLOR_TEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in CONDS], fontsize=S.SIZE_TICK)
    ax.set_ylabel("Families solved (out of 354)", fontsize=S.SIZE_BODY)
    ax.set_ylim(0, max(solved) * 1.18)
    ax.set_yticks(np.arange(0, max(solved) * 1.15, 30))

    # Group brackets above bars
    y_top = max(solved) * 1.10
    bracket_y = y_top + 5
    # control group spans bars 1,2 and bar 4
    ax.annotate("", xy=(0.7, bracket_y), xytext=(2.3, bracket_y),
                arrowprops=dict(arrowstyle="-", color=S.COLOR_TEXT_MUTED,
                                linewidth=S.LW_HELPER))
    ax.text(1.5, bracket_y + 1, "corruption controls",
            ha="center", va="bottom",
            fontsize=S.SIZE_ANNOTATION, color=S.COLOR_TEXT_MUTED, style="italic")
    ax.annotate("", xy=(3.7, bracket_y), xytext=(4.3, bracket_y),
                arrowprops=dict(arrowstyle="-", color=S.COLOR_TEXT_MUTED,
                                linewidth=S.LW_HELPER))
    ax.text(4, bracket_y + 1, "control",
            ha="center", va="bottom",
            fontsize=S.SIZE_ANNOTATION, color=S.COLOR_TEXT_MUTED, style="italic")

    ax.spines["bottom"].set_color(S.COLOR_TEXT)
    ax.spines["bottom"].set_linewidth(S.LW_MAIN)
    ax.spines["left"].set_color(S.COLOR_TEXT)
    ax.spines["left"].set_linewidth(S.LW_MAIN)
    ax.tick_params(axis="x", length=0, labelcolor=S.COLOR_TEXT)
    ax.tick_params(axis="y", length=3, color=S.COLOR_TEXT, labelcolor=S.COLOR_TEXT)

    fig.tight_layout()
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")
    print(f"  values: {dict(zip([c[1].replace(chr(10), ' ') for c in CONDS], solved))}")


if __name__ == "__main__":
    main()
