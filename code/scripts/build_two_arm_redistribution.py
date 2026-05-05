"""Figure 5: Two-arm RL longitudinal readout (paper §5.4).

Two-panel:
  Left  — state composition stacked bars for {Qwen-base, OutcomeRL-2K, MilestoneRL-2K}
          with MATH score below each bar and "any-probe count" above.
  Right — delta-bars: change in family count per parent-probe outcome (OutcomeRL/MilestoneRL minus Qwen-base).

Headline visual: two RL arms with similar MATH gain move families through different parent-probe outcomes.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

import figure_style as S

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/latex/figures/two_arm_redistribution.pdf"

# Canonical 16K data (verified from data/logs/rl/oracle_panel_16k/*.jsonl)
STATES = ["Direct", "Roadmap-needed", "Answers-needed", "Unrecovered"]
COUNTS = {
    "Direct":       [76, 106, 96],
    "Roadmap-needed": [50,  50, 47],
    "Answers-needed": [26,  14, 22],
    "Unrecovered":  [202, 184, 189],
}
MODELS = ["Qwen-base", "OutcomeRL-2K", "MilestoneRL-2K"]
MATH_SCORE = [0.454, 0.654, 0.634]

STATE_COLOR = {
    "Direct":       S.COLOR_DIRECT,
    "Roadmap-needed": S.COLOR_ROADMAP,
    "Answers-needed": S.COLOR_ANSWERS,
    "Unrecovered":  S.COLOR_UNRECOVERED,
}


def main():
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.0, 3.4),
        gridspec_kw={"width_ratios": [0.45, 0.55]}
    )

    # =====  LEFT: stacked composition bars  =====
    x = np.arange(3)
    bottom = np.zeros(3)
    for state in STATES:
        vals = np.array(COUNTS[state])
        c = STATE_COLOR[state]
        axL.bar(x, vals, S.BAR_WIDTH, bottom=bottom, color=c,
                edgecolor=S.BAR_EDGE_COLOR, linewidth=S.LW_BAR_EDGE)
        for i, v in enumerate(vals):
            if v >= 18:
                font_color = "white" if state in ("Direct", "Roadmap-needed", "Answers-needed") else S.COLOR_TEXT
                axL.text(x[i], bottom[i] + v / 2, str(int(v)),
                         ha="center", va="center",
                         fontsize=S.SIZE_ANNOTATION, color=font_color,
                         fontweight="bold")
        bottom += vals

    # "any-probe" ceiling annotation above each bar
    ceilings = [COUNTS["Direct"][i] + COUNTS["Roadmap-needed"][i] + COUNTS["Answers-needed"][i]
                for i in range(3)]
    for i, c in enumerate(ceilings):
        axL.text(i, 360, f"any probe = {c}",
                 ha="center", va="top",
                 fontsize=S.SIZE_ANNOTATION, color=S.COLOR_TEXT_MUTED, style="italic")

    # x-axis: model name + MATH score
    axL.set_xticks(x)
    axL.set_xticklabels(
        [f"{m}\nMATH = {p:.0%}" for m, p in zip(MODELS, MATH_SCORE)],
        fontsize=S.SIZE_TICK)
    axL.set_ylabel("Families (out of 354)", fontsize=S.SIZE_BODY)
    axL.set_ylim(0, 380)
    axL.set_yticks(np.arange(0, 355, 50))
    axL.spines["bottom"].set_color(S.COLOR_TEXT)
    axL.spines["bottom"].set_linewidth(S.LW_MAIN)
    axL.spines["left"].set_color(S.COLOR_TEXT)
    axL.spines["left"].set_linewidth(S.LW_MAIN)
    axL.tick_params(axis="x", length=0)
    axL.tick_params(axis="y", length=2, color=S.COLOR_TEXT)

    # State legend (compact, below left panel)
    handles = [mpatches.Patch(facecolor=STATE_COLOR[s], edgecolor="white", label=s)
               for s in STATES]
    axL.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.42), ncol=4,
               frameon=False, fontsize=S.SIZE_LEGEND,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

    # =====  RIGHT: delta bars  =====
    deltas_outcome = [COUNTS[s][1] - COUNTS[s][0] for s in STATES]
    deltas_milestone = [COUNTS[s][2] - COUNTS[s][0] for s in STATES]

    n = len(STATES)
    xpos = np.arange(n)
    width = 0.36

    bars_o = axR.bar(xpos - width/2, deltas_outcome, width,
                     color=S.COLOR_OUTCOME_RL, edgecolor=S.BAR_EDGE_COLOR,
                     linewidth=S.LW_BAR_EDGE, label="OutcomeRL-2K  − Qwen-base")
    bars_m = axR.bar(xpos + width/2, deltas_milestone, width,
                     color=S.COLOR_MILESTONE_RL, edgecolor=S.BAR_EDGE_COLOR,
                     linewidth=S.LW_BAR_EDGE, label="MilestoneRL-2K − Qwen-base")

    # numeric labels
    for bar, val in list(zip(bars_o, deltas_outcome)) + list(zip(bars_m, deltas_milestone)):
        ha_y = bar.get_height()
        offset = 1.5 if ha_y >= 0 else -1.5
        va = "bottom" if ha_y >= 0 else "top"
        axR.text(bar.get_x() + bar.get_width()/2, ha_y + offset, f"{val:+d}",
                 ha="center", va=va, fontsize=S.SIZE_ANNOTATION,
                 color=S.COLOR_TEXT, fontweight="bold")

    axR.axhline(0, color=S.COLOR_DIVIDER, linewidth=S.LW_HELPER)
    axR.set_xticks(xpos)
    axR.set_xticklabels(STATES, fontsize=S.SIZE_TICK)
    axR.set_ylabel("Change in family count vs. Qwen-base", fontsize=S.SIZE_BODY)
    axR.set_ylim(-22, 36)
    axR.spines["bottom"].set_color(S.COLOR_TEXT)
    axR.spines["bottom"].set_linewidth(S.LW_MAIN)
    axR.spines["left"].set_color(S.COLOR_TEXT)
    axR.spines["left"].set_linewidth(S.LW_MAIN)
    axR.tick_params(axis="x", length=0)
    axR.tick_params(axis="y", length=2, color=S.COLOR_TEXT)
    axR.grid(axis="y", linestyle=":", alpha=0.4, linewidth=S.LW_HELPER)
    axR.set_axisbelow(True)
    axR.legend(loc="upper right", fontsize=S.SIZE_LEGEND, frameon=False,
               handlelength=1.2, handletextpad=0.4)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
