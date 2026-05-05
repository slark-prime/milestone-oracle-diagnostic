"""Cross-model fingerprint stacked bars (Appendix C — supporting Figure 4).

Same recovery-state palette as Figure 5 (two-arm). Short model names.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import figure_style as S

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "data/logs/rl/panel_fingerprints_16k.json"
OUT = REPO / "docs/latex/figures/panel_fingerprints_16k.pdf"

MODEL_ORDER = [
    ("qwen3_8b_pre_rl",        "Qwen-base"),
    ("base_2k_step_180",       "OutcomeRL-2K"),
    ("mile_2k_step_180",       "MilestoneRL-2K"),
    ("gpt_oss_20b",            "gpt-oss-20b"),
    ("llama_3_3_70b_instruct", "Llama-70B"),
    ("deepseek_v3_1",          "DeepSeek-V3.1"),
]

STATE_KEYS  = ["direct", "desc_only", "gold_only", "unrecovered"]
STATE_LABEL = ["Direct", "Roadmap-needed", "Answers-needed", "Unrecovered"]
STATE_COLOR = [S.COLOR_DIRECT, S.COLOR_ROADMAP, S.COLOR_ANSWERS, S.COLOR_UNRECOVERED]


def main():
    fp = json.load(open(IN))

    fig, ax = plt.subplots(figsize=(6.7, 2.8))
    y_positions = list(range(len(MODEL_ORDER)))[::-1]

    for y_pos, (slug, label) in zip(y_positions, MODEL_ORDER):
        data = fp[slug]
        total = data["total"]
        left = 0
        for state, color in zip(STATE_KEYS, STATE_COLOR):
            val = data[state]
            ax.barh(y_pos, val, left=left, color=color,
                    edgecolor=S.BAR_EDGE_COLOR, linewidth=S.LW_BAR_EDGE)
            if val / total >= 0.05:
                font_color = "white" if color in (S.COLOR_DIRECT, S.COLOR_ROADMAP, S.COLOR_ANSWERS) else S.COLOR_TEXT
                ax.text(left + val / 2, y_pos, str(int(val)),
                        ha="center", va="center",
                        fontsize=S.SIZE_ANNOTATION, color=font_color,
                        fontweight="bold")
            left += val

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in MODEL_ORDER], fontsize=S.SIZE_TICK)
    ax.set_xlim(0, 354)
    ax.set_xlabel("Families on the $n{=}354$ diagnostic set", fontsize=S.SIZE_BODY)
    ax.spines["bottom"].set_color(S.COLOR_TEXT)
    ax.spines["bottom"].set_linewidth(S.LW_MAIN)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=2, color=S.COLOR_TEXT)
    ax.tick_params(axis="y", length=0)

    handles = [Patch(facecolor=c, edgecolor="white", label=l)
               for c, l in zip(STATE_COLOR, STATE_LABEL)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.40),
              ncol=4, frameon=False, fontsize=S.SIZE_LEGEND,
              handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
