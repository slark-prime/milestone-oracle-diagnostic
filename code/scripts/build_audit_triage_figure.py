"""Figure 6: Audit triage of the unrecovered bucket (paper §5.5).

One horizontal stacked bar (n=100). Four categories, plain-language labels.
No upper-bound marker (kept in caption / appendix to avoid figure clutter).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import figure_style as S

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/latex/figures/audit_triage.pdf"

CATEGORIES = [
    ("Artifacts",                  37, S.COLOR_AUDIT_ARTIFACT),
    ("Composition-related",         57, S.COLOR_AUDIT_STRONG),
    ("Beyond capability",           5, S.COLOR_AUDIT_BEYOND),
    ("Not standalone",              1, S.COLOR_AUDIT_NOTSTANDALONE),
]


def main():
    fig, ax = plt.subplots(figsize=(6.5, 1.7))
    left = 0
    for label, count, color in CATEGORIES:
        ax.barh(0, count, left=left, height=0.5, color=color,
                edgecolor=S.BAR_EDGE_COLOR, linewidth=S.LW_BAR_EDGE)
        # Inline number for any segment ≥ 4 wide
        if count >= 4:
            font_color = "white" if color in (S.COLOR_AUDIT_STRONG, S.COLOR_AUDIT_BEYOND) else S.COLOR_TEXT
            ax.text(left + count / 2, 0, str(count),
                    ha="center", va="center",
                    fontsize=S.SIZE_BODY, color=font_color, fontweight="bold")
        left += count

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Audited unrecovered families ($n{=}100$)", fontsize=S.SIZE_BODY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(S.COLOR_TEXT)
    ax.spines["bottom"].set_linewidth(S.LW_MAIN)
    ax.tick_params(axis="x", length=2, color=S.COLOR_TEXT, labelsize=S.SIZE_TICK)

    # Legend below
    handles = [mpatches.Patch(color=c, label=l) for l, _, c in CATEGORIES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -1.0),
              ncol=2, fontsize=S.SIZE_LEGEND, frameon=False,
              handlelength=1.4, handletextpad=0.4, columnspacing=1.4)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.45)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
