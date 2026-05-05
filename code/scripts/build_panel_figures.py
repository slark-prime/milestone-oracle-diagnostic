"""Build cross-model fingerprint figure from panel data.

Figure: horizontal stacked bars, one per model, segments = Direct / Roadmap-Needed
/ Answers-Needed / Unrecovered. Numbers annotated on each segment >=5%.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "data/logs/rl/panel_fingerprints.json"
OUT = REPO / "docs/latex/figures/panel_fingerprints.pdf"


def main():
    fp = json.load(open(IN))

    # Row order: strongest to weakest by direct-solve, but keep Qwen lineage together
    model_order = [
        ("qwen3_8b_pre_rl",   "Qwen3-8B-Base (pre-RL, 8B)"),
        ("base_2k_step_180",  "Base-2k (Qwen3-8B + RL, 8B)"),
        ("mile_2k_step_180",  "Mile-2k (+ milestone RL, 8B)"),
        ("gpt_oss_20b",       "gpt-oss-20b (OpenAI, 20B)"),
        ("llama_3_3_70b_instruct", "Llama-3.3-70B-Instruct (Meta, 70B)"),
        ("deepseek_v3_1",     "DeepSeek-V3.1 (DeepSeek, 671B MoE)"),
    ]

    state_order = ["direct", "desc_only", "gold_only", "unrecovered"]
    state_labels = ["DIRECT", "ROADMAP-ONLY", "ANSWERS-ONLY", "UNRECOVERED"]
    colors = {
        "direct":      "#2a8a3e",   # green
        "desc_only":   "#5dade2",   # blue
        "gold_only":   "#f5b041",   # amber
        "unrecovered": "#a93226",   # red
    }

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    y_positions = list(range(len(model_order)))[::-1]

    for y_pos, (slug, label) in zip(y_positions, model_order):
        data = fp[slug]
        total = data["total"]
        left = 0
        for state in state_order:
            val = data[state]
            pct = val / total
            ax.barh(y_pos, val, left=left, color=colors[state], edgecolor="white", linewidth=0.5)
            if pct >= 0.05:
                ax.text(left + val / 2, y_pos, f"{val}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
            left += val

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in model_order], fontsize=9)
    ax.set_xlim(0, 354)
    ax.set_xlabel("Families on the $n{=}354$ diagnostic set")
    ax.set_title("Cross-model failure fingerprints ($K{=}8$ rollouts, same oracle every model)", fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[s], edgecolor="white", label=l)
               for s, l in zip(state_order, state_labels)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
