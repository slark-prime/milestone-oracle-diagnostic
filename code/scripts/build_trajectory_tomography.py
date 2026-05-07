"""E5 analysis: state-transition tomography across the RL training trajectory.

Reads:
  data/logs/rl/oracle_panel_16k/qwen3_8b_pre_rl.jsonl       (step 0)
  data/logs/rl/oracle_trajectory/{base,mile}_2k_step_{030,060,090,120,150}.jsonl
  data/logs/rl/oracle_panel_16k/{base,mile}_2k_step_180.jsonl  (step 180)

For each arm, builds:
  - trajectory_states_{arm}.json: per-checkpoint state counts + ceiling
  - trajectory_transitions_{arm}.json: 4x4 state transition matrix
    pre-RL -> step_180 (and per-step pairwise)
  - figure: docs/latex/figures/trajectory_tomography.pdf
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

REPO = Path(__file__).resolve().parent.parent
PANEL_16K = REPO / "data/logs/rl/oracle_panel_16k"
TRAJ_DIR = REPO / "data/logs/rl/oracle_trajectory"
OUT_FIG = REPO / "docs/latex/figures/trajectory_tomography.pdf"
OUT_JSON = REPO / "data/logs/rl/trajectory_states.json"

if not TRAJ_DIR.exists():
    sys.exit(
        f"ERROR: required input directory {TRAJ_DIR} not found.\n"
        "This script needs the per-checkpoint oracle_trajectory/ directory "
        "(intermediate Tinker checkpoints at steps 30/60/90/120/150 for both "
        "RL arms), which is not packaged with the public release. The "
        "precomputed trajectory_tomography.pdf is included under "
        "code/docs/latex/figures/. See README for which scripts run end-to-end "
        "on this release."
    )

STEPS = [0, 30, 60, 90, 120, 150, 180]
ARMS = {
    "OutcomeRL-2K": {
        0: PANEL_16K / "qwen3_8b_pre_rl.jsonl",
        30: TRAJ_DIR / "base_2k_step_030.jsonl",
        60: TRAJ_DIR / "base_2k_step_060.jsonl",
        90: TRAJ_DIR / "base_2k_step_090.jsonl",
        120: TRAJ_DIR / "base_2k_step_120.jsonl",
        150: TRAJ_DIR / "base_2k_step_150.jsonl",
        180: PANEL_16K / "base_2k_step_180.jsonl",
    },
    "MilestoneRL-2K": {
        0: PANEL_16K / "qwen3_8b_pre_rl.jsonl",
        30: TRAJ_DIR / "mile_2k_step_030.jsonl",
        60: TRAJ_DIR / "mile_2k_step_060.jsonl",
        90: TRAJ_DIR / "mile_2k_step_090.jsonl",
        120: TRAJ_DIR / "mile_2k_step_120.jsonl",
        150: TRAJ_DIR / "mile_2k_step_150.jsonl",
        180: PANEL_16K / "mile_2k_step_180.jsonl",
    },
}

STATES = ["DIRECT", "ROADMAP_NEEDED", "ANSWERS_NEEDED", "UNRECOVERED"]
STATE_LABELS = {
    "DIRECT": "Direct",
    "ROADMAP_NEEDED": "Roadmap-Needed",
    "ANSWERS_NEEDED": "Answers-Needed",
    "UNRECOVERED": "Unrecovered",
}
COLORS = {
    "DIRECT": "#2c7bb6",
    "ROADMAP_NEEDED": "#abd9e9",
    "ANSWERS_NEEDED": "#fdae61",
    "UNRECOVERED": "#d7191c",
}


def load_states(fn):
    """Return {pid: state} from one panel jsonl."""
    if not fn.exists():
        return None
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
            states[pid] = "ROADMAP_NEEDED"
        elif pid in solves["C3_gold_answers"]:
            states[pid] = "ANSWERS_NEEDED"
        else:
            states[pid] = "UNRECOVERED"
    return states


def main():
    arm_data = {}
    for arm, step_files in ARMS.items():
        arm_data[arm] = {}
        for step, fn in step_files.items():
            states = load_states(fn)
            if states is None:
                print(f"[{arm} step {step}] missing: {fn}")
                continue
            counts = {s: 0 for s in STATES}
            for s in states.values():
                counts[s] += 1
            arm_data[arm][step] = {"states": states, "counts": counts}
            ceil = counts["DIRECT"] + counts["ROADMAP_NEEDED"] + counts["ANSWERS_NEEDED"]
            print(f"  [{arm} step {step:3d}] direct={counts['DIRECT']:3d} roadmap={counts['ROADMAP_NEEDED']:3d} answers={counts['ANSWERS_NEEDED']:3d} unrec={counts['UNRECOVERED']:3d} ceil={ceil}")

    # Save state counts
    out = {}
    for arm in arm_data:
        out[arm] = {step: arm_data[arm][step]["counts"] for step in arm_data[arm]}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # ===== Figure: trajectory of state composition across training =====
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharey=True)
    for ax, arm in zip(axes, arm_data):
        steps_avail = sorted(arm_data[arm].keys())
        if not steps_avail:
            continue
        # Stacked area by state across steps
        x = np.array(steps_avail)
        bottoms = np.zeros(len(x))
        for state in STATES:
            vals = np.array([arm_data[arm][s]["counts"][state] for s in steps_avail])
            ax.fill_between(x, bottoms, bottoms + vals, color=COLORS[state],
                            edgecolor="white", linewidth=0.5, label=STATE_LABELS[state])
            bottoms += vals
        ax.set_title(arm, fontsize=11)
        ax.set_xlabel("Training step", fontsize=10)
        ax.set_xticks(STEPS)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 354)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Ceiling annotation
        for s in steps_avail:
            c = arm_data[arm][s]["counts"]
            ceil = c["DIRECT"] + c["ROADMAP_NEEDED"] + c["ANSWERS_NEEDED"]
            ax.annotate(f"{ceil}", xy=(s, ceil), xytext=(0, 3),
                        textcoords="offset points", ha="center",
                        fontsize=7, color="#555555")
    axes[0].set_ylabel("Families (out of 354)", fontsize=10)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.suptitle("Trajectory tomography: state composition across the RL training trajectory (16K, K=8)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
