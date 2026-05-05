"""Unified figure style for all paper figures.

Import this module from every figure-build script. Edits here propagate
to the entire paper. Designed for skimming reviewers — colorblind-safe,
consistent vocabulary, minimal decoration.
"""
from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Colors — colorblind-safe, semantic, fixed across every figure.
# ---------------------------------------------------------------------------

# Parent-probe outcomes (Figures 1, 4, 5; Table 1 if colored)
COLOR_DIRECT = "#6B7280"           # gray — baseline, no help
COLOR_ROADMAP = "#3B82F6"          # blue — C2 / roadmap
COLOR_ANSWERS = "#8B5CF6"          # purple — C3 / milestone answers
COLOR_UNRECOVERED = "#D1D5DB"      # light gray — not solved

# Reasoning-gap categories (Figure 4)
COLOR_COMPOSITION = "#F97316"         # orange — the headline finding
COLOR_CAPABILITY = "#B91C1C"       # red-brown — hard failure
COLOR_MISSING_MS = "#D97706"       # yellow-brown — partial capability issue
COLOR_ROADMAP_GAP = "#3B82F6"      # blue — same as roadmap probe
COLOR_MS_EXEC_GAP = "#8B5CF6"      # purple — same as answers probe

# Controls (Figure 2)
COLOR_CONTROL = "#FCA5A5"          # light red — corruption controls
COLOR_CONTROL_HATCH = "////"       # hatch pattern for controls when single-color

# Two-arm RL (Figure 5) — aligned with C2 / C3 hues so the deck reuses one palette
COLOR_OUTCOME_RL = COLOR_ROADMAP    # blue, same as Roadmap (outcome-only RL ≈ "C2-direction")
COLOR_MILESTONE_RL = COLOR_ANSWERS  # purple, same as Answers (milestone RL ≈ "C3-direction")

# Training-paradigm colors for Figure 3 (model lookup) — reuse Tier-1 hues
COLOR_PARADIGM_BASE = COLOR_DIRECT   # gray:    base / no RL / instruction-tuned
COLOR_PARADIGM_OUTCOME = COLOR_ROADMAP   # blue:   outcome-only RL
COLOR_PARADIGM_MILESTONE = COLOR_ANSWERS # purple: milestone-augmented RL / reasoning-trained

# Audit triage (Figure 6)
COLOR_AUDIT_ARTIFACT = "#9CA3AF"   # gray — data-quality artifacts
COLOR_AUDIT_STRONG = "#F97316"     # orange — strong-evidence (matches Composition)
COLOR_AUDIT_BEYOND = "#B91C1C"     # red-brown — beyond capability
COLOR_AUDIT_NOTSTANDALONE = "#E5E7EB"  # very light gray — excluded

# Text / divider grays
COLOR_TEXT = "#1F2937"
COLOR_TEXT_MUTED = "#6B7280"
COLOR_DIVIDER = "#E5E7EB"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]

# Sizes (in pt) — designed for full-width papers, post-scaling minimum 6pt
SIZE_TITLE = 9
SIZE_BOX_TITLE = 8
SIZE_BODY = 7
SIZE_ANNOTATION = 6.5
SIZE_TICK = 7
SIZE_LEGEND = 7

# Line widths
LW_MAIN = 0.8
LW_HELPER = 0.5
LW_ARROW = 0.7
LW_BAR_EDGE = 0.6

# Corner radius (in axis units; matplotlib doesn't natively support pt for rect)
RADIUS_BOX = 0.04

# ---------------------------------------------------------------------------
# Bar / scatter conventions
# ---------------------------------------------------------------------------

BAR_EDGE_COLOR = "white"            # break ties between adjacent stacked bars
BAR_WIDTH = 0.62
SCATTER_SIZE = 80
SCATTER_EDGE_LW = 0.6


# ---------------------------------------------------------------------------
# Apply rcParams once on import so plt.subplots picks them up automatically
# ---------------------------------------------------------------------------
def apply_rc():
    """Idempotent: set matplotlib rcParams to the paper-wide defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": SIZE_BODY,
        "axes.titlesize": SIZE_TITLE,
        "axes.labelsize": SIZE_BODY,
        "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "legend.fontsize": SIZE_LEGEND,
        "axes.linewidth": LW_MAIN,
        "axes.edgecolor": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,         # TrueType, embeds correctly in PDF
        "ps.fonttype": 42,
    })


# Run on import so any script that just does `import figure_style` gets the rc set.
apply_rc()


# ---------------------------------------------------------------------------
# Convenience: lookup table for stacked-bar legends
# ---------------------------------------------------------------------------
RECOVERY_STATE_COLOR = {
    "Direct": COLOR_DIRECT,
    "Roadmap-needed": COLOR_ROADMAP,
    "Answers-needed": COLOR_ANSWERS,
    "Unrecovered": COLOR_UNRECOVERED,
}

REASONING_GAP_COLOR = {
    "Direct": COLOR_DIRECT,
    "Roadmap gap": COLOR_ROADMAP_GAP,
    "Milestone-execution gap": COLOR_MS_EXEC_GAP,
    "Composition gap": COLOR_COMPOSITION,
    "Missing-milestone gap": COLOR_MISSING_MS,
    "Capability gap": COLOR_CAPABILITY,
}

# Short model display names (avoid "pre-RL Qwen", "step 180", etc. in figures)
MODEL_DISPLAY = {
    "qwen3_8b_pre_rl": "Qwen-base",
    "base_2k_step_180": "OutcomeRL-2K",
    "mile_2k_step_180": "MilestoneRL-2K",
    "gpt_oss_20b": "gpt-oss-20b",
    "llama_3_3_70b_instruct": "Llama-70B",
    "deepseek_v3_1": "DeepSeek-V3.1",
}
