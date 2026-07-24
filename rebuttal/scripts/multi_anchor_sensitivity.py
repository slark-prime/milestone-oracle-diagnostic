"""Multi-anchor sensitivity: is the cross-model comparison Qwen-flavored?

Reviewer concern. The 354-family diagnostic set is selected by anchoring on
Qwen-base — i.e. families where Qwen-base passes Stage 0 and fails the parent.
A reviewer worries that the cross-model failure-shape comparison may therefore
favor narratives that look reasonable when read through Qwen-base's filter,
and might invert under a different anchor.

What this script does. For each of the 6 panel models taken as ANCHOR M:
  1. Restrict to the M-anchored subset of the 354 set:
       M passes Stage 0 (all non-INTEGRATE milestones n_correct >= τ)
       AND
       M's C1_direct == 0 (model fails the parent directly)
  2. Compute the per-probing-model bucket counts on this subset:
       Direct, Roadmap-Needed, Answers-Needed, Unrecovered (composition gap)
  3. Report whether the cross-model ranking of "composition-gap count" stays
     consistent across choice of anchor.

If the rank order of probers by composition-gap count is the same (or close)
across all 6 anchor choices, the cross-model comparison is anchor-invariant
in the sense that matters for the paper's claim.

Data sources (all existing, no new compute):
  - data/logs/rl/diagnostic_multi_families_repaired.jsonl  (354 families)
  - data/logs/rl/oracle_panel_16k/<model>.jsonl            (parent probes per model)
  - data/logs/rl/stage0_panel/<model>.jsonl                (Stage 0, 4K — main)
  - data/logs/rl/stage0_panel_16k/<model>.jsonl            (Stage 0, 16K — if available)

Usage:
  python scripts/experiments/multi_anchor_sensitivity.py
  python scripts/experiments/multi_anchor_sensitivity.py --stage0-dir stage0_panel_16k
  python scripts/experiments/multi_anchor_sensitivity.py --tau 1   # default
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("data/logs/rl")
EVAL_SET = ROOT / "diagnostic_multi_families_repaired.jsonl"
PANEL_DIR = ROOT / "oracle_panel_16k"

MODELS = [
    "qwen3_8b_pre_rl",
    "base_2k_step_180",
    "mile_2k_step_180",
    "gpt_oss_20b",
    "llama_3_3_70b_instruct",
    "deepseek_v3_1",
]
LABELS = {
    "qwen3_8b_pre_rl":          "Qwen-base",
    "base_2k_step_180":         "OutcomeRL-2K",
    "mile_2k_step_180":         "MilestoneRL-2K",
    "gpt_oss_20b":              "gpt-oss-20b",
    "llama_3_3_70b_instruct":   "Llama-70B",
    "deepseek_v3_1":            "DeepSeek-V3.1",
}


def load_stage0(slug, dir_name):
    fn = ROOT / dir_name / f"{slug}.jsonl"
    if not fn.exists():
        return None
    out = defaultdict(dict)
    with open(fn) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ms_type") == "INTEGRATE":
                continue
            out[r["pid"]][r["ms_idx"]] = r["n_correct"]
    return dict(out)


def load_panel(slug):
    fn = PANEL_DIR / f"{slug}.jsonl"
    out = defaultdict(dict)
    with open(fn) as f:
        for line in f:
            r = json.loads(line)
            out[r["pid"]][r["condition"]] = r["n_correct"]
    return dict(out)


def stage0_pass(per_ms_counts, tau):
    if not per_ms_counts:
        return False
    return all(n >= tau for n in per_ms_counts.values())


def parent_outcome(probes, tau=1):
    """Smallest help level needed."""
    if probes.get("C1_direct", 0) >= tau:
        return "Direct"
    if probes.get("C2_descriptions", 0) >= tau:
        return "Roadmap-Needed"
    if probes.get("C3_gold_answers", 0) >= tau:
        return "Answers-Needed"
    return "Unrecovered"


def kendall_tau(rank_a, rank_b):
    """Kendall's tau between two rank lists of the same items."""
    items = list(rank_a.keys())
    n = len(items)
    concord = 0
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i, a_j = rank_a[items[i]], rank_a[items[j]]
            b_i, b_j = rank_b[items[i]], rank_b[items[j]]
            sign_a = (a_i > a_j) - (a_i < a_j)
            sign_b = (b_i > b_j) - (b_i < b_j)
            if sign_a == 0 or sign_b == 0:
                continue
            if sign_a == sign_b:
                concord += 1
            else:
                discord += 1
    total = concord + discord
    return (concord - discord) / total if total else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0-dir", default="stage0_panel",
                    help="stage0_panel (4K, default) or stage0_panel_16k")
    ap.add_argument("--tau", type=int, default=1, help="Stage 0 / parent threshold")
    args = ap.parse_args()

    # Load eval set just to know the universe
    pids_354 = [json.loads(l)["pid"] for l in open(EVAL_SET)]
    print(f"Universe: {len(pids_354)} families from {EVAL_SET}")
    print(f"Stage 0 source: {args.stage0_dir}, threshold τ={args.tau}\n")

    s0 = {}
    panel = {}
    for slug in MODELS:
        s0[slug] = load_stage0(slug, args.stage0_dir)
        if s0[slug] is None:
            print(f"  [{slug}] missing {args.stage0_dir} — skip"); s0[slug] = {}
        panel[slug] = load_panel(slug)

    # For each anchor M, derive the M-anchored subset within the 354
    print(f"{'anchor':25s}  pass_stage0  fail_parent_C1  anchored_subset")
    anchored_sets = {}
    for M in MODELS:
        subset = []
        n_passS0 = 0
        n_failC1 = 0
        for pid in pids_354:
            ms = s0[M].get(pid, {})
            passes_s0 = stage0_pass(ms, args.tau)
            fails_c1 = panel[M].get(pid, {}).get("C1_direct", 0) < args.tau
            if passes_s0:
                n_passS0 += 1
            if fails_c1:
                n_failC1 += 1
            if passes_s0 and fails_c1:
                subset.append(pid)
        anchored_sets[M] = subset
        print(f"{LABELS[M]:25s}  {n_passS0:11d}  {n_failC1:14d}  {len(subset):15d}")

    # For each anchor M's subset, compute per-prober composition-gap counts.
    # Only count probers that are NOT M (otherwise the prober trivially fails
    # parent C1 by construction — that's the anchor screen).
    print(f"\n{'='*70}")
    print(f"COMPOSITION-GAP counts per (anchor, prober). Rows omit prober == anchor.")
    print(f"{'='*70}")
    label_anchor = "anchor / prober"
    header = f"{label_anchor:22s}  " + "  ".join(f"{LABELS[m][:12]:>12s}" for m in MODELS)
    print(header)
    comp_gap_counts = {}  # anchor -> prober -> count
    for M in MODELS:
        comp_gap_counts[M] = {}
        row_vals = []
        for P in MODELS:
            if P == M:
                comp_gap_counts[M][P] = None
                row_vals.append("    ---     ")
                continue
            count = 0
            for pid in anchored_sets[M]:
                # Stage 0 PASS for prober P on this anchored subset?
                passes_s0_P = stage0_pass(s0[P].get(pid, {}), args.tau)
                outc = parent_outcome(panel[P].get(pid, {}), tau=args.tau)
                if passes_s0_P and outc == "Unrecovered":
                    count += 1
            comp_gap_counts[M][P] = count
            row_vals.append(f"{count:>12d}")
        print(f"{LABELS[M]:22s}  " + "  ".join(row_vals))

    # Rank consistency: within each anchor, rank probers by composition-gap count
    # (excluding the anchor itself). Compare ranks across anchors.
    print(f"\n{'='*70}")
    print("RANK STABILITY: rank of each prober (1=most composition gaps), per anchor")
    print(f"{'='*70}")
    rank_per_anchor = {}
    print(f"{'anchor':22s}  " + "  ".join(f"{LABELS[m][:12]:>12s}" for m in MODELS))
    for M in MODELS:
        # Build a rank dict for probers != M
        items = [(P, comp_gap_counts[M][P]) for P in MODELS if P != M]
        # Higher count → lower rank number (rank 1 is the most)
        items.sort(key=lambda kv: kv[1] or -1, reverse=True)
        ranks = {P: i + 1 for i, (P, _) in enumerate(items)}
        rank_per_anchor[M] = ranks
        cells = []
        for P in MODELS:
            if P == M:
                cells.append("    ---     ")
            else:
                cells.append(f"     #{ranks[P]:<2d}     ")
        print(f"{LABELS[M]:22s}  " + "  ".join(cells))

    # Pairwise Kendall's tau across anchor rank lists (using the common probers)
    print(f"\n{'='*70}")
    print("Kendall's tau between anchor rank lists (1.0 = identical, -1.0 = reversed)")
    print("Computed on the 4 probers each pair has in common.")
    print(f"{'='*70}")
    print(f"{'':22s}  " + "  ".join(f"{LABELS[m][:12]:>12s}" for m in MODELS))
    for M1 in MODELS:
        cells = []
        for M2 in MODELS:
            if M1 == M2:
                cells.append("    1.00    ")
                continue
            common = [P for P in MODELS if P != M1 and P != M2]
            r1 = {P: rank_per_anchor[M1][P] for P in common}
            r2 = {P: rank_per_anchor[M2][P] for P in common}
            tau = kendall_tau(r1, r2)
            cells.append(f"   {tau:+.2f}    ")
        print(f"{LABELS[M1]:22s}  " + "  ".join(cells))

    # Headline summary
    print(f"\n{'='*70}\nHEADLINE\n{'='*70}")
    # mean tau (off-diagonal only)
    taus = []
    for M1 in MODELS:
        for M2 in MODELS:
            if M1 >= M2:
                continue
            common = [P for P in MODELS if P != M1 and P != M2]
            r1 = {P: rank_per_anchor[M1][P] for P in common}
            r2 = {P: rank_per_anchor[M2][P] for P in common}
            taus.append(kendall_tau(r1, r2))
    mean_tau = sum(taus) / len(taus) if taus else 0.0
    pos_tau_frac = sum(1 for t in taus if t > 0) / len(taus) if taus else 0.0
    print(f"  Pairwise anchor agreement (Kendall's τ on prober ranks): mean = {mean_tau:+.3f}")
    print(f"  Fraction of anchor-pairs with positive agreement (τ > 0): {pos_tau_frac:.0%}")
    print(f"  → If mean τ is high (>0.5) and fraction > 80%, the cross-model")
    print(f"    failure-shape comparison is anchor-invariant in rank order.")


if __name__ == "__main__":
    main()
