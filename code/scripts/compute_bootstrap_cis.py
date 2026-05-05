"""Bootstrap 95% CIs for oracle rates and transition counts.

1000 family-level resamples with replacement. Reports solved-count CIs for
each condition on main (n=354) and held-out (n=32) sets, plus CIs for
Base->Mile transition flows.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(0)

REPO = Path(__file__).resolve().parent.parent


def load_oracle(path):
    out = defaultdict(lambda: defaultdict(dict))
    for line in open(path):
        r = json.loads(line)
        out[r["checkpoint"]][r["condition"]][r["pid"]] = r["n_correct"]
    return out


def load_manifest_354():
    return [json.loads(l) for l in open(REPO / "data/logs/rl/manifest.jsonl")
            if json.loads(l)["in_354_multi_milestone"]]


def percentile(values, p):
    """Return the p-th percentile of a sorted list (p in [0, 100])."""
    values_sorted = sorted(values)
    idx = max(0, min(len(values_sorted) - 1, int(round(p / 100 * (len(values_sorted) - 1)))))
    return values_sorted[idx]


def bootstrap_solved(n_correct_by_pid, pids, K=8, B=1000):
    """Return (mean, 95% CI low, 95% CI high) for count-of-solved families."""
    counts = []
    srs = []
    n = len(pids)
    for _ in range(B):
        samp = [random.choice(pids) for _ in range(n)]
        solved = sum(1 for pid in samp if n_correct_by_pid.get(pid, 0) >= 1)
        correct = sum(n_correct_by_pid.get(pid, 0) for pid in samp)
        counts.append(solved)
        srs.append(correct / (n * K))
    return {
        "solved_mean": sum(counts) / B,
        "solved_ci": (percentile(counts, 2.5), percentile(counts, 97.5)),
        "sr_mean": sum(srs) / B,
        "sr_ci": (percentile(srs, 2.5), percentile(srs, 97.5)),
    }


def bootstrap_transition(manifest_354, src_state, dst_state, B=1000):
    """Bootstrap a family-level transition count."""
    counts = []
    pids = [r["pid"] for r in manifest_354]
    lookup = {r["pid"]: (r["base_state"], r["mile_state"]) for r in manifest_354}
    for _ in range(B):
        samp = [random.choice(pids) for _ in range(len(pids))]
        n = sum(1 for pid in samp if lookup[pid] == (src_state, dst_state))
        counts.append(n)
    return {
        "mean": sum(counts) / B,
        "ci": (percentile(counts, 2.5), percentile(counts, 97.5)),
    }


def main():
    # Main set
    print("=" * 60)
    print("BOOTSTRAP CIs — main set (n=354, Mile-2k step 180, K=8)")
    print("=" * 60)
    oracle_main = load_oracle(REPO / "data/logs/rl/oracle_canonical.jsonl")
    mile_main = oracle_main["mile_2k_step_180"]
    base_main = oracle_main["base_2k_step_180"]

    for cond in ["C1_direct", "C2_random", "C2_generic", "C2_descriptions",
                 "C3_mismatched", "C3_gold_answers"]:
        pids = list(mile_main[cond].keys())
        res = bootstrap_solved(mile_main[cond], pids)
        print(f"  Mile  {cond:22s}  solved={res['solved_mean']:5.1f} [{res['solved_ci'][0]}, {res['solved_ci'][1]}]   sr={res['sr_mean']:.3f} [{res['sr_ci'][0]:.3f}, {res['sr_ci'][1]:.3f}]")
        if cond in base_main:
            res_b = bootstrap_solved(base_main[cond], pids)
            print(f"  Base  {cond:22s}  solved={res_b['solved_mean']:5.1f} [{res_b['solved_ci'][0]}, {res_b['solved_ci'][1]}]   sr={res_b['sr_mean']:.3f} [{res_b['sr_ci'][0]:.3f}, {res_b['sr_ci'][1]:.3f}]")

    # Held-out
    print()
    print("=" * 60)
    print("BOOTSTRAP CIs — held-out (n=32, Mile-2k step 180, K=8)")
    print("=" * 60)
    oracle_ho = load_oracle(REPO / "data/logs/rl/oracle_held_out.jsonl")
    mile_ho = oracle_ho["mile_2k_step_180"]
    for cond in ["C1_direct", "C2_random", "C2_generic", "C2_descriptions",
                 "C3_mismatched", "C3_gold_answers"]:
        pids = list(mile_ho[cond].keys())
        res = bootstrap_solved(mile_ho[cond], pids)
        print(f"  {cond:22s}  solved={res['solved_mean']:5.1f} [{res['solved_ci'][0]}, {res['solved_ci'][1]}]   sr={res['sr_mean']:.3f} [{res['sr_ci'][0]:.3f}, {res['sr_ci'][1]:.3f}]")

    # Transition counts
    print()
    print("=" * 60)
    print("BOOTSTRAP CIs — transition counts (n=354)")
    print("=" * 60)
    manifest_354 = load_manifest_354()
    key_cells = [
        ("desc_only", "direct", "Description -> direct (absorption)"),
        ("gold_only", "direct", "Answer -> direct"),
        ("unrecovered", "direct", "Unrecovered -> direct"),
        ("direct", "unrecovered", "Direct -> unrecovered (hard regression)"),
        ("direct", "desc_only", "Direct -> desc-only (soft regression)"),
    ]
    for src, dst, label in key_cells:
        r = bootstrap_transition(manifest_354, src, dst)
        print(f"  {label:<45s}  mean={r['mean']:5.1f}  95% CI=[{r['ci'][0]}, {r['ci'][1]}]")


if __name__ == "__main__":
    main()
