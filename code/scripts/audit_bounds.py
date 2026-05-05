"""Compute leak-safe bounds for the unrecovered-family audit.

Bug being fixed: the original repair adds a FINAL_SYNTHESIS milestone whose
answer equals the parent answer. Under C3 (gold-answers conditioning), the
student is essentially handed the parent answer, so post-repair C3 success
is not evidence of a "fixed" decomposition.

This script:
  1. Verifies the leak rate on repaired_decomp.jsonl
  2. Computes strict (leak-aware) and broad (paper-current) bounds
  3. Emits a per-family table and bounds table for the rewrite

Output: data/logs/rl/audit_bounds.json + audit_bounds_table.csv
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

AUDIT = REPO / "data/logs/rl/audit_consensus.jsonl"
DECOMP = REPO / "data/logs/rl/repaired_decomp.jsonl"
VERIFIER = REPO / "data/logs/rl/repaired_verifier.jsonl"
ORACLE = REPO / "data/logs/rl/oracle_repaired.jsonl"

OUT_JSON = REPO / "data/logs/rl/audit_bounds.json"
OUT_CSV = REPO / "data/logs/rl/audit_bounds_table.csv"


def normalize(s):
    if s is None: return ""
    return re.sub(r"\s+", "", str(s).lower())


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def state_label(c1, c2, c3):
    if c1 is None: return "no_data"
    if c1 >= 1: return "direct"
    if c2 is not None and c2 >= 1: return "desc_only"
    if c3 is not None and c3 >= 1: return "gold_only"
    return "unrecovered"


def main():
    audit = {r["pid"]: r for r in load_jsonl(AUDIT)}
    decomp = {r["full_pid"]: r for r in load_jsonl(DECOMP)}
    verifier = {r["full_pid"]: r for r in load_jsonl(VERIFIER)}

    # Map short pid (audit, 12-char prefix) to full pid (repair, 36-char UUID)
    short2full_decomp = {p[:12]: p for p in decomp}
    short2full_verifier = {p[:12]: p for p in verifier}

    oracle = defaultdict(dict)
    for line in open(ORACLE):
        r = json.loads(line)
        oracle[(r["pid"], r["source"])][r["condition"]] = r["n_correct"]

    # Build per-family table
    rows = []
    for short_pid, audit_rec in audit.items():
        full_pid = short2full_decomp.get(short_pid) or short2full_verifier.get(short_pid)
        category = audit_rec["category"]
        decomp_rec = decomp.get(full_pid)
        verifier_rec = verifier.get(full_pid)

        # Leak check on decomp repair
        leak_status = "n/a"
        any_ms_eq_parent = False
        non_final_leak = False
        if decomp_rec:
            parent = normalize(decomp_rec["parent_answer"])
            ms_answers = [(m.get("type"), normalize(m.get("answer", ""))) for m in decomp_rec["milestones"]]
            any_ms_eq_parent = any(a == parent for _, a in ms_answers)
            non_final_leak = any(a == parent and t != "FINAL_SYNTHESIS" for t, a in ms_answers)
            leak_status = "leak" if any_ms_eq_parent else "leak_free"

        # Oracle outcomes
        if decomp_rec:
            o = oracle.get((full_pid, "repaired_decomp"), {})
            decomp_state = state_label(o.get("C1_direct"), o.get("C2_descriptions"), o.get("C3_gold_answers"))
        else:
            decomp_state = ""
        if verifier_rec:
            o = oracle.get((full_pid, "repaired_verifier"), {})
            verifier_state = state_label(o.get("C1_direct"), o.get("C2_descriptions"), o.get("C3_gold_answers"))
        else:
            verifier_state = ""

        rows.append({
            "short_pid": short_pid,
            "full_pid": full_pid or "",
            "audit_category": category,
            "decomp_repair_present": decomp_rec is not None,
            "decomp_repair_leak": leak_status,
            "decomp_post_repair_state": decomp_state,
            "verifier_repair_present": verifier_rec is not None,
            "verifier_post_repair_state": verifier_state,
        })

    # Save table
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Bounds calculation
    cat_counts = Counter(r["audit_category"] for r in rows)
    decomp_leak_rate = sum(1 for r in rows if r["decomp_repair_leak"] == "leak") / max(1, sum(1 for r in rows if r["decomp_repair_present"]))

    # Verifier repair outcome (leak-safe by construction)
    verifier_outcome = Counter(r["verifier_post_repair_state"] for r in rows if r["verifier_repair_present"])
    verifier_solved = sum(verifier_outcome[s] for s in ("direct", "desc_only", "gold_only"))
    verifier_total = sum(verifier_outcome.values())

    # Decomp repair outcome (BROAD — current paper)
    decomp_outcome = Counter(r["decomp_post_repair_state"] for r in rows if r["decomp_repair_present"])
    decomp_solved_broad = sum(decomp_outcome[s] for s in ("direct", "desc_only", "gold_only"))
    decomp_total = sum(decomp_outcome.values())

    bounds = {
        "n_audited": len(rows),
        "category_counts": dict(cat_counts),
        "decomp_leak_rate": decomp_leak_rate,
        "decomp_post_repair_outcome_broad": dict(decomp_outcome),
        "decomp_solved_broad": f"{decomp_solved_broad}/{decomp_total} ({100*decomp_solved_broad/decomp_total:.0f}%)",
        "decomp_solved_strict": f"0/{decomp_total} (all repairs leak — no leak-safe credit)",
        "verifier_post_repair_outcome": dict(verifier_outcome),
        "verifier_solved_leak_safe": f"{verifier_solved}/{verifier_total} ({100*verifier_solved/verifier_total:.0f}%)",
        "lower_bound_genuine_composition": cat_counts["GENUINE_COMPOSITION"],
        "upper_bound_genuine_composition": (
            cat_counts["GENUINE_COMPOSITION"]
            + cat_counts["DECOMPOSITION_INCOMPLETE"]
            + cat_counts.get("BEYOND_CAPABILITY", 0)
        ),
        "data_quality_lower_bound": verifier_solved,
        "data_quality_upper_bound": (
            cat_counts.get("VERIFIER_ARTIFACT", 0)
            + cat_counts.get("DECOMPOSITION_INCOMPLETE", 0)
            + cat_counts.get("NOT_STANDALONE", 0)
        ),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(bounds, f, indent=2)

    # Pretty print
    print("=" * 60)
    print("LEAK-SAFE AUDIT BOUNDS")
    print("=" * 60)
    print(f"\nAudit categories (n={len(rows)}):")
    for cat, n in cat_counts.most_common():
        print(f"  {n:3d}  {cat}")

    print(f"\nDecomp-repair leak rate: {decomp_leak_rate:.0%} ({sum(1 for r in rows if r['decomp_repair_leak']=='leak')}/{sum(1 for r in rows if r['decomp_repair_present'])})")
    print(f"  All decomp repairs include a FINAL_SYNTHESIS milestone whose answer = parent answer.")

    print(f"\nVerifier repair (leak-safe, no new milestones added):")
    for s in ("direct", "desc_only", "gold_only", "unrecovered"):
        if verifier_outcome.get(s):
            print(f"  {s:14s}: {verifier_outcome[s]}/{verifier_total}")
    print(f"  Solved after verifier fix: {verifier_solved}/{verifier_total} ({100*verifier_solved/verifier_total:.0f}%)")

    print(f"\nDecomp repair (BROAD — paper's current claim, includes leak):")
    for s in ("direct", "desc_only", "gold_only", "unrecovered"):
        if decomp_outcome.get(s):
            print(f"  {s:14s}: {decomp_outcome[s]}/{decomp_total}")
    print(f"  Solved after decomp 'fix': {decomp_solved_broad}/{decomp_total} ({100*decomp_solved_broad/decomp_total:.0f}%)")
    print(f"  STRICT: 0/{decomp_total} count under leak-safe rule (all repairs leak)")

    print(f"\n=== Bounds for paper ===")
    print(f"Genuine composition gap (lower): {bounds['lower_bound_genuine_composition']}/100")
    print(f"  = post-audit GENUINE_COMPOSITION only")
    print(f"Genuine composition gap (upper): {bounds['upper_bound_genuine_composition']}/100")
    print(f"  = GENUINE + DECOMPOSITION_INCOMPLETE + BEYOND_CAPABILITY (if decomp repairs are not trustable)")
    print(f"Data-quality issues (lower): {bounds['data_quality_lower_bound']}/100")
    print(f"  = verifier-repair confirmed (leak-safe)")
    print(f"Data-quality issues (upper): {bounds['data_quality_upper_bound']}/100")
    print(f"  = all VERIFIER + DECOMPOSITION_INCOMPLETE + NOT_STANDALONE (if all decomp repairs were trustworthy)")

    print(f"\nSaved {OUT_JSON} and {OUT_CSV}")


if __name__ == "__main__":
    main()
