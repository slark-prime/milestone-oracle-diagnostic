#!/usr/bin/env python3
"""Split milestone-trace attainment by whether the rollout solved the parent.

If milestones are real waypoints, rollouts that reach the correct final answer
should pass through them at a much higher rate than failed rollouts, and both
should beat the coincidence rate (same trace, another family's gold values).
Pure offline analysis over the released format-audit C1 traces; the parent
verdict per rollout is recomputed with the same deterministic verifier.
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

from decomposer.verifier.verifier import VerifierModule
from milestone_trace_alignment import present_in_trace, FAM_FN, AUDIT_DIR

SEED = 42
OUT_FN = ROOT / "data/logs/rl/trace_attainment_by_correctness.json"


def main():
    fams = {json.loads(l)["pid"]: json.loads(l) for l in open(FAM_FN)}
    all_pids = sorted(fams)
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    rng = random.Random(SEED)

    results = {}
    for audit_f in sorted(AUDIT_DIR.glob("*.jsonl")):
        slug = audit_f.stem
        g = {"correct": collections.Counter(), "failed": collections.Counter()}
        for line in open(audit_f):
            r = json.loads(line)
            if r["condition"] != "C1_direct":
                continue
            fam = fams.get(r["pid"])
            if not fam:
                continue
            resp = r["response"] or ""
            ok = v.verify(response=resp, answer=fam["parent_answer"],
                          note=fam.get("parent_note", ""))["label"] == "ACCEPT"
            grp = g["correct" if ok else "failed"]
            other = fams[rng.choice([p for p in all_pids if p != r["pid"]])]
            for m in fam["milestones"]:
                grp["m_hit"] += int(present_in_trace(resp, str(m.get("answer", ""))))
                grp["m_tot"] += 1
            for m in other["milestones"]:
                grp["c_hit"] += int(present_in_trace(resp, str(m.get("answer", ""))))
                grp["c_tot"] += 1
            grp["rollouts"] += 1

        row = {}
        for k, c in g.items():
            row[k] = {"rollouts": c["rollouts"],
                      "attainment": round(c["m_hit"] / c["m_tot"], 4) if c["m_tot"] else None,
                      "coincidence": round(c["c_hit"] / c["c_tot"], 4) if c["c_tot"] else None,
                      "m": f'{c["m_hit"]}/{c["m_tot"]}', "c": f'{c["c_hit"]}/{c["c_tot"]}'}
        results[slug] = row
        cr = row["correct"]; fr = row["failed"]
        print(f"{slug:26s} correct: n={cr['rollouts']:4d} att={cr['attainment']} (coinc {cr['coincidence']})"
              f" | failed: n={fr['rollouts']:4d} att={fr['attainment']} (coinc {fr['coincidence']})")

    json.dump(results, open(OUT_FN, "w"), indent=2)
    print(f"\nwritten: {OUT_FN}")


if __name__ == "__main__":
    main()
