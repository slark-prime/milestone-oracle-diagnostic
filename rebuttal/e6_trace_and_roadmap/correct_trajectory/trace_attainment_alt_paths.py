#!/usr/bin/env python3
"""Do correct rollouts walk SOME valid path, if not necessarily ours?

On the 39 families with two independent decompositions (GPT-5.4 and Inkling
atomic), split C1 rollouts by parent correctness and measure per-milestone
attainment against each decomposition and rollout-level any-waypoint hits for
their union. Offline; same containment test and verifier as E6.
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
from taxonomy_agreement import alt_families

SEED = 42
OUT_FN = ROOT / "data/logs/rl/trace_attainment_alt_paths.json"


def main():
    fams = {json.loads(l)["pid"]: json.loads(l) for l in open(FAM_FN)}
    alts = {f["pid"]: f for f in alt_families()}
    all_pids = sorted(fams)
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    rng = random.Random(SEED)

    results = {}
    for audit_f in sorted(AUDIT_DIR.glob("*.jsonl")):
        slug = audit_f.stem
        g = {"correct": collections.Counter(), "failed": collections.Counter()}
        for line in open(audit_f):
            r = json.loads(line)
            if r["condition"] != "C1_direct" or r["pid"] not in alts:
                continue
            fam, alt = fams[r["pid"]], alts[r["pid"]]
            resp = r["response"] or ""
            ok = v.verify(response=resp, answer=fam["parent_answer"],
                          note=fam.get("parent_note", ""))["label"] == "ACCEPT"
            grp = g["correct" if ok else "failed"]
            o1, o2 = rng.sample([p for p in all_pids if p != r["pid"]], 2)
            other = fams[o1]
            other2 = fams[o2]
            own_hits = [present_in_trace(resp, str(m.get("answer", ""))) for m in fam["milestones"]]
            alt_hits = [present_in_trace(resp, str(m.get("answer", ""))) for m in alt["milestones"]]
            ctl_hits = [present_in_trace(resp, str(m.get("answer", ""))) for m in other["milestones"]]
            ctl2_hits = ctl_hits + [present_in_trace(resp, str(m.get("answer", ""))) for m in other2["milestones"]]
            grp["own_h"] += sum(own_hits); grp["own_t"] += len(own_hits)
            grp["alt_h"] += sum(alt_hits); grp["alt_t"] += len(alt_hits)
            grp["ctl_h"] += sum(ctl_hits); grp["ctl_t"] += len(ctl_hits)
            grp["union_any"] += int(any(own_hits) or any(alt_hits))
            grp["ctl_any"] += int(any(ctl2_hits))
            grp["rollouts"] += 1

        row = {}
        for k, c in g.items():
            n = c["rollouts"]
            row[k] = {"rollouts": n,
                      "own": round(c["own_h"]/c["own_t"], 3) if c["own_t"] else None,
                      "alt": round(c["alt_h"]/c["alt_t"], 3) if c["alt_t"] else None,
                      "ctl": round(c["ctl_h"]/c["ctl_t"], 3) if c["ctl_t"] else None,
                      "union_any_rate": round(c["union_any"]/n, 3) if n else None,
                      "ctl_any_rate": round(c["ctl_any"]/n, 3) if n else None}
        results[slug] = row
        cr, fr = row["correct"], row["failed"]
        print(f"{slug:24s} correct n={cr['rollouts']:3} own={cr['own']} alt={cr['alt']} ctl={cr['ctl']} anyValid={cr['union_any_rate']} anyCtl={cr['ctl_any_rate']}")
        print(f"{'':24s} failed  n={fr['rollouts']:3} own={fr['own']} alt={fr['alt']} ctl={fr['ctl']} anyValid={fr['union_any_rate']} anyCtl={fr['ctl_any_rate']}")

    json.dump(results, open(OUT_FN, "w"), indent=2)
    print(f"\nwritten: {OUT_FN}")


if __name__ == "__main__":
    main()
