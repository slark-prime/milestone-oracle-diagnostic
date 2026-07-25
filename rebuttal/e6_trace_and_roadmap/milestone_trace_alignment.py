#!/usr/bin/env python3
"""AC-Q2: is the milestone roadmap a faithful representation of reasoning PROGRESS,
or an externally imposed decomposition?

Test: in the student's own UNAIDED (C1) traces, do the teacher's milestone results
actually appear, in order? If a roadmap tracks real reasoning progress, then
(a) milestone results should show up in unaided traces far more often than
    results from a random other family's roadmap (specificity), and
(b) attainment should be prefix-shaped (early milestones reached more often than
    late ones), i.e. traces advance along the roadmap and then stop.

Grading uses the same deterministic cascade, applied to the trace as a whole
(a milestone counts as "attained" if its gold value is verifiably present).
No new sampling: reads the released format-audit C1 traces.
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from decomposer.verifier.verifier import VerifierModule
from decomposer.verifier.math_reward import is_equiv, strip_string

FAM_FN = ROOT / "data/logs/rl/diagnostic_multi_families_repaired.jsonl"
AUDIT_DIR = ROOT / "data/logs/rl/format_audit"
STAGE0_DIR = ROOT / "data/logs/rl/stage0_panel_16k"
OUT_FN = ROOT / "data/logs/rl/milestone_trace_alignment.json"
SEED = 42


def present_in_trace(trace: str, gold: str) -> bool:
    """Deterministic containment test: is the gold value verifiably present?

    Conservative — requires the normalized gold string to occur in the normalized
    trace, or an exact numeric match on one of the trace's LaTeX-ish tokens.
    """
    g = (gold or "").strip()
    if not g or len(g) > 60:
        return False
    t = trace or ""
    if g in t:
        return True
    try:
        gn = strip_string(g)
        tn = strip_string(t)
        if gn and gn in tn:
            return True
    except Exception:
        pass
    # numeric gold: match as a standalone token
    try:
        val = float(g.replace(",", ""))
        for tok in t.replace("$", " ").replace("\\", " ").replace("{", " ").replace("}", " ").split():
            try:
                if abs(float(tok.strip(".,;:()[]")) - val) < 1e-9:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass
    return False


def main():
    fams = {json.loads(l)["pid"]: json.loads(l) for l in open(FAM_FN)}
    rng = random.Random(SEED)
    all_pids = sorted(fams)

    results = {}
    for audit_f in sorted(AUDIT_DIR.glob("*.jsonl")):
        slug = audit_f.stem
        traces = collections.defaultdict(list)
        for line in open(audit_f):
            r = json.loads(line)
            if r["condition"] == "C1_direct":
                traces[r["pid"]].append(r["response"] or "")
        if not traces:
            continue

        # Stage 0 (16K): which milestones can this model solve alone?
        s0 = {}
        s0f = STAGE0_DIR / f"{slug}.jsonl"
        if s0f.exists():
            for line in open(s0f):
                d = json.loads(line)
                s0[(d["pid"], d["ms_idx"])] = d["n_correct"]

        matched = collections.Counter()   # attained | total, by milestone position
        control = collections.Counter()
        prefix_shape = collections.Counter()
        s0_split = {"s0_pass_attained": 0, "s0_pass_total": 0,
                    "s0_fail_attained": 0, "s0_fail_total": 0}

        for pid, resp_list in traces.items():
            fam = fams.get(pid)
            if not fam:
                continue
            ms = fam["milestones"]
            # control: milestone golds drawn from a different random family
            other = fams[rng.choice([p for p in all_pids if p != pid])]
            for resp in resp_list:
                attained = []
                for i, m in enumerate(ms):
                    hit = present_in_trace(resp, str(m.get("answer", "")))
                    attained.append(hit)
                    matched[("attained", i)] += int(hit)
                    matched[("total", i)] += 1
                    n_ok = s0.get((pid, i))
                    if n_ok is not None:
                        k = "s0_pass" if n_ok >= 1 else "s0_fail"
                        s0_split[f"{k}_attained"] += int(hit)
                        s0_split[f"{k}_total"] += 1
                for j, m in enumerate(other["milestones"]):
                    control[("attained", j)] += int(present_in_trace(resp, str(m.get("answer", ""))))
                    control[("total", j)] += 1
                # prefix shape: length of the leading run of attained milestones
                run = 0
                for hit in attained:
                    if not hit:
                        break
                    run += 1
                prefix_shape[f"{run}/{len(ms)}"] += 1

        def rate(c):
            a = sum(v for (k, _), v in c.items() if k == "attained")
            t = sum(v for (k, _), v in c.items() if k == "total")
            return a, t, (a / t if t else 0.0)

        ma, mt, mr = rate(matched)
        ca, ct, cr = rate(control)
        by_pos = {i: (matched[("attained", i)] / matched[("total", i)])
                  for i in range(6) if matched[("total", i)]}
        results[slug] = {
            "families": len(traces),
            "own_roadmap_attained": [ma, mt, round(mr, 4)],
            "control_roadmap_attained": [ca, ct, round(cr, 4)],
            "specificity_ratio": round(mr / cr, 2) if cr else None,
            "attainment_by_position": {k: round(v, 3) for k, v in by_pos.items()},
            "prefix_run_distribution": dict(prefix_shape.most_common()),
            "stage0_conditional": {
                "attained_if_stage0_pass": round(s0_split["s0_pass_attained"] / s0_split["s0_pass_total"], 4)
                if s0_split["s0_pass_total"] else None,
                "attained_if_stage0_fail": round(s0_split["s0_fail_attained"] / s0_split["s0_fail_total"], 4)
                if s0_split["s0_fail_total"] else None,
                "counts": s0_split,
            },
        }
        print(f"{slug}: own {mr:.1%} vs control {cr:.1%} (x{results[slug]['specificity_ratio']}) "
              f"| by-position {results[slug]['attainment_by_position']} "
              f"| S0pass {results[slug]['stage0_conditional']['attained_if_stage0_pass']} "
              f"S0fail {results[slug]['stage0_conditional']['attained_if_stage0_fail']}", flush=True)

    json.dump(results, open(OUT_FN, "w"), indent=2)
    print(f"\nwritten: {OUT_FN}")


if __name__ == "__main__":
    main()
