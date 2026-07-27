#!/usr/bin/env python3
"""Second-anchor intersection analysis.

Families the ORIGINAL anchor (Qwen3-8B) failed were decomposed and probed in the
second-corpus suite. Here we take the subset of those families that a SECOND
anchor (gpt-oss-20b) also fails, and recompute the six-condition recovery table
on that doubly-anchor-failed subset, reusing the existing probe rollouts. No new
sampling and no teacher calls.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

import second_corpus_suite as s

SLUG = "gpt-oss-20b"
CONDS = ["C1_direct", "C2_descriptions", "C2_random", "C2_generic",
         "C3_gold_answers", "C3_mismatched"]


def anchor_failed(corpus):
    fn = ROOT / "data/logs/rl/second_corpus/anchor_screens" / SLUG / corpus / "c1_screen.jsonl"
    solved, seen = defaultdict(int), defaultdict(int)
    done = set()
    for l in open(fn):
        d = json.loads(l)
        if "error" in d or tuple(d["key"]) in done:
            continue
        done.add(tuple(d["key"]))
        seen[d["pid"]] += 1
        if d["accept"]:
            solved[d["pid"]] += 1
    rows = s.load_corpus(corpus, None)
    return {r["train_id"] for r in rows
            if seen[r["train_id"]] >= 1 and solved[r["train_id"]] == 0}


def main():
    out = {}
    for corpus, thresh in (("math500", 3), ("aime", 1)):
        fails2 = anchor_failed(corpus)
        fams = s.eligible_families(corpus)
        probed = {f["pid"] for f in fams}
        inter = probed & fails2
        pr = defaultdict(int)
        for l in open(s.paths(corpus)["probes"]):
            d = json.loads(l)
            if "error" not in d and d["accept"]:
                pr[(d["pid"], d["condition"])] += 1
        rec = {c: sum(1 for p in inter if pr[(p, c)] >= thresh) for c in CONDS}
        rec1 = {c: sum(1 for p in inter if pr[(p, c)] >= 1) for c in CONDS}
        out[corpus] = {"second_anchor": SLUG, "threshold": f">={thresh}/8",
                       "probed_families": len(probed),
                       "second_anchor_failed_of_probed": len(inter),
                       "recovery_at_threshold": rec,
                       "recovery_at_1of8": rec1}
        print(f"[{corpus}] probed {len(probed)}, also-failed-by-{SLUG}: {len(inter)}")
        print(f"  recovery @>= {thresh}/8: " + ", ".join(f"{c.split('_')[0]}{'-'+c.split('_')[1] if len(c.split('_'))>1 else ''}={rec[c]}" for c in CONDS))
        print(f"  recovery @>=1/8:    " + ", ".join(f"{c}={rec1[c]}" for c in CONDS))
    fn = ROOT / "data/logs/rl/second_corpus/anchor_screens" / SLUG / "intersection_report.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"\nwritten: {fn}")


if __name__ == "__main__":
    main()
