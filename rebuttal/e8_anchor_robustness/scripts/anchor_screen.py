#!/usr/bin/env python3
"""Full anchor screen on math500+aime with a non-Qwen anchor.
Same protocol as second_corpus_suite stage_screen (k=4, 16K, same prompts and
verifier); only the anchor model changes. Output goes to a separate directory so
the original Qwen screen files stay untouched."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

import second_corpus_suite as s

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
SLUG = MODEL.split("/")[-1].replace(".", "_")
s.ANCHOR = (MODEL, "role_colon")

_orig_paths = s.paths
def paths(corpus):
    d = ROOT / "data/logs/rl/second_corpus/anchor_screens" / SLUG / corpus
    d.mkdir(parents=True, exist_ok=True)
    p = _orig_paths(corpus)
    p["screen"] = d / "c1_screen.jsonl"
    return p
s.paths = paths

for corpus in ("math500", "aime"):
    print(f"=== {MODEL} screen on {corpus} ===", flush=True)
    s.stage_screen(corpus, None)
    solved = defaultdict(int)
    for l in open(paths(corpus)["screen"]):
        d = json.loads(l)
        if "error" not in d and d["accept"]:
            solved[d["pid"]] += 1
    rows = s.load_corpus(corpus, None)
    fails = [r["train_id"] for r in rows if solved[r["train_id"]] == 0]
    print(f"[{corpus}] anchor-failed: {len(fails)}/{len(rows)} ({len(fails)/len(rows):.0%})", flush=True)
