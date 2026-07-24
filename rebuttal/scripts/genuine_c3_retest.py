#!/usr/bin/env python3
"""Re-test the 18 author-audit GENUINE_COMPOSITION families at C3 with raw-output
capture, so the composition-vs-artifact question can be settled by reading what
the model actually writes when handed all gold milestone answers.

Protocol identical to the 16K oracle panel (same eval set, same build_c3 prompt,
same renderer/model via Tinker, max_tokens 16384); k=4 instead of 8 (hand-grading
budget). Raw text is persisted per rollout.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

from oracle_panel import build_c3  # byte-identical C3 prompt construction

import tinker
from tinker import types as tt
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = os.environ.get("RETEST_MODEL", "openai/gpt-oss-20b")
RENDERER = os.environ.get("RETEST_RENDERER", "role_colon")
K = int(os.environ.get("RETEST_K", "4"))
MAX_TOKENS = 16384
EVAL_SET = ROOT / "data/logs/rl/diagnostic_multi_families_repaired.jsonl"
OUT_FN = ROOT / "data/logs/rl/genuine_retest" / (MODEL.split("/")[-1].lower().replace("-", "_").replace(".", "_") + "_c3.jsonl")

GENUINE_PIDS = [
    "0ad0ce27", "20dfe8b1", "29eb7827", "2e647bfc", "330cbc5a", "40a06b35",
    "5ce4a16b", "60f12f06", "713ff8c1", "723ca360", "a0146340", "a0759695",
    "a3a636a8", "ad098d0c", "c52e7627", "cb760e44", "eeab310c", "ff548f26",
]


def main():
    fams = [json.loads(l) for l in open(EVAL_SET)]
    targets = [f for f in fams if any(f["pid"].startswith(p) for p in GENUINE_PIDS)]
    print(f"target families: {len(targets)}/18", flush=True)

    OUT_FN.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_FN.exists():
        for line in open(OUT_FN):
            try:
                r = json.loads(line)
                done.add((r["pid"], r["rollout"]))
            except Exception:
                pass

    sc = tinker.ServiceClient()
    cli = sc.create_sampling_client(base_model=MODEL)
    tok = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER, tokenizer=tok)

    jobs = []
    for fam in targets:
        prompt_text = build_c3(fam["parent_prompt"], fam["milestones"])
        for ki in range(K):
            if (fam["pid"], ki) in done:
                continue
            jobs.append((fam, ki, prompt_text))
    print(f"rollouts to sample: {len(jobs)}", flush=True)

    def sample(job):
        fam, ki, prompt_text = job
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}])
        params = tt.SamplingParams(
            max_tokens=MAX_TOKENS, temperature=0.7,
            stop=renderer.get_stop_sequences())
        fut = cli.sample(prompt=model_input, num_samples=1, sampling_params=params)
        res = fut.result()
        parsed, _ = renderer.parse_response(res.sequences[0].tokens)
        return parsed["content"]

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=128) as ex:
        futs = {ex.submit(sample, j): j for j in jobs}
        for fut in as_completed(futs):
            fam, ki, _ = futs[fut]
            try:
                content = fut.result()
                row = {"pid": fam["pid"], "rollout": ki, "model": MODEL,
                       "condition": "C3_gold_answers",
                       "parent_answer": fam["parent_answer"],
                       "parent_note": fam.get("parent_note", ""),
                       "response": content}
            except Exception as e:
                row = {"pid": fam["pid"], "rollout": ki,
                       "error": f"{type(e).__name__}: {e}"[:200]}
            with open(OUT_FN, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            print(f"[{n}/{len(jobs)}] {fam['pid'][:8]} r{ki} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
