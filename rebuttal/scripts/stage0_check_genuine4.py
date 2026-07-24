#!/usr/bin/env python3
"""Close the loop on the sufficiency-certificate argument.

The 354-set's milestone-pass certificate was established for Qwen3-8B-Base (the
anchor). The E1 retest used Qwen/Qwen3-8B (no-think) as the runnable proxy, so we
verify directly that this proxy also solves every non-INTEGRATE milestone
individually on the four families it failed at C3. Protocol matches
stage0_panel_16k.py (same STUDENT_SYSTEM_PROMPT, max_tokens=16384, K=8, tau>=1).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from decomposer.common.prompts import STUDENT_SYSTEM_PROMPT
from decomposer.verifier.verifier import VerifierModule

import tinker
from tinker import types as tt
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "Qwen/Qwen3-8B"
RENDERER = "qwen3_disable_thinking"
K = 8
MAX_TOKENS = 16384
TARGETS = ["0ad0ce27", "a0759695", "ad098d0c", "ff548f26"]
EVAL_SET = ROOT / "data/logs/rl/diagnostic_multi_families_repaired.jsonl"
OUT_FN = ROOT / "data/logs/rl/genuine_retest/qwen3_8b_stage0_check.jsonl"


def main():
    fams = [json.loads(l) for l in open(EVAL_SET)]
    targets = [f for f in fams if any(f["pid"].startswith(t) for t in TARGETS)]
    print(f"families: {len(targets)}", flush=True)

    verifier = VerifierModule(llm_client=None, llm_client_nothink=None)
    sc = tinker.ServiceClient()
    cli = sc.create_sampling_client(base_model=MODEL)
    tok = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER, tokenizer=tok)

    jobs = []
    for fam in targets:
        for idx, ms in enumerate(fam["milestones"]):
            if ms.get("type") == "INTEGRATE":
                continue
            for ki in range(K):
                jobs.append((fam["pid"], idx, ms, ki))
    print(f"rollouts: {len(jobs)}", flush=True)

    def sample(job):
        pid, idx, ms, ki = job
        convo = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                 {"role": "user", "content": ms["prompt"]}]
        gp = renderer.build_generation_prompt(convo)
        params = tt.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0,
                                   stop=renderer.get_stop_sequences())
        res = cli.sample(prompt=gp, num_samples=1, sampling_params=params).result()
        parsed, _ = renderer.parse_response(res.sequences[0].tokens)
        return parsed["content"]

    correct = defaultdict(int)
    OUT_FN.parent.mkdir(parents=True, exist_ok=True)
    t0, n = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=128) as ex:
        futs = {ex.submit(sample, j): j for j in jobs}
        for fut in as_completed(futs):
            pid, idx, ms, ki = futs[fut]
            try:
                content = fut.result()
                ok = verifier.verify(response=content, answer=ms["answer"],
                                     note=ms.get("note", ""))["label"] == "ACCEPT"
            except Exception:
                ok = False
            if ok:
                correct[(pid, idx)] += 1
            n += 1
            if n % 40 == 0:
                print(f"{n}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)

    with open(OUT_FN, "w") as fh:
        for fam in targets:
            per_ms = []
            for idx, ms in enumerate(fam["milestones"]):
                if ms.get("type") == "INTEGRATE":
                    continue
                per_ms.append({"idx": idx, "type": ms.get("type"),
                               "n_correct": correct[(fam["pid"], idx)], "k": K})
            passes = all(m["n_correct"] >= 1 for m in per_ms)
            row = {"pid": fam["pid"], "model": MODEL, "milestone_test_pass": passes,
                   "milestones": per_ms}
            fh.write(json.dumps(row) + "\n")
            print(f"{fam['pid'][:8]} milestone-test-pass={passes} "
                  f"{[(m['type'], m['n_correct']) for m in per_ms]}", flush=True)


if __name__ == "__main__":
    main()
