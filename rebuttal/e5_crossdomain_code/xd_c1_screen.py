#!/usr/bin/env python3
"""Cross-domain pilot, Track 1: C1 direct screen of Qwen3-8B on LCB medium
(stdin-type) problems. k=4, 16K tokens, graded by test execution."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lcb_utils import load_problems, grade_solution, extract_code

import tinker
from tinker import types as tt
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "Qwen/Qwen3-8B"
RENDERER = "qwen3_disable_thinking"
K = 4
OUT_FN = ROOT / "data/logs/rl/cg2/xdomain/c1_screen_qwen3_8b.jsonl"

PROMPT = """Solve the following competitive programming problem.

{statement}

Write a complete Python program that reads from standard input and writes to standard output. Put the final program in a single ```python code block."""


def main():
    probs = load_problems("medium")
    OUT_FN.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_FN.exists():
        for line in open(OUT_FN):
            try:
                r = json.loads(line)
                done.add((r["qid"], r["rollout"]))
            except Exception:
                pass

    sc = tinker.ServiceClient()
    cli = sc.create_sampling_client(base_model=MODEL)
    tok = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER, tokenizer=tok)

    jobs = [(p, k) for p in probs for k in range(K) if (p["qid"], k) not in done]
    print(f"problems={len(probs)} rollouts todo={len(jobs)}", flush=True)

    def sample(job):
        p, k = job
        gp = renderer.build_generation_prompt(
            [{"role": "user", "content": PROMPT.format(statement=p["statement"])}])
        params = tt.SamplingParams(max_tokens=16384, temperature=0.7,
                                   stop=renderer.get_stop_sequences())
        res = cli.sample(prompt=gp, num_samples=1, sampling_params=params).result()
        parsed, _ = renderer.parse_response(res.sequences[0].tokens)
        return parsed["content"]

    t0, n = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=128) as ex:
        futs = {ex.submit(sample, j): j for j in jobs}
        for fut in as_completed(futs):
            p, k = futs[fut]
            try:
                content = fut.result()
                code = extract_code(content)
                g = grade_solution(code, p["tests"]) if code else {"pass": False, "n_pass": 0, "n_run": 0}
                row = {"qid": p["qid"], "rollout": k, "accept": bool(g["pass"]),
                       "n_pass": g["n_pass"], "n_run": g["n_run"], "had_code": code is not None}
            except Exception as e:
                row = {"qid": p["qid"], "rollout": k, "error": f"{type(e).__name__}: {e}"[:150]}
            with open(OUT_FN, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            n += 1
            if n % 40 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
