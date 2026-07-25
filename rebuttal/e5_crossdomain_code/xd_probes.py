#!/usr/bin/env python3
"""Cross-domain pilot, Track 3: Stage 0 / C2 / C3 probes over compile-verified
code packets. All grading by execution.

Phases (XD_PHASE):
  parts  Qwen3-8B implements each helper from signature+docstring+tests (gold hidden);
         graded by running the helper's own asserts. K=8.
  c2     Qwen3-8B solves the parent given the decomposition plan (signatures+docstrings+main_plan).
  c3     Same as c2 plus the verified gold helper implementations verbatim.
  suff20 gpt-oss-20b at c3 (behavioral sufficiency, in addition to the compile-time proof).
  c1     K=8 direct arm (the screen was k=4; this makes C1 comparable to c2/c3).
  c2rand C2 with another problem's plan (corruption control).
  c2gen  C2 with a generic decomposition instruction, no problem-matched content.
  c3mis  C3 with helper implementations taken from other problems (corruption control).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lcb_utils import load_problems, grade_solution, run_program, extract_code

import tinker
from tinker import types as tt
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

XD = ROOT / "data/logs/rl/cg2/xdomain"
K = 8

PARTS_PROMPT = """Implement the following Python function.

{signature}
    \"\"\"{docstring}\"\"\"

Your implementation must pass these tests:
{tests}

Context — the function is one sub-goal of this larger problem (do NOT solve the larger problem):
{statement}

Write only the function. Put it in a single ```python code block."""

C2_PROMPT = """Solve the following competitive programming problem.

{statement}

A solution plan decomposes it into these helper functions:
{plan}

Overall approach: {main_plan}

Write a complete Python program that reads from standard input and writes to standard output, following the plan. Put the final program in a single ```python code block."""

C3_PROMPT = """Solve the following competitive programming problem.

{statement}

A solution plan decomposes it into helper functions, and correct, tested implementations of every helper are provided below. You may copy them verbatim.

{helpers_code}

Overall approach: {main_plan}

Write a complete Python program that reads from standard input and writes to standard output, using the provided helpers. Put the final program in a single ```python code block."""


C1_PROMPT = """Solve the following competitive programming problem.

{statement}

Write a complete Python program that reads from standard input and writes to standard output. Put the final program in a single ```python code block."""

GENERIC_PLAN = """- def helper_one(...):
    A first sub-step of the problem.
- def helper_two(...):
    A second sub-step of the problem.
- def helper_three(...):
    A third sub-step of the problem."""


def load_packets():
    rows = [json.loads(l) for l in open(XD / "inkling_code_packets.jsonl")]
    return [r for r in rows if "error" not in r]


def main():
    phase = os.environ["XD_PHASE"]
    packets = load_packets()
    probs = {p["qid"]: p for p in load_problems("medium")}
    out_fn = XD / f"{phase}_results.jsonl"

    # deterministic "some other problem" mapping for the corruption controls
    import random as _r
    _ids = sorted(p["qid"] for p in packets)
    _rng = _r.Random(42)
    _by_qid = {x["qid"]: x for x in packets}
    OTHER = {}
    for q in _ids:
        other_q = _rng.choice([z for z in _ids if z != q])   # choose once, then look up
        OTHER[q] = _by_qid[other_q]
    globals()["OTHER"] = OTHER

    model = "openai/gpt-oss-20b" if phase == "suff20" else "Qwen/Qwen3-8B"
    rend = "role_colon" if phase == "suff20" else "qwen3_disable_thinking"

    done = set()
    if out_fn.exists():
        for line in open(out_fn):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["key"])
            except Exception:
                pass

    jobs = []
    for r in packets:
        pk, qid = r["packet"], r["qid"]
        stmt = probs[qid]["statement"]
        if phase == "parts":
            for hi, h in enumerate(pk["helpers"]):
                prompt = PARTS_PROMPT.format(signature=h["signature"], docstring=h["docstring"],
                                             tests="\n".join(h["tests"]), statement=stmt[:1500])
                for k in range(K):
                    key = f"{qid}:{hi}:{k}"
                    if key not in done:
                        jobs.append({"key": key, "qid": qid, "hi": hi, "prompt": prompt,
                                     "tests": h["tests"]})
        else:
            plan = "\n".join(f"- {h['signature']}\n    {h['docstring']}" for h in pk["helpers"])
            helpers_code = "\n\n".join(h["gold_impl"] for h in pk["helpers"])
            if phase == "c1":
                prompt = C1_PROMPT.format(statement=stmt)
            elif phase == "c2":
                prompt = C2_PROMPT.format(statement=stmt, plan=plan, main_plan=pk["main_plan"])
            elif phase == "c2rand":
                o = OTHER[qid]
                oplan = "\n".join(f"- {h['signature']}\n    {h['docstring']}" for h in o["packet"]["helpers"])
                prompt = C2_PROMPT.format(statement=stmt, plan=oplan,
                                          main_plan=o["packet"]["main_plan"])
            elif phase == "c2gen":
                prompt = C2_PROMPT.format(statement=stmt, plan=GENERIC_PLAN,
                                          main_plan="Decompose the problem into helper functions, implement each, then combine them.")
            elif phase == "c3mis":
                o = OTHER[qid]
                ocode = "\n\n".join(h["gold_impl"] for h in o["packet"]["helpers"])
                prompt = C3_PROMPT.format(statement=stmt, helpers_code=ocode,
                                          main_plan=pk["main_plan"])
            else:
                prompt = C3_PROMPT.format(statement=stmt, helpers_code=helpers_code,
                                          main_plan=pk["main_plan"])
            for k in range(K):
                key = f"{qid}:{k}"
                if key not in done:
                    jobs.append({"key": key, "qid": qid, "prompt": prompt,
                                 "parent_tests": probs[qid]["tests"]})
    print(f"phase={phase} model={model} rollouts todo={len(jobs)}", flush=True)
    if not jobs:
        return

    sc = tinker.ServiceClient()
    cli = sc.create_sampling_client(base_model=model)
    tok = get_tokenizer(model)
    renderer = renderers.get_renderer(rend, tokenizer=tok)

    def sample(j):
        gp = renderer.build_generation_prompt([{"role": "user", "content": j["prompt"]}])
        params = tt.SamplingParams(max_tokens=16384, temperature=0.7,
                                   stop=renderer.get_stop_sequences())
        res = cli.sample(prompt=gp, num_samples=1, sampling_params=params).result()
        parsed, _ = renderer.parse_response(res.sequences[0].tokens)
        return parsed["content"]

    def grade(j, content):
        code = extract_code(content)
        if not code:
            return False, "no_code"
        if "tests" in j:  # parts: run helper asserts
            prog = code + "\n\n" + "\n".join(j["tests"]) + "\nprint('HELPER_OK')\n"
            status, out = run_program(prog, "", timeout=10)
            return (status == "ok" and "HELPER_OK" in out), status
        g = grade_solution(code, j["parent_tests"], timeout=15)
        return g["pass"], f"{g['n_pass']}/{g['n_run']}"

    t0, n = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=128) as ex:
        futs = {ex.submit(sample, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                content = fut.result()
                ok, note = grade(j, content)
                row = {"key": j["key"], "qid": j["qid"], "accept": ok, "note": str(note)[:60],
                       "tail": (content or "")[-400:]}
                if "hi" in j:
                    row["hi"] = j["hi"]
            except Exception as e:
                row = {"key": j["key"], "qid": j["qid"], "error": f"{type(e).__name__}: {e}"[:150]}
            with open(out_fn, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
