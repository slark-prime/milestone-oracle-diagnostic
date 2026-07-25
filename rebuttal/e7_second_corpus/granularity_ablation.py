#!/usr/bin/env python3
"""P1-1: controlled coarse/fine granularity ablation (FWVb Q2, AC-Q2).

Teacher held FIXED (GPT-5.4, the submission's teacher); only the requested
granularity varies. Same families, same student, same decoding protocol.
Grading is at the PARENT level only, so the compound-gold issue that blocked the
Inkling Stage-0 run does not apply here.

Stage A: re-decompose N families at COARSE (3-4 milestones) and FINE (8-10).
Stage B: probe C2/C3 on gpt-oss-20b, k=8, 16K; compare recovery with the
         submission's original roadmaps on the same families.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

from decomposer.common.data_types import Problem
from decomposer.common.llm_client import LLMClient
from decomposer.teacher.teacher import TeacherModule
from decomposer.verifier.verifier import VerifierModule
from oracle_panel import build_c2, build_c3  # byte-identical prompt builders

PRIME_BASE = "https://api.pinference.ai/api/v1"
TEACHER_MODEL = "openai/gpt-5.4"
STUDENT = "openai/gpt-oss-20b"
RENDERER = "role_colon"
K = 8
MAX_TOKENS = 16384
N_FAMILIES = 40
SEED = 42
EVAL_SET = ROOT / "data/logs/rl/diagnostic_multi_families_repaired.jsonl"
OUT_DIR = ROOT / "data/logs/rl/granularity_ablation"

LEVELS = {
    "coarse": ("GRANULARITY CONSTRAINT: produce exactly 3 milestones. Use large, "
               "self-contained steps; a single milestone may bundle several "
               "manipulations. All other requirements are unchanged."),
    "fine":   ("GRANULARITY CONSTRAINT: produce 8 to 10 milestones. Use small, "
               "atomic steps; split any step containing more than one substantive "
               "manipulation. All other requirements are unchanged."),
}


def build_problem(fam: dict) -> Problem:
    prompt = fam["parent_prompt"]
    marker = "Problem:\n"
    i = prompt.find(marker)
    statement = prompt[i + len(marker):].rstrip() if i >= 0 else prompt.strip()
    return Problem(problem_id=fam["pid"], statement=statement,
                   gold_answer=fam["parent_answer"], reference_solution="")


def stage_a(fams):
    key = os.environ["PRIME_API_KEY"]
    team = os.environ.get("PRIME_TEAM_ID", "").strip()
    kw = dict(api_key=key, base_url=PRIME_BASE)
    if team:
        kw["default_headers"] = {"X-Prime-Team-ID": team}
    client = LLMClient(model=TEACHER_MODEL, **kw)
    teacher = TeacherModule(llm_client=client, temperature=0.1, max_tokens=8000,
                            max_retries=3, skip_integrate_answer_check=True)
    for level, instr in LEVELS.items():
        out_fn = OUT_DIR / f"packets_{level}.jsonl"
        done = set()
        if out_fn.exists():
            for l in open(out_fn):
                r = json.loads(l)
                if "error" not in r:
                    done.add(r["pid"])
        todo = [f for f in fams if f["pid"] not in done]
        print(f"[{level}] todo={len(todo)}", flush=True)

        def work(fam):
            # Inject the granularity constraint into the problem statement block so it
            # reads as a hard requirement, not as upstream-milestone context. Everything
            # else (system prompt, mode, schema, teacher, temperature) is unchanged.
            p = build_problem(fam)
            p_constrained = Problem(
                problem_id=p.problem_id,
                statement=f"{instr}\n\n{p.statement}",
                gold_answer=p.gold_answer,
                reference_solution=p.reference_solution,
            )
            try:
                pkt = teacher.generate_packet(problem=p_constrained, mode="decompose")
                return {"pid": fam["pid"], "level": level,
                        "milestones": pkt.get("milestones", [])}
            except Exception as e:
                return {"pid": fam["pid"], "level": level,
                        "error": f"{type(e).__name__}: {e}"[:200]}

        with ThreadPoolExecutor(max_workers=40) as ex:
            for fut in as_completed([ex.submit(work, f) for f in todo]):
                r = fut.result()
                with open(out_fn, "a") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n = len(r.get("milestones", [])) if "error" not in r else "ERR"
                print(f"  {r['pid'][:8]} {n}", flush=True)


def stage_b(fams):
    import tinker
    from tinker import types as tt
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    fam_by_pid = {f["pid"]: f for f in fams}
    verifier = VerifierModule(llm_client=None, llm_client_nothink=None)
    sc = tinker.ServiceClient()
    cli = sc.create_sampling_client(base_model=STUDENT)
    tok = get_tokenizer(STUDENT)
    rend = renderers.get_renderer(RENDERER, tokenizer=tok)

    jobs = []
    for level in LEVELS:
        pkt_fn = OUT_DIR / f"packets_{level}.jsonl"
        if not pkt_fn.exists():
            continue
        for l in open(pkt_fn):
            r = json.loads(l)
            if "error" in r:
                continue
            fam = fam_by_pid[r["pid"]]
            ms = [m for m in r["milestones"] if m.get("type") != "INTEGRATE"]
            kept = [m for m in ms if verifier.verify(
                response="\\boxed{" + str(m.get("answer", "")) + "}",
                answer=fam["parent_answer"], note="")["label"] != "ACCEPT"]
            if len(kept) < 2:
                continue
            # adapt packet fields to the prompt builders' expected shape
            shaped = [{"prompt": f"\nMilestone:\n{m.get('description','')}\n\nOutput instruction:",
                       "answer": m.get("answer", "")} for m in kept]
            for cond, builder in (("C2_descriptions", build_c2), ("C3_gold_answers", build_c3)):
                text = builder(fam["parent_prompt"], shaped)
                for ki in range(K):
                    jobs.append((r["pid"], level, cond, ki, text,
                                 fam["parent_answer"], fam.get("parent_note", ""), len(kept)))

    out_fn = OUT_DIR / f"probes_{STUDENT.split('/')[-1]}.jsonl"
    done = set()
    if out_fn.exists():
        for l in open(out_fn):
            r = json.loads(l)
            done.add((r["pid"], r["level"], r["condition"], r["rollout"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"rollouts todo: {len(jobs)}", flush=True)

    def sample(job):
        pid, level, cond, ki, text, pa, pn, nms = job
        mi = rend.build_generation_prompt([{"role": "user", "content": text}])
        params = tt.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7,
                                   stop=rend.get_stop_sequences())
        res = cli.sample(prompt=mi, num_samples=1, sampling_params=params).result()
        parsed, _ = rend.parse_response(res.sequences[0].tokens)
        return parsed["content"]

    t0, n = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=128) as ex:
        futs = {ex.submit(sample, j): j for j in jobs}
        for fut in as_completed(futs):
            pid, level, cond, ki, _, pa, pn, nms = futs[fut]
            try:
                content = fut.result()
                acc = verifier.verify(response=content, answer=pa, note=pn)["label"] == "ACCEPT"
                row = {"pid": pid, "level": level, "condition": cond, "rollout": ki,
                       "n_milestones": nms, "accept": acc}
            except Exception as e:
                row = {"pid": pid, "level": level, "condition": cond, "rollout": ki,
                       "error": f"{type(e).__name__}: {e}"[:150]}
            with open(out_fn, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "ab"], default="ab")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    allf = [json.loads(l) for l in open(EVAL_SET)]
    fams = random.Random(SEED).sample(allf, N_FAMILIES)
    if args.stage in ("a", "ab"):
        stage_a(fams)
    if args.stage in ("b", "ab"):
        stage_b(fams)


if __name__ == "__main__":
    main()
