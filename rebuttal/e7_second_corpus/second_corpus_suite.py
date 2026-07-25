#!/usr/bin/env python3
"""AC-Q1: full protocol replication on a SECOND CORPUS (MATH500 / AIME).

Everything about the protocol is held identical to the submission — same teacher
(GPT-5.4), same student, same prompt builders, same six probe conditions, same
deterministic cascade, same K=8 at 16K, same >=1/K family-success rule. Only the
source corpus changes.

Stages (all resumable):
  screen     C1 direct on the corpus, k=4  -> keep families the anchor fails
  decompose  GPT-5.4 milestone packets for those parents
  stage0     milestone-only test, k=8      -> milestone-test pass/fail
  probes     C1 / C2-correct / C2-random / C2-generic / C3-gold / C3-mismatched, k=8
  report     specificity contrast + taxonomy distribution

Usage:
  python3 scripts/experiments/second_corpus_suite.py --corpus math500 --stage screen
  ... --stage decompose | stage0 | probes | report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

from decomposer.common.data_types import Problem
from decomposer.common.llm_client import LLMClient
from decomposer.common.prompts import STUDENT_SYSTEM_PROMPT
from decomposer.teacher.teacher import TeacherModule
from decomposer.verifier.verifier import VerifierModule
from oracle_panel import (build_c2, build_c2_generic, build_c2_random,
                          build_c3, build_c3_mismatched)

PRIME_BASE = "https://api.pinference.ai/api/v1"
TEACHER_MODEL = "openai/gpt-5.4"
ANCHOR = ("Qwen/Qwen3-8B", "qwen3_disable_thinking")   # runnable anchor/student
K_SCREEN, K = 4, 8
MAX_TOKENS = 16384
SEED = 42
CONDITIONS = ["C1_direct", "C2_descriptions", "C3_gold_answers",
              "C2_random", "C2_generic", "C3_mismatched"]


def paths(corpus: str):
    d = ROOT / "data/logs/rl/second_corpus" / corpus
    d.mkdir(parents=True, exist_ok=True)
    return {"dir": d, "screen": d / "c1_screen.jsonl", "packets": d / "packets.jsonl",
            "stage0": d / "stage0.jsonl", "probes": d / "probes.jsonl",
            "report": d / "report.json"}


def load_corpus(corpus: str, limit: int | None):
    rows = [json.loads(l) for l in open(ROOT / f"data/test/{corpus}.jsonl")]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def tinker_client():
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    name, rend_name = ANCHOR
    sc = tinker.ServiceClient()
    cli = sc.create_sampling_client(base_model=name)
    tok = get_tokenizer(name)
    return cli, renderers.get_renderer(rend_name, tokenizer=tok)


def sample_many(jobs, out_fn, done_key, row_builder, workers=192):
    """jobs: list of (key_tuple, prompt_text, meta). Resumable, append-only."""
    import tinker
    from tinker import types as tt
    cli, rend = tinker_client()
    done = set()
    if out_fn.exists():
        for l in open(out_fn):
            try:
                done.add(done_key(json.loads(l)))
            except Exception:
                pass
    todo = [j for j in jobs if j[0] not in done]
    print(f"  rollouts todo: {len(todo)} (done {len(done)})", flush=True)
    if not todo:
        return

    def run(job):
        key, text, meta = job
        convo = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                 {"role": "user", "content": text}]
        gp = rend.build_generation_prompt(convo)
        params = tt.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0,
                                   stop=rend.get_stop_sequences())
        res = cli.sample(prompt=gp, num_samples=1, sampling_params=params).result()
        parsed, _ = rend.parse_response(res.sequences[0].tokens)
        return parsed["content"]

    t0, n = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run, j): j for j in todo}
        for fut in as_completed(futs):
            key, text, meta = futs[fut]
            try:
                row = row_builder(key, meta, fut.result())
            except Exception as e:
                row = {"key": list(key), "error": f"{type(e).__name__}: {e}"[:150]}
            with open(out_fn, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.monotonic()-t0:.0f}s", flush=True)


def stage_screen(corpus, limit):
    p = paths(corpus)
    rows = load_corpus(corpus, limit)
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    print(f"[screen] {len(rows)} problems x k={K_SCREEN}", flush=True)
    jobs = [((r["train_id"], ki), r["prompt"], r) for r in rows for ki in range(K_SCREEN)]
    sample_many(
        jobs, p["screen"], lambda d: tuple(d["key"]),
        lambda key, meta, content: {
            "key": list(key), "pid": key[0], "rollout": key[1],
            "accept": v.verify(response=content, answer=meta["verifier"]["expected_answer"],
                               note=meta["verifier"].get("note", ""))["label"] == "ACCEPT"})


def stage_decompose(corpus, limit):
    p = paths(corpus)
    rows = {r["train_id"]: r for r in load_corpus(corpus, limit)}
    solved = defaultdict(int)
    for l in open(p["screen"]):
        d = json.loads(l)
        if "error" not in d and d["accept"]:
            solved[d["pid"]] += 1
    fails = [pid for pid in rows if solved[pid] == 0]
    print(f"[decompose] anchor-failed parents: {len(fails)}/{len(rows)}", flush=True)

    done = set()
    if p["packets"].exists():
        for l in open(p["packets"]):
            d = json.loads(l)
            if "error" not in d:
                done.add(d["pid"])
    todo = [pid for pid in fails if pid not in done]
    print(f"  to decompose: {len(todo)}", flush=True)
    if not todo:
        return
    kw = dict(api_key=os.environ["PRIME_API_KEY"], base_url=PRIME_BASE)
    teacher = TeacherModule(llm_client=LLMClient(model=TEACHER_MODEL, **kw),
                            temperature=0.1, max_tokens=8000, max_retries=3,
                            skip_integrate_answer_check=True)

    def work(pid):
        r = rows[pid]
        stmt = r["prompt"]
        i = stmt.find("Problem:\n")
        stmt = stmt[i + 9:].rstrip() if i >= 0 else stmt
        prob = Problem(problem_id=pid, statement=stmt,
                       gold_answer=r["verifier"]["expected_answer"], reference_solution="")
        try:
            pkt = teacher.generate_packet(problem=prob, mode="decompose")
            return {"pid": pid, "parent_prompt": r["prompt"],
                    "parent_answer": r["verifier"]["expected_answer"],
                    "parent_note": r["verifier"].get("note", ""),
                    "milestones": pkt.get("milestones", [])}
        except Exception as e:
            return {"pid": pid, "error": f"{type(e).__name__}: {e}"[:200]}

    with ThreadPoolExecutor(max_workers=48) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, pid) for pid in todo]), 1):
            r = fut.result()
            with open(p["packets"], "a") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


def eligible_families(corpus):
    """Packets that pass the submission's structural filter: >=2 non-INTEGRATE,
    leak-safe (no milestone gold equal to the parent answer)."""
    p = paths(corpus)
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    out = []
    for l in open(p["packets"]):
        d = json.loads(l)
        if "error" in d:
            continue
        ms = [m for m in d["milestones"] if m.get("type") != "INTEGRATE"]
        kept = [m for m in ms if v.verify(response="\\boxed{" + str(m.get("answer", "")) + "}",
                                          answer=d["parent_answer"], note="")["label"] != "ACCEPT"]
        if len(kept) >= 2:
            d = dict(d)
            d["milestones"] = [{"prompt": f"\nMilestone:\n{m.get('description','')}\n\nOutput instruction: Put your final answer in \\boxed{{}}.",
                                "answer": m.get("answer", ""), "note": m.get("note", ""),
                                "type": m.get("type")} for m in kept]
            out.append(d)
    return out


def stage_stage0(corpus):
    p = paths(corpus)
    fams = eligible_families(corpus)
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    print(f"[stage0] eligible families: {len(fams)}", flush=True)
    jobs = [((f["pid"], i, ki), f["milestones"][i]["prompt"], (f, i))
            for f in fams for i in range(len(f["milestones"])) for ki in range(K)]
    sample_many(
        jobs, p["stage0"], lambda d: tuple(d["key"]),
        lambda key, meta, content: {
            "key": list(key), "pid": key[0], "ms_idx": key[1], "rollout": key[2],
            "accept": v.verify(response=content, answer=meta[0]["milestones"][meta[1]]["answer"],
                               note=meta[0]["milestones"][meta[1]].get("note", ""))["label"] == "ACCEPT"})


def stage_probes(corpus):
    p = paths(corpus)
    fams = eligible_families(corpus)
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    rng = random.Random(SEED)
    all_ms = [(f["pid"], m) for f in fams for m in f["milestones"]]
    rand_assign, mismatch = {}, {}
    for f in fams:
        other = [m for pid, m in all_ms if pid != f["pid"]]
        rand_assign[f["pid"]] = rng.choices(other, k=len(f["milestones"])) if other else []
        wrong = []
        for m in f["milestones"]:
            for _ in range(20):
                _, o = rng.choice(all_ms)
                if o["answer"] != m["answer"]:
                    wrong.append(o["answer"]); break
            else:
                wrong.append("0")
        mismatch[f["pid"]] = wrong

    jobs = []
    for f in fams:
        pp, ms = f["parent_prompt"], f["milestones"]
        texts = {
            "C1_direct": pp,
            "C2_descriptions": build_c2(pp, ms),
            "C3_gold_answers": build_c3(pp, ms),
            "C2_random": build_c2_random(pp, rand_assign[f["pid"]]),
            "C2_generic": build_c2_generic(pp),
            "C3_mismatched": build_c3_mismatched(pp, ms, mismatch[f["pid"]]),
        }
        for cond in CONDITIONS:
            for ki in range(K):
                jobs.append(((f["pid"], cond, ki), texts[cond], f))
    print(f"[probes] {len(fams)} families x {len(CONDITIONS)} conditions x k={K}", flush=True)
    sample_many(
        jobs, p["probes"], lambda d: tuple(d["key"]),
        lambda key, meta, content: {
            "key": list(key), "pid": key[0], "condition": key[1], "rollout": key[2],
            "accept": v.verify(response=content, answer=meta["parent_answer"],
                               note=meta["parent_note"])["label"] == "ACCEPT"})


def stage_report(corpus):
    p = paths(corpus)
    fams = {f["pid"]: f for f in eligible_families(corpus)}
    pr = defaultdict(int)
    for l in open(p["probes"]):
        d = json.loads(l)
        if "error" not in d and d["accept"]:
            pr[(d["pid"], d["condition"])] += 1
    s0 = defaultdict(int)
    for l in open(p["stage0"]):
        d = json.loads(l)
        if "error" not in d and d["accept"]:
            s0[(d["pid"], d["ms_idx"])] += 1

    n = len(fams)
    rec = {c: sum(1 for pid in fams if pr[(pid, c)] >= 1) for c in CONDITIONS}
    ms_pass = {pid: all(s0[(pid, i)] >= 1 for i in range(len(f["milestones"])))
               for pid, f in fams.items()}
    solved_any = {pid for pid in fams
                  if any(pr[(pid, c)] >= 1 for c in ("C1_direct", "C2_descriptions", "C3_gold_answers"))}
    tax = defaultdict(int)
    for pid, f in fams.items():
        if pr[(pid, "C1_direct")] >= 1:
            tax["DIRECT"] += 1
        elif pr[(pid, "C2_descriptions")] >= 1:
            tax["ROADMAP_GAP"] += 1
        elif pr[(pid, "C3_gold_answers")] >= 1:
            tax["MILESTONE_EXECUTION_GAP"] += 1
        elif ms_pass[pid]:
            tax["COMPOSITION_GAP"] += 1
        else:
            tax["MISSING_MILESTONE_OR_CAPABILITY"] += 1

    rep = {
        "corpus": corpus, "families": n, "student": ANCHOR[0], "K": K,
        "recovery_by_condition": {c: [rec[c], round(rec[c] / n, 3)] for c in CONDITIONS},
        "specificity": {
            "C2_correct_vs_random": [rec["C2_descriptions"], rec["C2_random"]],
            "C2_correct_vs_generic": [rec["C2_descriptions"], rec["C2_generic"]],
            "C3_gold_vs_mismatched": [rec["C3_gold_answers"], rec["C3_mismatched"]],
        },
        "milestone_test_pass": sum(ms_pass.values()),
        "solved_by_any_probe": len(solved_any),
        "taxonomy": dict(tax),
        "composition_gap_share": round(tax["COMPOSITION_GAP"] / n, 3),
    }
    json.dump(rep, open(p["report"], "w"), indent=2)
    print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="math500")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stage", required=True,
                    choices=["screen", "decompose", "stage0", "probes", "report"])
    a = ap.parse_args()
    {"screen": lambda: stage_screen(a.corpus, a.limit),
     "decompose": lambda: stage_decompose(a.corpus, a.limit),
     "stage0": lambda: stage_stage0(a.corpus),
     "probes": lambda: stage_probes(a.corpus),
     "report": lambda: stage_report(a.corpus)}[a.stage]()


if __name__ == "__main__":
    main()
