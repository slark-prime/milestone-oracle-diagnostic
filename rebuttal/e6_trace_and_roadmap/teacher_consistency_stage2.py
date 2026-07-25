#!/usr/bin/env python3
"""P0-c stage 2: behavioral teacher-consistency — C2/C3 probes with Inkling
roadmaps on gpt-oss-20b, compared with the existing GPT-5.4-roadmap panel
results on the same families.

Prompt text formats replicate oracle_panel.py's build_c2/build_c3 exactly,
except milestone descriptions/answers come from Inkling packet fields directly
(Inkling packets carry `description`/`answer`, not rendered `prompt`s).
Leak-safe rule applied: non-INTEGRATE milestones whose gold answer is
verifier-equivalent to the parent answer are dropped; INTEGRATE excluded
(panel protocol).
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

from decomposer.verifier.verifier import VerifierModule

import tinker
from tinker import types as tt
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "openai/gpt-oss-20b"
RENDERER = "role_colon"
K = 8
MAX_TOKENS = 16384
PACKETS = ROOT / "data/logs/rl/teacher_consistency/inkling_packets.jsonl"
EVAL_SET = ROOT / "data/logs/rl/diagnostic_multi_families_repaired.jsonl"
OUT_FN = ROOT / "data/logs/rl/teacher_consistency/inkling_c2c3_gpt_oss_20b.jsonl"


def build_c2_ink(parent_prompt: str, milestones: list[dict]) -> str:
    hints = "\n\n".join(f"Sub-goal {i+1}: {ms['description']}" for i, ms in enumerate(milestones))
    return parent_prompt.rstrip() + "\n\nHint — the following sub-goals structure the solution:\n" + hints


def build_c3_ink(parent_prompt: str, milestones: list[dict]) -> str:
    hints = "\n\n".join(
        f"Sub-goal {i+1}: {ms['description']}\n  Answer: {ms['answer']}"
        for i, ms in enumerate(milestones))
    return parent_prompt.rstrip() + "\n\nThe following sub-goals have been solved:\n" + hints + "\n\nUse these results to solve the original problem."


def main():
    fams = {json.loads(l)["pid"]: json.loads(l) for l in open(EVAL_SET)}
    packets = [json.loads(l) for l in open(PACKETS) if "error" not in json.loads(l)]
    verifier = VerifierModule(llm_client=None, llm_client_nothink=None)

    jobs_meta = []
    for pk in packets:
        fam = fams[pk["pid"]]
        parent_ans, parent_note = fam["parent_answer"], fam.get("parent_note", "")
        ms = [m for m in pk["milestones"] if m.get("type") != "INTEGRATE"]
        kept = []
        for m in ms:
            leak = verifier.verify(
                response="\\boxed{" + str(m.get("answer", "")) + "}",
                answer=parent_ans, note="")["label"] == "ACCEPT"
            if not leak:
                kept.append(m)
        if len(kept) < 2:
            continue  # multi-milestone requirement, matching the 354-set rule
        for cond, builder in (("C2_descriptions", build_c2_ink), ("C3_gold_answers", build_c3_ink)):
            prompt_text = builder(fam["parent_prompt"], kept)
            jobs_meta.append((pk["pid"], cond, parent_ans, parent_note, prompt_text, len(kept)))

    done = set()
    if OUT_FN.exists():
        for line in open(OUT_FN):
            try:
                r = json.loads(line)
                done.add((r["pid"], r["condition"], r["rollout"]))
            except Exception:
                pass

    sc = tinker.ServiceClient()
    cli = sc.create_sampling_client(base_model=MODEL)
    tok = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER, tokenizer=tok)

    jobs = []
    for pid, cond, pa, pn, ptext, nms in jobs_meta:
        for ki in range(K):
            if (pid, cond, ki) in done:
                continue
            jobs.append((pid, cond, pa, pn, ptext, nms, ki))
    print(f"families={len(packets)} prompt-cells={len(jobs_meta)} rollouts todo={len(jobs)}", flush=True)

    def sample(job):
        pid, cond, pa, pn, ptext, nms, ki = job
        model_input = renderer.build_generation_prompt([{"role": "user", "content": ptext}])
        params = tt.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7,
                                   stop=renderer.get_stop_sequences())
        res = cli.sample(prompt=model_input, num_samples=1, sampling_params=params).result()
        parsed, _ = renderer.parse_response(res.sequences[0].tokens)
        return parsed["content"]

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=128) as ex:
        futs = {ex.submit(sample, j): j for j in jobs}
        for fut in as_completed(futs):
            pid, cond, pa, pn, ptext, nms, ki = futs[fut]
            try:
                content = fut.result()
                verdict = verifier.verify(response=content, answer=pa, note=pn)["label"]
                row = {"pid": pid, "condition": cond, "rollout": ki, "n_milestones": nms,
                       "accept": verdict == "ACCEPT",
                       "boxed_tail": (content or "")[-300:]}
            except Exception as e:
                row = {"pid": pid, "condition": cond, "rollout": ki,
                       "error": f"{type(e).__name__}: {e}"[:200]}
            with open(OUT_FN, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
