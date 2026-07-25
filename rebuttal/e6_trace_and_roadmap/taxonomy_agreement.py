#!/usr/bin/env python3
"""FWVb W1 named this metric explicitly: *taxonomy agreement* between independently
generated milestone roadmaps. The submission could not report it (a second teacher's
compound gold answers were not gradable, so Stage 0 could not be run on them); the
atomic-gold few-shot fix removes that blocker.

Chain, all resumable:
  stage0  milestone-only test on the alternative roadmap, K=8
  probes  C1 / C2 / C3 on the parent under the alternative roadmap, K=8
  report  five-way taxonomy per family under each roadmap, and their agreement

Student, decoding, prompts, verifier and the >=1/K rule are identical to the panel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

from decomposer.common.prompts import STUDENT_SYSTEM_PROMPT
from decomposer.verifier.verifier import VerifierModule
from oracle_panel import build_c2, build_c3

STUDENT, RENDERER = "openai/gpt-oss-20b", "role_colon"
K, MAX_TOKENS = 8, 16384
ALT = ROOT / "data/logs/rl/teacher_consistency/inkling_atomic_packets.jsonl"
EVAL = ROOT / "data/logs/rl/diagnostic_multi_families_repaired.jsonl"
OUT = ROOT / "data/logs/rl/teacher_consistency"


def alt_families():
    fams = {json.loads(l)["pid"]: json.loads(l) for l in open(EVAL)}
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    out = []
    for l in open(ALT):
        r = json.loads(l)
        if "error" in r or r["pid"] not in fams:
            continue
        f = fams[r["pid"]]
        ms = [m for m in r["milestones"] if m.get("type") != "INTEGRATE"]
        kept = [m for m in ms
                if v.verify(response="\\boxed{" + str(m.get("answer", "")) + "}",
                            answer=f["parent_answer"], note="")["label"] != "ACCEPT"]
        if len(kept) < 2:
            continue
        out.append({
            "pid": r["pid"], "parent_prompt": f["parent_prompt"],
            "parent_answer": f["parent_answer"], "parent_note": f.get("parent_note", ""),
            "milestones": [{"prompt": f"\nMilestone:\n{m.get('description','')}\n\nOutput instruction: Put your final answer in \\boxed{{}}.",
                            "answer": m.get("answer", ""), "note": m.get("note", ""),
                            "type": m.get("type")} for m in kept]})
    return out


def sample_all(jobs, out_fn, verifier):
    import tinker
    from tinker import types as tt
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    cli = tinker.ServiceClient().create_sampling_client(base_model=STUDENT)
    rend = renderers.get_renderer(RENDERER, tokenizer=get_tokenizer(STUDENT))
    done = set()
    if out_fn.exists():
        for l in open(out_fn):
            try:
                done.add(tuple(json.loads(l)["key"]))
            except Exception:
                pass
    todo = [j for j in jobs if tuple(j[0]) not in done]
    print(f"  todo {len(todo)} (done {len(done)})", flush=True)

    def run(job):
        key, text, meta = job
        gp = rend.build_generation_prompt(
            [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
             {"role": "user", "content": text}])
        p = tt.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0,
                              stop=rend.get_stop_sequences())
        res = cli.sample(prompt=gp, num_samples=1, sampling_params=p).result()
        parsed, _ = rend.parse_response(res.sequences[0].tokens)
        return parsed["content"]

    t0, n = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=160) as ex:
        futs = {ex.submit(run, j): j for j in todo}
        for fut in as_completed(futs):
            key, text, meta = futs[fut]
            try:
                ans, note = meta
                ok = verifier.verify(response=fut.result(), answer=ans, note=note)["label"] == "ACCEPT"
                row = {"key": list(key), "accept": ok}
            except Exception as e:
                row = {"key": list(key), "error": f"{type(e).__name__}: {e}"[:120]}
            with open(out_fn, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            n += 1
            if n % 200 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.monotonic()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["stage0", "probes", "report"])
    a = ap.parse_args()
    fams = alt_families()
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    print(f"alt-roadmap families: {len(fams)}", flush=True)

    if a.stage == "stage0":
        jobs = [((f["pid"], i, k), f["milestones"][i]["prompt"],
                 (f["milestones"][i]["answer"], f["milestones"][i].get("note", "")))
                for f in fams for i in range(len(f["milestones"])) for k in range(K)]
        sample_all(jobs, OUT / "alt_stage0.jsonl", v)

    elif a.stage == "probes":
        jobs = []
        for f in fams:
            texts = {"C1_direct": f["parent_prompt"],
                     "C2_descriptions": build_c2(f["parent_prompt"], f["milestones"]),
                     "C3_gold_answers": build_c3(f["parent_prompt"], f["milestones"])}
            for c, txt in texts.items():
                for k in range(K):
                    jobs.append(((f["pid"], c, k), txt, (f["parent_answer"], f["parent_note"])))
        sample_all(jobs, OUT / "alt_probes.jsonl", v)

    else:
        s0 = defaultdict(int)
        for l in open(OUT / "alt_stage0.jsonl"):
            d = json.loads(l)
            if d.get("accept"):
                s0[(d["key"][0], d["key"][1])] += 1
        pr = defaultdict(int)
        for l in open(OUT / "alt_probes.jsonl"):
            d = json.loads(l)
            if d.get("accept"):
                pr[(d["key"][0], d["key"][1])] += 1
        panel = {}
        for l in open(ROOT / "data/logs/rl/oracle_panel_16k/gpt_oss_20b.jsonl"):
            d = json.loads(l)
            panel[(d["pid"], d["condition"])] = d["n_correct"]
        base_s0 = defaultdict(int)
        for l in open(ROOT / "data/logs/rl/stage0_panel_16k/gpt_oss_20b.jsonl"):
            d = json.loads(l)
            base_s0[(d["pid"], d["ms_idx"])] = d["n_correct"]
        base_fams = {json.loads(l)["pid"]: json.loads(l) for l in open(EVAL)}

        def tax(c1, c2, c3, ms_pass):
            if c1 >= 1: return "DIRECT"
            if c2 >= 1: return "ROADMAP_GAP"
            if c3 >= 1: return "MILESTONE_EXECUTION_GAP"
            return "COMPOSITION_GAP" if ms_pass else "MISSING_MILESTONE_OR_CAPABILITY"

        rows, agree = [], 0
        for f in fams:
            pid = f["pid"]
            alt_pass = all(s0[(pid, i)] >= 1 for i in range(len(f["milestones"])))
            a_t = tax(pr[(pid, "C1_direct")], pr[(pid, "C2_descriptions")],
                      pr[(pid, "C3_gold_answers")], alt_pass)
            n_base = len(base_fams[pid]["milestones"])
            base_pass = all(base_s0.get((pid, i), 0) >= 1 for i in range(n_base))
            b_t = tax(panel.get((pid, "C1_direct"), 0), panel.get((pid, "C2_descriptions"), 0),
                      panel.get((pid, "C3_gold_answers"), 0), base_pass)
            rows.append({"pid": pid, "submission_roadmap": b_t, "alt_roadmap": a_t})
            agree += (a_t == b_t)
        n = len(rows)
        conf = defaultdict(int)
        for r in rows:
            conf[(r["submission_roadmap"], r["alt_roadmap"])] += 1
        rep = {"n_families": n, "taxonomy_agreement": [agree, round(agree / n, 3)],
               "confusion": {f"{a} -> {b}": c for (a, b), c in sorted(conf.items(), key=lambda x: -x[1])},
               "per_family": rows}
        json.dump(rep, open(OUT / "taxonomy_agreement.json", "w"), indent=2)
        print(f"\nTAXONOMY AGREEMENT: {agree}/{n} = {agree/n:.0%}")
        for k, c in rep["confusion"].items():
            print(f"  {k}: {c}")


if __name__ == "__main__":
    main()
