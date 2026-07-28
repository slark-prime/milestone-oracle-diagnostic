#!/usr/bin/env python3
"""Anchor-candidate pilot over Prime-served models (provider chat template).
Same design as anchor_pilot.py: 30 MATH500 problems x k=2; measures boxed-format
compliance, solve rate, and all-fail count per problem."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from decomposer.common.prompts import STUDENT_SYSTEM_PROMPT
from decomposer.verifier.verifier import VerifierModule

from openai import OpenAI

MODELS = os.environ.get("PILOT_MODELS", "google/gemma-3-27b-it,meta-llama/Llama-3.2-3B-Instruct,mistralai/mistral-nemo").split(",")
N_PROBLEMS, K, MAX_TOKENS, SEED = 30, 2, 16384, 42
OUT = ROOT / "data/logs/rl/second_corpus/anchor_pilot"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    cli = OpenAI(api_key=os.environ["PRIME_API_KEY"],
                 base_url="https://api.pinference.ai/api/v1", timeout=1200)
    rows = [json.loads(l) for l in open(ROOT / "data/test/math500.jsonl")]
    random.Random(SEED).shuffle(rows)
    rows = rows[:N_PROBLEMS]
    v = VerifierModule(llm_client=None, llm_client_nothink=None)

    for model in MODELS:
        slug = model.split("/")[-1].replace(".", "_")
        out_fn = OUT / f"prime_{slug}.jsonl"
        done = set()
        if out_fn.exists():
            for l in open(out_fn):
                try:
                    done.add(tuple(json.loads(l)["key"]))
                except Exception:
                    pass
        jobs = [((r["train_id"], ki), r) for r in rows for ki in range(K)
                if (r["train_id"], ki) not in done]
        print(f"[{model}] todo {len(jobs)} (done {len(done)})", flush=True)

        def run(job):
            key, r = job
            resp = cli.chat.completions.create(
                model=model, temperature=1.0, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                          {"role": "user", "content": r["prompt"]}])
            return resp.choices[0].message.content or ""

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=48) as ex:
            futs = {ex.submit(run, j): j for j in jobs}
            for n, fut in enumerate(as_completed(futs), 1):
                key, r = futs[fut]
                try:
                    content = fut.result()
                    row = {"key": list(key), "boxed": "\\boxed" in content,
                           "accept": v.verify(response=content,
                                              answer=r["verifier"]["expected_answer"],
                                              note=r["verifier"].get("note", ""))["label"] == "ACCEPT",
                           "tail": content[-160:]}
                except Exception as e:
                    row = {"key": list(key), "error": f"{type(e).__name__}: {e}"[:150]}
                with open(out_fn, "a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if n % 15 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)

    print("\n=== PRIME PILOT SUMMARY ===")
    for model in MODELS:
        slug = model.split("/")[-1].replace(".", "_")
        rs = [json.loads(l) for l in open(OUT / f"prime_{slug}.jsonl")]
        ok = [r for r in rs if "error" not in r]
        err = len(rs) - len(ok)
        pids = {}
        for r in ok:
            pids.setdefault(r["key"][0], []).append(r["accept"])
        all_fail = sum(1 for vv in pids.values() if not any(vv))
        print(f"{model}: rollouts {len(ok)} (err {err}), boxed {sum(r['boxed'] for r in ok)}/{len(ok)}, "
              f"accept {sum(r['accept'] for r in ok)}/{len(ok)}, all-fail {all_fail}/{len(pids)}")


if __name__ == "__main__":
    main()
