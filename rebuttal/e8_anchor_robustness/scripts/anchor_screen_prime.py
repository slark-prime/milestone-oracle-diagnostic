#!/usr/bin/env python3
"""Full anchor screen (math500 + aime, k=4) with a Prime-served model.
Same rule as the suite screen: anchor-failed = 0 accepts in 4 attempts.
Served with the provider's chat template (disclosed protocol difference)."""
from __future__ import annotations

import json
import os
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
import second_corpus_suite as s
from openai import OpenAI

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-1B-Instruct"
SLUG = MODEL.split("/")[-1].replace(".", "_").lower()
K_SCREEN, MAX_TOKENS = 4, 16384
OUTD = ROOT / "data/logs/rl/second_corpus/anchor_screens" / SLUG
OUTD.mkdir(parents=True, exist_ok=True)


def main():
    cli = OpenAI(api_key=os.environ["PRIME_API_KEY"],
                 base_url="https://api.pinference.ai/api/v1", timeout=1200)
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    for corpus in ("math500", "aime"):
        rows = s.load_corpus(corpus, None)
        out_fn = OUTD / f"{corpus}_c1_screen.jsonl"
        done = set()
        if out_fn.exists():
            for l in open(out_fn):
                try:
                    done.add(tuple(json.loads(l)["key"]))
                except Exception:
                    pass
        jobs = [((r["train_id"], ki), r) for r in rows for ki in range(K_SCREEN)
                if (r["train_id"], ki) not in done]
        print(f"[{corpus}] todo {len(jobs)} (done {len(done)})", flush=True)

        def run(job):
            key, r = job
            resp = cli.chat.completions.create(
                model=MODEL, temperature=1.0, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                          {"role": "user", "content": r["prompt"]}])
            return resp.choices[0].message.content or ""

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=64) as ex:
            futs = {ex.submit(run, j): j for j in jobs}
            for n, fut in enumerate(as_completed(futs), 1):
                key, r = futs[fut]
                try:
                    content = fut.result()
                    row = {"key": list(key), "pid": key[0],
                           "accept": v.verify(response=content,
                                              answer=r["verifier"]["expected_answer"],
                                              note=r["verifier"].get("note", ""))["label"] == "ACCEPT"}
                except Exception as e:
                    row = {"key": list(key), "pid": key[0], "error": f"{type(e).__name__}: {e}"[:120]}
                with open(out_fn, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                if n % 200 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)

        solved, seen = defaultdict(int), set()
        dk = set()
        for l in open(out_fn):
            d = json.loads(l)
            if "error" in d or tuple(d["key"]) in dk:
                continue
            dk.add(tuple(d["key"]))
            seen.add(d["pid"])
            if d["accept"]:
                solved[d["pid"]] += 1
        fails = [p for p in seen if solved[p] == 0]
        print(f"[{corpus}] anchor-failed: {len(fails)}/{len(seen)} ({len(fails)/len(seen):.0%})", flush=True)


if __name__ == "__main__":
    main()
