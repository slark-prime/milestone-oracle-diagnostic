#!/usr/bin/env python3
"""Anchor-candidate pilot: does the model follow the \\boxed{} format at all,
and how often does it solve MATH500 problems? 30 problems x k=2 per model.
Decides whether Llama-3.2-3B is usable as a non-Qwen screening anchor."""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

from decomposer.common.prompts import STUDENT_SYSTEM_PROMPT
from decomposer.verifier.verifier import VerifierModule

import os
_env = os.environ.get("PILOT_MODELS")
MODELS = ([(m, "role_colon") for m in _env.split(",")] if _env else
          [("meta-llama/Llama-3.2-3B", "role_colon"),
           ("openai/gpt-oss-20b", "role_colon")])
N_PROBLEMS, K, MAX_TOKENS, SEED = 30, 2, 16384, 42
OUT = ROOT / "data/logs/rl/second_corpus/anchor_pilot"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    import tinker
    from tinker import types as tt
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    rows = [json.loads(l) for l in open(ROOT / "data/test/math500.jsonl")]
    random.Random(SEED).shuffle(rows)
    rows = rows[:N_PROBLEMS]
    v = VerifierModule(llm_client=None, llm_client_nothink=None)
    sc = tinker.ServiceClient()

    for model, rend_name in MODELS:
        out_fn = OUT / f"{model.split('/')[-1].replace('.', '_')}.jsonl"
        done = set()
        if out_fn.exists():
            for l in open(out_fn):
                try:
                    done.add(tuple(json.loads(l)["key"]))
                except Exception:
                    pass
        cli = sc.create_sampling_client(base_model=model)
        rend = renderers.get_renderer(rend_name, tokenizer=get_tokenizer(model))
        jobs = [((r["train_id"], ki), r) for r in rows for ki in range(K)
                if (r["train_id"], ki) not in done]
        print(f"[{model}] todo {len(jobs)} (done {len(done)})", flush=True)

        def run(job):
            key, r = job
            gp = rend.build_generation_prompt(
                [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                 {"role": "user", "content": r["prompt"]}])
            p = tt.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0,
                                  stop=rend.get_stop_sequences())
            res = cli.sample(prompt=gp, num_samples=1, sampling_params=p).result()
            parsed, _ = rend.parse_response(res.sequences[0].tokens)
            return parsed["content"]

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=64) as ex:
            futs = {ex.submit(run, j): j for j in jobs}
            for n, fut in enumerate(as_completed(futs), 1):
                key, r = futs[fut]
                try:
                    content = fut.result()
                    row = {"key": list(key),
                           "boxed": "\\boxed" in content,
                           "n_tokens_hint": len(content),
                           "accept": v.verify(response=content,
                                              answer=r["verifier"]["expected_answer"],
                                              note=r["verifier"].get("note", ""))["label"] == "ACCEPT",
                           "tail": content[-200:]}
                except Exception as e:
                    row = {"key": list(key), "error": f"{type(e).__name__}: {e}"[:150]}
                with open(out_fn, "a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if n % 10 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} {time.monotonic()-t0:.0f}s", flush=True)

    print("\n=== PILOT SUMMARY ===")
    for model, _ in MODELS:
        out_fn = OUT / f"{model.split('/')[-1].replace('.', '_')}.jsonl"
        rs = [json.loads(l) for l in open(out_fn)]
        ok = [r for r in rs if "error" not in r]
        pids = {}
        for r in ok:
            pids.setdefault(r["key"][0], []).append(r["accept"])
        all_fail = sum(1 for v_ in pids.values() if not any(v_))
        print(f"{model}: rollouts {len(ok)}, boxed {sum(r['boxed'] for r in ok)}/{len(ok)}, "
              f"accept {sum(r['accept'] for r in ok)}/{len(ok)}, "
              f"problems all-fail {all_fail}/{len(pids)}")


if __name__ == "__main__":
    main()
