#!/usr/bin/env python3
"""P0-c stage 1: second-teacher (Inkling via Tinker) re-decomposition of a
60-family sample from the 354 set, for teacher-consistency analysis.

Reuses the exact pipeline path (TeacherModule, mode=decompose,
skip_integrate_answer_check=True, same as recover_integrate_milestones.py);
only the LLM client changes: thinkingmachines/Inkling sampled through Tinker.

Usage:
  python3 scripts/experiments/teacher_consistency_inkling.py --n 2   # dev check
  python3 scripts/experiments/teacher_consistency_inkling.py        # 60 families, resumable
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

# transformers 5.14.1 duplicate-kwarg bug for Inkling's tokenizer config
import transformers.tokenization_utils_tokenizers as _tut
_orig_init = _tut.TokenizersBackend.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("fix_mistral_regex", None)
    return _orig_init(self, *args, **kwargs)
_tut.TokenizersBackend.__init__ = _patched_init

from decomposer.common.data_types import Problem
from decomposer.teacher.teacher import TeacherModule

INPUT_FN = ROOT / "data/logs/rl/diagnostic_multi_families_with_integrate.jsonl"
OUT_DIR = ROOT / "data/logs/rl/teacher_consistency"
OUT_FN = OUT_DIR / "inkling_packets.jsonl"
MODEL = "thinkingmachines/Inkling"
N_FAMILIES = 60
SEED = 42
CONCURRENCY = 60
MAX_TOKENS = 24000


class InklingTinkerClient:
    """Minimal chat() adapter: Inkling chat template in, content_text out."""

    def __init__(self):
        import tinker
        from transformers import AutoTokenizer

        self._types = __import__("tinker").types
        self.sampling = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
        self.tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    def chat(self, messages, *, model=None, temperature=0.1, max_tokens=MAX_TOKENS, extra_body=None):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        params = self._types.SamplingParams(max_tokens=max_tokens, temperature=temperature)
        prompt = self._types.ModelInput.from_ints(ids)
        res = self.sampling.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        text = self.tok.decode(res.sequences[0].tokens)
        if "<|content_text|>" in text:
            text = text.split("<|content_text|>")[-1]
        return text.split("<|content_model_end_sampling|>")[0].strip()


def build_problem(fam: dict) -> Problem:
    prompt = fam["parent_prompt"]
    marker = "Problem:\n"
    idx = prompt.find(marker)
    statement = prompt[idx + len(marker):].rstrip() if idx >= 0 else prompt.strip()
    return Problem(
        problem_id=fam["pid"],
        statement=statement,
        gold_answer=fam["parent_answer"],
        reference_solution="",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_FAMILIES)
    args = ap.parse_args()

    fams = [json.loads(l) for l in open(INPUT_FN)]
    rng = random.Random(SEED)
    sample = rng.sample(fams, N_FAMILIES)[: args.n]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_FN.exists():
        for line in open(OUT_FN):
            try:
                row = json.loads(line)
                if "error" not in row:  # errored families retry on resume
                    done.add(row["pid"])
            except Exception:
                pass
    todo = [f for f in sample if f["pid"] not in done]
    print(f"sample={len(sample)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    client = InklingTinkerClient()
    teacher = TeacherModule(
        llm_client=client,
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        max_retries=2,
        skip_integrate_answer_check=True,
    )

    def work(fam: dict) -> dict:
        t0 = time.time()
        try:
            packet = teacher.generate_packet(problem=build_problem(fam), mode="decompose")
            return {
                "pid": fam["pid"],
                "teacher": MODEL,
                "milestones": packet.get("milestones", []),
                "elapsed_secs": round(time.time() - t0, 1),
            }
        except Exception as e:
            return {
                "pid": fam["pid"],
                "teacher": MODEL,
                "error": f"{type(e).__name__}: {e}"[:300],
                "elapsed_secs": round(time.time() - t0, 1),
            }

    n_done = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(work, f): f for f in todo}
        for fut in as_completed(futs):
            row = fut.result()
            with open(OUT_FN, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
            status = "ERR" if "error" in row else f"{len(row.get('milestones', []))} ms"
            print(f"[{n_done}/{len(todo)}] {row['pid'][:8]} {status} {row['elapsed_secs']}s", flush=True)


if __name__ == "__main__":
    main()
