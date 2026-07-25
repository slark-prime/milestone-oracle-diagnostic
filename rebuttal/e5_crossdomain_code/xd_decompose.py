#!/usr/bin/env python3
"""Cross-domain pilot, Track 2: Inkling decomposes LCB problems into
helper-function milestones with an executable compile-time verification loop.

Packet schema (JSON from the teacher):
{
  "helpers": [{"name": str, "signature": str, "docstring": str,
                "tests": [assert-statement strings], "gold_impl": str}],
  "main_plan": str,
  "gold_main": str   # full stdin->stdout program that may inline the helpers
}

Compile-time checks (all executable, no judgment calls):
  C-a each helper's gold_impl passes its own asserts;
  C-b gold_main passes ALL parent tests  -> roadmap sufficiency PROVEN by construction.
Packets failing either check are rejected (one retry with error feedback).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lcb_utils import load_problems, grade_solution, run_program

import transformers.tokenization_utils_tokenizers as _tut
_orig_init = _tut.TokenizersBackend.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("fix_mistral_regex", None)
    return _orig_init(self, *args, **kwargs)
_tut.TokenizersBackend.__init__ = _patched_init

import tinker
from tinker import types as tt
from transformers import AutoTokenizer

MODEL = "thinkingmachines/Inkling"
OUT_FN = ROOT / "data/logs/rl/cg2/xdomain/inkling_code_packets.jsonl"
MAX_TOKENS = 24000

PROMPT = """You are decomposing a competitive-programming problem into verifiable sub-goal milestones for a diagnostic evaluation.

Problem:
{statement}

Produce a JSON object, and nothing else, with this exact schema:
{{
  "helpers": [
    {{"name": "<function name>",
      "signature": "def name(args) -> ret",
      "docstring": "<what it computes, self-contained>",
      "tests": ["assert name(...) == ...", "assert name(...) == ..."],
      "gold_impl": "def name(...):\\n    ..."}}
  ],
  "main_plan": "<how a main program combines the helpers to solve the problem>",
  "gold_main": "<complete Python program reading stdin and writing stdout; it may redefine the helpers inline>"
}}

Requirements:
- 2 to 5 helpers, each a meaningful sub-goal (not trivial one-liners), each independently implementable from its signature + docstring alone.
- Each helper has >= 2 assert tests that its gold_impl passes.
- gold_main must correctly solve the problem end-to-end.
- Helpers must not read stdin; only gold_main does I/O.
- Before writing each assert, compute its expected output by hand and double-check it against your gold_impl; every assert must actually hold.
- Prefer small, pure, deterministic helpers (no randomness, no I/O, no global state).
- Output raw JSON only. No markdown fences, no commentary."""


class Client:
    def __init__(self):
        self.cli = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
        self.tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    def gen(self, text, temperature=0.1):
        ids = self.tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        res = self.cli.sample(prompt=tt.ModelInput.from_ints(ids), num_samples=1,
                              sampling_params=tt.SamplingParams(max_tokens=MAX_TOKENS,
                                                                temperature=temperature)).result()
        out = self.tok.decode(res.sequences[0].tokens)
        if "<|content_text|>" in out:
            out = out.split("<|content_text|>")[-1]
        return out.split("<|content_model_end_sampling|>")[0].split("<|end_message|>")[0].strip()


def parse_packet(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found")
    pk = json.loads(m.group(0))
    assert isinstance(pk.get("helpers"), list) and 2 <= len(pk["helpers"]) <= 6, "bad helpers"
    for h in pk["helpers"]:
        for k in ("name", "signature", "docstring", "tests", "gold_impl"):
            assert h.get(k), f"helper missing {k}"
        assert len(h["tests"]) >= 2, "need >=2 tests"
    assert pk.get("gold_main") and pk.get("main_plan"), "missing main"
    return pk


def compile_check(pk: dict, tests: list) -> tuple[bool, str]:
    # C-a: each gold helper passes its own asserts
    for h in pk["helpers"]:
        prog = h["gold_impl"] + "\n\n" + "\n".join(h["tests"]) + "\nprint('HELPER_OK')\n"
        status, out = run_program(prog, "", timeout=10)
        if status != "ok" or "HELPER_OK" not in out:
            return False, f"helper {h['name']} fails own tests: {out[:150]}"
    # C-b: gold_main passes all parent tests
    g = grade_solution(pk["gold_main"], tests, timeout=15)
    if not g["pass"]:
        return False, f"gold_main fails parent tests ({g['n_pass']}/{g['n_run']}): {str(g['first_fail'])[:150]}"
    return True, "ok"


def main():
    cands = set(json.load(open(ROOT / "data/logs/rl/cg2/xdomain/xd_candidate_qids.json")))
    probs = [p for p in load_problems("medium") if p["qid"] in cands]
    smoke = int(os.environ.get("XD_SMOKE", "0"))
    if smoke:
        probs = probs[:smoke]

    done = set()
    if OUT_FN.exists():
        for line in open(OUT_FN):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["qid"])
            except Exception:
                pass
    todo = [p for p in probs if p["qid"] not in done]
    print(f"candidates={len(probs)} todo={len(todo)}", flush=True)
    if not todo:
        return

    client = Client()

    def work(p):
        t0 = time.time()
        err_note = ""
        for attempt in range(3):
            try:
                text = client.gen(PROMPT.format(statement=p["statement"]) +
                                  (f"\n\nYour previous attempt failed verification: {err_note}. Fix it."
                                   if err_note else ""))
                pk = parse_packet(text)
                ok, msg = compile_check(pk, p["tests"])
                if ok:
                    return {"qid": p["qid"], "title": p["title"], "packet": pk,
                            "attempts": attempt + 1, "elapsed": round(time.time() - t0, 1)}
                err_note = msg
            except Exception as e:
                err_note = f"{type(e).__name__}: {e}"[:200]
        return {"qid": p["qid"], "error": err_note, "elapsed": round(time.time() - t0, 1)}

    n = 0
    with ThreadPoolExecutor(max_workers=31) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for fut in as_completed(futs):
            row = fut.result()
            with open(OUT_FN, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            status = "ERR " + row["error"][:60] if "error" in row else \
                f"{len(row['packet']['helpers'])} helpers, verified, {row['attempts']} attempt(s)"
            print(f"[{n}/{len(todo)}] {row['qid'][:20]} {status} {row['elapsed']}s", flush=True)


if __name__ == "__main__":
    main()
