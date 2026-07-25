#!/usr/bin/env python3
"""Few-shot answer-atomicity augmentation for the teacher prompt.

The submission's teacher prompt constrains the gold answer only with "(short)".
GPT-5.4 reads that as one atomic value (95% atomic, median 7 chars); Inkling does
not (31% atomic, median 55 chars), which made two experiments unrunnable:
  - Stage 0 on Inkling roadmaps (0/53 families fully gradable)
  - the decisive AC-Q2 test (this roadmap vs an independent valid decomposition,
    which needs comparably atomic golds on both sides)

This module supplies an *additive* system-prompt suffix with explicit atomicity
rules and few-shot good/bad examples. `decomposer/common/prompts.py` is left
untouched so the submission's protocol is unchanged.

Usage:
  python3 scripts/experiments/atomic_gold_fewshot.py --n 10        # measure atomic rate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import transformers.tokenization_utils_tokenizers as _tut
_orig_init = _tut.TokenizersBackend.__init__
def _patched_init(self, *a, **k):
    k.pop("fix_mistral_regex", None)
    return _orig_init(self, *a, **k)
_tut.TokenizersBackend.__init__ = _patched_init

from decomposer.common.data_types import Problem
from decomposer.teacher.teacher import TeacherModule
from teacher_consistency_inkling import InklingTinkerClient, build_problem

ATOMIC_SUFFIX = r"""

ANSWER ATOMICITY RULE (required — the answer is graded by a symbolic verifier, not a human):
- "answer" MUST be ONE gradable object: a single value, expression, equation, interval, or tuple.
- It MUST NOT chain multiple results, MUST NOT contain prose, and MUST NOT use ";" to join claims.
- Write LaTeX, never Unicode math (use \sqrt{2} not √2; \in not ∈; \leq not ≤; ^2 not ²).
- Prefer under 30 characters.
- If a milestone naturally produces several results, SPLIT it into several milestones — one result each.

GOOD answers:
  "3"
  "\frac{19}{4}"
  "c = 2a"
  "(\sqrt{7}, \sqrt{7}, 0)"
  "n \neq 2^{k}"
  "a \leq -2"

BAD answers (and why):
  "|PB|^2=(5p^2-16p+16)/4; |PD|^2=(5p^2-8p+16)/4; PB\cdot PD=(5p^2-12p)/4"   -> three results chained; split into three milestones
  "Pairs: (1,2),(2,4),(3,6),(4,8); largest a=4 => c=8"                        -> prose + two results; split
  "N = 10001a + 1010b + 100c with a\in[1,9]; maximize N subject to 101|N"     -> prose constraint, not a gradable value
  "Palindrome confirmed; 49894 = 101 \times 494"                              -> prose + result
  "u\cdot v = |u||v|\cos\theta; squaring gives ..."                           -> derivation, not an answer
"""


def is_atomic(gold: str) -> bool:
    g = (gold or "").strip()
    if not g or len(g) > 60:
        return False
    if ";" in g:
        return False
    if re.search(r"[√≤≥≠∈∉π°×·−∞⇒⇔ℝℤℕ²³±≈∑∫]", g):
        return False
    words = re.findall(r"[A-Za-z]{4,}", re.sub(r"\\[a-zA-Z]+", " ", g))
    prose = {"pairs", "largest", "with", "subject", "confirmed", "gives", "where",
             "such", "that", "then", "maximize", "minimize", "and", "since", "thus"}
    return not any(w.lower() in prose for w in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="data/logs/rl/teacher_consistency/inkling_atomic_packets.jsonl")
    a = ap.parse_args()

    # target the families that overlap the released C1 traces, so the decisive
    # AC-Q2 comparison becomes runnable
    trace_pids = set()
    with open(ROOT / "data/logs/rl/format_audit/mile_2k_step_180.jsonl") as fh:
        for l in fh:
            r = json.loads(l)
            if r["condition"] == "C1_direct":
                trace_pids.add(r["pid"])
    fams = [json.loads(l) for l in open(ROOT / "data/logs/rl/diagnostic_multi_families_repaired.jsonl")]
    targets = [f for f in fams if f["pid"] in trace_pids][: a.n]
    print(f"targets: {len(targets)} families (all have released C1 traces)", flush=True)

    out_fn = ROOT / a.out
    out_fn.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_fn.exists():
        for l in open(out_fn):
            r = json.loads(l)
            if "error" not in r:
                done.add(r["pid"])
    todo = [f for f in targets if f["pid"] not in done]
    print(f"todo: {len(todo)}", flush=True)

    if todo:
        client = InklingTinkerClient()
        teacher = TeacherModule(llm_client=client, temperature=0.1, max_tokens=24000,
                                max_retries=2, skip_integrate_answer_check=True)
        # additive suffix: patch only this teacher instance's system prompt
        import decomposer.teacher.teacher as tmod
        orig_sp = tmod.TEACHER_SYSTEM_PROMPT
        tmod.TEACHER_SYSTEM_PROMPT = orig_sp + ATOMIC_SUFFIX

        def work(fam):
            try:
                pkt = teacher.generate_packet(problem=build_problem(fam), mode="decompose")
                return {"pid": fam["pid"], "teacher": "thinkingmachines/Inkling+atomic_fewshot",
                        "milestones": pkt.get("milestones", [])}
            except Exception as e:
                return {"pid": fam["pid"], "error": f"{type(e).__name__}: {e}"[:200]}

        with ThreadPoolExecutor(max_workers=min(20, len(todo))) as ex:
            for fut in as_completed([ex.submit(work, f) for f in todo]):
                r = fut.result()
                with open(out_fn, "a") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n = len(r.get("milestones", [])) if "error" not in r else "ERR"
                print(f"  {r['pid'][:8]} {n}", flush=True)
        tmod.TEACHER_SYSTEM_PROMPT = orig_sp

    rows = [json.loads(l) for l in open(out_fn)]
    ok = [r for r in rows if "error" not in r]
    golds = [str(m.get("answer", "")) for r in ok for m in r["milestones"] if m.get("type") != "INTEGRATE"]
    if not golds:
        print("no golds"); return
    at = [g for g in golds if is_atomic(g)]
    lens = sorted(len(g) for g in golds)
    print(f"\n=== Inkling + atomic few-shot ===")
    print(f"packets ok: {len(ok)}/{len(rows)} | non-INTEGRATE golds: {len(golds)}")
    print(f"ATOMIC RATE: {len(at)}/{len(golds)} = {len(at)/len(golds):.0%}   (baseline Inkling 31%, GPT-5.4 95%)")
    print(f"median gold length: {lens[len(lens)//2]}   (baseline Inkling 55, GPT-5.4 7)")
    print("samples:", [g[:38] for g in golds[:8]])


if __name__ == "__main__":
    main()
