#!/usr/bin/env python3
"""P0-b rebuttal experiment: frontier LLM-as-judge vs strict symbolic cascade.

Replicates the App S paired-audit protocol with the judge swapped from
Qwen3-8B no-think to a frontier model (default openai/gpt-5.5 via Prime).
The judge prompt is byte-identical to the paper's (VERIFIER_SYSTEM_PROMPT +
build_verifier_prompt); the strict side is the local cascade with no LLM
fallback. Comparison target, as in the paper: the canonical answer.

Sources:
  in-dist  data/logs/rl/format_audit/*.jsonl   (parent rollouts, 4 models, raw text)
  OOD      data/logs/rl/eval_results/responses/*aime*.jsonl

Usage:
  python3 scripts/experiments/frontier_judge_audit.py --dry-run       # 10 cases
  python3 scripts/experiments/frontier_judge_audit.py                 # full run, resumable
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from decomposer.common.prompts import VERIFIER_SYSTEM_PROMPT, build_verifier_prompt
from decomposer.verifier.verifier import VerifierModule

JUDGE_MODEL = "openai/gpt-5.5"
PRIME_BASE = "https://api.pinference.ai/api/v1"
N_INDIST = 700
N_AIME = 300
SEED = 42
CONCURRENCY = 64
RESPONSE_CHAR_CAP = 12000  # keep the tail; boxed answers live at the end
OUT_DIR = ROOT / "data/logs/rl/frontier_judge_audit"
OUT_FILE = OUT_DIR / "paired_gpt_5_5.jsonl"
SUMMARY_FILE = OUT_DIR / "summary_gpt_5_5.json"


def load_canonicals() -> dict[str, tuple[str, str]]:
    out = {}
    with open(ROOT / "data/logs/rl/diagnostic_multi_families_repaired.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["pid"]] = (d["parent_answer"], d.get("parent_note") or "ACCEPT IF equivalent final answer is present.")
    return out


def build_pool() -> list[dict]:
    rng = random.Random(SEED)
    canon = load_canonicals()

    indist = []
    for f in sorted((ROOT / "data/logs/rl/format_audit").glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                if d["pid"] not in canon:
                    continue
                ans, note = canon[d["pid"]]
                indist.append({
                    "source": "numina_parent",
                    "model": d["model"],
                    "condition": d["condition"],
                    "bucket": d["bucket"],
                    "pid": d["pid"],
                    "rollout": d.get("rollout"),
                    "response": d["response"],
                    "answer": ans,
                    "note": note,
                })
    rng.shuffle(indist)
    indist = indist[:N_INDIST]

    aime = []
    for f in sorted((ROOT / "data/logs/rl/eval_results/responses").glob("*aime*.jsonl")):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                for i, resp in enumerate(d.get("responses") or []):
                    aime.append({
                        "source": "aime",
                        "model": f.stem,
                        "condition": "direct",
                        "bucket": None,
                        "pid": str(d["pid"]),
                        "rollout": i,
                        "response": resp,
                        "answer": str(d["answer"]),
                        "note": "ACCEPT IF equivalent final answer is present.",
                    })
    rng.shuffle(aime)
    aime = aime[:N_AIME]

    pool = indist + aime
    for i, c in enumerate(pool):
        c["case_id"] = f"{c['source']}:{c['pid']}:{c['model']}:{c['condition']}:{c['rollout']}"
    return pool


def judge_one(client, case: dict) -> dict:
    resp_text = case["response"][-RESPONSE_CHAR_CAP:]
    user = build_verifier_prompt(answer=case["answer"], note=case["note"], response=resp_text)
    r = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=2000,
        timeout=600,
    )
    text = r.choices[0].message.content or ""
    # brace-blind {...} regex loses verdicts whose reason contains LaTeX braces;
    # match the label field directly instead
    m = re.search(r'"label"\s*:\s*"(ACCEPT|NOT_ACCEPT)"', text)
    label = m.group(1) if m else "PARSE_FAIL"
    rm = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    reason = (rm.group(1) if rm else text)[:200]
    usage = getattr(r, "usage", None)
    cost = None
    if usage is not None:
        cost = getattr(usage, "cost", None) or (usage.model_dump().get("cost") if hasattr(usage, "model_dump") else None)
    return {"judge_label": label, "judge_reason": reason, "judge_cost": cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from openai import OpenAI
    key = os.environ.get("PRIME_API_KEY")
    if not key:
        print("ERROR: PRIME_API_KEY not set (source ~/.bashrc)")
        sys.exit(1)
    client = OpenAI(api_key=key, base_url=PRIME_BASE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    if "error" not in row:  # errored cases retry on resume
                        done.add(row["case_id"])
                except Exception:
                    pass

    pool = build_pool()
    if args.dry_run:
        pool = pool[:10]
    todo = [c for c in pool if c["case_id"] not in done]
    print(f"pool={len(pool)} done={len(done)} todo={len(todo)}", flush=True)

    strict = VerifierModule(llm_client=None, llm_client_nothink=None)
    lock = threading.Lock()

    def work(case: dict) -> dict:
        # grade strict on the same truncated input the judge sees
        sv = strict.verify(response=case["response"][-RESPONSE_CHAR_CAP:], answer=case["answer"], note=case["note"])
        jv = judge_one(client, case)
        row = {k: case[k] for k in ("case_id", "source", "model", "condition", "bucket", "pid", "rollout")}
        row["strict_label"] = sv["label"]
        row["strict_reason"] = str(sv["reason"])[:120]
        row.update(jv)
        return row

    t0 = time.monotonic()
    n_done = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for fut in as_completed(futs):
            case = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"case_id": case["case_id"], "error": f"{type(e).__name__}: {e}"[:200]}
            with lock:
                with open(OUT_FILE, "a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    print(f"{n_done}/{len(todo)} elapsed={time.monotonic()-t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(OUT_FILE)]
    rows = [r for r in rows if "error" not in r and r.get("judge_label") in ("ACCEPT", "NOT_ACCEPT")]
    n = len(rows)
    both_a = sum(1 for r in rows if r["strict_label"] == "ACCEPT" and r["judge_label"] == "ACCEPT")
    both_n = sum(1 for r in rows if r["strict_label"] == "NOT_ACCEPT" and r["judge_label"] == "NOT_ACCEPT")
    over = [r for r in rows if r["strict_label"] == "NOT_ACCEPT" and r["judge_label"] == "ACCEPT"]
    under = [r for r in rows if r["strict_label"] == "ACCEPT" and r["judge_label"] == "NOT_ACCEPT"]
    costs = [r["judge_cost"] for r in rows if r.get("judge_cost")]
    summary = {
        "judge_model": JUDGE_MODEL,
        "n_paired": n,
        "both_accept": both_a,
        "both_not_accept": both_n,
        "judge_accept_strict_reject": len(over),
        "judge_reject_strict_accept": len(under),
        "over_accept_rate_of_total": round(len(over) / n, 4) if n else None,
        "under_accept_rate_of_total": round(len(under) / n, 4) if n else None,
        "total_judge_cost_usd": round(sum(costs), 2) if costs else None,
        "by_source": {},
    }
    for src in ("numina_parent", "aime"):
        sub = [r for r in rows if r["source"] == src]
        if sub:
            summary["by_source"][src] = {
                "n": len(sub),
                "over": sum(1 for r in sub if r["strict_label"] == "NOT_ACCEPT" and r["judge_label"] == "ACCEPT"),
                "under": sum(1 for r in sub if r["strict_label"] == "ACCEPT" and r["judge_label"] == "NOT_ACCEPT"),
            }
    with open(SUMMARY_FILE, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
