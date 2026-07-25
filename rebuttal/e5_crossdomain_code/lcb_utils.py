"""LiveCodeBench loading + deterministic execution grading for the cross-domain pilot.

Scope: stdin-type problems only (codeforces/atcoder style: full program, stdin->stdout).
Grading = run the program on every test input in a subprocess sandbox and compare
normalized stdout. Fully deterministic; no LLM judge anywhere.
"""
from __future__ import annotations

import base64
import json
import pickle
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

LCB_PATH = (os.path.expanduser("~/.cache/huggingface/hub/datasets--livecodebench--"
            "code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl")


def load_problems(difficulty: str | None = "medium") -> list[dict]:
    rows = [json.loads(l) for l in open(LCB_PATH)]
    out = []
    for r in rows:
        pub = json.loads(r["public_test_cases"])
        if not pub or pub[0].get("testtype") != "stdin":
            continue
        if difficulty and r["difficulty"] != difficulty:
            continue
        try:
            priv = json.loads(r["private_test_cases"])
        except Exception:
            priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"]))))
        tests = [(t["input"], t["output"]) for t in pub + priv if t.get("testtype") == "stdin"]
        if len(tests) < 2:
            continue
        out.append({
            "qid": r["question_id"],
            "title": r["question_title"],
            "platform": r["platform"],
            "difficulty": r["difficulty"],
            "date": r["contest_date"],
            "statement": r["question_content"],
            "tests": tests,
        })
    return out


def _norm(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def run_program(code: str, stdin_text: str, timeout: float = 10.0) -> tuple[str, str]:
    """Returns (status, stdout). status in {ok, error, timeout}."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], input=stdin_text,
                           capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return "error", (p.stderr or "")[-500:]
        return "ok", p.stdout
    except subprocess.TimeoutExpired:
        return "timeout", ""
    finally:
        Path(path).unlink(missing_ok=True)


def grade_solution(code: str, tests: list[tuple[str, str]], timeout: float = 10.0,
                   max_tests: int = 12) -> dict:
    """Run code on up to max_tests cases; pass iff all executed cases match."""
    n_run = n_pass = 0
    first_fail = None
    for inp, exp in tests[:max_tests]:
        status, out = run_program(code, inp, timeout=timeout)
        n_run += 1
        if status == "ok" and _norm(out) == _norm(exp):
            n_pass += 1
        elif first_fail is None:
            first_fail = {"status": status, "got": out[:200], "expected": exp[:200]}
    return {"pass": n_run > 0 and n_pass == n_run, "n_pass": n_pass, "n_run": n_run,
            "first_fail": first_fail}


def extract_code(response: str) -> str | None:
    """Pull the last ```python ...``` block (or last ``` block) from a response."""
    import re
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response or "", re.DOTALL)
    return blocks[-1].strip() if blocks else None
