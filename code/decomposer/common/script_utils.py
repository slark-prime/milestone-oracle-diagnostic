"""Shared utilities for run_teacher, run_student, run_verifier scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from decomposer.common.data_types import Problem
from decomposer.common.schemas import STANDARD_PROBLEM_SCHEMA, validate_schema


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def to_problem(row: dict[str, Any], *, line_no: int) -> Problem:
    validate_schema(row, STANDARD_PROBLEM_SCHEMA, title=f"problem line {line_no}")
    return Problem(
        problem_id=row["problem_id"],
        statement=row["statement"],
        gold_answer=row["gold_answer"],
        reference_solution=row["reference_solution"],
        meta=row.get("meta", {}),
    )


def select_indices(
    *,
    total: int,
    line: int | None,
    random_k: int | None,
    seed: int,
) -> list[int]:
    if total == 0:
        raise ValueError("No rows found in problems JSONL")

    if line is not None:
        if line < 1 or line > total:
            raise ValueError(f"--line must be in [1, {total}], got {line}")
        return [line - 1]

    if random_k is not None:
        if random_k < 1:
            raise ValueError("--random-k must be >= 1")
        k = min(random_k, total)
        rng = random.Random(seed)
        return rng.sample(range(total), k)

    return [0]
