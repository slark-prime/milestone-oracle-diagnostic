from __future__ import annotations

import uuid
from typing import Optional

from decomposer.common.data_types import Milestone, Problem, TrainingRow
from decomposer.common.prompts import build_student_milestone_prompt, build_student_problem_prompt


def build_problem_training_row(problem: Problem, *, p_hat: Optional[float] = None) -> TrainingRow:
    meta: dict = {"source": "core_loop_keep"}
    if p_hat is not None:
        meta["p_hat"] = p_hat
    return TrainingRow(
        train_id=str(uuid.uuid4()),
        source_problem_id=problem.problem_id,
        source_item="problem",
        prompt=build_student_problem_prompt(problem.statement),
        target=problem.gold_answer,
        verifier={
            "expected_answer": problem.gold_answer,
            "note": "ACCEPT IF equivalent final answer is present.",
        },
        meta=meta,
    )


def build_milestone_training_row(
    problem: Problem,
    milestone: Milestone,
    *,
    p_hat: Optional[float] = None,
    depth: Optional[int] = None,
) -> TrainingRow:
    prompt = build_student_milestone_prompt(
        original_problem=problem.statement,
        milestone_description=milestone.description,
        include_original_problem=True,
        hints=milestone.hints,
    )
    meta: dict = {
        "milestone_id": milestone.id,
        "milestone_type": milestone.type,
        "source": "core_loop_decompose",
    }
    if p_hat is not None:
        meta["p_hat"] = p_hat
    if depth is not None:
        meta["depth"] = depth
    return TrainingRow(
        train_id=str(uuid.uuid4()),
        source_problem_id=problem.problem_id,
        source_item="milestone",
        prompt=prompt,
        target=milestone.answer,
        milestone_id=milestone.id,
        verifier={
            "expected_answer": milestone.answer,
            "note": milestone.note,
        },
        meta=meta,
    )
