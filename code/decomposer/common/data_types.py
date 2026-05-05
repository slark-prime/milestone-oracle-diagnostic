from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

ItemKind = Literal["problem", "milestone"]


@dataclass(slots=True)
class Problem:
    problem_id: str
    statement: str
    gold_answer: str
    reference_solution: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_problem_block(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "statement": self.statement,
            "solution_reference": self.reference_solution,
            "gold_answer": self.gold_answer,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Milestone:
    id: str
    type: str
    description: str
    answer: str
    note: str
    hints: Optional[list[str]] = None
    decompose_advice: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v is not None}


@dataclass(slots=True)
class Attempt:
    problem_id: str
    milestone_id: Optional[str]
    student_run_id: str
    response: str
    verdict: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PassStats:
    problem_id: str
    milestone_id: Optional[str]
    k: int
    passed: int
    p_hat: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "milestone_id": self.milestone_id,
            "k": self.k,
            "pass": self.passed,
            "p_hat": self.p_hat,
        }


@dataclass(slots=True)
class TrainingRow:
    train_id: str
    source_problem_id: str
    source_item: ItemKind
    prompt: str
    target: str
    milestone_id: Optional[str] = None
    verifier: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
