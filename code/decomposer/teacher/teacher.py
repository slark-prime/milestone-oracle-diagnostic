from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from decomposer.common.prompts import (
    HINTS_FORMAT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    build_teacher_user_prompt,
)
from decomposer.common.schemas import MILESTONE_PACKET_SCHEMA, validate_schema
from decomposer.common.data_types import Problem
from decomposer.verifier.symbolic import is_equivalent

logger = logging.getLogger(__name__)


class TeacherModule:
    def __init__(
        self,
        *,
        llm_client: Any,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        max_retries: int = 3,
        skip_integrate_answer_check: bool = False,
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(1, int(max_retries))
        self.skip_integrate_answer_check = bool(skip_integrate_answer_check)

    def generate_packet(
        self,
        *,
        problem: Problem,
        mode: str = "decompose",
        target_statement: Optional[str] = None,
        target_answer: Optional[str] = None,
        target_decompose_advice: Optional[list[str]] = None,
        target_hints: Optional[list[str]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        user_prompt = build_teacher_user_prompt(
            statement=problem.statement,
            solution_reference=problem.reference_solution,
            mode=mode,
            target_statement=target_statement,
            target_answer=target_answer,
            target_decompose_advice=target_decompose_advice,
            target_hints=target_hints,
        )
        system_prompt = HINTS_FORMAT_SYSTEM_PROMPT if mode == "hints" else TEACHER_SYSTEM_PROMPT
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = self.llm_client.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body=extra_body,
                )
                raw = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
                return self.parse_and_validate(raw_output=raw, problem=problem, mode=mode)
            except ValueError as e:
                last_error = e
                if attempt + 1 < self.max_retries:
                    logger.info("retry %d/%d after: %s", attempt + 2, self.max_retries, e)
                else:
                    raise
        raise last_error or RuntimeError("Teacher generate_packet failed")

    def generate_packets_batch(
        self,
        *,
        problem: Problem,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            self.generate_packet(
                problem=problem,
                mode=str(request.get("mode", "decompose")),
                target_statement=request.get("target_statement"),
                target_answer=request.get("target_answer"),
                target_decompose_advice=request.get("target_decompose_advice"),
                target_hints=request.get("target_hints"),
                extra_body=request.get("extra_body"),
            )
            for request in requests
        ]

    def parse_and_validate(self, *, raw_output: str, problem: Problem, mode: str) -> dict[str, Any]:
        payload = self._parse_json(raw_output)
        # Assign milestone IDs programmatically (teacher prompt no longer produces them)
        self._assign_milestone_ids(payload, mode=mode)
        # Keep canonical private reference fields from the dataset.
        payload["problem"] = problem.to_problem_block()
        validate_schema(payload, MILESTONE_PACKET_SCHEMA, title="teacher packet")
        self._enforce_invariants(payload=payload, problem=problem, mode=mode)
        return payload

    @staticmethod
    def _assign_milestone_ids(payload: dict[str, Any], *, mode: str) -> None:
        """Assign IDs based on mode: core→C1..Cn, decompose→D1..Dn, hints→H1..Hn."""
        prefix_map = {"core": "C", "decompose": "D", "hints": "H"}
        prefix = prefix_map.get(mode, "M")
        for idx, milestone in enumerate(payload.get("milestones", []), start=1):
            milestone["id"] = f"{prefix}{idx}"

    @staticmethod
    def _parse_json(raw_output: str) -> dict[str, Any]:
        raw = (raw_output or "").strip()
        parsed = None

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Recover if model wrapped JSON with text.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Teacher output contains malformed JSON (parse error at char {e.pos}): "
                        f"{raw[:500]}{'...' if len(raw) > 500 else ''}"
                    ) from e
            if parsed is None:
                raise ValueError(
                    f"Teacher output is not valid JSON (no {{...}} found): "
                    f"{raw[:500]}{'...' if len(raw) > 500 else ''}"
                )

        if not isinstance(parsed, dict):
            raise ValueError("Teacher output must be a JSON object")
        return parsed

    def _enforce_invariants(self, *, payload: dict[str, Any], problem: Problem, mode: str) -> None:
        milestones = payload.get("milestones", [])
        if mode != "hints" and not any(milestone.get("type") == "INTEGRATE" for milestone in milestones):
            raise ValueError("Teacher packet must include at least one INTEGRATE milestone")

        for milestone in milestones:
            if "hints" in milestone and "decompose_advice" in milestone:
                del milestone["hints"]
        if mode == "hints":
            return
        if self.skip_integrate_answer_check:
            return
        integrate_milestones = [m for m in milestones if m.get("type") == "INTEGRATE"]
        last_integrate = integrate_milestones[-1]
        equivalent, reason = is_equivalent(
            last_integrate.get("answer", ""),
            problem.gold_answer,
        )
        if not equivalent:
            raise ValueError(
                "INTEGRATE milestone answer is not equivalent to gold_answer: "
                f"{reason}"
            )
