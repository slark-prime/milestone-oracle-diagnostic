from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from decomposer.common.data_types import Attempt, Milestone, PassStats, Problem, TrainingRow
from decomposer.pipeline.build_training import (
    build_milestone_training_row,
    build_problem_training_row,
)

logger = logging.getLogger(__name__)


def _pid(problem: Problem) -> str:
    """Short problem ID prefix for log lines."""
    return problem.problem_id[:8]


def _log_packet(packet: dict[str, Any], *, label: str, problem: Problem) -> None:
    """Log packet as JSON (excludes problem block for brevity)."""
    if not logger.isEnabledFor(logging.INFO):
        return
    out = {k: v for k, v in packet.items() if k != "problem"}
    try:
        s = json.dumps(out, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(out)
    logger.info("[%s] --- %s ---\n%s\n--- end ---", _pid(problem), label, s)


@dataclass(slots=True)
class CoreLoopConfig:
    k: int = 8
    keep_threshold: float = 0.6
    max_depth: int = 2
    student_temperature: float = 0.7
    student_max_tokens: int = 512
    include_original_problem_in_milestone_prompt: bool = True
    student_sample_concurrency: int = 1
    include_full_problem_when_decomposed: bool = False
    problem_concurrency: int = 1
    # When True, run student rollouts on teacher-generated endpoint milestones
    # (hints or atomic/max-depth nodes) and record p_hat in the training row meta.
    evaluate_endpoint_pass_rate: bool = False


@dataclass(slots=True)
class CoreLoopResult:
    training_rows: list[TrainingRow] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    stats: list[PassStats] = field(default_factory=list)
    packets: list[dict[str, Any]] = field(default_factory=list)


class CoreLoop:
    def __init__(self, *, teacher_module: Any, student_module: Any, verifier_module: Any, config: CoreLoopConfig):
        self.teacher = teacher_module
        self.student = student_module
        self.verifier = verifier_module
        self.config = config

    def run(self, problems: list[Problem]) -> CoreLoopResult:
        concurrency = max(1, min(self.config.problem_concurrency, len(problems)))
        if concurrency == 1:
            result = CoreLoopResult()
            for idx, problem in enumerate(problems, start=1):
                logger.info("problem %d/%d: %s", idx, len(problems), problem.problem_id)
                try:
                    self._process_problem(problem=problem, result=result)
                except Exception:
                    logger.exception("[%s] failed, skipping", _pid(problem))
            return result

        def _run_one(args: tuple[int, Problem]) -> CoreLoopResult:
            idx, problem = args
            logger.info("problem %d/%d: %s", idx, len(problems), problem.problem_id)
            r = CoreLoopResult()
            try:
                self._process_problem(problem=problem, result=r)
            except Exception:
                logger.exception("[%s] failed, skipping", _pid(problem))
            return r

        merged = CoreLoopResult()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for r in executor.map(_run_one, enumerate(problems, start=1)):
                merged.training_rows.extend(r.training_rows)
                merged.attempts.extend(r.attempts)
                merged.stats.extend(r.stats)
                merged.packets.extend(r.packets)
        return merged

    def _process_problem(self, *, problem: Problem, result: CoreLoopResult) -> None:
        p = _pid(problem)
        logger.info("[%s] evaluating problem (k=%d)...", p, self.config.k)
        stat = self._evaluate_problem(problem=problem, result=result)
        logger.info("[%s] p_hat=%.2f (threshold=%.2f)", p, stat.p_hat, self.config.keep_threshold)
        if stat.p_hat >= self.config.keep_threshold:
            logger.info("[%s] -> KEEP (problem passed)", p)
            result.training_rows.append(build_problem_training_row(problem, p_hat=stat.p_hat))
            return

        logger.info("[%s] -> DECOMPOSE (teacher mode=core)", p)
        packet = self.teacher.generate_packet(problem=problem, mode="core")
        _log_packet(packet, label="teacher packet (mode=core)", problem=problem)
        result.packets.append(packet)
        milestones = self._milestones_from_packet(packet)
        logger.info("[%s] got %d milestones: %s", p, len(milestones), [m.id for m in milestones])
        self._process_milestones(problem=problem, milestones=milestones, depth=1, result=result)

        if self.config.include_full_problem_when_decomposed:
            result.training_rows.append(build_problem_training_row(problem))

    def _process_milestones(
        self,
        *,
        problem: Problem,
        milestones: list[Milestone],
        depth: int,
        result: CoreLoopResult,
    ) -> None:
        p = _pid(problem)
        hard_for_decompose: list[Milestone] = []
        hard_for_hints: list[Milestone] = []

        # Evaluate all milestones in parallel
        def _eval_one(milestone: Milestone) -> tuple[Milestone, PassStats]:
            logger.info("[%s] d%d evaluating %s (answer=%s)...", p, depth, milestone.id, milestone.answer[:60])
            local_result = CoreLoopResult()
            stat = self._evaluate_milestone(problem=problem, milestone=milestone, result=local_result)
            logger.info("[%s] d%d %s p_hat=%.2f", p, depth, milestone.id, stat.p_hat)
            return milestone, stat, local_result

        concurrency = max(1, self.config.student_sample_concurrency)
        if len(milestones) <= 1 or concurrency <= 1:
            eval_results = [_eval_one(m) for m in milestones]
        else:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(milestones))) as executor:
                eval_results = list(executor.map(_eval_one, milestones))

        # Merge results and make decisions
        for milestone, stat, local_result in eval_results:
            result.attempts.extend(local_result.attempts)
            result.stats.extend(local_result.stats)
            result.training_rows.append(build_milestone_training_row(problem, milestone, p_hat=stat.p_hat, depth=depth))

            if stat.p_hat >= self.config.keep_threshold:
                logger.info("[%s] d%d %s -> KEEP", p, depth, milestone.id)
                continue

            can_decompose = depth <= self.config.max_depth and milestone.decompose_advice is not None
            if can_decompose:
                logger.info("[%s] d%d %s -> DECOMPOSE", p, depth, milestone.id)
                hard_for_decompose.append(milestone)
                continue

            logger.info("[%s] d%d %s -> HINTS", p, depth, milestone.id)
            hard_for_hints.append(milestone)

        # Parallelize all teacher decompose + hints calls
        concurrency = max(1, self.config.student_sample_concurrency)

        def _do_decompose(milestone: Milestone) -> None:
            try:
                child_packets = self._generate_teacher_packets(
                    problem=problem,
                    requests=[{
                        "mode": "decompose",
                        "target_statement": milestone.description,
                        "target_answer": milestone.answer,
                        "target_decompose_advice": milestone.decompose_advice,
                    }],
                )
                child_packet = child_packets[0]
                _log_packet(child_packet, label=f"teacher packet (decompose {milestone.id})", problem=problem)
                child_packet = self._prefix_packet_milestone_ids(child_packet, prefix=milestone.id)
                result.packets.append(child_packet)
                children = self._milestones_from_packet(child_packet)
                self._process_milestones(
                    problem=problem,
                    milestones=children,
                    depth=depth + 1,
                    result=result,
                )
            except Exception:
                logger.exception("[%s] d%d decompose %s failed, continuing", p, depth, milestone.id)

        def _do_hints(milestone: Milestone) -> None:
            try:
                hint_packets = self._generate_teacher_packets(
                    problem=problem,
                    requests=[{
                        "mode": "hints",
                        "target_statement": milestone.description,
                        "target_answer": milestone.answer,
                        "target_hints": milestone.hints,
                    }],
                )
                hint_packet = hint_packets[0]
                _log_packet(hint_packet, label=f"teacher packet (hints {milestone.id})", problem=problem)
                hint_packet = self._prefix_packet_milestone_ids(hint_packet, prefix=milestone.id)
                result.packets.append(hint_packet)
                hint_milestones = self._milestones_from_packet(hint_packet)
                if self.config.evaluate_endpoint_pass_rate and hint_milestones:
                    self._evaluate_endpoint_milestones(
                        problem=problem, milestones=hint_milestones, depth=depth, result=result,
                    )
                else:
                    for hm in hint_milestones:
                        result.training_rows.append(build_milestone_training_row(problem, hm, depth=depth))
            except Exception:
                logger.exception("[%s] d%d hints %s failed, continuing", p, depth, milestone.id)

        all_tasks = [(m, "decompose") for m in hard_for_decompose] + [(m, "hints") for m in hard_for_hints]
        if all_tasks:
            ids_d = ",".join(m.id for m in hard_for_decompose)
            ids_h = ",".join(m.id for m in hard_for_hints)
            if hard_for_decompose:
                logger.info("[%s] d%d teacher decompose [%s] x%d", p, depth, ids_d, len(hard_for_decompose))
            if hard_for_hints:
                logger.info("[%s] d%d teacher hints [%s] x%d", p, depth, ids_h, len(hard_for_hints))

            if len(all_tasks) == 1:
                m, kind = all_tasks[0]
                (_do_decompose if kind == "decompose" else _do_hints)(m)
            else:
                with ThreadPoolExecutor(max_workers=min(concurrency, len(all_tasks))) as executor:
                    futures = []
                    for m, kind in all_tasks:
                        fn = _do_decompose if kind == "decompose" else _do_hints
                        futures.append(executor.submit(fn, m))
                    for f in as_completed(futures):
                        try:
                            f.result()
                        except Exception:
                            pass  # already caught inside _do_decompose/_do_hints

    def _evaluate_endpoint_milestones(
        self,
        *,
        problem: Problem,
        milestones: list[Milestone],
        depth: int,
        result: CoreLoopResult,
    ) -> None:
        """Evaluate endpoint milestones (hints output) in parallel."""
        p = _pid(problem)

        def _eval_endpoint(hm: Milestone) -> tuple[Milestone, PassStats, CoreLoopResult]:
            logger.info("[%s] d%d %s endpoint eval (k=%d)...", p, depth, hm.id, self.config.k)
            local_result = CoreLoopResult()
            stat = self._evaluate_milestone(problem=problem, milestone=hm, result=local_result)
            logger.info("[%s] d%d %s endpoint p_hat=%.2f", p, depth, hm.id, stat.p_hat)
            return hm, stat, local_result

        concurrency = max(1, self.config.student_sample_concurrency)
        if len(milestones) <= 1 or concurrency <= 1:
            eval_results = [_eval_endpoint(hm) for hm in milestones]
        else:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(milestones))) as executor:
                eval_results = list(executor.map(_eval_endpoint, milestones))

        for hm, stat, local_result in eval_results:
            result.attempts.extend(local_result.attempts)
            result.stats.extend(local_result.stats)
            result.training_rows.append(
                build_milestone_training_row(problem, hm, p_hat=stat.p_hat, depth=depth)
            )

    def _evaluate_problem(self, *, problem: Problem, result: CoreLoopResult) -> PassStats:
        responses = self.student.sample_problem(
            problem,
            k=self.config.k,
            temperature=self.config.student_temperature,
            max_tokens=self.config.student_max_tokens,
            max_concurrency=self.config.student_sample_concurrency,
        )
        return self._score_attempts(
            problem=problem,
            milestone=None,
            responses=responses,
            answer=problem.gold_answer,
            note="ACCEPT IF equivalent final answer is present.",
            result=result,
        )

    def _evaluate_milestone(
        self,
        *,
        problem: Problem,
        milestone: Milestone,
        result: CoreLoopResult,
    ) -> PassStats:
        responses = self.student.sample_milestone(
            problem_statement=problem.statement,
            milestone=milestone,
            k=self.config.k,
            include_original_problem=self.config.include_original_problem_in_milestone_prompt,
            temperature=self.config.student_temperature,
            max_tokens=self.config.student_max_tokens,
            max_concurrency=self.config.student_sample_concurrency,
        )
        return self._score_attempts(
            problem=problem,
            milestone=milestone,
            responses=responses,
            answer=milestone.answer,
            note=milestone.note,
            result=result,
        )

    def _score_attempts(
        self,
        *,
        problem: Problem,
        milestone: Milestone | None,
        responses: list[str],
        answer: str,
        note: str,
        result: CoreLoopResult,
    ) -> PassStats:
        if hasattr(self.verifier, "verify_batch"):
            verdicts = self.verifier.verify_batch(
                responses=responses,
                answer=answer,
                note=note,
            )
        else:
            verdicts = [
                self.verifier.verify(
                    response=response,
                    answer=answer,
                    note=note,
                )
                for response in responses
            ]

        passed = 0
        for idx, (response, verdict) in enumerate(zip(responses, verdicts), start=1):
            result.attempts.append(
                Attempt(
                    problem_id=problem.problem_id,
                    milestone_id=milestone.id if milestone else None,
                    student_run_id=f"{problem.problem_id}:{milestone.id if milestone else 'problem'}:{idx}",
                    response=response,
                    verdict=verdict,
                )
            )
            if verdict["label"] == "ACCEPT":
                passed += 1

        stat = PassStats(
            problem_id=problem.problem_id,
            milestone_id=milestone.id if milestone else None,
            k=len(responses),
            passed=passed,
            p_hat=(passed / len(responses)) if responses else 0.0,
        )
        result.stats.append(stat)
        return stat

    def _generate_teacher_packets(
        self,
        *,
        problem: Problem,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self.teacher.generate_packets_batch(
            problem=problem,
            requests=requests,
        )

    @staticmethod
    def _milestones_from_packet(packet: dict[str, Any]) -> list[Milestone]:
        milestones: list[Milestone] = []
        for raw in packet.get("milestones", []):
            milestones.append(
                Milestone(
                    id=raw["id"],
                    type=raw["type"],
                    description=raw["description"],
                    answer=raw["answer"],
                    note=raw["note"],
                    hints=raw.get("hints"),
                    decompose_advice=raw.get("decompose_advice"),
                )
            )
        return milestones

    @staticmethod
    def _prefix_packet_milestone_ids(packet: dict[str, Any], *, prefix: str) -> dict[str, Any]:
        if not prefix:
            return packet
        copied = dict(packet)
        copied_milestones: list[dict[str, Any]] = []
        for raw in packet.get("milestones", []):
            raw_copy = dict(raw)
            raw_copy["id"] = f"{prefix}-{raw_copy['id']}"
            copied_milestones.append(raw_copy)
        copied["milestones"] = copied_milestones
        return copied
