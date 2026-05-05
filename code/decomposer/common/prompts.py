from __future__ import annotations

from typing import Literal, Optional

from decomposer.common.schemas import MILESTONE_TYPES


# ---------------- SYSTEM PROMPT ----------------
# Teacher outputs {"milestones":[...]}; code adds the "problem" block after parsing.
TEACHER_SYSTEM_PROMPT = f"""You are a Mathematics Mentor. Your ONLY job is to output a milestone plan in valid JSON.

Inputs you will receive:
- ORIGINAL_PROBLEM (string)
- PRIVATE_SOLUTION_REFERENCE (string; private reference only)

CRITICAL STANDALONE RULE (non-negotiable):
- Each milestone must be an individually solvable mini-problem.
- A solver must be able to answer a milestone using ONLY that milestone's own "description" (plus the ORIGINAL_PROBLEM context).
- You MUST NOT rely on any unstated lemma, theorem, definition, or earlier milestone text.
- If a milestone needs any prior fact/result, you MUST restate that fact/result explicitly inside the milestone’s "description".
- Do NOT reference other milestones by phrases like “apply the lemma”, “from above”, “using the previous result”, “as shown earlier”.

NO UNDEFINED SYMBOLS RULE:
- Every nonstandard symbol/variable used in a milestone’s description/answer must be defined inside that description.
- Standard symbols like i, π, ℂ, ℝ, gcd are fine without definition.

COVERAGE RULE (no difficult space left):
- The milestone set must cover the entire solution path so that following milestones sequentially leaves no hidden hard steps.
- Avoid “magic-jump” wording (e.g., “clearly”, “observe”, “it follows”, “simplify”) unless the milestone also states an explicit, checkable target.
- If a step still contains nontrivial reasoning, split it into additional milestones.

INTEGRATE RULE (assembly only):
- Include at least one INTEGRATE milestone.
- INTEGRATE must only assemble previously established milestone outputs (but restated inside its description).
- INTEGRATE must NOT introduce new lemmas, new transformations, or fresh nontrivial computations.
- INTEGRATE’s answer must match the ORIGINAL_PROBLEM’s requested final output exactly (format/objects).

You MUST output a single JSON object with this shape:
{{
  "milestones": [
    {{
      "type": "{'|'.join(MILESTONE_TYPES)}",
      "description": "Standalone milestone problem statement. It cannot require reading other milestones; any needed facts must be restated here.",
      "answer": "Canonical gold result for this milestone (short).",
      "note": "Verifier guidance: ACCEPT IF ...; ALLOW ...; keep it concise and easy for a small verifier LLM.",
      "hints": ["...","..."]  // include ONLY if this milestone should NOT be further decomposed
      // OR
      "decompose_advice": ["...","..."]  // include ONLY if this milestone IS decomposable
    }}
  ]
}}

Tag definitions (use exactly one tag per milestone):
- MODEL: Translate into formal math: define variables/objects, write equations/constraints, restate the goal.
- NORMALIZE: Apply a WLOG/scaling/coordinate/ordering/gcd reduction that simplifies the problem while preserving equivalence.
- WARMUP: A simpler stand-alone practice problem using the same technique needed soon.
- KEY_MOVE: The main reframing/substitution/invariant/transform that unlocks the solution; output is the transformed formulation ready for computation.
- LEMMA: A reusable claim/identity/property used later; output is the lemma statement (and proof if requested by the milestone).
- COMPUTE: Carry out the main solving/computation after setup; output is a concrete value/result.
- SANITY: Quick validation/checkpoint (units, sign, boundary case, dimensional consistency, back-substitution).
- INTEGRATE: Assemble milestone outputs to conclude the original target; no new hard steps.

Global constraints:
- Every milestone must include exactly one of: hints OR decompose_advice (never both).
- You should only use the hints if the milestone is already atomic, otherwise use decompose_advice.
- Hints should include part of the solution
- Type tags must come from the restricted taxonomy above.
- Use PRIVATE_SOLUTION_REFERENCE only to ensure the milestone set correctly leads to the final gold answer.
- Output JSON only. No markdown. No extra text.
"""

HINTS_FORMAT_SYSTEM_PROMPT = f"""You format tutoring hints into a milestone JSON packet.

Task type:
- You are organizing a provided focus problem + hints into standalone student-facing milestones.

Output requirements:
- Return JSON only with shape {{"milestones":[...]}}.
- Every milestone must include part of the provided hints.
- Incorporate the provided hints into milestone descriptions. Each milestone must have a "hints" field; use hints: [] when the hint is fully incorporated into the description.

You MUST output a single JSON object with this shape:
{{
  "milestones": [
    {{
      "type": "{'|'.join(MILESTONE_TYPES)}",
      "description": "Standalone milestone problem statement with hint incorporated.",
      "answer": "Canonical gold result for this milestone (short).",
      "note": "Verifier guidance: ACCEPT IF ...; ALLOW ...; keep it concise and easy for a small verifier LLM.",
      "hints": []
    }}
  ]
}}
"""

STUDENT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}. If the answer is a formula or number, use standard LaTeX notation (e.g. \\boxed{\\frac{1}{2}}, \\boxed{2\\sqrt{3}}, \\boxed{60^\\circ}). For multiple choice questions, put the choice letter in \\boxed{} (e.g. \\boxed{D})."

VERIFIER_SYSTEM_PROMPT = """You are a strict verifier. Your only job is to decide whether a student's response should be accepted.

- Use the Canonical answer as ground truth.
- Follow the Verifier note (grading policy) exactly.
- If uncertain, choose NOT_ACCEPT.
- First briefly state how the student's answer aligns with or violates the grading policy, then output your verdict as JSON.

Example output:
The student gives X which satisfies/violates the note because Y.
{"label": "ACCEPT", "reason": "brief reason"}"""
# ---------------- USER PROMPT BUILDER ----------------
TeacherMode = Literal["core", "decompose", "hints"]


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _build_mode_block(mode: TeacherMode) -> str:
    if mode == "core":
        return (
            "Mode: CORE (original problem decomposition).\n"
            "- Produce 4-8 core milestones.\n"
            "- Decompose ORIGINAL_PROBLEM into standalone milestones that cover the full solution path.\n"
            "- PREFER decompose_advice over hints: use decompose_advice for any milestone that has nontrivial sub-steps "
            "(MODEL, KEY_MOVE, COMPUTE, LEMMA, NORMALIZE, WARMUP, SANITY). Use hints only for INTEGRATE or truly atomic steps.\n"
            "- decompose_advice should list 1-3 concrete suggestions for how to split this milestone further (e.g., 'Compute the cross product first', 'State the cyclicity claim before the angle equality').\n"
            "- Include at least one INTEGRATE milestone.\n"
        )
    if mode == "decompose":
        return (
            "Mode: DECOMPOSE (targeted milestone decomposition).\n"
            "- The provided focus target is the standalone problem to refine.\n"
            "- Use decompose_advice when a produced milestone can still be split further.\n"
            "- Use hints only for milestones that are already atomic.\n"
            "- Include an INTEGRATE milestone if it cleanly completes the focus target.\n"
            "- If the focus target seems under-specified, use ORIGINAL_PROBLEM and optional reference answer as context.\n"
        )
    return (
        "Mode: HINTS (standalone tutoring/formatting endpoint; NOT decomposition).\n"
        "- Treat the focus target as the problem to tutor directly.\n"
        "- Combine provided hints into concise student-facing milestones.\n"
        "- Every milestone MUST include a hints field (use hints: [] when the hint is incorporated into the description).\n"
        "- Prefer a milestone for each hint.\n"
    )


def _build_focus_target_block(
    *,
    mode: TeacherMode,
    target_statement: Optional[str],
    target_answer: Optional[str],
) -> str:
    if not target_statement:
        return ""

    lines = ["Focus standalone problem:", f"- focus_problem: {target_statement}"]
    if target_answer:
        lines.append(f"- optional_reference_answer: {target_answer}")

    if mode == "decompose":
        lines.append("Instruction: Refine this standalone problem into smaller standalone milestones.")
    elif mode == "hints":
        lines.append("Instruction: Tutor this problem directly and organize the provided hints.")
    else:
        lines.append("Instruction: Keep this target in scope.")
    return "\n".join(lines)


def _build_optional_list_block(*, title: str, items: Optional[list[str]], closing_line: str) -> str:
    if not items:
        return ""
    return f"{title}\n{_format_bullets(items)}\n{closing_line}"


def build_teacher_user_prompt(
    *,
    statement: str,
    solution_reference: str,
    mode: TeacherMode,
    target_statement: Optional[str] = None,
    target_answer: Optional[str] = None,
    target_decompose_advice: Optional[list[str]] = None,
    target_hints: Optional[list[str]] = None,
) -> str:
    """
    Build the mode-specific user prompt for Teacher.

    Modes:
    - "core": original-problem decomposition (4-8 milestones)
    - "decompose": decomposition of a specific target milestone
    - "hints": standalone tutoring endpoint (no further decomposition)
    """
    if mode not in {"core", "decompose", "hints"}:
        raise ValueError(f"Unknown teacher mode: {mode}")

    if mode == "decompose" and target_hints:
        raise ValueError("decompose mode does not accept target_hints")
    if mode == "hints" and target_decompose_advice:
        raise ValueError("hints mode does not accept target_decompose_advice")

    sections = [
        "Build a milestone packet in JSON format.",
        f"Restricted milestone types: {', '.join(MILESTONE_TYPES)}",
        _build_mode_block(mode),
        _build_focus_target_block(
            mode=mode,
            target_statement=target_statement,
            target_answer=target_answer,
        ),
        _build_optional_list_block(
            title="Given decomposition advice from upstream milestone:",
            items=target_decompose_advice,
            closing_line=(
                "Use this advice as supplemental context; output milestones must remain standalone."
            ),
        ),
        _build_optional_list_block(
            title="Given hint list from upstream milestone:",
            items=target_hints,
            closing_line=(
                "Combine these hints into student-facing milestones;"
            ),
        ),
        f"ORIGINAL_PROBLEM:\n{statement}",
        f"PRIVATE_SOLUTION_REFERENCE (private):\n{solution_reference}",
        'Return JSON only with the shape: {"milestones":[...]}.',
    ]
    return "\n\n".join(section for section in sections if section)


def build_student_problem_prompt(statement: str) -> str:
    return (
        "Solve the following problem. Provide your final answer clearly.\n\n"
        f"Problem:\n{statement}\n"
    )


def build_student_milestone_prompt(
    *,
    original_problem: str,
    milestone_description: str,
    include_original_problem: bool = True,
    hints: list[str] | None = None,
) -> str:
    context_block = ""
    if include_original_problem:
        context_block = f"Original problem (context):\n{original_problem}\n\n"
    hints_block = ""
    if hints:
        hints_text = "\n".join(f"- {h}" for h in hints)
        hints_block = f"\nHints:\n{hints_text}\n"
    return (
        "You are solving one milestone from a larger problem.\n"
        "Keep your response focused on this milestone only.\n\n"
        f"{context_block}"
        f"Milestone:\n{milestone_description}\n"
        f"{hints_block}"
    )


def build_verifier_prompt(*, answer: str, note: str, response: str) -> str:
    return f"""You are a strict verifier. Decide if the student's response should be accepted.

Rules:
- Use the Canonical answer as ground truth.
- The Verifier note is the grading policy; follow it.
- If uncertain, choose NOT_ACCEPT.
- Output valid JSON only (no markdown, no extra keys).

Canonical answer:
<<<
{answer}
>>>

Verifier note (grading policy):
<<<
{note}
>>>

Student response:
<<<
{response}
>>>

Return JSON:
{{
  "label": "ACCEPT" | "NOT_ACCEPT",
  "reason": "brief reason (<= 30 words)"
}}
"""


if __name__ == "__main__":
    # Manual local test harness (no CLI).
    # Edit these values directly, then run:
    #   python3.11 -m decomposer.common.prompts
    test_mode = "core"  # one of: core, decompose, hints
    test_statement = (
        "Solve for x: x^2 - 5x + 6 = 0. "
        "Return all real solutions."
    )
    test_solution_reference = (
        "Factor: (x-2)(x-3)=0, so x=2 or x=3."
    )
    test_target_statement = "Factor the quadratic and extract roots."
    test_target_answer = "x=2,3"

    teacher_user_prompt = build_teacher_user_prompt(
        statement=test_statement,
        solution_reference=test_solution_reference,
        mode=test_mode,
        target_statement=test_target_statement if test_mode in {"decompose", "hints"} else None,
        target_answer=test_target_answer if test_mode in {"decompose", "hints"} else None,
    )

    sample_milestone = (
        "Given x^2 - 5x + 6 = 0, factor the polynomial over reals and list all real roots."
    )
    verifier_prompt = build_verifier_prompt(
        answer="x=2,3",
        note="ACCEPT IF both roots 2 and 3 are present; ALLOW set notation.",
        response="x in {2, 3}",
    )

    print("=== TEACHER_SYSTEM_PROMPT ===")
    print(TEACHER_SYSTEM_PROMPT)
    print("\n=== build_teacher_user_prompt(...) ===")
    print(teacher_user_prompt)
    print("\n=== build_student_problem_prompt(...) ===")
    print(build_student_problem_prompt(test_statement))
    print("\n=== build_student_milestone_prompt(include_original_problem=True) ===")
    print(
        build_student_milestone_prompt(
            original_problem=test_statement,
            milestone_description=sample_milestone,
            include_original_problem=True,
        )
    )
    print("\n=== build_student_milestone_prompt(include_original_problem=False) ===")
    print(
        build_student_milestone_prompt(
            original_problem=test_statement,
            milestone_description=sample_milestone,
            include_original_problem=False,
        )
    )
    print("\n=== build_verifier_prompt(...) ===")
    print(verifier_prompt)
