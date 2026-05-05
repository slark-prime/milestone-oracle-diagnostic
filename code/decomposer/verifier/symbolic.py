from __future__ import annotations

import re

from sympy import simplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
_RATIO_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*$"
)


def _extract_candidate(text: str) -> str:
    stripped = (text or "").strip().replace("$", "")
    if not stripped:
        return ""

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""

    candidate = lines[-1]
    if "=" in candidate:
        candidate = candidate.split("=")[-1].strip()
    if ":" in candidate and candidate.lower().startswith(("answer", "final")):
        candidate = candidate.split(":")[-1].strip()
    ratio_match = _RATIO_RE.match(candidate)
    if ratio_match:
        candidate = f"({ratio_match.group(1)})/({ratio_match.group(2)})"
    return candidate


def _try_parse(expr_text: str):
    candidate = _extract_candidate(expr_text)
    if not candidate:
        return None
    try:
        return parse_expr(candidate, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return None


def is_equivalent(lhs_text: str, rhs_text: str) -> tuple[bool, str]:
    lhs_expr = _try_parse(lhs_text)
    rhs_expr = _try_parse(rhs_text)
    if lhs_expr is None or rhs_expr is None:
        return False, "PARSE_ERROR: could not parse one side"

    try:
        if simplify(lhs_expr - rhs_expr) == 0:
            return True, "symbolic equivalence"
    except Exception:
        pass

    return False, "NOT_EQUIVALENT: expressions differ"


class SymbolicVerifier:
    def verify(self, *, response: str, gold_answer: str, note: str = "") -> dict[str, str]:
        equivalent, reason = is_equivalent(response, gold_answer)
        return {
            "label": "ACCEPT" if equivalent else "NOT_ACCEPT",
            "reason": reason,
        }
