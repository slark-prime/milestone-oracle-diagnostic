"""Strict symbolic verifier cascade — no LLM judge.

Pipeline (matches paper Section 3 / Appendix N):
    1. Strip </think> tags
    2. Extract \\boxed{...} from response (NOT_ACCEPT if missing)
    3. Try math_reward.is_equiv (handles common normalizations)
    4. Try schema-aware symbolic verifier
    5. Try math-verify (LaTeX -> SymPy equivalence)
    6. Otherwise NOT_ACCEPT

There is no LLM-judge fallback. The cascade abstains by emitting NOT_ACCEPT
when none of the symbolic stages can resolve equivalence.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from decomposer.verifier.math_reward import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)
from decomposer.verifier.symbolic import SymbolicVerifier

logger = logging.getLogger(__name__)


class VerifierModule:
    """Strict symbolic verifier cascade.

    Use either ``.verify(response=..., answer=..., note=...)`` for a
    structured ``{label, reason}`` dict, or the module-level
    ``verify_answer(candidate, gold, note=...)`` convenience wrapper.
    """

    def __init__(self, *, strip_think: bool = True):
        self.strip_think = strip_think
        self.symbolic = SymbolicVerifier()

    def verify(
        self,
        *,
        response: str,
        answer: str,
        note: str = "",
    ) -> dict[str, str]:
        # 1. Strip </think>
        if self.strip_think and "</think>" in response:
            response = response.split("</think>")[-1].strip()

        # 2. Boxed extraction
        boxed_str = last_boxed_only_string(response)
        if boxed_str is None:
            return {"label": "NOT_ACCEPT", "reason": "no \\boxed found in response"}

        extracted = remove_boxed(boxed_str)

        # 3. math_reward equivalence
        if is_equiv(extracted, answer):
            return {"label": "ACCEPT", "reason": "boxed match (math_reward)"}

        # 4. Schema-aware symbolic
        symbolic_verdict = self.symbolic.verify(
            response=extracted, gold_answer=answer, note=note,
        )
        if symbolic_verdict["label"] == "ACCEPT":
            return {"label": "ACCEPT", "reason": str(symbolic_verdict["reason"])}

        # 5. math-verify
        mv_verdict = self._try_math_verify(extracted, answer)
        if mv_verdict is not None:
            return mv_verdict

        # 6. Strict cascade abstains: no LLM fallback.
        return {"label": "NOT_ACCEPT", "reason": str(symbolic_verdict["reason"])}

    def verify_batch(
        self,
        *,
        responses: list[str],
        answer: str,
        note: str = "",
    ) -> list[dict[str, str]]:
        return [self.verify(response=r, answer=answer, note=note) for r in responses]

    @staticmethod
    def _normalize_for_mv(text: str) -> str:
        s = text.strip()
        s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
        s = re.sub(r";", ",", s)
        s = re.sub(r"^[a-zA-Z_]\w*\s*=\s*", "", s)
        return s

    @staticmethod
    def _try_math_verify(extracted: str, gold_answer: str) -> dict[str, str] | None:
        try:
            from math_verify import parse, verify as mv_verify
        except ImportError:
            return None

        gold_norm = VerifierModule._normalize_for_mv(gold_answer)
        pred_norm = VerifierModule._normalize_for_mv(extracted)

        try:
            gold_parsed = parse(r"\boxed{" + gold_norm + "}")
            if not gold_parsed:
                return None
            pred_parsed = parse(r"\boxed{" + pred_norm + "}")
            if not pred_parsed:
                return None
            if mv_verify(gold_parsed, pred_parsed):
                return {"label": "ACCEPT", "reason": "math-verify equivalence"}
            return None
        except Exception:
            return None


# Module-level convenience wrapper used in the README quickstart.
_DEFAULT_VERIFIER: Optional[VerifierModule] = None


def verify_answer(candidate: str, gold: str, note: str = "") -> dict[str, str]:
    """Strict-cascade verification convenience wrapper.

    Returns a dict ``{label, reason}`` where ``label`` is ``ACCEPT`` or
    ``NOT_ACCEPT``. ``candidate`` is the raw model response; ``gold`` is
    the canonical answer; ``note`` is the verifier acceptance note (e.g.,
    ``ACCEPT IF equivalent final answer is present``).
    """
    global _DEFAULT_VERIFIER
    if _DEFAULT_VERIFIER is None:
        _DEFAULT_VERIFIER = VerifierModule()
    return _DEFAULT_VERIFIER.verify(response=candidate, answer=gold, note=note)
