from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

MILESTONE_TYPES = [
    "MODEL",
    "NORMALIZE",
    "WARMUP",
    "KEY_MOVE",
    "LEMMA",
    "COMPUTE",
    "SANITY",
    "INTEGRATE",
]

STANDARD_PROBLEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "problem_id",
        "statement",
        "gold_answer",
        "reference_solution",
        "meta",
    ],
    "properties": {
        "problem_id": {"type": "string"},
        "statement": {"type": "string"},
        "gold_answer": {"type": "string"},
        "reference_solution": {"type": "string"},
        "meta": {"type": "object"},
    },
    "additionalProperties": False,
}

MILESTONE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["type", "description", "answer", "note"],
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string", "enum": MILESTONE_TYPES},
        "description": {"type": "string"},
        "answer": {"type": "string"},
        "note": {"type": "string"},
        "hints": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
        },
        "decompose_advice": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "anyOf": [
        {"required": ["hints"]},
        {"required": ["decompose_advice"]},
    ],
    "additionalProperties": False,
}

MILESTONE_PACKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["problem", "milestones"],
    "properties": {
        "problem": {
            "type": "object",
            "required": ["statement", "solution_reference", "gold_answer"],
            "properties": {
                "problem_id": {"type": "string"},
                "statement": {"type": "string"},
                "solution_reference": {"type": "string"},
                "gold_answer": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "milestones": {
            "type": "array",
            "items": MILESTONE_SCHEMA,
            "minItems": 1,
        },
    },
    "additionalProperties": False,
}

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["label", "reason"],
    "properties": {
        "label": {"type": "string", "enum": ["ACCEPT", "NOT_ACCEPT"]},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}


def validate_schema(payload: dict[str, Any], schema: dict[str, Any], *, title: str = "payload") -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: err.path)
    if not errors:
        return
    joined = "; ".join(error.message for error in errors)
    raise ValueError(f"{title} failed schema validation: {joined}")
