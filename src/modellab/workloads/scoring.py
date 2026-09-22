"""Deterministic scorers for reproducible workload evaluation."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from modellab.workloads.models import ExpectedAnswer


def score_answer(output: str, expected: ExpectedAnswer | None) -> float | None:
    if expected is None:
        return None
    scorer = expected.scorer
    if scorer == "exact_match":
        return float(output.strip() == str(expected.value).strip())
    if scorer in {"normalized_exact_match", "multiple_choice"}:
        return float(_normalize(output) == _normalize(str(expected.value)))
    if scorer == "contains_all":
        values = expected.value if isinstance(expected.value, list) else [expected.value]
        normalized_output = _normalize(output)
        return float(all(_normalize(str(value)) in normalized_output for value in values))
    if scorer == "numeric_tolerance":
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", output.replace(",", ""))
        if not numbers or not isinstance(expected.value, (int, float)):
            return 0.0
        return float(
            math.isclose(
                float(numbers[0]),
                float(expected.value),
                abs_tol=expected.tolerance or 0.0,
                rel_tol=0.0,
            )
        )
    try:
        parsed = json.loads(_extract_json(output))
    except (json.JSONDecodeError, ValueError):
        return 0.0
    if scorer == "valid_json":
        return 1.0
    if scorer == "json_schema":
        return float(_matches_schema(parsed, expected.schema_definition or {}))
    raise ValueError(f"Unknown scorer: {scorer}")


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _extract_json(output: str) -> str:
    stripped = output.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1])
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:].lstrip()
    return stripped


def _matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Validate the small JSON Schema subset used by built-in workloads."""

    expected_type = schema.get("type")
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    if expected_type in type_map and not isinstance(value, type_map[expected_type]):
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if isinstance(value, dict):
        if any(key not in value for key in schema.get("required", [])):
            return False
        for key, child_schema in schema.get("properties", {}).items():
            if key in value and not _matches_schema(value[key], child_schema):
                return False
    if isinstance(value, list) and "items" in schema:
        return all(_matches_schema(item, schema["items"]) for item in value)
    return True
