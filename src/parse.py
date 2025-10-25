from __future__ import annotations

import json
import logging
import re
from typing import List, Sequence

from .schemas import PARSE_SCHEMA
from .vllm_structured import StructuredCallError, run_structured

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SCHEMA_TEXT = json.dumps(PARSE_SCHEMA, indent=2, sort_keys=True)
_BULLET_RE = re.compile(r"^(?:\d+[.)]|[a-z]\)|[-*+])\s+")


def parse_with_model(
    model,
    *,
    session_id: str,
    candidate_blob: str,
    system_prompt: str,
    retries: int = 1,
    temperature: float | None = 0.0,
    max_tokens: int | None = 512,
    validate: bool = True,
) -> List[str]:
    """Attempt model-based parsing with deterministic fallback."""
    blob = candidate_blob or ""
    if not blob.strip():
        return []

    user_prompt = (
        "Parse the candidate reduction steps into a JSON object that matches the schema.\n"
        "Return only JSON, no prose.\n\n"
        "Candidate steps (verbatim):\n"
        f"{blob}\n\n"
        "Schema:\n"
        f"{_SCHEMA_TEXT}\n"
    )

    try:
        result = run_structured(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=PARSE_SCHEMA,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            validate=validate,
        )
        steps = _normalize_steps(result.data.get("steps", []))
        if steps:
            return steps
        logger.info("Model returned empty steps; falling back to deterministic split.")
    except StructuredCallError as exc:
        logger.info("Parse model failed (%s); using deterministic split.", exc)

    return deterministic_split(blob)


def deterministic_split(blob: str) -> List[str]:
    """Split candidate text into steps using simple bullet heuristics."""
    lines = [ln.rstrip() for ln in blob.splitlines()]
    steps: List[str] = []
    current: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current:
                steps.append(" ".join(current).strip())
                current = []
            continue

        if _BULLET_RE.match(line):
            if current:
                steps.append(" ".join(current).strip())
                current = []
            line = _BULLET_RE.sub("", line, count=1).strip()
            if line:
                current.append(line)
            continue

        current.append(line)

    if current:
        steps.append(" ".join(current).strip())

    steps = [step for step in steps if step]
    if steps:
        return steps

    fallback = [ln.strip() for ln in lines if ln.strip()]
    return fallback


def _normalize_steps(raw_steps: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for item in raw_steps:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


__all__ = ["parse_with_model", "deterministic_split"]
