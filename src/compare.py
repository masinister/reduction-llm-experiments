from __future__ import annotations

import json
import logging
from difflib import ndiff
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .schemas import GT_COMPARE_SCHEMA
from .vllm_structured import StructuredCallError, StructuredResult, run_structured

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_COMPARE_SYSTEM_PROMPT = (
    "Compare the candidate reduction with the provided ground truth."
    " Respond only in JSON matching the schema."
)


def compare_to_ground_truth(
    model,
    *,
    session_id: str,
    context: Mapping[str, Any],
    steps: List[str],
    retries: int = 2,
    temperature: float | None = 0.0,
    max_tokens: int | None = 1024,
    system_prompt: str = DEFAULT_COMPARE_SYSTEM_PROMPT,
) -> Tuple[Dict[str, Any], Optional[StructuredResult]]:
    """Run ground-truth comparison with fallback diff heuristics."""
    payload = {
        "ground_truth": context.get("ground_truth", ""),
        "candidate_steps": steps,
    }
    user_prompt = (
        "Compare the candidate reduction to the ground-truth description.\n"
        "Return only JSON.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
    )

    try:
        result = run_structured(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=GT_COMPARE_SCHEMA,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        return dict(result.data), result
    except StructuredCallError as exc:
        logger.warning("Ground-truth comparison failed: %s", exc)
        fallback = _fallback_compare(context=context, steps=steps)
        return fallback, None


def _fallback_compare(*, context: Mapping[str, Any], steps: List[str]) -> Dict[str, Any]:
    gt = context.get("ground_truth", "")
    candidate = "\n".join(steps)
    diff_lines = list(ndiff(gt.splitlines(), candidate.splitlines()))
    differences: List[str] = []
    for line in diff_lines:
        if line.startswith("- ") or line.startswith("+ "):
            differences.append(line)
    consistent = len(differences) == 0

    issues: List[Dict[str, Any]] = []
    if not consistent:
        issues.append(
            {
                "id": "gt-diff",
                "title": "Mismatch with ground truth",
                "description": "Candidate differs from ground truth. Inspect differences list.",
                "severity": "medium",
                "category": "logic",
                "step_index": None,
            }
        )

    return {
        "consistent": consistent,
        "differences": differences,
        "issues": issues,
    }


__all__ = ["compare_to_ground_truth", "DEFAULT_COMPARE_SYSTEM_PROMPT"]
