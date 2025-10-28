from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .schemas import STEP_EVAL_SCHEMA
from .vllm_structured import StructuredCallError, StructuredResult, run_structured

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_STEP_SYSTEM_PROMPT = (
    "You are a strict formal reduction evaluator. "
    "Respond only with JSON that matches the provided schema."
)


def evaluate_step(
    model,
    *,
    session_base: str,
    context: Mapping[str, Any],
    steps: List[str],
    index: int,
    retries: int = 2,
    temperature: float | None = 0.0,
    max_tokens: int | None = 512,
    system_prompt: str = DEFAULT_STEP_SYSTEM_PROMPT,
) -> Tuple[Dict[str, Any], Optional[StructuredResult]]:
    """Evaluate a single step, returning parsed data and raw result."""
    step_text = steps[index] if 0 <= index < len(steps) else ""

    numbered_steps = "\n".join(f"{i}: {text}" for i, text in enumerate(steps))
    user_payload = {
        "source_text": context.get("source_text", ""),
        "target_text": context.get("target_text", ""),
        "ground_truth": context.get("ground_truth", ""),
        "step_index": index,
        "step_text": step_text,
        "all_steps": numbered_steps,
    }
    user_prompt = (
        "Evaluate the highlighted step within the reduction.\n"
        "Return JSON that conforms to the schema.\n\n"
        f"Payload:\n{json.dumps(user_payload, indent=2, ensure_ascii=False)}\n"
    )

    session_id = f"{session_base}-step-{index}"

    try:
        result = run_structured(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=STEP_EVAL_SCHEMA,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        data = dict(result.data)
        if "step_index" not in data:
            data["step_index"] = index
        if "step_text" not in data:
            data["step_text"] = step_text
        return data, result
    except StructuredCallError as exc:
        logger.warning("Step evaluation failed for index %d: %s", index, exc)
        extracted = _extract_reason_list(str(exc))
        fallback = {
            "step_index": index,
            "step_text": step_text,
            "classification": "other",
            "passes": False,
            "confidence_score": 0.0,
            "reasons": extracted or ["model_failed_to_return_valid_json"],
        }
        if extracted:
            issues = []
            for pos, detail in enumerate(extracted):
                issues.append(
                    {
                        "id": f"step{index}-schema-detail-{pos}",
                        "title": "Auto-captured critique",
                        "description": detail,
                        "severity": "high",
                        "category": "soundness",
                        "step_index": index,
                    }
                )
            fallback["issues"] = issues
        else:
            fallback["issues"] = [
                {
                    "id": f"step{index}-parsing-fail",
                    "title": "Model JSON parse failure",
                    "description": str(exc),
                    "severity": "high",
                    "category": "other",
                    "step_index": index,
                }
            ]
        return fallback, None


def evaluate_steps(
    model,
    *,
    session_base: str,
    context: Mapping[str, Any],
    steps: List[str],
    retries: int = 2,
    temperature: float | None = 0.0,
    max_tokens: int | None = 512,
    system_prompt: str = DEFAULT_STEP_SYSTEM_PROMPT,
) -> Tuple[List[Dict[str, Any]], List[StructuredResult]]:
    """Evaluate all steps sequentially."""
    parsed: List[Dict[str, Any]] = []
    raw_results: List[StructuredResult] = []
    for idx in range(len(steps)):
        data, raw = evaluate_step(
            model,
            session_base=session_base,
            context=context,
            steps=steps,
            index=idx,
            retries=retries,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        parsed.append(data)
        if raw is not None:
            raw_results.append(raw)
    return parsed, raw_results


__all__ = ["evaluate_step", "evaluate_steps", "DEFAULT_STEP_SYSTEM_PROMPT"]


def _extract_reason_list(message: str) -> List[str]:
    match = re.search(r"\[(?:\s*'[^']*'(?:,\s*)?)+\s*\]", message)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(0))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    cleaned: List[str] = []
    for item in parsed:
        if isinstance(item, str):
            cleaned.append(item.replace("\t", " ").strip())
    return [text for text in cleaned if text]
