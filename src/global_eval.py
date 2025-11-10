from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .schemas import GLOBAL_EVAL_SCHEMA
from .vllm_structured import StructuredCallError, StructuredResult, run_structured
from .inference import get_max_tokens_from_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_GLOBAL_SYSTEM_PROMPT = (
    "You are an expert reduction-proof auditor. "
    "Review the candidate steps and decide whether they form a self-contained argument. "
    "Flag any place where the steps rely on context, ground truth snippets, or unstated lemmas instead of introducing the needed definitions themselves. "
    "List hidden assumptions, missing definitions, or global logical gaps. "
    "Explain every issue, cite the affected step index when possible, and assign severities (high for correctness blockers, medium for unclear dependencies, low for minor clarity). "
    "Output strictly JSON conforming to the provided schema."
)

_MAX_TEXT_CHARS = 2000
_MAX_STEPS = 20


def _truncate_text(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "... [truncated]"


def _truncate_steps(steps: List[str], limit: int = _MAX_STEPS) -> List[str]:
    subset = list(steps[:limit])
    if len(steps) > limit:
        subset.append(f"... [truncated {len(steps) - limit} additional steps]")
    return subset


def run_global_evaluation(
    model,
    *,
    session_id: str,
    context: Mapping[str, Any],
    steps: List[str],
    retries: int = 2,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    system_prompt: str = DEFAULT_GLOBAL_SYSTEM_PROMPT,
) -> Tuple[Dict[str, Any], Optional[StructuredResult]]:
    """Run global evaluation.
    
    Args:
        max_tokens: Maximum tokens for response. If None, loads from config.ini
    """
    if max_tokens is None:
        max_tokens = get_max_tokens_from_config()
    """Run the global evaluation structured call with heuristics fallback."""
    payload = {
        "source_text": _truncate_text(context.get("source_text", "")),
        "target_text": _truncate_text(context.get("target_text", "")),
        "ground_truth": _truncate_text(context.get("ground_truth", "")),
        "steps": _truncate_steps(steps),
    }
    user_prompt = (
        "Determine whether the candidate steps constitute a self-contained reduction proof.\n"
        "Use the source, target, and ground-truth texts only to detect dependencies -- do not assume the final argument may reference them.\n"
        "Tasks: (1) Set `relies_on_ground_truth` true if any step depends on those external texts or leaves definitions implicit.\n"
        "(2) Add concise strings to `reasons` summarizing the main findings.\n"
        "(3) Populate `issues` with detailed findings, unique `id` values, severities, categories, and step indices when applicable.\n"
        "Return JSON only.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
    )

    try:
        result = run_structured(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=GLOBAL_EVAL_SCHEMA,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        data = dict(result.data)
        return data, result
    except StructuredCallError as exc:
        logger.warning("Global evaluation failed: %s", exc)
        fallback = _fallback_global_eval(context=context, steps=steps)
        return fallback, None


def _fallback_global_eval(*, context: Mapping[str, Any], steps: Iterable[str]) -> Dict[str, Any]:
    return {
        "relies_on_ground_truth": False,
        "reasons": ["Structured global evaluation unavailable."],
        "issues": [],
    }


__all__ = ["run_global_evaluation", "DEFAULT_GLOBAL_SYSTEM_PROMPT"]
