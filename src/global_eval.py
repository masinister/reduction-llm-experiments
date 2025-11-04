from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .schemas import GLOBAL_EVAL_SCHEMA
from .vllm_structured import StructuredCallError, StructuredResult, run_structured
from .inference import get_max_tokens_from_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_GLOBAL_SYSTEM_PROMPT = (
    "You are verifying whether the whole reduction relies on hidden assumptions or "
    "leaks the target ground truth. "
    "In issue descriptions, specify which steps leak information and what should be removed or changed. "
    "Respond only with JSON that matches the schema."
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
        "Assess whether the reduction relies on the ground truth or has global issues.\n"
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
    """Cheap heuristic substitute when the model cannot respond."""
    gt = context.get("ground_truth", "")
    gt_tokens = _tokenize(gt)
    step_tokens = _tokenize(" ".join(steps))
    overlap = _ngram_overlap(gt_tokens, step_tokens)
    relies = overlap > 0.6 if gt_tokens else False
    undefined_symbols = _find_undefined_symbols(steps)

    issues: List[Dict[str, Any]] = []
    if relies:
        issues.append(
            {
                "id": "global-ground-truth-overlap",
                "title": "Possible ground-truth leakage",
                "description": "Large lexical overlap between steps and ground truth.",
                "severity": "medium",
                "category": "ground-truth-leak",
                "step_index": None,
            }
        )
    if undefined_symbols:
        issues.append(
            {
                "id": "global-undefined-symbols",
                "title": "Undefined notation detected",
                "description": "Undefined tokens: " + ", ".join(sorted(undefined_symbols)),
                "severity": "medium",
                "category": "notation",
                "step_index": None,
            }
        )

    reasons: List[str] = []
    if relies:
        reasons.append("Ground truth tokens appear heavily in the candidate steps.")
    if undefined_symbols:
        reasons.append("Found unused uppercase tokens or placeholders without definitions.")
    if not reasons:
        reasons.append("No strong global issues detected via heuristics.")

    return {
        "relies_on_ground_truth": relies,
        "reasons": reasons,
        "issues": issues,
    }


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in text.split() if tok]


def _ngram_overlap(gt_tokens: List[str], step_tokens: List[str]) -> float:
    if not gt_tokens or not step_tokens:
        return 0.0
    gt_counts = Counter(gt_tokens)
    step_counts = Counter(step_tokens)
    intersection = sum(min(gt_counts[tok], step_counts[tok]) for tok in gt_counts)
    total = sum(gt_counts.values())
    return intersection / total if total else 0.0


def _find_undefined_symbols(steps: Iterable[str]) -> List[str]:
    seen = set()
    undefined = set()
    for text in steps:
        tokens = text.split()
        if not tokens:
            continue
        defined = tokens[0].isalpha() and tokens[0][0].islower()
        for tok in tokens:
            if tok.isupper() and tok.isalpha():
                if defined:
                    seen.add(tok)
                elif tok not in seen:
                    undefined.add(tok)
    return sorted(undefined)


__all__ = ["run_global_evaluation", "DEFAULT_GLOBAL_SYSTEM_PROMPT"]
