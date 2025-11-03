from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Tuple

from .vllm_structured import _stable_id

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def synthesize_issues(
    step_results: Iterable[Dict[str, Any]],
    global_result: Dict[str, Any],
    compare_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Merge issues from step/global/compare evaluations deterministically."""
    consolidated: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    # Debug logging: show raw issues from evaluators
    logger.debug("Synthesizing issues from evaluators:")
    logger.debug("  Step results: %d steps", len(list(step_results)))
    for step in step_results:
        idx = step.get("step_index")
        step_issues = step.get("issues", [])
        logger.debug("    Step %d: %d raw issues", idx, len(step_issues))
        for issue in step_issues:
            logger.debug("      Raw issue: %s", issue)
            normalized = _normalize_issue(issue, fallback_step=idx)
            key = (_dedupe_key(normalized), normalized.get("step_index"))
            consolidated[key] = _merge(consolidated.get(key), normalized)

    global_issues = global_result.get("issues", [])
    logger.debug("  Global evaluation: %d raw issues", len(global_issues))
    for issue in global_issues:
        logger.debug("    Global issue: %s", issue)
        normalized = _normalize_issue(issue, fallback_step=None)
        key = (_dedupe_key(normalized), normalized.get("step_index"))
        consolidated[key] = _merge(consolidated.get(key), normalized)

    compare_issues = compare_result.get("issues", [])
    logger.debug("  Ground truth comparison: %d raw issues", len(compare_issues))
    for issue in compare_issues:
        logger.debug("    Compare issue: %s", issue)
        normalized = _normalize_issue(issue, fallback_step=None)
        key = (_dedupe_key(normalized), normalized.get("step_index"))
        consolidated[key] = _merge(consolidated.get(key), normalized)

    issues = list(consolidated.values())
    issues.sort(key=lambda item: (_SEVERITY_ORDER.get(item.get("severity", "medium"), 1), item.get("id", "")))
    return issues


def _normalize_issue(issue: Dict[str, Any], *, fallback_step) -> Dict[str, Any]:
    title = (issue.get("title") or issue.get("description") or "Issue").strip()
    description = (issue.get("description") or issue.get("title") or "").strip()
    severity = issue.get("severity", "medium")
    category = issue.get("category", "other")
    step_index = issue.get("step_index", fallback_step)

    norm_text = f"{title}\n{description}".lower()
    norm_text = _collapse_whitespace(norm_text)
    issue_id = _stable_id("issue", f"{norm_text}|{step_index}")
    
    logger.debug("    Normalized: title=%s, step_index=%s -> issue_id=%s", title[:50], step_index, issue_id)

    normalized = {
        "id": issue_id,
        "title": title,
        "description": description or title,
        "severity": severity if severity in _SEVERITY_ORDER else "medium",
        "category": category,
        "step_index": step_index,
    }
    if issue.get("meta"):
        normalized["meta"] = issue["meta"]
    return normalized


def _dedupe_key(issue: Dict[str, Any]) -> str:
    parts = [issue.get("title", ""), issue.get("description", ""), issue.get("category", "")]
    return _collapse_whitespace("|".join(parts).lower())


def _merge(existing: Dict[str, Any] | None, new_issue: Dict[str, Any]) -> Dict[str, Any]:
    if existing is None:
        return new_issue

    severity_rank_new = _SEVERITY_ORDER.get(new_issue.get("severity", "medium"), 1)
    severity_rank_old = _SEVERITY_ORDER.get(existing.get("severity", "medium"), 1)
    if severity_rank_new < severity_rank_old:
        existing["severity"] = new_issue.get("severity", existing.get("severity"))

    if "meta" in new_issue:
        meta = dict(existing.get("meta", {}))
        meta.update(new_issue["meta"])
        existing["meta"] = meta
    return existing


def count_medium_high(issues: Iterable[Dict[str, Any]]) -> int:
    return sum(1 for issue in issues if issue.get("severity") in {"medium", "high"})


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


__all__ = ["synthesize_issues", "count_medium_high"]
