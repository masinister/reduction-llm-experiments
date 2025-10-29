from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Tuple

from .schemas import REPAIR_SCHEMA
from .vllm_structured import StructuredCallError, StructuredResult, run_structured

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_REPAIR_SYSTEM_PROMPT = (
    "You propose precise edits to repair the reduction. Output JSON only,"
    " matching the schema."
)

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def repair_with_model(
    model,
    *,
    session_id: str,
    steps: List[str],
    issues: List[Dict[str, Any]],
    retries: int = 2,
    temperature: float | None = 0.0,
    max_tokens: int | None = 1024,
    max_edits: int = 5,
    system_prompt: str = DEFAULT_REPAIR_SYSTEM_PROMPT,
) -> Tuple[Dict[str, Any], StructuredResult | None]:
    """Request repair plan from the model with edit limiting."""
    payload = {
        "steps": steps,
        "issues": issues,
        "max_edits": max_edits,
    }
    user_prompt = (
        "Propose repairs for the reduction. Respect max_edits and return only JSON.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
    )

    try:
        result = run_structured(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=REPAIR_SCHEMA,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        data = dict(result.data)
        data["edits"] = _truncate_edits(data.get("edits", []), data.get("todo", []), max_edits)
        sanitized, dropped = _sanitize_repair_plan(data, issues)
        if dropped:
            logger.info(
                "Repair plan sanitized: removed %d todos and %d edits with invalid issue ids.",
                dropped["todos"],
                dropped["edits"],
            )
        return sanitized, result
    except StructuredCallError as exc:
        logger.warning("Repair generation failed: %s", exc)
        fallback = {"todo": [], "edits": [], "resolved_issue_ids": [], "notes": str(exc)}
        return fallback, None


def apply_edits(
    steps: List[str],
    edits: Iterable[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply edits deterministically, returning new steps, applied, skipped."""
    working = list(steps)
    applied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    validated = []
    for edit in edits:
        info = _validate_edit(edit)
        if info["valid"]:
            validated.append(info)
        else:
            skipped.append({"edit": edit, "reason": info["reason"]})

    deletes_replaces = [e for e in validated if e["op"] in {"delete", "replace"}]
    deletes_replaces.sort(key=lambda e: e["index"], reverse=True)
    moves = [e for e in validated if e["op"] == "move"]
    inserts = [e for e in validated if e["op"] == "insert"]
    inserts.sort(key=lambda e: e["index"])

    for edit in deletes_replaces:
        idx = edit["index"]
        if not 0 <= idx < len(working):
            skipped.append({"edit": edit["raw"], "reason": "index_out_of_bounds"})
            continue
        if edit["op"] == "delete":
            removed = working.pop(idx)
            applied.append({"op": "delete", "index": idx, "removed": removed})
        else:
            old = working[idx]
            working[idx] = edit["content"]
            applied.append({"op": "replace", "index": idx, "previous": old, "content": edit["content"]})

    for edit in moves:
        idx = edit["index"]
        target = edit["to_index"]
        if not 0 <= idx < len(working):
            skipped.append({"edit": edit["raw"], "reason": "index_out_of_bounds"})
            continue
        snippet = working.pop(idx)
        length_before = len(working) + 1
        if target is None:
            new_index = len(working)
        else:
            if target < 0:
                skipped.append({"edit": edit["raw"], "reason": "target_index_out_of_bounds"})
                working.insert(idx, snippet)
                continue
            if target >= length_before:
                new_index = len(working)
            elif idx < target:
                new_index = max(target - 1, 0)
            else:
                new_index = target
            new_index = min(new_index, len(working))
        working.insert(new_index, snippet)
        applied.append({"op": "move", "from": idx, "to": new_index, "content": snippet})

    for edit in inserts:
        idx = edit["index"]
        content = edit["content"]
        if idx < 0:
            skipped.append({"edit": edit["raw"], "reason": "index_out_of_bounds"})
            continue
        if idx > len(working):
            idx = len(working)
        working.insert(idx, content)
        applied.append({"op": "insert", "index": idx, "content": content})

    return working, applied, skipped


def collect_touched_indices(applied: Iterable[Dict[str, Any]]) -> List[int]:
    touched = set()
    for edit in applied:
        if edit["op"] == "insert":
            idx = edit.get("index")
            if isinstance(idx, int):
                touched.add(idx)
        elif edit["op"] == "delete":
            idx = edit.get("index")
            if isinstance(idx, int):
                touched.add(idx)
        elif edit["op"] == "replace":
            idx = edit.get("index")
            if isinstance(idx, int):
                touched.add(idx)
        elif edit["op"] == "move":
            frm = edit.get("from")
            to = edit.get("to")
            if isinstance(frm, int):
                touched.add(frm)
            if isinstance(to, int):
                touched.add(to)
    return sorted(touched)


def _truncate_edits(edits: List[Dict[str, Any]], todo: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(edits) <= limit:
        return edits
    priority_map: Dict[str, int] = {}
    for item in todo:
        issue_ids = item.get("linked_issue_ids", [])
        priority = item.get("priority", "medium")
        score = _PRIORITY_ORDER.get(priority, 1)
        for issue_id in issue_ids:
            priority_map[issue_id] = min(priority_map.get(issue_id, score), score)

    indexed = list(enumerate(edits))

    def _rank(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        idx, edit = item
        linked = edit.get("linked_issue_ids", [])
        if not linked:
            return (_PRIORITY_ORDER.get("medium", 1), idx)
        best = min(priority_map.get(issue, _PRIORITY_ORDER.get("medium", 1)) for issue in linked)
        return (best, idx)

    ordered = sorted(indexed, key=_rank)
    keep_indices = {idx for idx, _ in ordered[:limit]}
    return [edit for idx, edit in enumerate(edits) if idx in keep_indices]


def _validate_edit(edit: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(edit, dict):
        return {"valid": False, "reason": "edit_not_a_dict", "raw": edit}
    op = edit.get("op")
    if op not in {"insert", "replace", "delete", "move"}:
        return {"valid": False, "reason": "unsupported_op", "raw": edit}
    index = edit.get("index")
    if not isinstance(index, int) or index < 0:
        return {"valid": False, "reason": "invalid_index", "raw": edit}
    content = edit.get("content")
    if op in {"insert", "replace"}:
        if not isinstance(content, str) or not content.strip():
            return {"valid": False, "reason": "missing_content", "raw": edit}
    to_index = edit.get("to_index") if op == "move" else None
    if op == "move" and to_index is not None and (not isinstance(to_index, int) or to_index < 0):
        return {"valid": False, "reason": "invalid_target_index", "raw": edit}
    return {
        "valid": True,
        "op": op,
        "index": index,
        "content": content.strip() if isinstance(content, str) else content,
        "to_index": to_index,
        "raw": edit,
    }


def _sanitize_repair_plan(
    plan: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    valid_issue_ids = {issue.get("id") for issue in issues if issue.get("id")}

    removed_todos = 0
    filtered_todo: List[Dict[str, Any]] = []
    for entry in plan.get("todo", []) or []:
        linked = [issue_id for issue_id in entry.get("linked_issue_ids", []) if issue_id in valid_issue_ids]
        if not linked:
            removed_todos += 1
            continue
        copy = dict(entry)
        copy["linked_issue_ids"] = linked
        filtered_todo.append(copy)
    plan["todo"] = filtered_todo

    removed_edits = 0
    filtered_edits: List[Dict[str, Any]] = []
    for edit in plan.get("edits", []) or []:
        linked = [issue_id for issue_id in edit.get("linked_issue_ids", []) if issue_id in valid_issue_ids]
        if not linked:
            removed_edits += 1
            continue
        copy = dict(edit)
        copy["linked_issue_ids"] = linked
        filtered_edits.append(copy)
    plan["edits"] = filtered_edits

    resolved = [issue_id for issue_id in plan.get("resolved_issue_ids", []) if issue_id in valid_issue_ids]
    plan["resolved_issue_ids"] = resolved

    meta = plan.get("notes")
    if removed_todos or removed_edits:
        message = (
            "Sanitized repair plan: dropped entries referencing unknown issues."
            if not meta
            else f"{meta}\nSanitized repair plan: dropped entries referencing unknown issues."
        )
        plan["notes"] = message

    dropped = {"todos": removed_todos, "edits": removed_edits}
    return plan, dropped


__all__ = [
    "repair_with_model",
    "apply_edits",
    "collect_touched_indices",
    "DEFAULT_REPAIR_SYSTEM_PROMPT",
]
