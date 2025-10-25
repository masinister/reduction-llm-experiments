#!/usr/bin/env python3
"""Verify a reduction by translating it into granular steps and self-checking each step.

Pipeline per row:
- Extract a candidate sequence of atomic steps from the ground-truth reduction text, with a type per step
  (definition|construction|claim|justification), using the LLM.
- For each step i, run a dedicated evaluation call (with the full context of all steps, source/target, and
  the ground-truth reduction) that decides pass/fail and optionally proposes a repair. Repeat up to N iterations
  or until pass.
- Return the final sequence of steps and a summary of issues encountered.

Input CSV columns (required):
  - source_text
  - target_text
  - reduction_full_text

Output CSV columns (added):
    - verified_steps_json     # JSON list with fields: index, type, text
    - issues_summary          # Concise summary of issues and repairs
    - standalone_verdict      # pass|fail from standalone global check
    - consistency_verdict     # pass|fail from consistency global check
    - verify_tokens           # total tokens generated across all calls for this row (best-effort)
    - verify_latency_s        # total latency across all calls for this row

Usage examples:
  python examples/reduction_verification.py \
    --input_csv karp.csv \
    --output_csv verified.csv \
    --max_iters 6 \
    --min_steps 8 --max_steps 20 \
    --thinking on

Notes:
- This script uses the shared vLLM wrapper in src/inference.py and reads model params from config.ini.
- Per-step evaluation is isolated by step but includes the full steps context in each prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.inference import Model, load_from_config


# -----------------------------
# Data structures
# -----------------------------

StepType = str  # one of {"definition","construction","claim","justification"}


@dataclass
class Step:
    index: int
    text: str
    type: StepType


# -----------------------------
# Prompt construction
# -----------------------------

def create_extraction_messages(
    source_text: str,
    target_text: str,
    reduction_full_text: str,
    min_steps: int,
    max_steps: int,
    thinking: str = "on",
) -> List[Dict[str, str]]:
    """Ask the model to rewrite the reduction into granular, typed steps.

    The model is asked to output pure JSON: {"steps": [{"index":1,"type":"definition|construction|claim|justification","text":"..."}, ...]}
    """

    system = f"""detailed thinking {thinking}

You are a meticulous complexity-theory verifier. Rewrite the provided reduction into a rigorous sequence of small, atomic steps.
Each step must be one short sentence. Label each step as one of: definition, construction, claim, justification.
Aim for {min_steps}-{max_steps} steps and stay within this range.
Return strictly JSON and nothing else. Do not emit LaTeX or backslashes; use plain text (e.g., write 'not x' instead of '\\lnot x').
"""

    user = f"""
SOURCE PROBLEM:
{source_text}

TARGET PROBLEM:
{target_text}

GROUND-TRUTH REDUCTION (to translate):
{reduction_full_text}

REQUIREMENTS:
- Break the reduction into small steps that are clear, precise, and relevant.
- Types:
  - definition: introduce objects, notation, or properties.
  - construction: describe an explicit mapping or transformation.
  - claim: assert a property that must be evident or justified by other steps.
  - justification: explain why a claim holds.
- Keep steps single-purpose and avoid verbosity; avoid references like "as above"—be explicit.
- Do not include LaTeX environments; translate them to plain text when needed.
 - Must include at least:
     - explicit construction mapping from source instance to target instance;
     - an instance validity/counting step (e.g., clause/variable bounds like "each literal appears at most twice" if applicable);
     - a polynomial-time argument;
     - correctness in both directions: one step for (⇒) and one for (⇐).

        FORMAT:
        Output valid JSON with a single top-level key "steps":
{{
  "steps": [
    {{"index":1, "type":"definition", "text":"..."}},
    ...
  ]
}}
"""

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


def create_evaluation_messages(
    source_text: str,
    target_text: str,
    reduction_full_text: str,
    all_steps: List[Step],
    step_idx: int,
    thinking: str = "on",
) -> List[Dict[str, str]]:
    """Create messages to evaluate one step given the full steps and ground truth.

    The model should return JSON with fields:
    {"verdict":"pass|fail","issues":["..."],"repair":"rewritten step text or empty if pass","reclassify":"optional new type or empty"}
    """

    system = f"""detailed thinking {thinking}

You are an exacting proof assistant. Evaluate exactly one step from a proposed reduction.
Criteria for pass:
- If type is definition or construction or justification: the step must be clear, precise, and relevant (no missing context, no ambiguity), and not overly terse or verbose.
- If type is claim: it must be immediately evident or justified by other steps (earlier or later), or by standard definitions; if not, propose a concrete justification or weaken the claim.
- The step must be consistent with the provided ground-truth reduction.
Return strictly JSON and nothing else. Do not emit LaTeX or backslashes in fields.
"""

    # Build JSON context blocks safely
    steps_json_block = json.dumps([
        {"index": s.index, "type": s.type, "text": s.text.replace("\\", " ")}
        for s in all_steps
    ], ensure_ascii=False)

    focus = all_steps[step_idx]
    focus_json_block = json.dumps({
        "index": focus.index,
        "type": focus.type,
        "text": focus.text.replace("\\", " ")
    }, ensure_ascii=False)

    user = f"""
SOURCE PROBLEM: {source_text}
TARGET PROBLEM: {target_text}

GROUND-TRUTH REDUCTION:
{reduction_full_text}

FULL CANDIDATE STEPS (context):
{steps_json_block}

FOCUS STEP (evaluate only this one):
{focus_json_block}

EVALUATION INSTRUCTIONS:
- Decide pass/fail.
- If fail, list concrete issues and propose a precise repair for the single step; keep it one sentence, and keep or update the type if needed.
- Also flag excessive brevity or verbosity.

FORMAT:
{{
  "verdict": "pass" | "fail",
  "issues": ["..."],
  "repair": "...",              # empty if pass
  "reclassify": "" | "definition|construction|claim|justification"
}}
"""

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


# -----------------------------
# Global verification prompts
# -----------------------------

def create_extraction_refine_messages(
    source_text: str,
    target_text: str,
    reduction_full_text: str,
    current_steps: List[Step],
    min_steps: int,
    max_steps: int,
    thinking: str = "on",
) -> List[Dict[str, str]]:
    """Ask the model to adjust the step list to fit within bounds and include required elements.

    Returns JSON {"steps": [...]} with the same step object schema.
    """

    system = f"""detailed thinking {thinking}

You are refining a step-by-step reduction. Adjust the list so it has exactly {min_steps} one-sentence steps and includes:
- explicit construction mapping;
- an instance validity/counting step (e.g., bounds like "each literal appears at most twice" if applicable);
- a polynomial-time argument;
- correctness in both directions: one step for (⇒) and one for (⇐).
Keep type labels as one of: definition, construction, claim, justification.
Return strictly JSON and nothing else. No LaTeX or backslashes.
"""

    steps_json_block = json.dumps([
        {"index": s.index, "type": s.type, "text": s.text.replace("\\", " ")}
        for s in current_steps
    ], ensure_ascii=False)

    user = f"""
SOURCE PROBLEM: {source_text}
TARGET PROBLEM: {target_text}

GROUND-TRUTH REDUCTION (for reference during refinement):
{reduction_full_text}

CURRENT STEPS (adjust to fit requirements):
{steps_json_block}

FORMAT:
{{
  "steps": [
    {{"index":1, "type":"definition|construction|claim|justification", "text":"..."}},
    ...
  ]
}}
"""

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]

def create_global_standalone_messages(
    source_text: str,
    target_text: str,
    all_steps: List[Step],
    thinking: str = "on",
) -> List[Dict[str, str]]:
    """Ask the model to judge whether steps form a complete standalone proof (no ground truth)."""

    system = f"""detailed thinking {thinking}

You are a rigorous verifier. Determine whether the provided steps alone form a complete, correct, and self-contained reduction proof from the source problem to the target problem. Do not rely on any external text.

Required elements for PASS:
- Clear problem definitions/notation as needed for understanding (or explicit reliance on standard definitions is fine if stated).
- Precise construction: explicit mapping from source instance to target instance, including exact gadgets/clauses.
- Instance validity: argument that each literal appears at most twice (for 2-occurrence 1-in-3 SAT), and that the target instance remains in the proper clause form.
- Polynomial-time construction.
- Correctness both directions: (⇒) and (⇐), with concrete reasoning.
- No incorrect statements.
- Steps are single-purpose, concise but sufficiently precise.

If FAIL, propose a minimal repaired full step list that satisfies the above. Keep one-sentence steps and the type labels.
Return strictly JSON and nothing else. Do not emit LaTeX or backslashes.
"""

    steps_json_block = json.dumps([
        {"index": s.index, "type": s.type, "text": s.text.replace("\\", " ")}
        for s in all_steps
    ], ensure_ascii=False)

    user = f"""
SOURCE PROBLEM: {source_text}
TARGET PROBLEM: {target_text}

STEPS (candidate standalone proof):
{steps_json_block}

FORMAT:
{{
  "verdict": "pass" | "fail",
  "issues": ["..."],
  "patched_steps": [ {{"index":1, "type":"...", "text":"..."}}, ... ]  # optional; include only if fail
}}
"""

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


def create_global_consistency_messages(
    source_text: str,
    target_text: str,
    reduction_full_text: str,
    all_steps: List[Step],
    thinking: str = "on",
) -> List[Dict[str, str]]:
    """Ask the model to compare steps against ground truth for consistency (as a reference)."""

    system = f"""detailed thinking {thinking}

You are a careful comparator. Treat the ground-truth reduction text as a reference and check whether the provided steps are consistent with it: no contradictions, no missing essential parts, and equivalent intent.
If inconsistencies or gaps exist, propose a minimally changed full step list that aligns with the reference.
Return strictly JSON and nothing else. Do not emit LaTeX or backslashes.
"""

    steps_json_block = json.dumps([
        {"index": s.index, "type": s.type, "text": s.text.replace("\\", " ")}
        for s in all_steps
    ], ensure_ascii=False)

    user = f"""
SOURCE PROBLEM: {source_text}
TARGET PROBLEM: {target_text}

GROUND-TRUTH REDUCTION (reference for comparison only):
{reduction_full_text}

STEPS (candidate proof):
{steps_json_block}

FORMAT:
{{
  "verdict": "pass" | "fail",
  "issues": ["..."],
  "patched_steps": [ {{"index":1, "type":"...", "text":"..."}}, ... ]  # optional; include only if fail
}}
"""

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


# -----------------------------
# Utilities
# -----------------------------

def _step_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "type": {"type": "string", "enum": ["definition", "construction", "claim", "justification"]},
            "text": {"type": "string"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }


def _extraction_schema(min_items: int, max_items: int) -> Dict[str, Any]:
    # Note: Some JSON Schema features may not be fully enforced by the backend.
    # We still provide min/max bounds and require presence of key step types via 'contains'.
    steps_array: Dict[str, Any] = {
        "type": "array",
        "items": _step_schema(),
        "minItems": int(max(1, min_items)),
        "maxItems": int(max_items) if max_items and max_items >= min_items else int(max(1, min_items)),
    }

    # Encourage presence of construction, claim, and justification types at least once.
    # (If 'contains' isn't supported fully, the model still sees prompt requirements above.)
    steps_array["allOf"] = [
        {"contains": {"type": "object", "properties": {"type": {"const": "construction"}}}},
        {"contains": {"type": "object", "properties": {"type": {"const": "claim"}}}},
        {"contains": {"type": "object", "properties": {"type": {"const": "justification"}}}},
    ]

    return {
        "type": "object",
        "properties": {
            "steps": steps_array,
        },
        "required": ["steps"],
        "additionalProperties": False,
    }


def _evaluation_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "issues": {"type": "array", "items": {"type": "string"}},
            "repair": {"type": "string"},
            "reclassify": {"type": "string", "enum": ["", "definition", "construction", "claim", "justification"]},
        },
        "required": ["verdict", "issues", "repair", "reclassify"],
        "additionalProperties": False,
    }


def _global_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "issues": {"type": "array", "items": {"type": "string"}},
            "patched_steps": {"type": "array", "items": _step_schema() },
        },
        "required": ["verdict", "issues"],
        "additionalProperties": False,
    }

def _extract_json(text: str) -> Optional[dict]:
    """Try to parse first JSON object or array from text.
    Returns None if parsing fails.
    """
    text = text.strip()
    # Common case: already pure JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Heuristic: find outermost JSON object/array
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if m:
        snippet = m.group(1)
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _extract_json_with_key(text: str, key: str) -> Optional[dict]:
    """Scan text for any JSON object that contains a given top-level key.
    Useful when the model wraps JSON in prose or code fences.
    """
    text = text.strip()
    # Fast path: plain JSON object
    try:
        data = json.loads(text)
        if isinstance(data, dict) and key in data:
            return data
    except Exception:
        pass

    # Heuristic: iterate over all candidate JSON snippets (objects or arrays)
    for m in re.finditer(r"(\{.*?\}|\[.*?\])", text, flags=re.S):
        snippet = m.group(1)
        try:
            data = json.loads(snippet)
            if isinstance(data, dict) and key in data:
                return data
        except Exception:
            continue
    return None


def _coerce_steps(obj: Any) -> List[Step]:
    """Convert model JSON to List[Step], with minimal validation."""
    if not isinstance(obj, dict) or "steps" not in obj:
        raise ValueError("Extraction JSON must be an object with key 'steps'")
    steps_raw = obj["steps"]
    if not isinstance(steps_raw, list):
        raise ValueError("'steps' must be a list")

    steps: List[Step] = []
    for i, s in enumerate(steps_raw, start=1):
        if not isinstance(s, dict):
            raise ValueError("Each step must be an object")
        idx = int(s.get("index", i))
        t = str(s.get("type", ""))
        if t.lower() not in {"definition", "construction", "claim", "justification"}:
            t = "construction" if i == 1 else "claim"  # simple fallback
        txt = str(s.get("text", "")).strip()
        if not txt:
            txt = "[empty step]"
        steps.append(Step(index=idx, text=txt, type=t.lower()))
    # normalize indices to 1..n
    for j, s in enumerate(steps, start=1):
        s.index = j
    return steps


def _parse_eval_json(text: str) -> Tuple[str, List[str], str, str]:
    """Return (verdict, issues, repair, reclassify). Fallbacks are conservative."""
    data = _extract_json(text) or {}
    verdict = str(data.get("verdict", "fail")).lower()
    if verdict not in {"pass", "fail"}:
        verdict = "fail"
    issues = data.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    repair = str(data.get("repair", ""))
    reclassify = str(data.get("reclassify", "")).lower()
    if reclassify and reclassify not in {"definition", "construction", "claim", "justification"}:
        reclassify = ""
    return verdict, [str(x) for x in issues], repair, reclassify


def _parse_global_json(text: str) -> Tuple[str, List[str], Optional[List[Step]]]:
    """Parse global verification JSON: returns (verdict, issues, patched_steps or None)."""
    data = _extract_json(text) or {}
    verdict = str(data.get("verdict", "fail")).lower()
    if verdict not in {"pass", "fail"}:
        verdict = "fail"
    issues = data.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    patched_steps = None
    if "patched_steps" in data and isinstance(data["patched_steps"], list):
        try:
            patched_steps = _coerce_steps({"steps": data["patched_steps"]})
        except Exception:
            patched_steps = None
    return verdict, [str(x) for x in issues], patched_steps


# -----------------------------
# Core verification loop
# -----------------------------

def verify_reduction(
    model: Model,
    source_text: str,
    target_text: str,
    reduction_full_text: str,
    min_steps: int,
    max_steps: int,
    max_iters: int,
    global_iters: int,
    consistency_iters: int,
    enable_thinking: bool,
    session_prefix: str,
) -> Tuple[List[Step], List[str], float, int, Optional[str], Optional[str]]:
    """Run extraction then per-step evaluation/repair.

    Returns: (final_steps, issues_log, total_latency_s, total_tokens)
    """
    total_latency = 0.0
    total_tokens = 0
    issues_log: List[str] = []

    # 1) Extraction
    extract_msgs = create_extraction_messages(
        source_text, target_text, reduction_full_text, min_steps, max_steps, "on" if enable_thinking else "off"
    )
    res = model.infer(
        prompt=extract_msgs[1]["content"],
        system_prompt=extract_msgs[0]["content"],
        session_id=f"{session_prefix}-extract",
        enable_thinking=enable_thinking,
        json_schema=_extraction_schema(min_steps, max_steps),
    )
    total_latency += res["latency_s"]
    total_tokens += res["tokens"]

    data = _extract_json_with_key(res["text"], "steps") or _extract_json(res["text"]) or {}
    try:
        steps = _coerce_steps(data)
    except Exception as e:
        # Fallback: create a single construction step containing a condensed version of the text
        condensed = res["text"].strip()
        if len(condensed) > 400:
            condensed = condensed[:400] + " ..."
        steps = [Step(index=1, text=condensed, type="construction")]
        issues_log.append(f"extraction-json-failure: {e}")

    # If outside requested bounds, run a short refinement loop to expand/compress.
    if len(steps) < min_steps or (max_steps and len(steps) > max_steps):
        issues_log.append(
            f"extraction-size-adjust: had {len(steps)} steps, targeting {min_steps}-{max_steps}"
        )
        for ref_iter in range(2):
            refine_msgs = create_extraction_refine_messages(
                source_text, target_text, reduction_full_text, steps, min_steps, max_steps, "on" if enable_thinking else "off"
            )
            rr = model.infer(
                prompt=refine_msgs[1]["content"],
                system_prompt=refine_msgs[0]["content"],
                session_id=f"{session_prefix}-extract-refine-{ref_iter+1}",
                enable_thinking=enable_thinking,
                json_schema=_extraction_schema(min_steps, min_steps),
            )
            total_latency += rr["latency_s"]
            total_tokens += rr["tokens"]
            d2 = _extract_json_with_key(rr["text"], "steps") or _extract_json(rr["text"]) or {}
            try:
                steps2 = _coerce_steps(d2)
                if min_steps <= len(steps2) <= max_steps:
                    steps = steps2
                    break
                else:
                    steps = steps2  # keep best effort even if still off
            except Exception:
                # keep previous steps if parse failed
                pass

    # 2) Per-step evaluation loop
    for i in range(len(steps)):
        iter_count = 0
        while iter_count < max_iters:
            eval_msgs = create_evaluation_messages(
                source_text, target_text, reduction_full_text, steps, i, "on" if enable_thinking else "off"
            )
            r = model.infer(
                prompt=eval_msgs[1]["content"],
                system_prompt=eval_msgs[0]["content"],
                session_id=f"{session_prefix}-step-{i+1}-iter-{iter_count+1}",
                enable_thinking=enable_thinking,
                json_schema=_evaluation_schema(),
            )
            total_latency += r["latency_s"]
            total_tokens += r["tokens"]

            verdict, issues, repair, reclassify = _parse_eval_json(r["text"])

            if verdict == "pass":
                # Keep as-is; optionally log any warnings if present
                if issues:
                    issues_log.append(f"step-{i+1}: pass-with-notes: " + "; ".join(issues))
                break

            # Fail: apply repair if provided; otherwise append issue and continue once
            if issues:
                issues_log.append(f"step-{i+1}: " + "; ".join(issues))

            updated = False
            if repair:
                steps[i].text = repair.strip()
                updated = True
            if reclassify:
                steps[i].type = reclassify
                updated = True

            if not updated:
                # If no concrete repair, try a minimal rewrite prompt: ask for a single-sentence, concrete rewrite
                steps[i].text = steps[i].text + " (clarify)"

            iter_count += 1

    # 3) Global standalone verification loop (no ground truth in prompt)
    standalone_verdict: Optional[str] = None
    for g in range(global_iters):
        g_msgs = create_global_standalone_messages(
            source_text, target_text, steps, "on" if enable_thinking else "off"
        )
        gr = model.infer(
            prompt=g_msgs[1]["content"],
            system_prompt=g_msgs[0]["content"],
            session_id=f"{session_prefix}-global-standalone-{g+1}",
            enable_thinking=enable_thinking,
            json_schema=_global_schema(),
        )
        total_latency += gr["latency_s"]
        total_tokens += gr["tokens"]
        verdict, g_issues, patched = _parse_global_json(gr["text"])
        standalone_verdict = verdict
        if g_issues:
            issues_log.append("global-standalone: " + "; ".join(g_issues))
        if verdict == "pass":
            break
        if verdict == "fail" and not patched and not g_issues:
            issues_log.append("global-standalone: fail (no patched_steps provided)")
        if patched:
            steps = patched
            # Re-run per-step checks quickly to validate updated steps
            for i in range(len(steps)):
                iter_count = 0
                while iter_count < min(3, max_iters):
                    eval_msgs = create_evaluation_messages(
                        source_text, target_text, reduction_full_text, steps, i, "on" if enable_thinking else "off"
                    )
                    r = model.infer(
                        prompt=eval_msgs[1]["content"],
                        system_prompt=eval_msgs[0]["content"],
                        session_id=f"{session_prefix}-globalfix-step-{i+1}-iter-{iter_count+1}",
                        enable_thinking=enable_thinking,
                        json_schema=_evaluation_schema(),
                    )
                    total_latency += r["latency_s"]
                    total_tokens += r["tokens"]
                    v, iss, rep, rec = _parse_eval_json(r["text"])
                    if iss:
                        issues_log.append(f"globalfix-step-{i+1}: " + "; ".join(iss))
                    if v == "pass":
                        break
                    if rep:
                        steps[i].text = rep.strip()
                    if rec:
                        steps[i].type = rec
                    iter_count += 1

    # 4) Global consistency verification loop (with ground truth as reference)
    consistency_verdict: Optional[str] = None
    for g in range(consistency_iters):
        c_msgs = create_global_consistency_messages(
            source_text, target_text, reduction_full_text, steps, "on" if enable_thinking else "off"
        )
        cr = model.infer(
            prompt=c_msgs[1]["content"],
            system_prompt=c_msgs[0]["content"],
            session_id=f"{session_prefix}-global-consistency-{g+1}",
            enable_thinking=enable_thinking,
            json_schema=_global_schema(),
        )
        total_latency += cr["latency_s"]
        total_tokens += cr["tokens"]
        verdict, c_issues, patched = _parse_global_json(cr["text"])
        consistency_verdict = verdict
        if c_issues:
            issues_log.append("global-consistency: " + "; ".join(c_issues))
        if verdict == "pass":
            break
        if verdict == "fail" and not patched and not c_issues:
            issues_log.append("global-consistency: fail (no patched_steps provided)")
        if patched:
            steps = patched
            # Validate patched steps with a quick per-step pass
            for i in range(len(steps)):
                iter_count = 0
                while iter_count < min(3, max_iters):
                    eval_msgs = create_evaluation_messages(
                        source_text, target_text, reduction_full_text, steps, i, "on" if enable_thinking else "off"
                    )
                    r = model.infer(
                        prompt=eval_msgs[1]["content"],
                        system_prompt=eval_msgs[0]["content"],
                        session_id=f"{session_prefix}-consistencyfix-step-{i+1}-iter-{iter_count+1}",
                        enable_thinking=enable_thinking,
                        json_schema=_evaluation_schema(),
                    )
                    total_latency += r["latency_s"]
                    total_tokens += r["tokens"]
                    v, iss, rep, rec = _parse_eval_json(r["text"])
                    if iss:
                        issues_log.append(f"consistencyfix-step-{i+1}: " + "; ".join(iss))
                    if v == "pass":
                        break
                    if rep:
                        steps[i].text = rep.strip()
                    if rec:
                        steps[i].type = rec
                    iter_count += 1

    return steps, issues_log, total_latency, total_tokens, standalone_verdict, consistency_verdict


# -----------------------------
# CLI & main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reduction verification via granular steps and self-evaluation")
    p.add_argument("--toy", action="store_true", help="Use tiny model for local testing (overrides config.ini)")
    p.add_argument("--input_csv", required=True, help="CSV with source_text,target_text,reduction_full_text")
    p.add_argument("--output_csv", default="verified_results.csv", help="Output CSV path")
    p.add_argument("--min_steps", type=int, default=8, help="Minimum target number of steps for extraction")
    p.add_argument("--max_steps", type=int, default=20, help="Maximum target number of steps for extraction")
    p.add_argument("--max_iters", type=int, default=10, help="Max repair iterations per step")
    p.add_argument("--global_iters", type=int, default=2, help="Max global standalone verification iterations")
    p.add_argument("--consistency_iters", type=int, default=2, help="Max global consistency verification iterations")
    p.add_argument("--limit_rows", type=int, default=None, help="If set, only process the first N rows")
    p.add_argument("--row_offset", type=int, default=0, help="Skip the first N rows")

    # Model params (overrides)
    p.add_argument("--thinking", choices=["on", "off"], default="on", help="Enable think tokens if supported")
    p.add_argument("--temperature", type=float, help="Override temperature from config")
    p.add_argument("--top_p", type=float, help="Override top_p from config")
    p.add_argument("--top_k", type=int, help="Override top_k from config")
    p.add_argument("--max_tokens", type=int, help="Override max_tokens from config")
    p.add_argument("--tensor_parallel_size", type=int, help="Override tensor_parallel_size from config")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load CSV
    in_path = os.path.expanduser(args.input_csv)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")
    df = pd.read_csv(in_path)
    required_cols = ["source_text", "target_text", "reduction_full_text"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Slice if requested
    if args.row_offset:
        df = df.iloc[args.row_offset :].reset_index(drop=True)
    if args.limit_rows is not None:
        df = df.iloc[: args.limit_rows].reset_index(drop=True)

    print(f"[verify] Loaded {len(df)} rows from {in_path}")

    # Build model with config + overrides
    model_kwargs: Dict[str, Any] = {"toy": args.toy}
    if args.tensor_parallel_size is not None:
        model_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.temperature is not None:
        model_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        model_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        model_kwargs["top_k"] = args.top_k
    if args.max_tokens is not None:
        model_kwargs["max_tokens"] = args.max_tokens

    model = load_from_config(**model_kwargs)
    enable_thinking = args.thinking.lower() == "on"

    # Outputs to collect
    out_steps_json: List[str] = []
    out_issue_summaries: List[str] = []
    out_latencies: List[float] = []
    out_tokens: List[int] = []
    out_standalone: List[Optional[str]] = []
    out_consistency: List[Optional[str]] = []

    t0 = time.time()
    for idx, row in df.iterrows():
        src = str(row["source_text"]) 
        tgt = str(row["target_text"]) 
        red = str(row["reduction_full_text"]) 

        steps, issues, latency, tokens, v_standalone, v_consistency = verify_reduction(
            model=model,
            source_text=src,
            target_text=tgt,
            reduction_full_text=red,
            min_steps=int(args.min_steps),
            max_steps=int(args.max_steps),
            max_iters=int(args.max_iters),
            global_iters=int(args.global_iters),
            consistency_iters=int(args.consistency_iters),
            enable_thinking=enable_thinking,
            session_prefix=f"verify-row-{idx}",
        )

        out_steps_json.append(json.dumps([s.__dict__ for s in steps], ensure_ascii=False))
        out_issue_summaries.append("; ".join(issues))
        out_latencies.append(latency)
        out_tokens.append(tokens)
        out_standalone.append(v_standalone)
        out_consistency.append(v_consistency)

        if (idx + 1) % 5 == 0:
            recent = sum(out_latencies[-5:]) / min(5, len(out_latencies[-5:]))
            print(f"[verify] {idx+1}/{len(df)} rows | avg last-5 latency: {recent:.2f}s")

    # Save
    df["verified_steps_json"] = out_steps_json
    df["issues_summary"] = out_issue_summaries
    df["standalone_verdict"] = out_standalone
    df["consistency_verdict"] = out_consistency
    df["verify_latency_s"] = out_latencies
    df["verify_tokens"] = out_tokens

    out_path = os.path.expanduser(args.output_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)

    total = time.time() - t0
    print(f"[verify] Saved results to {out_path}")
    print(f"[verify] Total wall time: {total:.1f}s | Avg: {total/len(df):.2f}s/row")


if __name__ == "__main__":
    main()
