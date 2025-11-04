from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .compare import DEFAULT_COMPARE_SYSTEM_PROMPT, compare_to_ground_truth
from .evaluator import DEFAULT_STEP_SYSTEM_PROMPT, evaluate_steps
from .global_eval import DEFAULT_GLOBAL_SYSTEM_PROMPT, run_global_evaluation
from .parse import parse_with_model
from .repair import DEFAULT_REPAIR_SYSTEM_PROMPT, apply_edits, collect_touched_indices, repair_with_model
from .synthesize import count_medium_high, synthesize_issues
from .vllm_structured import StructuredResult, _stable_id

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_PARSE_SYSTEM_PROMPT = "You extract orderly reduction steps and respond in JSON only."


@dataclass
class PipelineResult:
    steps: List[str]
    issues: List[Dict[str, Any]]
    iterations: int
    history: List[Dict[str, Any]]
    final_summary: Dict[str, Any]
    artifacts_dir: Optional[str] = None


class ArtifactLogger:
    def __init__(self, root: Optional[str]):
        self.root = root
        self.audit_path = os.path.join(root, "audit_log.jsonl") if root else None
        if self.root:
            os.makedirs(self.root, exist_ok=True)

    def write_json(self, relative_path: str, payload: Any) -> None:
        if not self.root:
            return
        path = os.path.join(self.root, relative_path)
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def log_event(self, event: Dict[str, Any]) -> None:
        if not self.audit_path:
            return
        record = dict(event)
        record.setdefault("timestamp", time.time())
        with open(self.audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(
    model,
    *,
    source_text: str,
    target_text: str,
    ground_truth: str,
    candidate_blob: str,
    run_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    system_prompts: Optional[Dict[str, str]] = None,
    max_iters: int = 6,
    max_edits_per_iteration: int = 5,
    retries: int = 2,
) -> PipelineResult:
    run_identifier = run_id or _stable_id("run", str(int(time.time() * 1000)))
    artifacts_root = os.path.join(output_dir, run_identifier) if output_dir else None
    artifacts = ArtifactLogger(artifacts_root)

    prompts = {
        "parse": DEFAULT_PARSE_SYSTEM_PROMPT,
        "step": DEFAULT_STEP_SYSTEM_PROMPT,
        "global": DEFAULT_GLOBAL_SYSTEM_PROMPT,
        "compare": DEFAULT_COMPARE_SYSTEM_PROMPT,
        "repair": DEFAULT_REPAIR_SYSTEM_PROMPT,
    }
    if system_prompts:
        prompts.update(system_prompts)

    parse_session = f"{run_identifier}-parse"
    steps = parse_with_model(
        model,
        session_id=parse_session,
        candidate_blob=candidate_blob,
        system_prompt=prompts["parse"],
        retries=retries,
        temperature=0.0,
        max_tokens=512,
    )
    artifacts.write_json("parsed_steps.json", {"steps": steps})
    artifacts.log_event({"event": "parse_completed", "steps": len(steps)})

    context = {
        "source_text": source_text,
        "target_text": target_text,
        "ground_truth": ground_truth,
    }

    history: List[Dict[str, Any]] = []
    last_step_results: List[Dict[str, Any]] = []
    last_issues: List[Dict[str, Any]] = []
    mechanical_stall_count = 0  # Counts iterations with no edits applied
    semantic_stall_count = 0    # Counts iterations with edits but no issue reduction
    previous_medium_high = -1   # Track issue count from previous iteration
    evaluation_matches_steps = False

    for iteration in range(1, max_iters + 1):
        iteration_dir = f"iteration-{iteration}"
        step_session_base = f"{run_identifier}-iter{iteration}"

        step_results, step_raw = evaluate_steps(
            model,
            session_base=step_session_base,
            context=context,
            steps=steps,
            retries=retries,
            temperature=0.0,
            max_tokens=512,
            system_prompt=prompts["step"],
        )
        global_result, global_raw = run_global_evaluation(
            model,
            session_id=f"{run_identifier}-global-{iteration}",
            context=context,
            steps=steps,
            retries=retries,
            temperature=0.0,
            max_tokens=1024,
            system_prompt=prompts["global"],
        )
        compare_result, compare_raw = compare_to_ground_truth(
            model,
            session_id=f"{run_identifier}-compare-{iteration}",
            context=context,
            steps=steps,
            retries=retries,
            temperature=0.0,
            max_tokens=1024,
            system_prompt=prompts["compare"],
        )

        issues = synthesize_issues(step_results, global_result, compare_result)
        medium_high = count_medium_high(issues)

        artifacts.write_json(f"{iteration_dir}/step_evaluations.json", step_results)
        artifacts.write_json(f"{iteration_dir}/global_eval.json", global_result)
        artifacts.write_json(f"{iteration_dir}/gt_compare.json", compare_result)
        artifacts.write_json(f"{iteration_dir}/issues.json", issues)
        raw_payload = _collect_raw_outputs(step_raw, global_raw, compare_raw)
        artifacts.write_json(f"{iteration_dir}/raw_model_outputs.json", raw_payload)

        history_entry: Dict[str, Any] = {
            "iteration": iteration,
            "step_results": step_results,
            "global_eval": global_result,
            "gt_compare": compare_result,
            "issues": issues,
        }

        artifacts.log_event({
            "event": "iteration_evaluated",
            "iteration": iteration,
            "issues_medium_high": medium_high,
        })

        last_step_results = step_results
        last_issues = issues
        evaluation_matches_steps = True

        if medium_high == 0:
            history.append(history_entry)
            break

        # Debug: Check issues before repair
        logger.debug("Issues before repair call: %d issues", len(issues))
        for i, issue in enumerate(issues[:3]):  # Show first 3
            logger.debug("  Issue %d keys: %s", i, list(issue.keys()))
            logger.debug("    id=%s, title=%s", issue.get("id"), issue.get("title", "")[:50])

        repair_plan, repair_raw = repair_with_model(
            model,
            session_id=f"{run_identifier}-repair-{iteration}",
            steps=steps,
            issues=issues,
            context=context,
            retries=retries,
            temperature=0.0,
            max_tokens=None,
            max_edits=max_edits_per_iteration,
            system_prompt=prompts["repair"],
        )
        history_entry["repair_plan"] = repair_plan
        artifacts.write_json(f"{iteration_dir}/repair_plan.json", repair_plan)
        if repair_raw is not None:
            artifacts.write_json(
                f"{iteration_dir}/repair_raw.json",
                _structured_result_to_dict(repair_raw),
            )

        new_steps, applied_edits, skipped_edits = apply_edits(steps, repair_plan.get("edits", []))
        history_entry["applied_edits"] = applied_edits
        history_entry["skipped_edits"] = skipped_edits
        artifacts.write_json(f"{iteration_dir}/applied_edits.json", applied_edits)
        if skipped_edits:
            artifacts.write_json(f"{iteration_dir}/skipped_edits.json", skipped_edits)

        touched = collect_touched_indices(applied_edits)
        artifacts.log_event({
            "event": "edits_applied",
            "iteration": iteration,
            "applied": len(applied_edits),
            "skipped": len(skipped_edits),
            "touched_indices": touched,
        })

        history.append(history_entry)

        # Update stall detection logic
        if not applied_edits:
            # Mechanical stall: no edits were applied
            mechanical_stall_count += 1
            semantic_stall_count = 0  # Reset semantic stall
            logger.info("Iteration %d: No edits applied (mechanical stall %d/2)", iteration, mechanical_stall_count)
        else:
            mechanical_stall_count = 0  # Reset mechanical stall
            evaluation_matches_steps = False
            
            # Check for semantic stall: edits applied but issues didn't decrease
            if previous_medium_high >= 0 and medium_high >= previous_medium_high:
                semantic_stall_count += 1
                logger.info(
                    "Iteration %d: Edits applied but medium/high issues unchanged (%d -> %d), semantic stall %d/2",
                    iteration, previous_medium_high, medium_high, semantic_stall_count
                )
            else:
                semantic_stall_count = 0  # Reset if we made progress

        previous_medium_high = medium_high
        steps = new_steps
        artifacts.write_json(f"{iteration_dir}/post_edit_steps.json", {"steps": steps})

        # Stop if we've stalled mechanically or semantically for 2 consecutive iterations
        if mechanical_stall_count >= 2:
            logger.info("Stopping: No edits applied for two consecutive iterations (mechanical stall).")
            break
        if semantic_stall_count >= 2:
            logger.info("Stopping: Issues not decreasing despite edits for two consecutive iterations (semantic stall).")
            break
    else:
        iteration = max_iters

    if not evaluation_matches_steps and steps:
        verification_dir = "final_verification"
        step_results, step_raw = evaluate_steps(
            model,
            session_base=f"{run_identifier}-final",
            context=context,
            steps=steps,
            retries=retries,
            temperature=0.0,
            max_tokens=512,
            system_prompt=prompts["step"],
        )
        global_result, global_raw = run_global_evaluation(
            model,
            session_id=f"{run_identifier}-final-global",
            context=context,
            steps=steps,
            retries=retries,
            temperature=0.0,
            max_tokens=1024,
            system_prompt=prompts["global"],
        )
        compare_result, compare_raw = compare_to_ground_truth(
            model,
            session_id=f"{run_identifier}-final-compare",
            context=context,
            steps=steps,
            retries=retries,
            temperature=0.0,
            max_tokens=1024,
            system_prompt=prompts["compare"],
        )
        last_step_results = step_results
        last_issues = synthesize_issues(step_results, global_result, compare_result)
        artifacts.write_json(f"{verification_dir}/step_evaluations.json", step_results)
        artifacts.write_json(f"{verification_dir}/global_eval.json", global_result)
        artifacts.write_json(f"{verification_dir}/gt_compare.json", compare_result)
        artifacts.write_json(
            f"{verification_dir}/raw_model_outputs.json",
            _collect_raw_outputs(step_raw, global_raw, compare_raw),
        )

    final_summary = _build_summary(last_step_results, last_issues)
    artifacts.write_json("final_summary.json", final_summary)
    artifacts.log_event({"event": "run_complete", "iterations": iteration})

    return PipelineResult(
        steps=steps,
        issues=last_issues,
        iterations=iteration,
        history=history,
        final_summary=final_summary,
        artifacts_dir=artifacts_root,
    )


def _collect_raw_outputs(
    step_raw: Sequence[StructuredResult],
    global_raw: Optional[StructuredResult],
    compare_raw: Optional[StructuredResult],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": [_structured_result_to_dict(item) for item in step_raw],
    }
    if global_raw is not None:
        payload["global"] = _structured_result_to_dict(global_raw)
    if compare_raw is not None:
        payload["compare"] = _structured_result_to_dict(compare_raw)
    return payload


def _structured_result_to_dict(result: StructuredResult) -> Dict[str, Any]:
    return {
        "raw_text": result.raw_text,
        "cleaned_text": result.cleaned_text,
        "latency_s": result.latency_s,
        "tokens": result.tokens,
        "attempts": result.attempts,
        "reasoning_text": result.reasoning_text,
        "reasoning_tokens": result.reasoning_tokens,
        "data": result.data,
    }


def _build_summary(step_results: Sequence[Dict[str, Any]], issues: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if step_results:
        pass_count = sum(1 for item in step_results if item.get("passes"))
        rate = pass_count / max(len(step_results), 1)
    else:
        rate = 0.0
    remaining = len(issues)
    remaining_severe = count_medium_high(issues)
    recommend_continue = remaining_severe > 0
    return {
        "per_step_pass_rate": rate,
        "remaining_issues_count": remaining,
        "remaining_high_medium": remaining_severe,
        "recommend_continue": recommend_continue,
    }


__all__ = ["run_pipeline", "PipelineResult"]
