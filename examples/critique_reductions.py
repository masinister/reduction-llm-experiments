from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from src import config
from src.core_backend import Backend


# ============================================================================
# Pydantic Models
# ============================================================================


class ProblemDefinition(BaseModel):
    name: str
    input_format: str
    yes_condition: str


class Reduction(BaseModel):
    source_problem: str
    target_problem: str
    source_definition: ProblemDefinition
    target_definition: ProblemDefinition
    reduction_steps: list[str]
    forward_proof: str
    backward_proof: str
    key_insight: str


class StepClarification(BaseModel):
    step_kind: Literal["definition", "claim", "unclear"]
    rewritable: bool
    explicit_statement: str | None
    missing_information: list[str]
    confidence: Literal["high", "medium", "low"]


class ReductionSummary(BaseModel):
    rewritable_steps: int
    total_steps: int
    blockers: list[str]
    overall_assessment: str


# ============================================================================
# Prompt
# ============================================================================


_STEP_CLARIFICATION_PROMPT = """SYSTEM: You are a precise reduction-clarification assistant.
Do NOT judge correctness. Do NOT fix the step.
Your task is ONLY to classify and restate.

COMPONENT:
"{step_content}"

TASK:
Classify this step and determine whether it can be rewritten as a single,
explicit mathematical statement WITHOUT introducing new ideas.

Definitions:
- DEFINITION: explicitly constructs or defines an object from given inputs
- CLAIM: asserts a property, implication, or equivalence
- UNCLEAR: mixes actions or is too vague to classify

EXAMPLES:

Input: "Given a graph G = (V, E), construct a formula φ with one variable x_v for each vertex v ∈ V."
Output:
{{
  "step_kind": "definition",
  "rewritable": true,
  "explicit_statement": "Define φ as a formula over variables {{x_v : v ∈ V}}.",
  "missing_information": [],
  "confidence": "high"
}}

Input: "If G has a clique of size k, then φ is satisfiable."
Output:
{{
  "step_kind": "claim",
  "rewritable": true,
  "explicit_statement": "For all graphs G: if G contains a clique of size k, then the formula φ constructed from G is satisfiable.",
  "missing_information": [],
  "confidence": "high"
}}

Input: "We then do the usual trick to handle the edges."
Output:
{{
  "step_kind": "unclear",
  "rewritable": false,
  "explicit_statement": null,
  "missing_information": ["what is 'the usual trick'", "how edges are handled"],
  "confidence": "high"
}}

NOW CLASSIFY THE COMPONENT ABOVE.

Answer the following:
1) step_kind: "definition", "claim", or "unclear"
2) rewritable:
   - true if the step could be rewritten as ONE explicit mathematical sentence
   - false if essential information is missing or hidden
3) explicit_statement:
   - if rewritable == true, write that single sentence
   - if rewritable == false, use null (not the string "null")
4) missing_information:
   - list concrete missing details that prevent rewriting
   - use short phrases
5) confidence: "high", "medium", or "low"

Return EXACTLY this JSON and nothing else:
{{
  "step_kind": "...",
  "rewritable": true|false,
  "explicit_statement": "..." or null,
  "missing_information": ["...", ...],
  "confidence": "..."
}}
"""


# ============================================================================
# Clarification Logic
# ============================================================================


def clarify_step(
    backend: Backend,
    step_content: str,
) -> StepClarification:
    """Ask the LLM to classify and restate a single step."""
    prompt = _STEP_CLARIFICATION_PROMPT.format(step_content=step_content)

    return backend.create(
        prompt,
        StepClarification,
        temperature=0.0,
        max_tokens=256,
    )


def clarify_reduction(
    backend: Backend,
    reduction: Reduction,
) -> tuple[list[dict[str, Any]], ReductionSummary]:
    """Clarify all components of a reduction."""

    components: list[tuple[str, str]] = []

    # Reduction steps
    for i, step in enumerate(reduction.reduction_steps):
        components.append((f"reduction_step_{i+1}", step))

    # High-level components
    components.append(("forward_proof", reduction.forward_proof))
    components.append(("backward_proof", reduction.backward_proof))
    components.append(("key_insight", reduction.key_insight))

    analyses: list[dict[str, Any]] = []
    blockers: list[str] = []

    rewritable_count = 0

    for label, content in components:
        if not content or not content.strip():
            continue

        try:
            clarification = clarify_step(backend, content)
        except ValidationError as e:
            clarification = StepClarification(
                step_kind="unclear",
                rewritable=False,
                explicit_statement=None,
                missing_information=["model output validation failed"],
                confidence="low",
            )

        if clarification.rewritable:
            rewritable_count += 1
        else:
            blockers.extend(clarification.missing_information)

        analyses.append(
            {
                "component": label,
                "original_text": content,
                "analysis": clarification.model_dump(),
            }
        )

    summary = ReductionSummary(
        rewritable_steps=rewritable_count,
        total_steps=len(analyses),
        blockers=sorted(set(blockers)),
        overall_assessment=(
            "All steps can be rewritten as explicit statements."
            if rewritable_count == len(analyses)
            else "Some steps lack sufficient specificity to be rewritten as explicit statements."
        ),
    )

    return analyses, summary


# ============================================================================
# IO Helpers
# ============================================================================


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as e:
                yield line_num, {"__json_error__": True, "error": str(e)}


def extract_reduction_dict(record: dict[str, Any]) -> dict[str, Any]:
    if "reduction" in record and isinstance(record["reduction"], dict):
        return record["reduction"]
    if "source_problem" in record and "target_problem" in record:
        return record
    raise ValueError("Record does not contain a valid reduction")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clarify reductions toward a more formalizable form",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("data") / "processed" / "karp_reductions.jsonl"),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)

    args = parser.parse_args()
    config.load()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / f"{input_path.stem}_critiqued{input_path.suffix}"
    )

    os.makedirs(output_path.parent, exist_ok=True)

    backend = Backend()

    processed = 0
    clarified = 0
    failed = 0

    total_records = count_jsonl_lines(input_path)

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for _, record in tqdm(iter_jsonl(input_path), total=total_records, desc="Critiquing"):
            if args.skip and processed < args.skip:
                processed += 1
                continue
            if args.limit is not None and processed >= args.skip + args.limit:
                break

            processed += 1
            entry_key = record.get("entry_key", f"record_{processed}")

            if record.get("__json_error__"):
                out_fh.write(json.dumps(record) + "\n")
                failed += 1
                continue

            try:
                reduction_dict = extract_reduction_dict(record)
                reduction = Reduction.model_validate(reduction_dict)

                step_analysis, summary = clarify_reduction(backend, reduction)

                # Preserve original fields, then add critique results
                output_record = {k: v for k, v in record.items()}
                output_record.update(
                    {
                        "entry_key": entry_key,
                        "reduction": reduction_dict,
                        "step_analysis": step_analysis,
                        "summary": summary.model_dump(),
                        "has_blockers": summary.rewritable_steps < summary.total_steps,
                    }
                )

                out_fh.write(
                    json.dumps(output_record, ensure_ascii=False) + "\n"
                )

                clarified += 1

            except Exception as e:
                failed += 1
                out_fh.write(
                    json.dumps(
                        {
                            "entry_key": entry_key,
                            "error": str(e),
                        }
                    )
                    + "\n"
                )

            gc.collect()

    print("\nClarification complete.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Processed: {processed}")
    print(f"Clarified: {clarified}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
