"""Critique generated reductions for correctness and coherence.

Loads structured reductions from a JSONL file and produces a critique identifying
major problems, logical gaps, or steps that don't make sense.

Works with output from either:
- extract_reductions.py (extraction pipeline)
- refine_reductions.py (refinement pipeline)
- zero_shot.py (zero-shot generation)

Output JSONL schema:
- `entry_key`: identifier from input
- `reduction`: the original reduction (preserved)
- `critique`: the generated critique
- `has_major_issues`: boolean flag for filtering
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from src import config
from src.core_backend import Backend


# ============================================================================
# Pydantic Models
# ============================================================================


class ProblemDefinition(BaseModel):
    """Definition of a computational problem."""

    name: str = Field(description="Name of the problem")
    input_format: str = Field(description="Description of what constitutes a valid input instance")
    yes_condition: str = Field(description="The condition that makes an instance a YES instance")


class Reduction(BaseModel):
    """Structured representation of a computational reduction proof."""

    source_problem: str = Field(description="The source problem being reduced from")
    target_problem: str = Field(description="The target problem being reduced to")
    source_definition: ProblemDefinition = Field(description="Formal definition of the source problem")
    target_definition: ProblemDefinition = Field(description="Formal definition of the target problem")
    reduction_steps: list[str] = Field(description="Step-by-step construction procedure")
    forward_proof: str = Field(description="Proof that YES source implies YES target")
    backward_proof: str = Field(description="Proof that YES target implies YES source")
    key_insight: str = Field(description="The key idea that makes this reduction work")


class StepCritique(BaseModel):
    """Critique of a single step or component."""

    is_valid: bool = Field(description="True if this component is concrete and correct, False if problematic")
    issue: str | None = Field(description="Description of the problem if is_valid=False, else null")
    severity: str = Field(description="'major' if correctness-affecting, 'minor' if just unclear, 'none' if valid")


class Critique(BaseModel):
    """Aggregate critique of a reduction's correctness and coherence."""

    summary: str = Field(description="Brief overall assessment")
    major_issues: list[str] = Field(
        description="Critical problems: incorrect logic, missing steps, flawed proofs (empty if none)"
    )
    minor_issues: list[str] = Field(
        description="Small issues that don't affect correctness (empty if none)"
    )


# ============================================================================
# Prompts
# ============================================================================


_CRITIQUE_RULES = """VALID steps specify the CONCRETE MAPPING or CONSTRUCTION:
- "For each variable x, introduce new variable y and replace ¬x with y" - says exactly what to create
- "Add vertex v connected to all vertices in G" - says exactly what to add
- "For each clause (a,b,c), add the triple {a,b,c} to the set S" - says exactly what mapping occurs

INVALID steps hide the key transformation behind vague language:
- "Convert clauses into a new form" - WHAT form? HOW?
- "Apply appropriate logical operations" - WHICH operations?
- "Transform the instance accordingly" - HOW?
- "Ensure the property holds through transformation" - WHAT transformation?
- "Map X to Y using a function" - WHAT function?

The test: Could someone IMPLEMENT this step from the description alone?
If the step just says "convert/transform/map X to Y" without saying HOW, it's invalid.

ALSO INVALID - logical errors:
- Circular reasoning or assuming what needs to be proved
- Proofs that restate the goal without justification
- Steps that contradict the problem definitions

IGNORE: Polynomial-time complexity arguments.

Evaluate THIS component independently. Output plain text—no LaTeX."""


def make_step_critique_prompt(
    reduction: dict[str, Any],
    step_type: str,
    step_index: int | None,
    step_content: str,
) -> str:
    """Create prompt to critique a single step/component."""
    reduction_json = json.dumps(reduction, ensure_ascii=False, indent=2)

    if step_index is not None:
        step_label = f"{step_type} #{step_index + 1}"
    else:
        step_label = step_type

    return f"""Critique this specific component of the reduction.

=== FULL REDUCTION (for context) ===
{reduction_json}

=== COMPONENT TO EVALUATE ===
{step_label}: "{step_content}"

{_CRITIQUE_RULES}

Ask yourself:
- Does this step specify the CONCRETE construction or mapping? (not just "convert X to Y")
- Could someone implement this step from the description alone?
- Is the logic correct and non-circular?

Return a JSON object:
- is_valid: true ONLY if the step specifies the concrete transformation
- issue: description of the problem if is_valid=false, else null  
- severity: "major" if vague/incorrect, "minor" if just unclear, "none" if valid
"""


def make_summary_prompt(
    reduction: dict[str, Any],
    component_critiques: list[dict[str, Any]],
) -> str:
    """Create prompt to summarize all component critiques."""
    reduction_json = json.dumps(reduction, ensure_ascii=False, indent=2)
    critiques_json = json.dumps(component_critiques, ensure_ascii=False, indent=2)

    return f"""Summarize the critique of this reduction based on the component-level analysis.

Return a JSON Critique object:
- summary: 1-2 sentence overall assessment
- major_issues: list of critical problems from the component critiques (can be empty)
- minor_issues: list of minor issues from the component critiques (can be empty)

=== REDUCTION ===
{reduction_json}

=== COMPONENT CRITIQUES ===
{critiques_json}
"""


# ============================================================================
# Per-Component Critique
# ============================================================================


def critique_reduction_components(
    backend: Backend,
    reduction: dict[str, Any],
) -> tuple[Critique, list[dict[str, Any]]]:
    """Critique each component of the reduction individually.

    Returns:
        Tuple of (final_critique, component_critiques_list)
    """
    component_critiques = []

    # Components to critique in order
    components = []

    # Add reduction steps
    steps = reduction.get("reduction_steps", [])
    for i, step in enumerate(steps):
        components.append(("reduction_step", i, step))

    # Add proofs
    components.append(("forward_proof", None, reduction.get("forward_proof", "")))
    components.append(("backward_proof", None, reduction.get("backward_proof", "")))
    components.append(("key_insight", None, reduction.get("key_insight", "")))

    # Critique each component independently (no accumulated context to prevent contamination)
    for step_type, step_index, step_content in components:
        if not step_content or not step_content.strip():
            continue

        step_critique = backend.create(
            make_step_critique_prompt(
                reduction, step_type, step_index, step_content
            ),
            StepCritique,
            temperature=0.0,
        )

        critique_record = {
            "component": step_type,
            "index": step_index,
            "content": step_content[:100] + "..." if len(step_content) > 100 else step_content,
            "is_valid": step_critique.is_valid,
            "issue": step_critique.issue,
            "severity": step_critique.severity,
        }
        component_critiques.append(critique_record)

    # Generate summary from component critiques
    summary_critique = backend.create(
        make_summary_prompt(reduction, component_critiques),
        Critique,
        temperature=0.0,
    )

    return summary_critique, component_critiques


# ============================================================================
# IO Helpers
# ============================================================================


def iter_jsonl(path: Path):
    """Iterate over JSONL file, yielding (line_num, record) pairs."""
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e


def extract_reduction_dict(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the reduction dict from a record.

    Supports both nested format {"reduction": {...}} and flat format.
    """
    if "reduction" in record and isinstance(record["reduction"], dict):
        return record["reduction"]
    # Assume flat format with reduction fields at top level
    if "source_problem" in record and "target_problem" in record:
        return record
    raise ValueError("Record does not contain a valid reduction")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Critique generated reductions for correctness and coherence",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("data") / "processed" / "karp_reductions.jsonl"),
        help="Input JSONL path containing reductions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: input path with '_critiqued' suffix)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N records")

    args = parser.parse_args()

    config.load()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_critiqued")

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    os.makedirs(output_path.parent, exist_ok=True)

    backend = Backend()

    processed = 0
    critiqued_ok = 0
    with_major_issues = 0
    failed = 0

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for _, record in tqdm(iter_jsonl(input_path), desc="Critiquing", unit="reduction"):
            if args.skip and processed < args.skip:
                processed += 1
                continue

            if args.limit is not None and (processed - args.skip) >= args.limit:
                break

            processed += 1

            entry_key = record.get("entry_key", f"record_{processed}")

            try:
                reduction_dict = extract_reduction_dict(record)

                # Critique each component individually
                critique, component_critiques = critique_reduction_components(
                    backend, reduction_dict
                )

                has_major = bool(critique.major_issues)

                # Count invalid components
                invalid_count = sum(1 for c in component_critiques if not c["is_valid"])
                major_count = sum(1 for c in component_critiques if c["severity"] == "major")

                out_record = {
                    "entry_key": entry_key,
                    "reduction": reduction_dict,
                    "critique": {
                        "summary": critique.summary,
                        "major_issues": critique.major_issues,
                        "minor_issues": critique.minor_issues,
                        "component_critiques": component_critiques,
                    },
                    "has_major_issues": has_major,
                }

                out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                critiqued_ok += 1

                if has_major:
                    with_major_issues += 1

                # Brief status
                status = "ISSUES" if has_major else "OK"
                component_info = f" ({invalid_count} invalid, {major_count} major)"
                print(f"[{entry_key}] {status}{component_info}: {critique.summary[:50]}...")

            except Exception as e:
                failed += 1
                err_record = {
                    "entry_key": entry_key,
                    "error": str(e),
                }
                out_fh.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                print(f"[{entry_key}] ERROR: {e}")

            gc.collect()

    print("\nCritique complete!")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Processed: {processed}")
    print(f"  Critiqued OK: {critiqued_ok}")
    print(f"  With major issues: {with_major_issues}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
