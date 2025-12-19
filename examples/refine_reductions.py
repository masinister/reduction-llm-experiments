"""Refine reductions based on critique output.

Loads critiqued reductions from a JSONL file (output of critique_reductions.py)
and applies fixes to address the flagged issues.

Input JSONL schema (from critique_reductions.py):
- `entry_key`: identifier
- `reduction`: the reduction to refine
- `critique`: the critique with major_issues and minor_issues
- `has_major_issues`: boolean flag

Output JSONL schema:
- `entry_key`: identifier (preserved)
- `reduction`: the refined Reduction object
- `input_critique`: the critique that was addressed
- `was_refined`: boolean indicating if changes were made
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

    source_problem: str = Field(description="The source problem being reduced from (e.g., '3-SAT')")
    target_problem: str = Field(description="The target problem being reduced to (e.g., 'CLIQUE')")
    source_definition: ProblemDefinition = Field(description="Formal definition of the source problem")
    target_definition: ProblemDefinition = Field(description="Formal definition of the target problem")
    reduction_steps: list[str] = Field(
        description="Step-by-step construction procedure transforming a source instance to a target instance"
    )
    forward_proof: str = Field(description="Proof that YES instance of source implies YES instance of target")
    backward_proof: str = Field(description="Proof that YES instance of target implies YES instance of source")
    key_insight: str = Field(description="The key idea or intuition that makes this reduction work")


# ============================================================================
# Prompt
# ============================================================================


_PLAIN_TEXT_RULES = """Output must be plain text only—no LaTeX or backslashes.
Use readable equivalents: AND, OR, NOT, "in", "subset of", "for all", "there exists", "implies", "iff".
"""


def make_refine_prompt(reduction: dict[str, Any], critique: dict[str, Any]) -> str:
    reduction_json = json.dumps(reduction, ensure_ascii=False, indent=2)
    critique_json = json.dumps(critique, ensure_ascii=False, indent=2)

    source_problem = reduction.get("source_problem", "")
    target_problem = reduction.get("target_problem", "")

    return f"""Apply minimal edits to fix the issues in the critique.

Constraints:
- Keep source_problem exactly as "{source_problem}"
- Keep target_problem exactly as "{target_problem}"
- Only fix what the critique flags; do not introduce unrelated changes

Return a JSON Reduction object with fields:
- source_problem, target_problem
- source_definition (name, input_format, yes_condition)
- target_definition (name, input_format, yes_condition)
- reduction_steps (list of strings)
- forward_proof, backward_proof
- key_insight

{_PLAIN_TEXT_RULES}

=== ORIGINAL REDUCTION ===
{reduction_json}

=== CRITIQUE ===
{critique_json}
"""


# ============================================================================
# IO Helpers
# ============================================================================


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e


def has_issues(critique: dict[str, Any]) -> bool:
    """Check if critique has any issues to address."""
    major = critique.get("major_issues", [])
    minor = critique.get("minor_issues", [])
    return bool(major) or bool(minor)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine reductions based on critique output",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("data") / "processed" / "karp_reductions_critiqued.jsonl"),
        help="Input JSONL path (output from critique_reductions.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: input path with '_refined' suffix)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N records")

    args = parser.parse_args()

    config.load()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_refined")

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    os.makedirs(output_path.parent, exist_ok=True)

    backend = Backend()

    processed = 0
    refined = 0
    skipped = 0
    failed = 0

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for _, record in tqdm(iter_jsonl(input_path), desc="Refining", unit="reduction"):
            if args.skip and processed < args.skip:
                processed += 1
                continue

            if args.limit is not None and (processed - args.skip) >= args.limit:
                break

            processed += 1

            entry_key = record.get("entry_key", f"record_{processed}")

            # Skip error records from critique step
            if "error" in record and "reduction" not in record:
                skipped += 1
                continue

            try:
                reduction_in = record.get("reduction", {})
                critique = record.get("critique", {})

                if not has_issues(critique):
                    # No issues to fix, pass through unchanged
                    out_record = {
                        "entry_key": entry_key,
                        "reduction": reduction_in,
                        "input_critique": critique,
                        "was_refined": False,
                    }
                    skipped += 1
                else:
                    # Refine based on critique
                    refined_reduction = backend.create(
                        make_refine_prompt(reduction_in, critique),
                        Reduction,
                        temperature=0.1,
                    )

                    out_record = {
                        "entry_key": entry_key,
                        "reduction": refined_reduction.model_dump(),
                        "input_critique": critique,
                        "was_refined": True,
                    }
                    refined += 1

                out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                status = "REFINED" if out_record["was_refined"] else "UNCHANGED"
                print(f"[{entry_key}] {status}")

            except Exception as e:
                failed += 1
                err_record = {
                    "entry_key": entry_key,
                    "error": str(e),
                }
                out_fh.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                print(f"[{entry_key}] ERROR: {e}")

            gc.collect()

    print("\nRefinement complete!")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Processed: {processed}")
    print(f"  Refined: {refined}")
    print(f"  Skipped (no issues): {skipped}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
