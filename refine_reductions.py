"""Refine extracted reduction representations via a two-step LLM pass.

Loads structured reductions from a JSONL file (default: data/processed/karp_reductions.jsonl),
critiques their rigor/quality (prompt 1), then applies edits to produce an improved
structured representation (prompt 2).

Output is written as JSONL with the refined Reduction fields plus an attached critique.
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
# Pydantic Models (match extraction schema)
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


class Critique(BaseModel):
    """Critique of a structured reduction representation."""

    summary: str = Field(description="One-paragraph summary of quality and key issues")
    major_issues: list[str] = Field(description="Critical correctness/rigor issues that must be fixed")
    minor_issues: list[str] = Field(description="Smaller clarity/formatting issues")
    suggested_edits: list[str] = Field(description="Actionable edits to improve correctness and clarity")
    confidence: float = Field(description="Confidence in critique from 0.0 to 1.0")


# ============================================================================
# Prompts
# ============================================================================


_PLAIN_TEXT_RULES = """*** CRITICAL: ALL OUTPUT MUST BE PLAIN TEXT - NO LATEX ***
Convert all math notation to readable plain text:
- \\land, \\wedge -> "AND"
- \\lor, \\vee -> "OR"
- \\neg, \\lnot -> "NOT"
- \\in -> "in"
- \\subseteq -> "subset of"
- \\forall -> "for all"
- \\exists -> "there exists"
- \\implies, \\Rightarrow -> "implies" or "=>"
- \\iff, \\Leftrightarrow -> "if and only if"
- $x_i$ -> "x_i"
- \\textsc{Name} -> "NAME"
- \\Big, \\big, \\left, \\right -> remove entirely
- Remove all \\begin{...}, \\end{...}, \\item, etc.
- Avoid backslashes in the final text.
"""


def make_critique_prompt(record: dict[str, Any]) -> str:
    record_json = json.dumps(record, ensure_ascii=False, indent=2)

    return f"""You are an expert in NP-completeness and polynomial-time reductions.

Your task: critique the following structured reduction representation for rigor, correctness, and clarity.

Focus on:
1) Definitions: do input_format and yes_condition match standard definitions? Any ambiguity?
2) Construction: are reduction_steps complete, deterministic, and polynomial-time? Are parameters (like k) defined?
3) Correctness: are forward_proof and backward_proof logically sound and consistent with steps?
4) Consistency: do the problems/definitions/steps/proofs align (same objects, symbols, direction)?
5) Completeness: is any crucial argument missing (e.g., iff, polynomial bounds, gadget correctness)?
6) Style: plain text only; avoid LaTeX; steps should be atomic and unambiguous.

Do NOT rewrite the whole reduction here. Only critique and propose concrete fixes.

Return a JSON object matching this schema:
- summary: string
- major_issues: list of strings
- minor_issues: list of strings
- suggested_edits: list of strings
- confidence: number in [0, 1]

{_PLAIN_TEXT_RULES}

=== STRUCTURED REDUCTION (INPUT) ===
{record_json}
"""


def make_refine_prompt(record: dict[str, Any], critique: Critique) -> str:
    record_json = json.dumps(record, ensure_ascii=False, indent=2)
    critique_json = critique.model_dump_json(indent=2)

    source_problem = record.get("source_problem", "")
    target_problem = record.get("target_problem", "")

    return f"""You are an expert editor of NP-completeness reductions.

You will improve the structured reduction representation using the critique.

Hard constraints:
- Keep source_problem EXACTLY as: "{source_problem}"
- Keep target_problem EXACTLY as: "{target_problem}"
- Keep problem definition names consistent with those.
- Output MUST be valid JSON for the specified schema.
- Output MUST be plain text (no LaTeX, no backslashes).
- Do not invent a completely different reduction. Only fix rigor/clarity/consistency issues.

Quality bar:
- reduction_steps: atomic, imperative, and complete enough to implement.
- forward_proof/backward_proof: logically valid, references the construction, and addresses any parameters.
- Mention polynomial-time / size bounds if they are relevant and missing.

Return ONLY the JSON for a Reduction object with fields:
- source_problem
- target_problem
- source_definition: {{name, input_format, yes_condition}}
- target_definition: {{name, input_format, yes_condition}}
- reduction_steps: list of strings
- forward_proof
- backward_proof
- key_insight

{_PLAIN_TEXT_RULES}

=== ORIGINAL STRUCTURED REDUCTION ===
{record_json}

=== CRITIQUE (USE THIS) ===
{critique_json}
"""


# ============================================================================
# IO helpers
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


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Critique and refine structured reductions from karp_reductions.jsonl",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("data") / "processed" / "karp_reductions.jsonl"),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("data") / "processed" / "karp_reductions_refined.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N records (after reading)",
    )

    args = parser.parse_args()

    config.load()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    os.makedirs(output_path.parent, exist_ok=True)

    backend = Backend()

    processed = 0
    refined_ok = 0
    failed = 0

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for _, record in tqdm(iter_jsonl(input_path), desc="Refining", unit="reduction"):
            if args.skip and processed < args.skip:
                processed += 1
                continue

            if args.limit is not None and (processed - args.skip) >= args.limit:
                break

            processed += 1

            entry_key = record.get("entry_key", "")
            try:
                critique = backend.create(make_critique_prompt(record), Critique)
                refined = backend.create(make_refine_prompt(record, critique), Reduction)

                # Preserve original metadata but replace the reduction fields.
                out_record = dict(record)
                out_record.update(refined.model_dump())
                out_record["refinement_critique"] = critique.model_dump()

                out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                refined_ok += 1
            except Exception as e:
                failed += 1
                # Keep a minimal error record so the run is resumable/debuggable.
                err_record = {
                    "entry_key": entry_key,
                    "error": str(e),
                }
                out_fh.write(json.dumps(err_record, ensure_ascii=False) + "\n")

            # Cleanup between iterations to reduce memory growth.
            gc.collect()

    print("\nRefinement complete!")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Processed: {processed}")
    print(f"  Refined OK: {refined_ok}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
