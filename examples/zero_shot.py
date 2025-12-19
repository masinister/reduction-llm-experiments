from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field
from tqdm import tqdm

from src import config
from src.core_backend import Backend


config.load()


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


# ============================================================================
# Prompt
# ============================================================================


def _format_definition(defn: ProblemDefinition) -> str:
    return (
        f"Name: {defn.name}\n"
        f"Input format: {defn.input_format}\n"
        f"YES condition: {defn.yes_condition}\n"
    )


def make_zero_shot_prompt(
    source_name: str,
    source_def: ProblemDefinition,
    target_name: str,
    target_def: ProblemDefinition,
) -> str:
    source_def_text = _format_definition(source_def)
    target_def_text = _format_definition(target_def)
    return f"""You are an expert in NP-completeness and polynomial-time many-one reductions.

Your task: produce a structured description of a polynomial-time reduction from the SOURCE problem to the TARGET problem.

Important constraints:
- You are NOT given the ground-truth reduction proof text.
- Use ONLY the problem definitions below plus your general knowledge.
- Output MUST be plain text content inside fields (no LaTeX, no backslashes).
- If the reduction uses parameters (like k), define them clearly.
- Make the construction deterministic and polynomial-time.

=== PROBLEM DEFINITIONS (given) ===

SOURCE PROBLEM: {source_name}
{source_def_text}

TARGET PROBLEM: {target_name}
{target_def_text}

=== OUTPUT FIELDS ===

Fill a Reduction object with:
1) source_problem: "{source_name}"
2) target_problem: "{target_name}"
3) source_definition: parse from the source definition above
4) target_definition: parse from the target definition above
5) reduction_steps: an ordered list of atomic steps mapping a source instance to a target instance
6) forward_proof: argue source YES => constructed target YES
7) backward_proof: argue constructed target YES => source YES
8) key_insight: the central idea that makes the reduction work

Remember: the field text must be plain English (no LaTeX)."""


# ============================================================================
# Processing
# ============================================================================


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e


def zero_shot_reduction(
    backend: Backend,
    source_name: str,
    source_def: ProblemDefinition,
    target_name: str,
    target_def: ProblemDefinition,
) -> Reduction | None:
    prompt = make_zero_shot_prompt(source_name, source_def, target_name, target_def)
    try:
        return backend.create(prompt, Reduction)
    except Exception as e:
        print(f"Zero-shot generation failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot generate Reduction models from source/target definitions (no proof text)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N rows")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/karp_reductions.jsonl",
        help="Input JSONL path (expects extracted reductions with plain-text definitions)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/zero_shot_reductions.jsonl",
        help="Output JSONL path",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print("Initializing backend...")
    backend = Backend()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    generated_ok = 0
    failed = 0
    processed = 0

    with open(args.output, "w", encoding="utf-8") as out_fh:
        print(f"\nProcessing zero-shot generations from: {input_path}")
        for _, record in tqdm(iter_jsonl(input_path), desc="Zero-shot", unit="reduction"):
            if args.skip and processed < args.skip:
                processed += 1
                continue

            if args.limit is not None and (processed - args.skip) >= args.limit:
                break

            processed += 1

            entry_key = record.get("entry_key", "")
            difficulty = record.get("difficulty")

            reduction_in = record.get("reduction")
            if not isinstance(reduction_in, dict):
                failed += 1
                out_fh.write(json.dumps({"entry_key": entry_key, "error": "missing_reduction_object"}) + "\n")
                gc.collect()
                continue
            
            source_name = reduction_in.get("source_problem", "")
            target_name = reduction_in.get("target_problem", "")

            try:
                source_def = ProblemDefinition.model_validate(reduction_in.get("source_definition", {}))
                target_def = ProblemDefinition.model_validate(reduction_in.get("target_definition", {}))
            except Exception as e:
                failed += 1
                out_fh.write(
                    json.dumps({"entry_key": entry_key, "error": f"invalid_definitions: {e}"}, ensure_ascii=False)
                    + "\n"
                )
                gc.collect()
                continue

            print(f"\n{'#' * 70}")
            print(f"# Zero-shot: {entry_key}")
            print(f"# {source_name} -> {target_name}")
            print(f"{'#' * 70}")

            reduction = zero_shot_reduction(
                backend=backend,
                source_name=source_name,
                source_def=source_def,
                target_name=target_name,
                target_def=target_def,
            )

            if reduction is None:
                failed += 1
                out_fh.write(json.dumps({"entry_key": entry_key, "error": "generation_failed"}) + "\n")
                gc.collect()
                continue

            out_record = {
                "entry_key": entry_key,
                "difficulty": difficulty,
                "reduction": reduction.model_dump(),
            }

            out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            generated_ok += 1

            # Cleanup between iterations to reduce memory growth.
            del reduction
            gc.collect()

    print("\nZero-shot generation complete!")
    print(f"  Successful: {generated_ok}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
