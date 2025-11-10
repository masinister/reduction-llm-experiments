from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.runner import run_pipeline
from src.schemas import PARSE_SCHEMA
from src.vllm_structured import run_structured, StructuredCallError

try:  # prefer real model if available
    from src.inference import load_from_config
except Exception:  # pragma: no cover - inference dependencies optional
    load_from_config = None  # type: ignore


RIGOROUS_SYSTEM_PROMPTS = {
    "generate": (
        "You are a complexity-theory expert writing formal reduction proofs. "
        "Given a source problem, target problem, and a ground-truth reference proof, "
        "generate a candidate proof as a discrete sequence of numbered steps. "
        "Each step should be a complete, standalone sentence that captures exactly one "
        "claim, definition, or construction. The candidate should capture the same "
        "reduction as the reference but may vary in wording, level of detail, or "
        "explicitness of justifications. The candidate may have gaps that need refinement. "
        "Use PLAIN TEXT ONLY - no LaTeX formatting, no special commands. Write mathematical "
        "notation using simple ASCII (e.g., 'x_i' instead of '$x_i$', 'phi' instead of '$\\phi$'). "
        "Output ONLY the numbered steps, nothing else."
    ),
    "parse": (
        "You are a formal proof editor extracting numbered reduction steps. "
        "Segment the narrative so that each step captures exactly one claim, "
        "definition, or construction obligation, written as a complete sentence "
        "with hypotheses and conclusions made explicit. Respond in JSON only."
    ),
    "step": (
        "You are a skeptical complexity-theory referee. For the highlighted step "
        "decide whether the argument is logically sound *and* whether every required "
        "justification is stated. Treat appeals to intuition, unstated lemmas, or missing "
        "directional arguments as failures. Always cite the exact obligation that is unmet "
        "or explain why the reasoning is airtight. Respond strictly with a single JSON object "
        "matching the schema; do not emit arrays or prose. The JSON must include the keys "
        "step_index, step_text, classification, passes, confidence_score, reasons, and issues. "
        "For example: {\"step_index\":0,\"step_text\":\"...\",\"classification\":\"claim\",\"passes\":false,\"confidence_score\":0.0,\"reasons\":[\"detail\"],\"issues\":[{\"id\":\"issue-1\",\"title\":\"...\",\"description\":\"...\",\"severity\":\"high\",\"category\":\"soundness\"}]} "
        "Replace the placeholder values but keep the exact structure."
    ),
    "global": (
        "Audit the entire reduction for hidden assumptions, missing correctness directions, "
        "and gaps in the witnessing argument. Flag any place where equivalence, faithfulness, "
        "or resource bounds are not justified. Respond only with JSON matching the schema."
    ),
    "compare": (
        "Compare the candidate to the ground-truth proof with special attention to the rigor "
        "of justifications. Call out omissions where the ground truth includes explicit reasoning "
        "that the candidate lacks, even if the high-level outline matches. Report at most three "
        "differences and three issues by merging redundant findings. Respond only with JSON."
    ),
    "repair": (
        "You revise reduction steps to restore formal rigor. Wherever an issue flags missing "
        "support, expand the step with explicit hypotheses, direction checks, or gadget analysis. "
        "Proposed edits must be concise but complete. Return exactly one JSON object conforming to "
        "the schema; never wrap the object in an array or add commentary. The object must contain "
        "todo, edits, resolved_issue_ids, and optional notes. Every todo and edit must cite one or "
        "more linked_issue_ids drawn from the provided issue list; omit entries if you cannot tie "
        "them to specific issues."
    ),
}


class FallbackModel:
    """Minimal stub that forces heuristic fallbacks in run_structured."""

    def infer(self, **_: Any) -> Dict[str, Any]:
        return {"text": "{}", "raw_text": "{}", "tokens": 1}


@dataclass
class ReductionRow:
    entry_key: str
    reduction_full_text: str
    source_text: str
    target_text: str

    @classmethod
    def load_all(cls, path: Path) -> list["ReductionRow"]:
        rows: list[ReductionRow] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, record in enumerate(reader, start=1):
                if not record:
                    continue
                entry_key = record.get("entry_key") or f"row-{idx}"
                rows.append(
                    cls(
                        entry_key=entry_key,
                        reduction_full_text=record.get("reduction_full_text", ""),
                        source_text=record.get("source_text", ""),
                        target_text=record.get("target_text", ""),
                    )
                )
        return rows


def load_model(*, use_real: bool, toy: bool) -> Any:
    if use_real and load_from_config is not None:
        return load_from_config(toy=toy)
    return FallbackModel()


def _safe_run_id(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return sanitized or "reduction"


def generate_initial_candidate(
    model: Any,
    *,
    source_text: str,
    target_text: str,
    ground_truth: str,
    session_id: str,
    system_prompt: str,
) -> str:
    """Generate initial candidate proof from ground truth using structured outputs.
    
    This creates a starting point for iterative refinement by asking the model
    to produce a candidate reduction proof as a discrete sequence of steps.
    Uses vLLM structured outputs to ensure proper JSON formatting.
    """
    user_prompt = (
        f"Generate a candidate reduction proof for the following:\n\n"
        f"**Source Problem:**\n{source_text}\n\n"
        f"**Target Problem:**\n{target_text}\n\n"
        f"**Reference Proof (for guidance):**\n{ground_truth}\n\n"
        f"Instructions:\n"
        f"- Generate a JSON object with a 'steps' array containing 8-15 discrete steps\n"
        f"- Each step must be a complete, standalone sentence\n"
        f"- Each step should capture exactly ONE claim, definition, or construction\n"
        f"- Include: problem setup, reduction construction, correctness proof (both directions)\n"
        f"- Be detailed and explicit - avoid combining multiple ideas in one step\n"
        f"- The candidate should closely follow the reference but may vary in wording\n"
        f"- Use PLAIN TEXT ONLY - strip all LaTeX commands (\\textsc, \\text, $, etc.)\n"
        f"- Write math using simple ASCII: 'x_i' not '$x_i$', 'phi' not '$\\phi$'\n"
        f"- Write problem names simply: '1-in-3 SAT' not '\\textsc{{1-in-3 SAT}}'\n"
        f"- Return only valid JSON matching the schema\n"
    )
    
    # Try structured output first (only works with real models, not FallbackModel)
    if not isinstance(model, FallbackModel):
        try:
            result = run_structured(
                model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=PARSE_SCHEMA,
                session_id=session_id,
                temperature=0.3,  # Lower temperature for more faithful reproduction
                max_tokens=4096,  # Increased to allow more detailed steps
                retries=2,
            )
            steps = result.data.get("steps", [])
            if steps:
                # Convert steps list back to numbered text format for consistency
                candidate = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
                return candidate
        except StructuredCallError as e:
            print(f"Warning: Structured generation failed ({e}), falling back to ground truth.")
            return ground_truth
    
    # Fallback to ground truth for FallbackModel or if structured generation fails
    return ground_truth


def _print_run_report(result: Any) -> None:
    print("Final Summary:")
    for key, value in result.final_summary.items():
        print(f"  {key}: {value}")

    print("\nFinal Steps:")
    for idx, step in enumerate(result.steps):
        print(f"[{idx}] {step}")

    if result.issues:
        print("\nRemaining Issues:")
        for issue in result.issues:
            print(f"- ({issue.get('severity')}) {issue.get('title')}: {issue.get('description')}")
    else:
        print("\nNo outstanding issues detected.")

    if result.history:
        latest = result.history[-1]
        step_results = latest.get("step_results", [])
        problematic = [item for item in step_results if not item.get("passes", False)]
        if problematic:
            print("\nSteps flagged for insufficient justification:")
            for item in problematic:
                idx = item.get("step_index")
                reasons = item.get("reasons", []) or ["No reason provided"]
                print(f"- step {idx}: {reasons[0]}")
                for extra in reasons[1:]:
                    print(f"  - {extra}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Iteratively refine karp.csv reductions through candidate generation and refinement.")
    parser.add_argument("--csv", default="karp.csv", help="Path to karp-style dataset (default: karp.csv)")
    parser.add_argument("--output-dir", default="out/examples", help="Directory for pipeline artifacts")
    parser.add_argument("--use-real-model", action="store_true", help="Load model via config.ini instead of stub")
    parser.add_argument("--toy", action="store_true", help="When using a real model, load toy_model_id")
    parser.add_argument("--max-iters", type=int, default=3, help="Maximum refinement iterations")
    parser.add_argument("--limit", type=int, help="Only process the first N rows from the CSV")
    parser.add_argument("--skip-generation", action="store_true", help="Skip candidate generation, use ground truth directly")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging that pretty-prints every prompt and model response",
    )
    args = parser.parse_args(argv)
    
    # Configure logging based on debug flag
    import logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        print("Debug mode enabled - verbose logging active\n")

    rows = ReductionRow.load_all(Path(args.csv))
    if args.limit is not None and args.limit >= 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("The CSV does not contain any reduction rows.")

    model = load_model(use_real=args.use_real_model, toy=args.toy)

    total = len(rows)
    print(f"Loaded {total} reductions. Starting pipeline...\n")

    for idx, row in enumerate(rows, start=1):
        display_key = row.entry_key or f"row-{idx}"
        run_id = _safe_run_id(f"{display_key}-{idx:04d}")
        print(f"=== [{idx}/{total}] Reduction '{display_key}' ===\n")

        # Generate initial candidate from ground truth
        if args.skip_generation or isinstance(model, FallbackModel):
            print("Using ground truth as initial candidate (generation skipped).\n")
            candidate_blob = row.reduction_full_text
        else:
            print("Generating initial candidate from ground truth...\n")
            candidate_blob = generate_initial_candidate(
                model,
                source_text=row.source_text,
                target_text=row.target_text,
                ground_truth=row.reduction_full_text,
                session_id=f"{run_id}-generate",
                system_prompt=RIGOROUS_SYSTEM_PROMPTS["generate"],
            )
            print(f"Generated candidate ({len(candidate_blob)} chars).\n")

        result = run_pipeline(
            model,
            source_text=row.source_text,
            target_text=row.target_text,
            ground_truth=row.reduction_full_text,
            candidate_blob=candidate_blob,
            run_id=run_id,
            output_dir=args.output_dir,
            max_iters=args.max_iters,
            system_prompts=RIGOROUS_SYSTEM_PROMPTS,
        )

        _print_run_report(result)

        if idx != total:
            print("\n")



if __name__ == "__main__":
    main()
