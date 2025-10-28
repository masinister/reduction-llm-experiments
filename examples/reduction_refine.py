from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.runner import run_pipeline

try:  # prefer real model if available
    from src.inference import load_from_config
except Exception:  # pragma: no cover - inference dependencies optional
    load_from_config = None  # type: ignore


RIGOROUS_SYSTEM_PROMPTS = {
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
        "For example: {\"step_index\":0,\"step_text\":\"...\",\"classification\":\"claim\",\"passes\":false,\"confidence_score\":0.0,\"reasons\":[""detail""],\"issues\":[{\"id\":\"issue-1\",\"title\":\"...\",\"description\":\"...\",\"severity\":\"high\",\"category\":\"soundness\"}]} "
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
        "that the candidate lacks, even if the high-level outline matches. Respond only with JSON."
    ),
    "repair": (
        "You revise reduction steps to restore formal rigor. Wherever an issue flags missing "
        "support, expand the step with explicit hypotheses, direction checks, or gadget analysis. "
        "Proposed edits must be concise but complete. Return exactly one JSON object conforming to "
        "the schema; never wrap the object in an array or add commentary. The object must contain "
        "todo, edits, resolved_issue_ids, and optional notes."
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
    parser = argparse.ArgumentParser(description="Iteratively refine the first karp.csv reduction.")
    parser.add_argument("--csv", default="karp.csv", help="Path to karp-style dataset (default: karp.csv)")
    parser.add_argument("--output-dir", default="out/examples", help="Directory for pipeline artifacts")
    parser.add_argument("--use-real-model", action="store_true", help="Load model via config.ini instead of stub")
    parser.add_argument("--toy", action="store_true", help="When using a real model, load toy_model_id")
    parser.add_argument("--max-iters", type=int, default=3, help="Maximum refinement iterations")
    parser.add_argument("--dry-run", action="store_true", help="Skip applying edits (planning only)")
    parser.add_argument("--limit", type=int, help="Only process the first N rows from the CSV")
    args = parser.parse_args(argv)

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

        result = run_pipeline(
            model,
            source_text=row.source_text,
            target_text=row.target_text,
            ground_truth=row.reduction_full_text,
            candidate_blob=row.reduction_full_text,
            run_id=run_id,
            output_dir=args.output_dir,
            max_iters=args.max_iters,
            dry_run=args.dry_run,
            system_prompts=RIGOROUS_SYSTEM_PROMPTS,
        )

        _print_run_report(result)

        if idx != total:
            print("\n")



if __name__ == "__main__":
    main()
