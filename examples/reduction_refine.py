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
    def from_csv(cls, path: Path) -> "ReductionRow":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first = next(reader)
        return cls(
            entry_key=first.get("entry_key", "unknown"),
            reduction_full_text=first.get("reduction_full_text", ""),
            source_text=first.get("source_text", ""),
            target_text=first.get("target_text", ""),
        )


def load_model(*, use_real: bool, toy: bool) -> Any:
    if use_real and load_from_config is not None:
        return load_from_config(toy=toy)
    return FallbackModel()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Iteratively refine the first karp.csv reduction.")
    parser.add_argument("--csv", default="karp.csv", help="Path to karp-style dataset (default: karp.csv)")
    parser.add_argument("--output-dir", default="out/examples", help="Directory for pipeline artifacts")
    parser.add_argument("--use-real-model", action="store_true", help="Load model via config.ini instead of stub")
    parser.add_argument("--toy", action="store_true", help="When using a real model, load toy_model_id")
    parser.add_argument("--max-iters", type=int, default=3, help="Maximum refinement iterations")
    parser.add_argument("--dry-run", action="store_true", help="Skip applying edits (planning only)")
    args = parser.parse_args(argv)

    row = ReductionRow.from_csv(Path(args.csv))
    model = load_model(use_real=args.use_real_model, toy=args.toy)

    print(f"Loaded reduction '{row.entry_key}'. Starting pipeline...\n")

    result = run_pipeline(
        model,
        source_text=row.source_text,
        target_text=row.target_text,
        ground_truth=row.reduction_full_text,
        candidate_blob=row.reduction_full_text,
        run_id=row.entry_key,
        output_dir=args.output_dir,
        max_iters=args.max_iters,
        dry_run=args.dry_run,
    )

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


if __name__ == "__main__":
    main()
