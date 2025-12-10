"""Test structured extraction with the Backend and strategies."""

import os
from pydantic import BaseModel

from src import config
from src.core_backend import Backend
from src.strategies import sequential_extract, hierarchical_extract

config.load()


class KeyPoints(BaseModel):
    """Key points extracted from text."""
    points: list[str]


def extract_prompt(text: str, previous: str | None = None) -> str:
    """Prompt for extracting key points."""
    context = f"\n\nPrevious extraction (continue, don't repeat):\n{previous}" if previous else ""
    return f"Extract key points from this text:{context}\n\n{text}"


def combine_prompt(partials: list[str]) -> str:
    """Prompt for combining partial extractions."""
    joined = "\n---\n".join(partials)
    return f"Combine these partial key point extractions into one:\n\n{joined}"


def simple_extract_prompt(text: str) -> str:
    """Simple prompt without memory context."""
    return f"Extract key points from this text:\n\n{text}"


def print_result(name: str, result: KeyPoints | None) -> None:
    """Pretty print extraction result."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print("=" * 60)
    if result:
        print(f"✅ Extracted {len(result.points)} key points:")
        for i, point in enumerate(result.points[:5], 1):
            print(f"   {i}. {point[:80]}...")
        if len(result.points) > 5:
            print(f"   ... and {len(result.points) - 5} more")
    else:
        print("❌ Failed to extract key points")


def main() -> None:
    """Test key point extraction with backend and strategies."""
    # Load test text
    file_path = os.path.join(os.path.dirname(__file__), "example_reduction.txt")
    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    backend = Backend()

    # 1. Direct extraction (no chunking needed for short text)
    print("\n--- Testing direct extraction ---")
    result = backend.create(simple_extract_prompt(text[:2000]), KeyPoints)
    print_result("DIRECT EXTRACTION", result)

    # 2. Sequential extraction (with memory between chunks)
    print("\n--- Testing sequential extraction ---")
    result = sequential_extract(
        backend=backend,
        text=text,
        response_model=KeyPoints,
        extract_prompt=extract_prompt,
        combine_prompt=combine_prompt,
    )
    print_result("SEQUENTIAL EXTRACTION", result)

    # 3. Hierarchical extraction (independent chunks)
    print("\n--- Testing hierarchical extraction ---")
    result = hierarchical_extract(
        backend=backend,
        text=text,
        response_model=KeyPoints,
        extract_prompt=simple_extract_prompt,
        combine_prompt=combine_prompt,
    )
    print_result("HIERARCHICAL EXTRACTION", result)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
