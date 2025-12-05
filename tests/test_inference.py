"""Test key point extraction using each strategy.

Tests:
  1. DirectStrategy - Single call, no chunking
  2. MapReduceStrategy - Map over chunks independently, then reduce
  3. SlidingWindowStrategy - Chunks with memory carried between windows
"""

from typing import List, Optional
from pydantic import BaseModel
from src.core_backend import CoreBackend
from src.strategies import (
    DirectStrategy,
    MapReduceStrategy,
    SlidingWindowStrategy,
)
import os


class KeyPoints(BaseModel):
    """Key points extracted from text."""
    points: List[str]


class Memory(BaseModel):
    """Memory for sliding window - summary of previous points."""
    summary: str


def keypoints_prompt(text: str) -> str:
    return f"Extract key points from the following text:\n\n{text}"


def keypoints_prompt_with_memory(text: str, memory: Optional[Memory]) -> str:
    if memory:
        return f"Previous context: {memory.summary}\n\nExtract key points from:\n\n{text}"
    return f"Extract key points from:\n\n{text}"


def combine_keypoints(outputs: List[KeyPoints]) -> KeyPoints:
    all_points = []
    for out in outputs:
        all_points.extend(out.points)
    return KeyPoints(points=all_points)


def update_memory(old: Optional[Memory], output: KeyPoints) -> Memory:
    summary = "; ".join(output.points[:3])
    if old:
        return Memory(summary=f"{old.summary} | {summary}")
    return Memory(summary=summary)


def print_result(name: str, result: Optional[KeyPoints]) -> None:
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
    """Test key point extraction with each strategy."""
    # Load test text
    file_path = os.path.join(os.path.dirname(__file__), "example_reduction.txt")
    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    core = CoreBackend()

    try:
        # 1. Direct Strategy
        direct = DirectStrategy(backend=core, output_model=KeyPoints)
        result = direct.process(text=text, prompt_fn=keypoints_prompt)
        print_result("DIRECT STRATEGY", result)

        # 2. MapReduce Strategy (uses config defaults for chunk_size/overlap)
        mapreduce = MapReduceStrategy(
            backend=core,
            output_model=KeyPoints,
        )
        result = mapreduce.process(
            text=text,
            prompt_fn=keypoints_prompt,
            combine_fn=combine_keypoints,
        )
        print_result("MAP-REDUCE STRATEGY", result)

        # 3. Sliding Window Strategy (uses config defaults for chunk_size/overlap)
        sliding = SlidingWindowStrategy(
            backend=core,
            output_model=KeyPoints,
            memory_model=Memory,
        )
        result = sliding.process(
            text=text,
            prompt_fn=keypoints_prompt_with_memory,
            combine_fn=combine_keypoints,
            update_memory_fn=update_memory,
        )
        print_result("SLIDING WINDOW STRATEGY", result)

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 60 + "\n")

    finally:
        if hasattr(core, 'llm') and core.llm is not None:
            del core.llm


if __name__ == "__main__":
    main()
