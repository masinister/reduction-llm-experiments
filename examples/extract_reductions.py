"""Example: Extract structured Reduction models from the dataset.

This script processes each reduction from the karp dataset and transforms
it into a structured Pydantic model using MapReduceStrategy to handle
long reduction texts. Each chunk extracts partial information which is
then combined into the final Reduction model.
"""

import logging
import os
import argparse
import json
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.core_backend import CoreBackend
from src.strategies import SlidingWindowStrategy

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class ReductionMemory(BaseModel):
    """Memory state carried between chunks for context-aware extraction."""
    
    steps_so_far: List[str] = Field(
        default_factory=list,
        description="Reduction steps extracted so far"
    )
    forward_parts_so_far: List[str] = Field(
        default_factory=list,
        description="Forward proof parts extracted so far"
    )
    backward_parts_so_far: List[str] = Field(
        default_factory=list,
        description="Backward proof parts extracted so far"
    )
    insights_so_far: List[str] = Field(
        default_factory=list,
        description="Key insights extracted so far"
    )


class ReductionChunk(BaseModel):
    """Partial extraction from a chunk of reduction text."""
    
    reduction_steps: List[str] = Field(
        default_factory=list,
        description="Any reduction procedure steps found in this chunk"
    )
    forward_proof_parts: List[str] = Field(
        default_factory=list,
        description="Parts of the forward direction proof (source YES → target YES)"
    )
    backward_proof_parts: List[str] = Field(
        default_factory=list,
        description="Parts of the backward direction proof (target YES → source YES)"
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights or ideas found in this chunk"
    )


class Reduction(BaseModel):
    """Structured representation of a computational reduction proof."""
    
    source_problem: str = Field(
        description="The source problem being reduced from"
    )
    target_problem: str = Field(
        description="The target problem being reduced to"
    )
    reduction_steps: List[str] = Field(
        description="Step-by-step procedure for the reduction"
    )
    forward_proof: str = Field(
        description="Proof that YES instance of source implies YES instance of target"
    )
    backward_proof: str = Field(
        description="Proof that YES instance of target implies YES instance of source"
    )
    key_insight: Optional[str] = Field(
        default=None,
        description="The key idea or insight that makes this reduction work"
    )


# ============================================================================
# Prompt and Combine Functions
# ============================================================================

def make_chunk_prompt(text: str, memory: Optional[ReductionMemory] = None) -> str:
    """Create extraction prompt for a chunk of reduction text with memory context."""
    context = ""
    if memory and (memory.steps_so_far or memory.forward_parts_so_far or memory.backward_parts_so_far):
        context_parts = []
        if memory.steps_so_far:
            context_parts.append(f"Already found steps: {memory.steps_so_far[:3]}")
        if memory.forward_parts_so_far:
            context_parts.append(f"Already found forward proof: {memory.forward_parts_so_far[:2]}")
        if memory.backward_parts_so_far:
            context_parts.append(f"Already found backward proof: {memory.backward_parts_so_far[:2]}")
        context = "\n\nPreviously extracted (do not repeat):\n" + "\n".join(context_parts)
    
    return f"""Extract information from this mathematical reduction proof.

Instructions:
- reduction_steps: List the SPECIFIC algorithmic steps. Quote the actual procedure from the text.
- forward_proof_parts: Copy the actual argument for why source YES => target YES (after $\\implies$ or "implies").
- backward_proof_parts: Copy the actual argument for why target YES => source YES (after $\\impliedby$ or "conversely").
- key_insights: The main clever idea that makes the reduction work.

Example output format:
- reduction_steps: ["Replace each variable x with k new variables x^(1),...,x^(k)", "Add linking clauses to force equality"]
- forward_proof_parts: ["Assign all x_i^(k) the same value as x_i, which satisfies the linking clauses"]
- backward_proof_parts: ["The linking clauses force all copies to have equal values, so we can recover the original assignment"]
{context}

TEXT:
{text}

Extract the actual content from the text above. Be specific and quote relevant parts."""


def combine_chunks(chunks: List[ReductionChunk]) -> ReductionChunk:
    """Combine extracted chunks into a single ReductionChunk."""
    combined = ReductionChunk()
    for chunk in chunks:
        combined.reduction_steps.extend(chunk.reduction_steps)
        combined.forward_proof_parts.extend(chunk.forward_proof_parts)
        combined.backward_proof_parts.extend(chunk.backward_proof_parts)
        combined.key_insights.extend(chunk.key_insights)
    return combined


def update_memory(old_memory: Optional[ReductionMemory], chunk_output: ReductionChunk) -> ReductionMemory:
    """Update memory with new chunk output for sequential processing."""
    if old_memory is None:
        old_memory = ReductionMemory()
    
    return ReductionMemory(
        steps_so_far=old_memory.steps_so_far + chunk_output.reduction_steps,
        forward_parts_so_far=old_memory.forward_parts_so_far + chunk_output.forward_proof_parts,
        backward_parts_so_far=old_memory.backward_parts_so_far + chunk_output.backward_proof_parts,
        insights_so_far=old_memory.insights_so_far + chunk_output.key_insights,
    )


def chunk_to_reduction(chunk: ReductionChunk, source: str, target: str) -> Reduction:
    """Convert combined chunk data to final Reduction model."""
    return Reduction(
        source_problem=source,
        target_problem=target,
        reduction_steps=chunk.reduction_steps,
        forward_proof=" ".join(chunk.forward_proof_parts) if chunk.forward_proof_parts else "Not extracted",
        backward_proof=" ".join(chunk.backward_proof_parts) if chunk.backward_proof_parts else "Not extracted",
        key_insight=chunk.key_insights[0] if chunk.key_insights else None,
    )


# ============================================================================
# Processing
# ============================================================================

def extract_reduction(
    strategy: SlidingWindowStrategy,
    text: str,
    source: str,
    target: str,
) -> Optional[Reduction]:
    """Extract structured reduction from text using SlidingWindow (sequential with memory)."""
    if not isinstance(text, str) or not text.strip():
        return None
    
    try:
        # Use SlidingWindow to process chunks sequentially with memory
        chunk_result = strategy.process(
            text=text,
            prompt_fn=make_chunk_prompt,
            combine_fn=combine_chunks,
            update_memory_fn=update_memory,
        )
        
        if chunk_result:
            return chunk_to_reduction(chunk_result, source, target)
        return None
    except Exception as e:
        logger.exception("Extraction failed: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured Reduction models from dataset."
    )
    parser.add_argument("--limit", type=int, help="Limit rows for testing")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/raw/karp.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/karp_reductions.jsonl",
        help="Output JSONL path"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    if args.limit:
        print(f"Limiting to first {args.limit} rows.")
        df = df.head(args.limit)
    
    print("Initializing LLM backend...")
    backend = CoreBackend()
    strategy = SlidingWindowStrategy(
        backend=backend,
        output_model=ReductionChunk,
        memory_model=ReductionMemory,
    )
    
    text_col = 'reduction_full_text'
    results = []
    failed = 0
    
    print(f"\nProcessing {len(df)} reductions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        entry_key = row.get('entry_key', f'row_{idx}')
        source = row.get('source', 'Unknown')
        target = row.get('target', 'Unknown')
        text = row.get(text_col, '')
        
        reduction = extract_reduction(strategy, text, source, target)
        
        if reduction:
            result = {
                'entry_key': entry_key,
                'difficulty': row.get('difficulty'),
                **reduction.model_dump()
            }
            results.append(result)
        else:
            failed += 1
            logger.warning("Failed to extract: %s", entry_key)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nExtraction complete!")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
