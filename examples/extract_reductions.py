"""Example: Extract structured Reduction models from the dataset.

This script processes each reduction from the karp dataset using
the sequential chunking strategy when needed.
"""

import argparse
import json
import os
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from src import config
from src.core_backend import Backend
from src.strategies import sequential_extract
from src.utils import model_field_descriptions

config.load()


# ============================================================================
# Pydantic Models
# ============================================================================

class Reduction(BaseModel):
    """Structured representation of a computational reduction proof."""
    
    source_problem: str = Field(description="The source problem being reduced from")
    target_problem: str = Field(description="The target problem being reduced to")
    reduction_steps: list[str] = Field(description="Step-by-step procedure for the reduction")
    forward_proof: str = Field(description="Proof that YES instance of source implies YES instance of target")
    backward_proof: str = Field(description="Proof that YES instance of target implies YES instance of source")
    key_insight: Optional[str] = Field(default=None, description="The key idea that makes this reduction work")


# ============================================================================
# Prompts
# ============================================================================

def extract_prompt(text: str, previous: str | None) -> str:
    """Create extraction prompt, optionally with context from previous chunk."""
    context = ""
    if previous:
        context = f"\n\nPreviously extracted (continue from here, do not repeat):\n{previous[:500]}..."
    
    field_descriptions = model_field_descriptions(Reduction)
    
    return f"""Extract information from this mathematical reduction proof.

Instructions:
{field_descriptions}
{context}

TEXT:
{text}

Extract the actual content from the text above. IMPORTANT: 
- Steps should be atomic and not depend on future steps.
- Convert ALL LaTeX commands to plain text."""


def combine_prompt(partials: list[str]) -> str:
    """Create prompt to combine partial extractions."""
    joined = "\n---\n".join(partials)
    return f"""You have multiple partial extractions from different chunks of a reduction proof.
Combine them into a single coherent Reduction.

- Merge reduction_steps into a complete sequence (remove duplicates)
- Combine forward_proof parts into one coherent proof
- Combine backward_proof parts into one coherent proof
- Pick the best key_insight

PARTIAL EXTRACTIONS:
{joined}

Produce the final combined Reduction."""


# ============================================================================
# Processing
# ============================================================================

def extract_reduction(backend: Backend, text: str) -> Optional[Reduction]:
    """Extract structured reduction from text."""
    if not text or not text.strip():
        return None
    
    try:
        return sequential_extract(
            backend=backend,
            text=text,
            response_model=Reduction,
            extract_prompt=extract_prompt,
            combine_prompt=combine_prompt,
        )
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract structured Reduction models from dataset.")
    parser.add_argument("--limit", type=int, help="Limit rows for testing")
    parser.add_argument("--input", type=str, default="data/raw/karp.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="data/processed/karp_reductions.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    if args.limit:
        print(f"Limiting to first {args.limit} rows.")
        df = df.head(args.limit)
    
    print("Initializing backend...")
    backend = Backend()
    
    results = []
    failed = 0
    
    print(f"\nProcessing {len(df)} reductions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        entry_key = row.get('entry_key', f'row_{idx}')
        text = row.get('reduction_full_text', '')
        
        reduction = extract_reduction(backend, text)
        
        if reduction:
            result = {
                'entry_key': entry_key,
                'difficulty': row.get('difficulty'),
                **reduction.model_dump()
            }
            results.append(result)
        else:
            failed += 1
    
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
