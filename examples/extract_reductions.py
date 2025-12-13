"""Example: Extract structured Reduction models from the dataset.

This script processes each reduction from the karp dataset using
the sequential chunking strategy when needed.
"""

import argparse
import gc
import json
import os

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from src import config
from src.core_backend import Backend
from src.strategies import sequential_extract

config.load()


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
    reduction_steps: list[str] = Field(description="Step-by-step construction procedure transforming a source instance to a target instance")
    forward_proof: str = Field(description="Proof that YES instance of source implies YES instance of target")
    backward_proof: str = Field(description="Proof that YES instance of target implies YES instance of source")
    key_insight: str = Field(description="The key idea or intuition that makes this reduction work")


# ============================================================================
# Prompts
# ============================================================================

def make_extract_prompt(source_name: str, source_def: str, target_name: str, target_def: str):
    """Create an extraction prompt function with source/target definitions baked in."""
    
    def extract_prompt(text: str, previous: str | None) -> str:
        context = ""
        if previous:
            context = f"\n\n[CONTEXT FROM PREVIOUS CHUNK - continue from here, do not repeat:]\n{previous[:1000]}..."
        
        return f"""Extract a structured representation of this computational reduction.

*** CRITICAL: ALL OUTPUT MUST BE PLAIN TEXT - NO LATEX ***
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
- \\textsc{{Name}} -> "NAME"
- \\Big, \\big, \\left, \\right -> remove entirely
- Remove all \\begin{{...}}, \\end{{...}}, \\item, etc.

=== PROBLEM DEFINITIONS (for reference) ===

SOURCE PROBLEM: {source_name}
{source_def}

TARGET PROBLEM: {target_name}
{target_def}

=== REDUCTION PROOF TEXT ===
{text}

=== FIELDS TO EXTRACT ===

1. source_problem: "{source_name}"

2. target_problem: "{target_name}"

3. source_definition: Parse from the source problem definition above:
   - name: Problem name
   - input_format: What is given as input
   - yes_condition: Condition for YES output

4. target_definition: Parse from the target problem definition above:
   - name: Problem name
   - input_format: What is given as input
   - yes_condition: Condition for YES output

5. reduction_steps: List of atomic construction steps from the reduction proof.
   Each step should be self-contained.

6. forward_proof: From the proof, extract the argument that:
   source is YES => constructed target is YES

7. backward_proof: From the proof, extract the argument that:
   constructed target is YES => source is YES

8. key_insight: The central idea that makes this reduction work.
{context}

Remember: Output ONLY plain text. No backslashes, no LaTeX commands."""
    
    return extract_prompt


def combine_prompt(partials: list[str]) -> str:
    """Create prompt to combine partial extractions."""
    joined = "\n---\n".join(partials)
    return f"""Combine these partial extractions from chunks of a reduction proof into a single coherent Reduction.

Merge rules:
- Use the most complete version of each definition
- Merge reduction_steps into complete sequence, maintain order
- Pick the best key_insight

PARTIAL EXTRACTIONS:
{joined}"""


# ============================================================================
# Processing
# ============================================================================

def extract_reduction(
    backend: Backend,
    text: str,
    source_name: str,
    source_def: str,
    target_name: str,
    target_def: str,
) -> Reduction:
    """Extract structured reduction from text with problem definitions."""
    if not text or not text.strip():
        return None
    
    extract_prompt = make_extract_prompt(source_name, source_def, target_name, target_def)
    
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
        source_name = row.get('source', '')
        source_def = row.get('source_text', '')
        target_name = row.get('target', '')
        target_def = row.get('target_text', '')
        
        print(f"\n{'#'*70}")
        print(f"# [{idx+1}/{len(df)}] Processing: {entry_key}")
        print(f"# {source_name} -> {target_name}")
        print(f"# Input text: {len(text):,} chars (~{len(text)//4:,} tokens)")
        print(f"{'#'*70}")
        
        reduction = extract_reduction(
            backend, text, source_name, source_def, target_name, target_def
        )
        
        if reduction:
            result = {
                'entry_key': entry_key,
                'difficulty': row.get('difficulty'),
                **reduction.model_dump()
            }
            results.append(result)
            print(f"[{entry_key}] Extraction successful:")
            print(json.dumps(result, indent=2))
        else:
            print(f"[{entry_key}] Extraction failed")
            failed += 1
        
        # Explicit cleanup to prevent resource leak
        del reduction
        gc.collect()
    
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
