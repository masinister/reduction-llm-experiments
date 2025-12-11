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

class ProblemDefinition(BaseModel):
    """Definition of a computational problem."""
    
    name: str = Field(description="Name of the problem")
    input_format: str = Field(description="Description of what constitutes a valid input instance")
    yes_condition: str = Field(description="The condition that makes an instance a YES instance")


class GadgetProperty(BaseModel):
    """A property of a gadget with its proof."""
    
    statement: str = Field(description="The property statement")
    proof: str = Field(description="Proof that the gadget satisfies this property")


class Gadget(BaseModel):
    """A gadget used in a reduction construction."""
    
    name: str = Field(description="Name or identifier for the gadget (e.g., 'variable gadget', 'clause gadget')")
    description: str = Field(description="What the gadget represents or encodes")
    construction: str = Field(description="How the gadget is constructed (vertices, edges, elements, etc.)")
    properties: Optional[list[GadgetProperty]] = Field(default=None, description="Properties the gadget satisfies, with proofs")


class Lemma(BaseModel):
    """An external technical result used in the reduction (e.g., CRT, algebraic identities)."""
    
    name: Optional[str] = Field(default=None, description="Name of the lemma/theorem if well-known (e.g., 'Chinese Remainder Theorem')")
    statement: str = Field(description="The statement of the lemma")
    usage: str = Field(description="How this lemma is applied in the reduction")


class Reduction(BaseModel):
    """Structured representation of a computational reduction proof."""
    
    # Problem definitions
    source_problem: str = Field(description="The source problem being reduced from")
    target_problem: str = Field(description="The target problem being reduced to")
    source_definition: Optional[ProblemDefinition] = Field(default=None, description="Formal definition of the source problem")
    target_definition: Optional[ProblemDefinition] = Field(default=None, description="Formal definition of the target problem")
    
    # Auxiliary content
    definitions: Optional[list[str]] = Field(default=None, description="Any auxiliary definitions introduced (e.g., terminology, notation)")
    lemmas: Optional[list[Lemma]] = Field(default=None, description="Technical lemmas required for the proof")
    gadgets: Optional[list[Gadget]] = Field(default=None, description="Gadgets used in the reduction construction")
    
    # Core reduction content
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

EXTRACTION GUIDELINES:

1. PROBLEM DEFINITIONS: Extract the formal definition of both problems.
   - input_format: What is given as input (e.g., "a graph G and integer k")
   - yes_condition: When is the answer YES (e.g., "G has a clique of size k")

2. GADGETS: If the reduction uses gadgets, extract each with:
   - name, description, construction
   - properties: Each property needs a rigorous proof

3. LEMMAS: If necessary, extract technical lemmas with full proofs (not sketches).

4. CORE CONTENT:
   - reduction_steps: Atomic construction steps
   - forward_proof / backward_proof: Full proofs
   - key_insight: The core idea
{context}

TEXT:
{text}

IMPORTANT: 
- Convert ALL LaTeX to plain text.
- Gadget properties require rigorous proofs.
- Only include definitions/lemmas if explicitly present."""


def combine_prompt(partials: list[str]) -> str:
    """Create prompt to combine partial extractions."""
    joined = "\n---\n".join(partials)
    return f"""You have multiple partial extractions from different chunks of a reduction proof.
Combine them into a single coherent Reduction.

MERGING RULES:
- source_definition / target_definition: Use the most complete version
- definitions: Merge lists, remove duplicates
- gadgets: Merge gadget lists; combine info for same gadget
- lemmas: Merge lists, remove duplicates
- reduction_steps: Merge into complete sequence, maintain order
- forward_proof / backward_proof: Combine into coherent proofs
- key_insight: Pick the best version

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
            print(f"[{entry_key}] Extraction successful:")
            print(json.dumps(result, indent=2))
        else:
            print(f"[{entry_key}] Extraction failed")
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
