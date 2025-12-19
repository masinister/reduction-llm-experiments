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


class Validation(BaseModel):
    """Validation result comparing extraction against ground truth."""
    
    agrees: bool = Field(description="True if extraction faithfully represents the ground truth reduction")
    major_errors: list[str] = Field(description="Critical errors where extraction contradicts the ground truth (empty if none)")
    minor_errors: list[str] = Field(description="Minor inaccuracies or omissions (empty if none)")
    explanation: str = Field(description="Brief explanation of the validation result")


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


def make_validation_prompt(extraction: Reduction, raw_text: str) -> str:
    """Create prompt to validate extraction for correctness."""
    extraction_json = extraction.model_dump_json(indent=2)
    return f"""Review this structured reduction for correctness and agreement with the raw text.

Be CONSERVATIVE:
- Only flag major_errors for genuine correctness problems or significant misrepresentations
- Minor wording differences, equivalent formulations, or small clarifications are acceptable
- If the core construction and proofs are sound, set agrees=True

Output must be plain text only—no LaTeX or backslashes.

Return a JSON Validation object:
- agrees: true if extraction is correct and faithful, false only if major errors exist
- major_errors: list of genuine correctness problems (can be empty)
- minor_errors: list of minor issues (can be empty)
- explanation: brief summary

=== STRUCTURED EXTRACTION ===
{extraction_json}

=== RAW REDUCTION TEXT ===
{raw_text}
"""


def make_repair_prompt(
    extraction: Reduction,
    validation: Validation,
    raw_text: str,
    source_name: str,
    target_name: str,
) -> str:
    """Create prompt to repair extraction based on validation errors."""
    extraction_json = extraction.model_dump_json(indent=2)
    errors_json = json.dumps({
        "major_errors": validation.major_errors,
        "minor_errors": validation.minor_errors,
        "explanation": validation.explanation,
    }, indent=2)
    
    return f"""Fix the extraction to address the validation errors.

Guidelines:
- Fix the flagged correctness issues
- Preserve parts that are already correct
- Keep the reduction faithful to the raw text
- You may add minor clarifications to improve rigor

Constraints:
- Keep source_problem exactly as "{source_name}"
- Keep target_problem exactly as "{target_name}"

Output must be plain text only—no LaTeX or backslashes.

Return a corrected JSON Reduction object.

=== CURRENT EXTRACTION ===
{extraction_json}

=== VALIDATION ERRORS ===
{errors_json}

=== RAW REDUCTION TEXT ===
{raw_text}
"""


# ============================================================================
# Validation & Repair
# ============================================================================

MAX_REPAIR_ATTEMPTS = 3


def validate_and_repair(
    backend: Backend,
    extraction: Reduction,
    raw_text: str,
    source_name: str,
    target_name: str,
) -> tuple[Reduction, list[Validation]]:
    """Validate extraction for correctness and repair if needed.
    
    Checks the extraction for correctness, using the raw reduction text
    as a reference. Allows minor improvements if they fix genuine issues.
    
    Returns:
        Tuple of (final_reduction, list_of_validations)
    """
    validations = []
    current = extraction
    
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        validation = backend.create(
            make_validation_prompt(current, raw_text),
            Validation,
            temperature=0.0,
        )
        validations.append(validation)
        
        if validation.agrees or not validation.major_errors:
            # No major errors, we're done
            break
        
        if attempt < MAX_REPAIR_ATTEMPTS:
            print(f"  [Repair attempt {attempt + 1}/{MAX_REPAIR_ATTEMPTS}] Fixing {len(validation.major_errors)} major error(s)...")
            current = backend.create(
                make_repair_prompt(current, validation, raw_text, source_name, target_name),
                Reduction,
                temperature=0.1,
            )
    
    return current, validations


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
    validate: bool = True,
) -> tuple[Reduction | None, list[Validation]]:
    """Extract structured reduction from text with problem definitions.
    
    Args:
        backend: LLM backend
        text: Raw reduction text
        source_name: Source problem name
        source_def: Source problem definition
        target_name: Target problem name  
        target_def: Target problem definition
        validate: Whether to validate against ground truth and repair
        
    Returns:
        Tuple of (reduction, list_of_validations). Validations empty if validate=False.
    """
    if not text or not text.strip():
        return None, []
    
    extract_prompt = make_extract_prompt(source_name, source_def, target_name, target_def)
    
    try:
        reduction = sequential_extract(
            backend=backend,
            text=text,
            response_model=Reduction,
            extract_prompt=extract_prompt,
            combine_prompt=combine_prompt,
        )
        
        if validate and reduction:
            reduction, validations = validate_and_repair(
                backend, reduction, text, source_name, target_name
            )
            return reduction, validations
        
        return reduction, []
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None, []


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
        
        reduction, validations = extract_reduction(
            backend, text, source_name, source_def, target_name, target_def
        )
        
        if reduction:
            result = {
                'entry_key': entry_key,
                'difficulty': row.get('difficulty'),
                'reduction': reduction.model_dump(),
            }
            # Include validation summary if available
            if validations:
                final_validation = validations[-1]
                result['validation'] = {
                    'agrees': final_validation.agrees,
                    'repair_attempts': len(validations) - 1,
                    'final_major_errors': final_validation.major_errors,
                    'final_minor_errors': final_validation.minor_errors,
                }
            results.append(result)
            print(f"[{entry_key}] Extraction successful:")
            if validations:
                v = validations[-1]
                status = "VALID" if v.agrees else f"ISSUES ({len(v.major_errors)} major)"
                print(f"  Validation: {status} after {len(validations)-1} repair(s)")
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
