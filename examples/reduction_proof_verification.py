#!/usr/bin/env python3
"""Granular proof verification through step-by-step unfolding.

This script unfolds reduction proofs into granular, verifiable steps and
validates each step individually. The process:
  1. Unfold: Extract atomic proof steps from the reduction
  2. Prove: Generate rigorous proofs for each individual step
  3. Verify: Check each proof for logical validity and completeness

The goal is to validate reductions by ensuring every granular step is sound,
rather than accepting the reduction as a monolithic whole.

The model determines the appropriate granularity and can use sub-steps
(e.g., 1, 2a, 2b, 3) to break down complex parts while combining simpler ones.

Input CSV columns (required):
  - source_text
  - target_text
  - reduction_full_text

Output CSV columns (added):
  - proof_steps_json      # List of unfolded proof steps
  - step_proofs_json      # List of {step, proof, verification} dicts
  - correctness_score     # Fraction of steps that verified successfully
  - num_steps
  - num_verified

Usage:
  python examples/reduction_proof_verification.py \
    --toy \
    --input_csv karp.csv \
    --output_csv verified_proofs.csv
"""

from __future__ import annotations

import argparse
import os
import json
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

from src.inference import Model, load_from_config


@dataclass
class ProofStep:
    """A single granular step in an unfolded proof."""
    index: int
    text: str
    proof: str = ""
    logically_valid: bool = False
    complete: bool = False
    uses_only_premises: bool = False
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
    
    @property
    def verified(self) -> bool:
        """True if all verification checks passed."""
        return self.logically_valid and self.complete and self.uses_only_premises
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "index": self.index,
            "step": self.text,
            "proof": self.proof,
            "verification": {
                "logically_valid": self.logically_valid,
                "complete": self.complete,
                "uses_only_premises": self.uses_only_premises,
                "verified": self.verified,
            },
            "issues": self.issues,
        }


def create_step_extraction_messages(
    source_text: str,
    target_text: str,
    reduction_full_text: str,
) -> List[Dict[str, str]]:
    """Create messages for unfolding a reduction into proof steps."""
    
    system_message = """You are a complexity theory expert breaking down reduction proofs.
Extract the essential logical steps. Keep it simple and focused on what matters."""

    user_message = f"""**Source problem:**
{source_text}

**Target problem:**
{target_text}

**Reduction proof:**
{reduction_full_text}

**Task:**
Unfold and organize this reduction into its essential steps. Generate concrete and precise claims that can be individually proved and verified. Omit all proof details, but explicitly state what needs to addressed in each step.

**Key steps to identify:**
1. Elements and properties of the source problem
2. Elements and properties of the target problem
3. High-level idea; correspondence between source and target elements
4. Relevant technical lemmas or gadget constructions (if any)
5. Full construction (what is built, including any gadgets)
6. Structural correctness: Valid source instances map to valid target instances
7. Forward direction: Why source satisfiable → target satisfiable
8. Reverse direction: Why target satisfiable → source satisfiable
9. Polynomial time analysis

**Format:**
Output ONLY a numbered list:
1. [Step]
2. [Step]
...
"""

    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def create_proof_messages(
    step: ProofStep,
    source_text: str,
    target_text: str,
    prior_steps: List[ProofStep],
) -> List[Dict[str, str]]:
    """Create messages for proving or explaining a step."""
    
    system_message = """You are a complexity theory expert. Provide clear, rigorous explanations."""

    prior_text = ""
    if prior_steps:
        prior_text = "\n**Prior steps:**\n"
        for ps in prior_steps:
            prior_text += f"{ps.index}. {ps.text}\n"
    
    user_message = f"""**Source problem:**
{source_text}

**Target problem:**
{target_text}
{prior_text}
**Step {step.index}:**
{step.text}

Address this step clearly and completely in a few sentences. Focus only on what this step asks for.
"""

    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def create_verification_messages(
    step: ProofStep,
    source_text: str,
    target_text: str,
    prior_steps: List[ProofStep],
) -> List[Dict[str, str]]:
    """Create messages for verifying a step."""
    
    system_message = """You are a proof verifier. Check if the response correctly addresses the step."""

    prior_text = ""
    if prior_steps:
        prior_text = "\n**Prior steps:**\n"
        for ps in prior_steps:
            prior_text += f"{ps.index}. {ps.text}\n"
    
    user_message = f"""**Source problem:**
{source_text}

**Target problem:**
{target_text}
{prior_text}
**Step {step.index}:**
{step.text}

**Response:**
{step.proof}

Evaluate this response:

(a) Logically valid - Is the reasoning sound and free of errors?
(b) Complete - Does it fully address what the step asks for?
(c) Uses only premises - Does it rely only on the problems, prior steps, and standard knowledge?

Answer in this format:
(a) YES/NO
(b) YES/NO
(c) YES/NO
Issues: [Explain any problems, or write "None"]
"""

    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def parse_steps(response_text: str) -> List[str]:
    """Parse numbered list of proof steps from model response.
    
    Handles both simple numbering (1, 2, 3) and sub-step numbering (2a, 2b, 3c).
    """
    steps = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Handle formats like "1. Step", "2a. Step", "1) Step", "3b) Step"
        if line[0].isdigit():
            # Find the first non-digit, non-letter, non-punctuation character
            i = 0
            while i < len(line) and (line[i].isdigit() or line[i].isalpha() or line[i] in '.-)]:'):
                i += 1
            step_text = line[i:].strip()
            if step_text:
                steps.append(step_text)
    return steps


def parse_verification(response_text: str) -> Dict[str, any]:
    """Parse verification response into structured data."""
    result = {
        "logically_valid": False,
        "complete": False,
        "uses_only_premises": False,
        "issues": []
    }
    
    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('(a)'):
            result["logically_valid"] = 'YES' in line.upper()
        elif line.startswith('(b)'):
            result["complete"] = 'YES' in line.upper()
        elif line.startswith('(c)'):
            result["uses_only_premises"] = 'YES' in line.upper()
        elif line.lower().startswith('issues:'):
            issue_text = line[7:].strip()
            if issue_text.lower() not in ['none', 'none.']:
                result["issues"].append(issue_text)
    
    return result


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def print_section_header(title: str):
    """Print a formatted section header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def verify_reduction(
    model: Model,
    row_index: int,
    source_text: str,
    target_text: str,
    reduction_text: str,
) -> Dict[str, any]:
    """Verify a reduction by unfolding into steps, proving each, and verifying each."""
    
    print_section_header(f"REDUCTION {row_index + 1}")
    print(f"Source: {source_text[:100]}...")
    print(f"Target: {target_text[:100]}...")
    print()
    
    # Phase 1: Unfold reduction into granular steps
    print("📋 PHASE 1: Unfolding reduction into granular steps...")
    print_separator('-')
    
    messages = create_step_extraction_messages(
        source_text, target_text, reduction_text
    )
    result = model.infer(
        messages[1]["content"],
        session_id=f"unfold-{row_index}",
        system_prompt=messages[0]["content"],
    )
    
    step_texts = parse_steps(result["text"])
    print(f"Unfolded into {len(step_texts)} steps:\n")
    for i, step_text in enumerate(step_texts, 1):
        print(f"  {i}. {step_text}")
    print()
    
    # Phase 2 & 3: Address and verify each step
    steps = []
    for i, step_text in enumerate(step_texts, 1):
        step = ProofStep(index=i, text=step_text)
        
        print_section_header(f"STEP {i}/{len(step_texts)}")
        print(f"Step: {step_text}\n")
        
        # Generate explanation or proof for this step
        print("✍️  Addressing step...")
        print_separator('-')
        proof_messages = create_proof_messages(
            step, source_text, target_text, steps[:i-1]
        )
        proof_result = model.infer(
            proof_messages[1]["content"],
            session_id=f"prove-{row_index}-{i}",
            system_prompt=proof_messages[0]["content"],
        )
        step.proof = proof_result["text"].strip()
        print(step.proof)
        print()
        
        # Verify the step
        print("🔍 Verifying step...")
        print_separator('-')
        verify_messages = create_verification_messages(
            step, source_text, target_text, steps[:i-1]
        )
        verify_result = model.infer(
            verify_messages[1]["content"],
            session_id=f"verify-{row_index}-{i}",
            system_prompt=verify_messages[0]["content"],
        )
        
        verification = parse_verification(verify_result["text"])
        step.logically_valid = verification["logically_valid"]
        step.complete = verification["complete"]
        step.uses_only_premises = verification["uses_only_premises"]
        step.issues = verification["issues"]
        
        print(verify_result["text"])
        print()
        
        # Summary for this step
        status = "✅ VERIFIED" if step.verified else "❌ FAILED"
        print(f"Status: {status}")
        if not step.verified:
            print(f"Issues: {'; '.join(step.issues) if step.issues else 'See verification output'}")
        print()
        
        steps.append(step)
    
    # Final summary
    num_verified = sum(1 for s in steps if s.verified)
    correctness_score = num_verified / len(steps) if steps else 0.0
    
    print_section_header("SUMMARY")
    print(f"Total steps: {len(steps)}")
    print(f"Verified: {num_verified}")
    print(f"Failed: {len(steps) - num_verified}")
    print(f"Correctness score: {correctness_score:.2%}")
    print()
    
    return {
        "proof_steps": [s.text for s in steps],
        "step_proofs": [s.to_dict() for s in steps],
        "num_steps": len(steps),
        "num_verified": num_verified,
        "correctness_score": correctness_score,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Granular proof verification through step-by-step unfolding")
    p.add_argument("--toy", action="store_true", help="Use tiny model for local testing (overrides config.ini)")
    p.add_argument("--input_csv", required=True, help="CSV with source_text,target_text,reduction_full_text")
    p.add_argument("--output_csv", default="verified_proofs.csv", help="Output CSV path")
    
    # Model params
    p.add_argument("--temperature", type=float, help="Override temperature from config")
    p.add_argument("--top_p", type=float, help="Override top_p from config")
    p.add_argument("--top_k", type=int, help="Override top_k from config")
    p.add_argument("--max_tokens", type=int, help="Override max_tokens from config")
    p.add_argument("--tensor_parallel_size", type=int, help="Override tensor_parallel_size from config")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Load CSV
    path = os.path.expanduser(args.input_csv)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    
    df = pd.read_csv(path)
    required_cols = ["source_text", "target_text", "reduction_full_text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"[proof-verify] Loaded {len(df)} rows from {path}")
    print()
    
    # Initialize model - model_id is sourced from config.ini
    # Only pass explicitly provided CLI arguments as overrides
    model_kwargs = {"toy": args.toy}
    if args.tensor_parallel_size is not None:
        model_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.temperature is not None:
        model_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        model_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        model_kwargs["top_k"] = args.top_k
    if args.max_tokens is not None:
        model_kwargs["max_tokens"] = args.max_tokens
    
    model = load_from_config(**model_kwargs)
    
    # Process rows - unfold, prove, and verify each reduction
    all_steps = []
    all_step_proofs = []
    all_num_steps = []
    all_num_verified = []
    all_scores = []
    
    for idx, row in df.iterrows():
        result = verify_reduction(
            model=model,
            row_index=idx,
            source_text=str(row["source_text"]),
            target_text=str(row["target_text"]),
            reduction_text=str(row["reduction_full_text"]),
        )
        
        all_steps.append(json.dumps(result["proof_steps"]))
        all_step_proofs.append(json.dumps(result["step_proofs"]))
        all_num_steps.append(result["num_steps"])
        all_num_verified.append(result["num_verified"])
        all_scores.append(result["correctness_score"])
    
    # Save results
    df["proof_steps_json"] = all_steps
    df["step_proofs_json"] = all_step_proofs
    df["num_steps"] = all_num_steps
    df["num_verified"] = all_num_verified
    df["correctness_score"] = all_scores
    
    out = os.path.expanduser(args.output_csv)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    
    print_separator('=')
    print(f"[proof-verify] Saved results to {out}")
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[proof-verify] Average correctness score: {avg_score:.2%}")
    print_separator('=')


if __name__ == "__main__":
    main()
