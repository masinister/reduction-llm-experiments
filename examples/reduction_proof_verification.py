#!/usr/bin/env python3
"""Structured proof verification for reductions (MVP).

Extracts key claims from each reduction, generates proofs for each claim,
and verifies each proof. Records which claims pass verification.

Input CSV columns (required):
  - source_text
  - target_text
  - reduction_full_text

Output CSV columns (added):
  - claims_json           # List of extracted claims
  - proofs_json           # List of {claim, proof, verification} dicts
  - correctness_score     # Fraction of claims that verified successfully
  - num_claims
  - num_verified

Usage:
  python examples/reduction_proof_verification.py \
    --toy \
    --input_csv karp.csv \
    --output_csv verified_proofs.csv \
    --num_claims 5
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
class ProofClaim:
    """A single claim in a structured proof."""
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
            "claim": self.text,
            "proof": self.proof,
            "verification": {
                "logically_valid": self.logically_valid,
                "complete": self.complete,
                "uses_only_premises": self.uses_only_premises,
                "verified": self.verified,
            },
            "issues": self.issues,
        }


def create_claim_extraction_messages(
    source_text: str,
    target_text: str,
    reduction_full_text: str,
    num_claims: int = 5,
) -> List[Dict[str, str]]:
    """Create messages for extracting key claims from a reduction."""
    
    system_message = """You are a complexity theory expert analyzing reduction proofs.
Your task is to extract the key claims that together constitute a complete proof of the reduction."""

    user_message = f"""**Source problem:**
{source_text}

**Target problem:**
{target_text}

**Reduction proof:**
{reduction_full_text}

**Task:**
Extract exactly {num_claims} key claims that together prove this reduction is correct.

**Guidelines:**
- Each claim should be one clear, verifiable statement
- Claims should build on each other logically
- Typical structure: construction definition → polynomial-time → correctness (both directions)
- Number them 1 through {num_claims}

**Format:**
Output ONLY a numbered list, one claim per line:
1. [First claim]
2. [Second claim]
...
"""

    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def create_proof_messages(
    claim: ProofClaim,
    source_text: str,
    target_text: str,
    prior_claims: List[ProofClaim],
) -> List[Dict[str, str]]:
    """Create messages for proving a specific claim."""
    
    system_message = """You are a complexity theory expert writing rigorous proofs.
Provide clear, logically sound proofs that are complete yet concise."""

    prior_text = ""
    if prior_claims:
        prior_text = "\n**Previously proven claims:**\n"
        for pc in prior_claims:
            prior_text += f"{pc.index}. {pc.text}\n"
    
    user_message = f"""**Source problem:**
{source_text}

**Target problem:**
{target_text}
{prior_text}
**Claim to prove (Claim {claim.index}):**
{claim.text}

**Task:**
Provide a rigorous but concise proof of this claim (3-8 sentences).

**You may use:**
- Definitions of the source and target problems
- Previously proven claims (listed above)
- Standard complexity theory facts

**Format:**
Output ONLY the proof text, no preamble.
"""

    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def create_verification_messages(
    claim: ProofClaim,
    source_text: str,
    target_text: str,
    prior_claims: List[ProofClaim],
) -> List[Dict[str, str]]:
    """Create messages for verifying a proof."""
    
    system_message = """You are a proof verification system.
Be rigorous and adversarial - actively look for gaps, errors, or unjustified steps.
A proof should only pass if it is genuinely sound and complete."""

    prior_text = ""
    if prior_claims:
        prior_text = "\n**Available prior claims:**\n"
        for pc in prior_claims:
            prior_text += f"{pc.index}. {pc.text}\n"
    
    user_message = f"""**Source problem:**
{source_text}

**Target problem:**
{target_text}
{prior_text}
**Claim:**
{claim.text}

**Proposed proof:**
{claim.proof}

**Task:**
Verify this proof by answering three questions:

(a) Is the proof logically valid (no logical errors)? YES or NO
(b) Is the proof complete (no gaps or missing steps)? YES or NO  
(c) Does the proof use only available premises (problem definitions + prior claims)? YES or NO

**Format:**
Output EXACTLY in this format:
(a) [YES/NO]
(b) [YES/NO]
(c) [YES/NO]
Issues: [If any answer is NO, briefly explain the specific problem. Otherwise write "None"]
"""

    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def parse_claims(response_text: str) -> List[str]:
    """Parse numbered list of claims from model response."""
    claims = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Handle formats like "1. Claim" or "1) Claim" or just numbered lines
        if line[0].isdigit():
            # Find the first non-digit, non-punctuation character
            i = 0
            while i < len(line) and (line[i].isdigit() or line[i] in '.-)]:'):
                i += 1
            claim_text = line[i:].strip()
            if claim_text:
                claims.append(claim_text)
    return claims


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
    num_claims: int,
) -> Dict[str, any]:
    """Verify a single reduction by extracting claims, proving, and verifying each."""
    
    print_section_header(f"REDUCTION {row_index + 1}")
    print(f"Source: {source_text[:100]}...")
    print(f"Target: {target_text[:100]}...")
    print()
    
    # Step 1: Extract claims
    print("📋 STEP 1: Extracting key claims...")
    print_separator('-')
    
    messages = create_claim_extraction_messages(
        source_text, target_text, reduction_text, num_claims
    )
    result = model.infer(
        messages[1]["content"],
        session_id=f"extract-{row_index}",
        system_prompt=messages[0]["content"],
    )
    
    claim_texts = parse_claims(result["text"])
    print(f"Extracted {len(claim_texts)} claims:\n")
    for i, claim_text in enumerate(claim_texts, 1):
        print(f"  {i}. {claim_text}")
    print()
    
    # Step 2 & 3: Prove and verify each claim
    claims = []
    for i, claim_text in enumerate(claim_texts, 1):
        claim = ProofClaim(index=i, text=claim_text)
        
        print_section_header(f"CLAIM {i}/{len(claim_texts)}")
        print(f"Claim: {claim_text}\n")
        
        # Generate proof
        print("✍️  Generating proof...")
        print_separator('-')
        proof_messages = create_proof_messages(
            claim, source_text, target_text, claims[:i-1]
        )
        proof_result = model.infer(
            proof_messages[1]["content"],
            session_id=f"prove-{row_index}-{i}",
            system_prompt=proof_messages[0]["content"],
        )
        claim.proof = proof_result["text"].strip()
        print(claim.proof)
        print()
        
        # Verify proof
        print("🔍 Verifying proof...")
        print_separator('-')
        verify_messages = create_verification_messages(
            claim, source_text, target_text, claims[:i-1]
        )
        verify_result = model.infer(
            verify_messages[1]["content"],
            session_id=f"verify-{row_index}-{i}",
            system_prompt=verify_messages[0]["content"],
        )
        
        verification = parse_verification(verify_result["text"])
        claim.logically_valid = verification["logically_valid"]
        claim.complete = verification["complete"]
        claim.uses_only_premises = verification["uses_only_premises"]
        claim.issues = verification["issues"]
        
        print(verify_result["text"])
        print()
        
        # Summary
        status = "✅ VERIFIED" if claim.verified else "❌ FAILED"
        print(f"Status: {status}")
        if not claim.verified:
            print(f"Issues: {', '.join(claim.issues) if claim.issues else 'See verification output'}")
        print()
        
        claims.append(claim)
    
    # Final summary
    num_verified = sum(1 for c in claims if c.verified)
    correctness_score = num_verified / len(claims) if claims else 0.0
    
    print_section_header("SUMMARY")
    print(f"Total claims: {len(claims)}")
    print(f"Verified: {num_verified}")
    print(f"Failed: {len(claims) - num_verified}")
    print(f"Correctness score: {correctness_score:.2%}")
    print()
    
    return {
        "claims": [c.text for c in claims],
        "proofs": [c.to_dict() for c in claims],
        "num_claims": len(claims),
        "num_verified": num_verified,
        "correctness_score": correctness_score,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Structured proof verification for reductions")
    p.add_argument("--model", help="Model id/path (or use --toy)")
    p.add_argument("--toy", action="store_true", help="Use tiny model for local testing")
    p.add_argument("--input_csv", required=True, help="CSV with source_text,target_text,reduction_full_text")
    p.add_argument("--output_csv", default="verified_proofs.csv", help="Output CSV path")
    p.add_argument("--num_claims", type=int, default=5, help="Number of claims to extract per reduction")
    
    # Model params
    p.add_argument("--temperature", type=float, help="Override temperature from config")
    p.add_argument("--top_p", type=float, help="Override top_p from config")
    p.add_argument("--top_k", type=int, help="Override top_k from config")
    p.add_argument("--max_tokens", type=int, help="Override max_tokens from config")
    p.add_argument("--tensor_parallel_size", type=int, help="Override tensor_parallel_size from config")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    if not args.model and not args.toy:
        raise SystemExit("Error: Specify either --model MODEL_ID or --toy")
    
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
    
    # Initialize model
    model_kwargs = {"toy": args.toy}
    if args.model is not None:
        model_kwargs["model_id"] = args.model
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
    
    # Process rows
    all_claims = []
    all_proofs = []
    all_num_claims = []
    all_num_verified = []
    all_scores = []
    
    for idx, row in df.iterrows():
        result = verify_reduction(
            model=model,
            row_index=idx,
            source_text=str(row["source_text"]),
            target_text=str(row["target_text"]),
            reduction_text=str(row["reduction_full_text"]),
            num_claims=args.num_claims,
        )
        
        all_claims.append(json.dumps(result["claims"]))
        all_proofs.append(json.dumps(result["proofs"]))
        all_num_claims.append(result["num_claims"])
        all_num_verified.append(result["num_verified"])
        all_scores.append(result["correctness_score"])
    
    # Save results
    df["claims_json"] = all_claims
    df["proofs_json"] = all_proofs
    df["num_claims"] = all_num_claims
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
