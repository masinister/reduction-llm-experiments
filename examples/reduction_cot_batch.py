#!/usr/bin/env python3
"""Batch chain-of-thought (CoT) generation for known reductions.

Input CSV columns (required):
  - source_text
  - target_text
  - reduction_full_text   # a natural-language description of the known reduction

Output CSV columns (added):
  - cot_generation      # cleaned model output
  - cot_generation_raw  # raw model output (may include think tokens)
  - cot_latency_s
  - cot_tokens

Usage:
  python examples/reduction_cot_batch.py \
    --toy \
    --input_csv data.csv \
    --output_csv cot_results.csv \
    --thinking on \
    --num_steps 5 \
    --temperature 0.7
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
from typing import List, Dict

from src.inference import Model, load_from_config


def create_cot_messages(
    source_text: str,
    target_text: str,
    reduction_full_text: str,
    thinking: str = "on",
    num_steps: int = 5,
) -> List[Dict[str, str]]:

    system_message = f"""detailed thinking {thinking}

You are a complexity-theory expert.  Your goal is **not** to solve the reduction from scratch, but to **invent** a realistic-sounding internal reasoning trace that an expert *could* have gone through, step by step, to arrive at the given reduction.  
- Each step should be a *brief* thought (1-2 sentences).  
- Do **not** restate the final reduction.
"""

    user_message = f"""
**Source problem:**  
{source_text}

**Target problem:**  
{target_text}

**Known reduction (for your reference):**  
{reduction_full_text}

**Instructions:**  
Please generate exactly {num_steps} synthetic CoT steps that an expert might have used to discover this reduction. Format your response as a bulleted list. For example:

- First, notice that...
- This suggests that...
- Hence, the reduction can be constructed by...
- Indeed, ...
"""

    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Batch chain-of-thought generation for reductions")
    p.add_argument("--model", help="Model id/path (or use --toy)")
    p.add_argument("--toy", action="store_true", help="Use tiny model for local testing")
    p.add_argument("--input_csv", required=True, help="CSV with source_text,target_text,reduction_full_text")
    p.add_argument("--output_csv", default="cot_results.csv", help="Output CSV path")

    # CoT controls
    p.add_argument("--thinking", choices=["on", "off"], default="on", help="Enable special thinking mode if supported")
    p.add_argument("--num_steps", type=int, default=5, help="Number of CoT steps to generate")

    # Model params (no defaults - let config.ini handle defaults)
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

    print(f"[cot-batch] Loaded {len(df)} rows from {path}")

    # Initialize model - only pass explicitly provided CLI arguments
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
    cot_generations = []
    cot_raw = []
    cot_latencies = []
    cot_tokens = []

    enable_thinking = args.thinking.lower() == "on"

    for idx, row in df.iterrows():
        messages = create_cot_messages(
            source_text=str(row["source_text"]),
            target_text=str(row["target_text"]),
            reduction_full_text=str(row["reduction_full_text"]),
            thinking=args.thinking,
            num_steps=int(args.num_steps),
        )

        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]

        # Each row gets its own session to avoid cross-contamination
        result = model.infer(
            user_prompt,
            session_id=f"cot-row-{idx}",
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )

        cot_generations.append(result["text"])  # cleaned text (after </think>)
        cot_raw.append(result["raw_text"])      # raw text (may include </think>)
        cot_latencies.append(result["latency_s"])
        cot_tokens.append(result["tokens"])

        if (idx + 1) % 10 == 0:
            avg_latency = sum(cot_latencies[-10:]) / min(10, len(cot_latencies[-10:]))
            print(f"[cot-batch] {idx + 1}/{len(df)} rows | avg latency: {avg_latency:.2f}s")

    # Save results
    df["cot_generation"] = cot_generations
    df["cot_generation_raw"] = cot_raw
    df["cot_latency_s"] = cot_latencies
    df["cot_tokens"] = cot_tokens

    out = os.path.expanduser(args.output_csv)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[cot-batch] Saved results to {out}")
    total = sum(cot_latencies)
    print(f"[cot-batch] Total time: {total:.1f}s | Avg: {total/len(df):.2f}s/row")


if __name__ == "__main__":
    main()
