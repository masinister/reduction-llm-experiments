#!/usr/bin/env python3
"""Per-sentence verification for known reductions.

For each input reduction, we:
  1) Split the reduction into sentences (deterministically, no LLM).
  2) For each sentence, prompt an LLM to classify the sentence (definition/claim/justification),
     and decide clarity/justification using the rest of the reduction as context.

We keep parsing simple: we store the model's feedback per sentence as-is (no complex JSON parsing).

Input CSV columns (required):
  - reduction_full_text

Optional input columns (pass-through / context):
  - source_text
  - target_text

Output CSV columns (added):
  - verification_sentences  # JSON array of objects: {sentence_id, sentence_text, feedback, raw_feedback, latency_s, tokens}
  - ver_total_latency_s     # sum of latencies across sentences
  - ver_total_tokens        # sum of tokens across sentences

Usage:
  python examples/reduction_verification.py \
    --input_csv data.csv \
    --output_csv verification_results.csv

Model configuration is read from config.ini via src.inference.load_from_config.
You can override parameters with CLI flags like --temperature, etc., same as examples/reduction_cot_batch.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Dict

import pandas as pd

from src.inference import load_from_config


# -------------------------------
# Sentence splitting (deterministic)
# -------------------------------
ABBREV = {
    "e.g.", "i.e.", "etc.", "vs.", "cf.", "Fig.", "Def.", "Prop.", "Thm.", "Cor.", "Eq.",
}


def split_into_sentences(text: str) -> List[str]:
    """A light-weight, deterministic sentence splitter.

    Heuristics:
      - Split on [.?!] followed by whitespace and a capital/number/open-paren.
      - Avoid splitting after common abbreviations.
      - Keep semicolon-separated clauses within the same sentence (by default).
    """
    if not text:
        return []

    # Normalize spaces
    t = re.sub(r"\s+", " ", str(text)).strip()
    if not t:
        return []

    # Quick path: if very short, return as single sentence
    if len(t) < 200 and t.count(".") == 0 and t.count("?") == 0 and t.count("!") == 0:
        return [t]

    sentences: List[str] = []
    start = 0
    for m in re.finditer(r"[\.\?\!]\s+(?=[A-Z0-9(])", t):
        end = m.end()
        segment = t[start:end].strip()
        # Check for abbreviation at the end
        tail = segment.split(" ")[-1]
        if any(segment.endswith(abbrev) for abbrev in ABBREV):
            continue
        if tail.endswith(".") and tail in ABBREV:
            continue
        if segment:
            sentences.append(segment)
        start = end

    last = t[start:].strip()
    if last:
        sentences.append(last)

    # Final cleanup: strip extra spaces
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# -------------------------------
# Prompt builder
# -------------------------------
def build_verification_messages(
    reduction_full_text: str,
    sentence_text: str,
    sentence_id: int,
    total_sentences: int,
    rest_of_reduction: str,
) -> List[Dict[str, str]]:
    """Construct messages for LLM classification with minimal, deterministic instructions.

    We ask the model to respond concisely; we won't parse complex JSON—store feedback as-is.
    """
    system_message = (
        "You verify reductions sentence-by-sentence. "
        "For the current sentence, briefly state: (a) type: definition/claim/justification/mixed, "
        "(b) clarity: clear/unclear, (c) justification: justified/unjustified/self-evident, and a one-line note. "
        "Keep output on one or two short lines."
    )

    user_message = f"""
Pre-approved reduction (verbatim):
{reduction_full_text}

Current sentence [{sentence_id+1}/{total_sentences}]:
{sentence_text}

Rest of reduction (all other sentences in order):
{rest_of_reduction}

Please respond succinctly in this template (free text acceptable):
type=<definition|claim|justification|mixed>; clarity=<clear|unclear>; justification=<justified|unjustified|self-evident>; note=<short rationale>
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# -------------------------------
# CLI and main
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Per-sentence verification for reductions")
    p.add_argument("--toy", action="store_true", help="Use tiny model for local testing (overrides config.ini)")
    p.add_argument("--input_csv", required=True, help="CSV with at least reduction_full_text column")
    p.add_argument("--output_csv", default="verification_results.csv", help="Output CSV path")

    # Model overrides (optional)
    p.add_argument("--temperature", type=float, help="Override temperature from config")
    p.add_argument("--top_p", type=float, help="Override top_p from config")
    p.add_argument("--top_k", type=int, help="Override top_k from config")
    p.add_argument("--max_tokens", type=int, help="Override max_tokens from config")
    p.add_argument("--tensor_parallel_size", type=int, help="Override tensor_parallel_size from config")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input CSV
    path = os.path.expanduser(args.input_csv)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    if "reduction_full_text" not in df.columns:
        raise ValueError("Missing required column: reduction_full_text")

    print(f"[verify] Loaded {len(df)} rows from {path}")

    # Initialize model using config.ini with optional overrides
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

    # Process each reduction
    all_feedback_arrays: List[List[Dict[str, object]]] = []
    total_latencies: List[float] = []
    total_tokens: List[int] = []

    for ridx, row in df.iterrows():
        reduction_text = str(row["reduction_full_text"])
        sentences = split_into_sentences(reduction_text)
        n = len(sentences)
        if n == 0:
            all_feedback_arrays.append([])
            total_latencies.append(0.0)
            total_tokens.append(0)
            continue

        print(f"\n[verify] Row {ridx}: {n} sentences", flush=True)
        preview = reduction_text.strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"[verify] Reduction preview: {preview}", flush=True)

        feedback_entries: List[Dict[str, object]] = []
        row_latencies = 0.0
        row_tokens = 0

        for sidx, sent in enumerate(sentences):
            print(f"\n[verify] S{ sidx + 1 }/{n}: {sent}", flush=True)
            rest = " ".join(sentences[:sidx] + sentences[sidx+1:])
            messages = build_verification_messages(
                reduction_full_text=reduction_text,
                sentence_text=sent,
                sentence_id=sidx,
                total_sentences=n,
                rest_of_reduction=rest,
            )

            # Use a dedicated session per sentence for isolation
            result = model.infer(
                messages[1]["content"],
                session_id=f"verify-row-{ridx}-s{ sidx }",
                system_prompt=messages[0]["content"],
                enable_thinking=False,
            )

            # Pretty-print immediate feedback
            fb_text = result.get("text", "").strip()
            raw_text = result.get("raw_text", "").strip()
            lat = float(result.get("latency_s", 0.0))
            toks = int(result.get("tokens", 0))
            print(f"  -> feedback: {fb_text}", flush=True)
            if fb_text != raw_text and raw_text:
                print(f"  -> raw: {raw_text}", flush=True)
            print(f"  -> latency: {lat:.2f}s | tokens: {toks}", flush=True)

            # Store feedback as-is; avoid complex parsing
            feedback_entries.append({
                "sentence_id": int(sidx),
                "sentence_text": sent,
                "feedback": fb_text,
                "raw_feedback": raw_text,
                "latency_s": lat,
                "tokens": toks,
            })

            row_latencies += lat
            row_tokens += toks

            if (sidx + 1) % 10 == 0:
                avg_lat = row_latencies / (sidx + 1)
                print(f"[verify] row {ridx}: {sidx + 1}/{n} sentences | avg latency: {avg_lat:.2f}s")

        all_feedback_arrays.append(feedback_entries)
        total_latencies.append(row_latencies)
        total_tokens.append(row_tokens)

        # Row summary
        print(f"\n[verify] Row {ridx} summary: total latency {row_latencies:.2f}s | total tokens {row_tokens}", flush=True)

    # Save results
    df["verification_sentences"] = [json.dumps(arr, ensure_ascii=False) for arr in all_feedback_arrays]
    df["ver_total_latency_s"] = total_latencies
    df["ver_total_tokens"] = total_tokens

    out = os.path.expanduser(args.output_csv)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[verify] Saved results to {out}")
    grand_total = sum(total_latencies)
    denom = sum(1 for arr in all_feedback_arrays for _ in arr) or 1
    print(f"[verify] Total time: {grand_total:.1f}s | Avg per sentence: {grand_total/denom:.2f}s")


if __name__ == "__main__":
    main()
