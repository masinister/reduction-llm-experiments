"""Example: Convert LaTeX to plaintext using StreamingStrategy."""

import logging
import os
import argparse
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.core_backend import CoreBackend
from src.strategies import StreamingStrategy

logger = logging.getLogger(__name__)


class PlaintextConversion(BaseModel):
    """Output model for LaTeX to plaintext conversion."""
    plaintext: str = Field(description="LaTeX converted to plaintext")


def make_prompt(text: str) -> str:
    """Create conversion prompt for a chunk."""
    return (
        "Convert the following LaTeX text to plaintext. "
        "Replace all LaTeX commands and symbols with their plaintext equivalents. "
        "Preserve the exact structure and format of the text.\n\n"
        f"LaTeX:\n{text}\n"
    )


def process_text(strategy: StreamingStrategy, text: str) -> str:
    """Process a single text field."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        return strategy.process(
            text=text,
            prompt_fn=make_prompt,
            extract_fn=lambda x: x.plaintext,
        )
    except Exception as e:
        logger.exception("Conversion failed: %s", e)
        return text


def main():
    parser = argparse.ArgumentParser(description="Convert LaTeX to plaintext.")
    parser.add_argument("--limit", type=int, help="Limit rows for testing")
    parser.add_argument("--chunk-size", type=int, default=512, help="Tokens per chunk")
    args = parser.parse_args()

    input_path = "data/raw/karp.csv"
    output_path = "data/processed/karp_plaintext.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    if args.limit:
        print(f"Limiting to first {args.limit} rows.")
        df = df.head(args.limit)
    
    print("Initializing...")
    backend = CoreBackend()
    strategy = StreamingStrategy(
        backend=backend,
        output_model=PlaintextConversion,
        chunk_size=args.chunk_size,
        overlap=args.chunk_size // 8,
    )
    
    cols_to_convert = ['reduction_full_text', 'source_text', 'target_text']
    tqdm.pandas()
    
    for col in cols_to_convert:
        if col in df.columns:
            print(f"Processing column: {col}")
            new_col = f"{col}_plaintext"
            df[new_col] = df[col].progress_apply(lambda x: process_text(strategy, x))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
