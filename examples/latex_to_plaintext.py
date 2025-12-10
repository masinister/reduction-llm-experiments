"""Example: Convert LaTeX to plaintext using hierarchical extraction."""

import argparse
import os

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from src import config
from src.core_backend import Backend
from src.strategies import hierarchical_extract

config.load()


class PlaintextConversion(BaseModel):
    """Output model for LaTeX to plaintext conversion."""
    plaintext: str = Field(description="LaTeX converted to plaintext")


def extract_prompt(text: str) -> str:
    """Create conversion prompt for a chunk."""
    return (
        "Convert the following LaTeX text to plaintext. "
        "Replace all LaTeX commands and symbols with their plaintext equivalents. "
        "Preserve the exact structure and format of the text.\n\n"
        f"LaTeX:\n{text}\n"
    )


def combine_prompt(partials: list[str]) -> str:
    """Combine partial conversions."""
    joined = "\n---\n".join(partials)
    return f"Combine these partial plaintext conversions into one coherent text:\n\n{joined}"


def process_text(backend: Backend, text: str) -> str:
    """Process a single text field."""
    if not text or not text.strip():
        return ""
    
    try:
        result = hierarchical_extract(
            backend=backend,
            text=text,
            response_model=PlaintextConversion,
            extract_prompt=extract_prompt,
            combine_prompt=combine_prompt,
        )
        return result.plaintext
    except Exception as e:
        print(f"Conversion failed: {e}")
        return text


def main():
    parser = argparse.ArgumentParser(description="Convert LaTeX to plaintext.")
    parser.add_argument("--limit", type=int, help="Limit rows for testing")
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
    backend = Backend()
    
    cols_to_convert = ['reduction_full_text', 'source_text', 'target_text']
    tqdm.pandas()
    
    for col in cols_to_convert:
        if col in df.columns:
            print(f"Processing column: {col}")
            new_col = f"{col}_plaintext"
            df[new_col] = df[col].progress_apply(lambda x: process_text(backend, x))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
