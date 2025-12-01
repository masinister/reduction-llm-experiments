import json
import logging
import os
import pandas as pd
from typing import Dict, Any
from pydantic import BaseModel, Field
from tqdm import tqdm
from src.core_backend import CoreBackend
from src.pipeline import Pipeline
from src.context_budget import ContextBudget
from src.utils import parse_structured_output
import argparse

logger = logging.getLogger(__name__)

# Define Pydantic Model for Structured Output
class PlaintextConversion(BaseModel):
    plaintext: str = Field(description="The input text with all LaTeX commands and symbols converted to plaintext equivalents.")

# Prompt Formatter
def conversion_prompt_formatter(context: Dict[str, Any]) -> str:
    text = context['text']
    
    prompt = "Convert the following LaTeX text to plaintext. "
    prompt += "Replace all LaTeX commands and symbols with their plaintext equivalents. "
    prompt += "Preserve the exact structure and format of the text.\n\n"
    prompt += f"LaTeX:\n{text}\n"
        
    return prompt

def process_text(pipeline: Pipeline, text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    context = {"text": text}
    try:
        out_json = pipeline.process(
            text=text,
            json_schema=PlaintextConversion.model_json_schema(),
            prompt_formatter=conversion_prompt_formatter,
            chunk_size_tokens=256,
            overlap_tokens=64
        )

        # Use the new utils for parsing
        parsed = parse_structured_output(out_json, PlaintextConversion)
        
        if parsed:
            return parsed.plaintext
        
        # Fallback: return original text if parsing failed
        logger.warning(f"Failed to parse output, returning original text")
        return text

    except Exception as e:
        # Log if you have logging; return original text on any failure
        logger.exception("Conversion failed: %s", e)
        return text
    
def main():
    parser = argparse.ArgumentParser(description="Convert LaTeX to plaintext.")
    parser.add_argument("--limit", type=int, help="Limit the number of rows to process for testing.") 
    args = parser.parse_args()

    # Load Data
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
    
    # Initialize Backend
    print("Initializing Pipeline...")
    core = CoreBackend()
    pipeline = Pipeline(
        core_backend=core,
        context_budget=ContextBudget(),
        merger_strategy="hierarchical",
        merger_batch_size=5,
        use_summarization=False  # Don't need summarization for conversion
    )
    
    # Define columns to convert
    # We focus on 'reduction_full_text' as it contains the main LaTeX content.
    # 'source_text' and 'target_text' might also contain LaTeX.
    cols_to_convert = ['reduction_full_text', 'source_text', 'target_text']
    
    tqdm.pandas()
    
    for col in cols_to_convert:
        if col in df.columns:
            print(f"Processing column: {col}")
            new_col = f"{col}_plaintext"
            
            # Use progress_apply with a lambda to pass the pipeline
            df[new_col] = df[col].progress_apply(lambda x: process_text(pipeline, x))
            
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()
