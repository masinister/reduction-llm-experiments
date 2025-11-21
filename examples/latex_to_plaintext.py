import json
import os
import pandas as pd
from typing import Dict, Any
from pydantic import BaseModel, Field
from tqdm import tqdm
from src.backend import Backend
import argparse

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

def process_text(backend: Backend, text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        context = {"text": text}
        out_json = backend.inference(
            context, 
            conversion_prompt_formatter, 
            PlaintextConversion.model_json_schema(),
            assemble_final=True,
            use_summarization=False,
            chunk_size_tokens=2000,
            overlap_tokens=200
        )
        
        parsed_output = json.loads(out_json)
        return parsed_output.get("plaintext", text)
    except Exception:
        # Return original text if conversion fails
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
    print("Initializing Backend...")
    backend = Backend()
    
    # Define columns to convert
    # We focus on 'reduction_full_text' as it contains the main LaTeX content.
    # 'source_text' and 'target_text' might also contain LaTeX.
    cols_to_convert = ['reduction_full_text', 'source_text', 'target_text']
    
    tqdm.pandas()
    
    for col in cols_to_convert:
        if col in df.columns:
            print(f"Processing column: {col}")
            new_col = f"{col}_plaintext"
            
            # Use progress_apply with a lambda to pass the backend
            df[new_col] = df[col].progress_apply(lambda x: process_text(backend, x))
            
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()
