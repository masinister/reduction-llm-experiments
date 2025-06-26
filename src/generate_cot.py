import os
import argparse
import torch
import pandas as pd
import transformers
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate chain-of-thought reasoning for Karp dataset")
    parser.add_argument("--input_csv", type=str, default="~/data/karp.csv",
                        help="Path to input karp.csv file")
    parser.add_argument("--output_csv", type=str, default="../data/karp_cot.csv",
                        help="Path to save augmented karp_cot.csv file")
    parser.add_argument("--cot_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Model to use for generating chain-of-thought reasoning")
    parser.add_argument("--max_new_tokens", type=int, default=32768,
                        help="Maximum sequence length for CoT model")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for CoT model generation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation (keep low for large models)")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume generation from specific index (for interrupted runs)")
    parser.add_argument("--thinking", type=str, default="on", choices=["on", "off"],
                        help="Enable detailed thinking mode for the model")
    return parser.parse_args()


def load_cot_model(args):
    """Load the chain-of-thought generation pipeline"""
    print(f"Loading CoT model: {args.cot_model}")
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.model_dtype]
    
    model_kwargs = {
        "torch_dtype": model_dtype,
        "trust_remote_code": True,
        "device_map": "auto" if args.device == "auto" else None
    }
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.cot_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.cot_model,
        tokenizer=tokenizer,
        max_new_tokens=32768,
        temperature=args.temperature,
        top_p=0.95,
        **model_kwargs
    )
    
    print("✅ CoT pipeline loaded successfully")
    return pipeline


def create_cot_messages(source_text: str, target_text: str, reduction_text: str, thinking: str = "on") -> list:
    """Create chat messages for generating chain-of-thought reasoning"""
    
    system_message = f"""detailed thinking {thinking}

You are an expert in computational complexity theory and mathematical reductions. Your task is to generate a detailed chain-of-thought reasoning that explains the step-by-step process of creating reductions between computational problems."""
    
    user_message = f"""I need you to analyze the following reduction between two computational problems and explain the reasoning behind it.

**Source Problem**: {source_text}
**Target Problem**: {target_text}

**Reduction**: {reduction_text}

Please explain the reasoning in 3 concise (1-2 sentence) chain-of-thought steps. Format your response as follows:

1. **Problem understanding**
    - [Describe the key characteristics of the source problem and target problem.]
2. **Reduction strategy**
    - [Provide a high-level overview of how you would approach the reduction.]
3. **Correctness**
    - [(=>) Why are solutions to the source problem transformed into solutions to the target problem?]
    - [(<=) Why do solutions to the target problem correspond to solutions to the source problem?]"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    return messages


def generate_single_cot(pipeline, example: Dict, args) -> str:
    """Generate chain-of-thought reasoning for a single example"""
    
    messages = create_cot_messages(
        example['source_text'],
        example['target_text'], 
        example['reduction_full_text'],
        args.thinking
    )
    
    try:
        # Use the pipeline with chat messages directly
        result = pipeline(messages)
        
        # Parse the output following the recommended approach
        response = ""
        for message in result[0]["generated_text"]:
            if message.get("role") == "assistant":
                response = message.get("content", "")
                break
        else:
            # If no assistant message found
            response = str(result[0]["generated_text"])
        
        # Filter out the <think> section if present
        if "<think>" in response and "</think>" in response:
            response = response.split("</think>", 1)[-1].strip()
        
        return response.strip() if response else "ERROR: No valid response generated"
        
    except Exception as e:
        return f'ERROR: {str(e)}'


def load_karp_dataset(file_path: str) -> pd.DataFrame:
    """Load the Karp dataset from CSV file"""
    file_path = os.path.expanduser(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Karp dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Check for required columns
    required_columns = ['source_text', 'target_text', 'reduction_full_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns in dataset: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df


def save_checkpoint(df: pd.DataFrame, output_path: str, current_index: int):
    """Save intermediate results as checkpoint"""
    checkpoint_path = output_path.replace('.csv', f'_checkpoint_{current_index}.csv')
    df.to_csv(checkpoint_path, index=False, encoding='utf-8')
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    args = parse_args()
    
    print("="*60)
    print("CHAIN-OF-THOUGHT GENERATION FOR KARP DATASET")
    print("="*60)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"CoT model: {args.cot_model}")
    print(f"Device: {args.device}")
    print(f"Model dtype: {args.model_dtype}")
    print(f"Temperature: {args.temperature}")
    print(f"Thinking mode: {args.thinking}")
    print(f"Resume from index: {args.resume_from}")
    print("="*60)
    
    # Load Karp dataset
    print("Loading Karp dataset...")
    df = load_karp_dataset(args.input_csv)
    print(f"Loaded {len(df)} examples from dataset")
    
    # Check if we're resuming from a previous run
    output_path = os.path.expanduser(args.output_csv)
    if args.resume_from > 0 and os.path.exists(output_path):
        print(f"Resuming from existing file at index {args.resume_from}")
        existing_df = pd.read_csv(output_path)
        if 'chain_of_thought' in existing_df.columns:
            # Use existing results up to resume point
            df = existing_df.copy()
        else:
            # Add empty CoT column
            df['chain_of_thought'] = ""
    else:
        # Add new column for chain-of-thought reasoning
        df['chain_of_thought'] = ""
    
    # Load CoT model
    pipeline = load_cot_model(args)
    
    # Generate chain-of-thought reasoning for each example
    print("Starting chain-of-thought generation...")
    
    for i in range(args.resume_from, len(df)):
        print(f"Generating CoT for example {i+1}/{len(df)}")
        
        # Skip if already generated
        if pd.notna(df.iloc[i]['chain_of_thought']) and df.iloc[i]['chain_of_thought'].strip():
            print(f"  Skipping example {i+1} (already has CoT)")
            continue
        
        example = df.iloc[i].to_dict()
        cot_reasoning = generate_single_cot(pipeline, example, args)
        
        # Update the dataframe
        df.iloc[i, df.columns.get_loc('chain_of_thought')] = cot_reasoning
        
        # Save progress every 10 examples
        if (i + 1) % 10 == 0:
            save_checkpoint(df, output_path, i + 1)
        
        # Print preview of generated reasoning
        if not cot_reasoning.startswith('ERROR:'):
            preview = cot_reasoning[:200] + "..." if len(cot_reasoning) > 200 else cot_reasoning
            print(f"  Generated CoT preview: {preview}")
        else:
            print(f"  Error generating CoT: {cot_reasoning}")
    
    # Save final results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Print summary
    print("\n" + "="*60)
    print("CHAIN-OF-THOUGHT GENERATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_path}")
    print(f"Total examples: {len(df)}")
    
    # Count successful generations
    successful = len(df[~df['chain_of_thought'].str.startswith('ERROR:', na=False) & 
                      df['chain_of_thought'].notna() & 
                      (df['chain_of_thought'].str.strip() != '')])
    errors = len(df[df['chain_of_thought'].str.startswith('ERROR:', na=False)])
    empty = len(df[(df['chain_of_thought'].isna()) | (df['chain_of_thought'].str.strip() == '')])
    
    print(f"Successfully generated: {successful}")
    print(f"Errors: {errors}")
    print(f"Empty/skipped: {empty}")
    print("="*60)
    
    # Print column information
    print(f"\nOutput dataset columns:")
    for col in df.columns:
        print(f"  - {col}")
    
    print(f"\nSample of generated chain-of-thought:")
    if successful > 0:
        # Find first successful example
        for idx, row in df.iterrows():
            if (pd.notna(row['chain_of_thought']) and 
                not row['chain_of_thought'].startswith('ERROR:') and 
                row['chain_of_thought'].strip()):
                print(f"Example {idx + 1}:")
                print(f"Source: {row['source_text'][:100]}...")
                print(f"Target: {row['target_text'][:100]}...")
                print(f"CoT: {row['chain_of_thought'][:300]}...")
                break
    else:
        print("No successful generations found.")


if __name__ == "__main__":
    main()
