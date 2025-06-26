import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate chain-of-thought reasoning for Karp dataset")
    parser.add_argument("--input_csv", type=str, default="~/data/karp.csv",
                        help="Path to input karp.csv file")
    parser.add_argument("--output_csv", type=str, default="data/karp_cot.csv",
                        help="Path to save augmented karp_cot.csv file")
    parser.add_argument("--cot_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Model to use for generating chain-of-thought reasoning")
    parser.add_argument("--max_length", type=int, default=32768,
                        help="Maximum sequence length for CoT model")
    parser.add_argument("--max_new_tokens", type=int, default=32768,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for CoT model generation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation (keep low for large models)")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume generation from specific index (for interrupted runs)")
    return parser.parse_args()


def load_cot_model(args):
    """Load the chain-of-thought generation model and tokenizer"""
    print(f"Loading CoT model: {args.cot_model}")
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.model_dtype]
    
    tokenizer = AutoTokenizer.from_pretrained(args.cot_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.cot_model,
        torch_dtype=model_dtype,
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=True,
    )
    
    model.config.use_cache = True
    model.eval()
    
    print("✅ CoT model loaded successfully")
    return model, tokenizer


def create_cot_messages(source_text: str,
                        target_text: str,
                        reduction_text: str,
                        thinking: str = "on",
                        num_steps: int = 5) -> list:
    """Produce a synthetic “chain of thought” leading to a known reduction."""
    
    system_message = """detailed thinking {thinking}

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
{reduction_text}

**Instructions:**  
Please generate exactly {num_steps} synthetic CoT steps that an expert might have used to discover this reduction. Format your response as a bulleted list. For example:

- First, notice that...
- This suggests that...
- Hence, the reduction can be constructed by...
- Indeed, ...
"""
    
    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user",   "content": user_message.strip()}
    ]



def generate_single_cot(model, tokenizer, example: Dict, args) -> str:
    """Generate chain-of-thought reasoning for a single example"""
    
    messages = create_cot_messages(
        example['source_text'],
        example['target_text'], 
        example['reduction_full_text']
    )
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True if args.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        cot_reasoning = response[len(prompt):].strip()
        
        # Parse output by taking everything AFTER the </think> tag
        if "</think>" in cot_reasoning:
            cot_reasoning = cot_reasoning.split("</think>", 1)[1].strip()
        
        return cot_reasoning
        
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
    model, tokenizer = load_cot_model(args)
    
    # Generate chain-of-thought reasoning for each example
    print("Starting chain-of-thought generation...")
    
    for i in range(args.resume_from, len(df)):
        print(f"Generating CoT for example {i+1}/{len(df)}")
        
        # Skip if already generated
        if pd.notna(df.iloc[i]['chain_of_thought']) and df.iloc[i]['chain_of_thought'].strip():
            print(f"  Skipping example {i+1} (already has CoT)")
            continue
        
        example = df.iloc[i].to_dict()
        cot_reasoning = generate_single_cot(model, tokenizer, example, args)
        
        # Update the dataframe
        df.iloc[i, df.columns.get_loc('chain_of_thought')] = cot_reasoning
        
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
