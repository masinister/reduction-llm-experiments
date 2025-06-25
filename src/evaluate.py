import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inference results using LLM-as-a-judge")
    parser.add_argument("--inference_results", type=str, required=True,
                        help="Path to inference results CSV file")
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Model to use as judge for evaluation")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length for judge model")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for judge model generation (low for consistency)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation (keep low for large models)")
    return parser.parse_args()


def load_judge_model(args):
    """Load the judge model and tokenizer"""
    print(f"Loading judge model: {args.judge_model}")
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.model_dtype]
    
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=model_dtype,
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=True,
    )
    
    model.config.use_cache = True
    model.eval()
    
    print("✅ Judge model loaded successfully")
    return model, tokenizer


def create_evaluation_prompt(source_text: str, target_text: str, 
                           ground_truth_reduction: str, generated_reduction: str) -> str:
    """Create a prompt for the judge model to evaluate the generated reduction"""
    
    prompt = f"""You are an expert in computational complexity theory and mathematical reductions. Your task is to evaluate whether a generated reduction between two computational problems is correct and complete.

**Source Problem**: {source_text}
**Target Problem**: {target_text}

**Ground Truth Reduction**: {ground_truth_reduction}

**Generated Reduction**: {generated_reduction}

Please evaluate the generated reduction on the following criteria:

1. **Similarity**: Does the reduction match the structure and intent of the ground truth reduction?
2. **Correctness**: Does the reduction correctly transform instances of the source problem to the target problem?

Then provide an overall assessment:
- **ACCEPT**: The generated reduction is substantially correct and complete
- **PARTIAL**: The generated reduction has the right idea but missing key details or has minor errors
- **REJECT**: The generated reduction is fundamentally incorrect or incomplete

"""

    return prompt





def evaluate_single_example(model, tokenizer, example: Dict, args) -> Dict:
    """Evaluate a single inference result using the judge model"""
    
    # Skip examples with generation errors
    if example['generated_reduction'].startswith('ERROR:'):
        return {
            'evaluation': 'ERROR: Generation failed',
        }
    
    prompt = create_evaluation_prompt(
        example['source_text'],
        example['target_text'], 
        example['ground_truth_reduction'],
        example['generated_reduction']
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=args.temperature,
                do_sample=True if args.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return {'evaluation': response}
        
    except Exception as e:
        return {'evaluation': f'ERROR: {str(e)}'}


def load_inference_results(file_path: str) -> pd.DataFrame:
    """Load inference results from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Inference results file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Check for required columns
    required_columns = ['source_text', 'target_text', 'ground_truth_reduction', 'generated_reduction']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in inference results: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        
        # Fill missing columns with empty strings
        for col in missing_columns:
            df[col] = ""
    
    return df


def main():
    args = parse_args()
    
    print("="*60)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("="*60)
    print(f"Inference results: {args.inference_results}")
    print(f"Judge model: {args.judge_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Model dtype: {args.model_dtype}")
    print("="*60)
    
    # Load inference results
    print("Loading inference results...")
    df = load_inference_results(args.inference_results)
    print(f"Loaded {len(df)} inference results")
      # Load judge model
    model, tokenizer = load_judge_model(args)
    
    # Evaluate each example
    print("Starting evaluation...")
    evaluation_results = []
    
    for i, row in df.iterrows():
        print(f"Evaluating example {i+1}/{len(df)}")
        
        eval_result = evaluate_single_example(model, tokenizer, row.to_dict(), args)
        
        # Combine original data with evaluation results
        combined_result = row.to_dict()
        combined_result.update(eval_result)
        evaluation_results.append(combined_result)
    
    # Create results dataframe
    results_df = pd.DataFrame(evaluation_results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract base name from inference results file
    base_name = os.path.splitext(os.path.basename(args.inference_results))[0]
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, f"evaluation_{base_name}.csv")
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_file}")
    print(f"Total examples: {len(df)}")
    print(f"Successfully evaluated: {len(results_df[~results_df['evaluation'].str.startswith('ERROR:', na=False)])}")
    print(f"Errors: {len(results_df[results_df['evaluation'].str.startswith('ERROR:', na=False)])}")
    print("="*60)


if __name__ == "__main__":
    main()