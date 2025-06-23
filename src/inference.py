import os
import argparse
import torch
import json
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned model from finetune.py")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the output directory from finetune.py")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name/path (same as used in finetune.py)")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--held_out_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--merge_adapters", action="store_true",
                        help="Merge LoRA adapters into base model")
    return parser.parse_args()


def load_test_data(args):
    csv_path = os.path.expanduser(args.csv_path)

    if args.held_out_file is None:
        # The held_out_indices.json should be in the same directory as the model
        held_out_file = os.path.join(args.model_path, "held_out_indices.json")
    else:
        held_out_file = os.path.expanduser(args.held_out_file)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(held_out_file):
        raise FileNotFoundError(f"Held-out indices not found: {held_out_file}")

    with open(held_out_file, "r") as f:
        held_out_info = json.load(f)

    raw = load_dataset("csv", data_files=csv_path, split="train")

    if len(raw) != held_out_info["dataset_size"]:
        raise ValueError("Dataset size mismatch")

    test_data = raw.select(held_out_info["test_indices"])
    val_data = raw.select(held_out_info["validation_indices"])

    return test_data, val_data, held_out_info

def load_model_and_tokenizer(args):
    print(f"Loading model from {args.model_path}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.model_dtype]

    # Find the latest checkpoint subdirectory
    checkpoint_dirs = [d for d in os.listdir(args.model_path) if d.startswith("checkpoint-")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {args.model_path}")

    # Use the latest checkpoint
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
    checkpoint_path = os.path.join(args.model_path, latest_checkpoint)
    print(f"Loading from latest checkpoint: {checkpoint_path}")

    # Load tokenizer from the checkpoint directory
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    print(f"Loading base model: {args.base_model}")

    # Load base model first
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=model_dtype,
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=True,
    )

    # Load PEFT adapters on top of base model
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=model_dtype,
        device_map="auto" if args.device == "auto" else None,
    )
    print("✅ Loaded PEFT model from checkpoint")

    # Merge adapters if requested
    if args.merge_adapters:
        print("Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
        print("✅ Adapters merged")

    model.config.use_cache = True
    model.eval()
    return model, tokenizer



def run_inference(model, tokenizer, dataset, dataset_name, args):
    results = []
    device = next(model.parameters()).device

    for i, example in enumerate(dataset):
        print(f"{dataset_name} example {i+1}/{len(dataset)}")

        messages = [
            {
                "role": "system",
                "content": "Write a natural-language LaTeX reduction given source and target."
            },
            {
                "role": "user", 
                "content": f"Source: {example['source_text']}\nTarget: {example['target_text']}"
            }
        ]
        
        try:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback to manual formatting if tokenizer doesn't support chat templates
            print(f"Warning: Chat template not supported, falling back to manual formatting: {e}")
            prompt = (
                f"System: {messages[0]['content']}\n"
                f"User: {messages[1]['content']}\n"
                f"Assistant: "
            )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_reduction = decoded[len(prompt):].strip()
        except Exception as e:
            generated_reduction = f"ERROR: {str(e)}"

        results.append({
            "dataset": dataset_name,
            "index": i,
            "source_text": example["source_text"],
            "target_text": example["target_text"],
            "ground_truth_reduction": example.get("reduction_full_text", ""),
            "generated_reduction": generated_reduction,
            "prompt": prompt
        })

    return results


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "inference_results.csv")

    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\nSaved results to {output_file}")
    
    # Print summary by dataset
    for dataset_name in df['dataset'].unique():
        dataset_results = df[df['dataset'] == dataset_name]
        succeeded = len(dataset_results[~dataset_results['generated_reduction'].str.startswith('ERROR:', na=False)])
        failed = len(dataset_results[dataset_results['generated_reduction'].str.startswith('ERROR:', na=False)])
        print(f"{dataset_name}: {succeeded} succeeded, {failed} failed")
    
    total_succeeded = len(df[~df['generated_reduction'].str.startswith('ERROR:', na=False)])
    total_failed = len(df[df['generated_reduction'].str.startswith('ERROR:', na=False)])
    print(f"Total: {total_succeeded} succeeded, {total_failed} failed")

    return output_file


def main():
    args = parse_args()

    print("="*50)
    print("INFERENCE SCRIPT FOR FINETUNE.PY OUTPUTS")
    print("="*50)
    print(f"Model path: {args.model_path}")
    print(f"Base model: {args.base_model}")
    print(f"CSV path: {args.csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.model_dtype}")
    print("="*50)
    
    model, tokenizer = load_model_and_tokenizer(args)
    test_data, val_data, held_out_info = load_test_data(args)

    # Always run inference on both test and validation sets
    print("\nRunning inference on test set...")
    test_results = run_inference(model, tokenizer, test_data, "test", args)
    
    print("\nRunning inference on validation set...")
    val_results = run_inference(model, tokenizer, val_data, "validation", args)
    
    # Combine all results
    all_results = test_results + val_results
    
    # Save combined results to single file
    output_file = save_results(all_results, args.output_dir)

    print(f"\nDone. Output file: {output_file}")


if __name__ == "__main__":
    main()
