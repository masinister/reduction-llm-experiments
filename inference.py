import os
import argparse
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--held_out_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--test_set", type=str, default="test", choices=["test", "validation", "both"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
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
        model_dir = os.path.dirname(args.model_path.rstrip("/"))
        if model_dir.endswith("final"):
            model_dir = os.path.dirname(model_dir)
        held_out_file = os.path.join(model_dir, "held_out_indices.json")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=True,
    )

    adapter_path = args.adapter_path or args.model_path
    adapter_config_file = os.path.join(adapter_path, "adapter_config.json")

    if os.path.exists(adapter_config_file):
        print(f"Loading LoRA adapters from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        if args.merge_adapters:
            print("Merging adapters...")
            model = model.merge_and_unload()
        else:
            print("⚠️ Adapters loaded but not merged. This may impact inference speed.")
    else:
        print("No adapters found. Using base model only.")

    # Ensure caching is enabled
    model.config.use_cache = True
    model.eval()

    # Final PEFT check
    if isinstance(model, PeftModel) and args.merge_adapters:
        raise RuntimeError("PEFT wrapper still present — merge_and_unload() likely failed.")

    return model, tokenizer


def run_inference(model, tokenizer, dataset, args):
    results = []
    device = next(model.parameters()).device

    for i, example in enumerate(dataset):
        print(f"Example {i+1}/{len(dataset)}")

        prompt = (
            "### Instruction:\nWrite a natural-language LaTeX reduction given source and target.\n"
            f"### Input:\nSource: {example['source_text']}\nTarget: {example['target_text']}\n### Output:\n"
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
            "index": i,
            "source_text": example["source_text"],
            "target_text": example["target_text"],
            "ground_truth_reduction": example.get("reduction_full_text", ""),
            "generated_reduction": generated_reduction,
            "prompt": prompt
        })

    return results


def save_results(results, output_dir, dataset_name):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"inference_results_{dataset_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {output_file}")
    print(f"{sum(1 for r in results if not r['generated_reduction'].startswith('ERROR:'))} succeeded")
    print(f"{sum(1 for r in results if r['generated_reduction'].startswith('ERROR:'))} failed")

    return output_file


def main():
    args = parse_args()

    print("="*50)
    print("INFERENCE SCRIPT FOR FSDP+PEFT MODELS")
    print("="*50)
    print(f"Model path: {args.model_path}")
    print(f"CSV path: {args.csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.model_dtype}")
    print("="*50)

    model, tokenizer = load_model_and_tokenizer(args)
    test_data, val_data, held_out_info = load_test_data(args)

    datasets_to_run = {}
    if args.test_set in ("test", "both"):
        datasets_to_run["test"] = test_data
    if args.test_set in ("validation", "both"):
        datasets_to_run["validation"] = val_data

    output_files = []
    for name, data in datasets_to_run.items():
        print(f"\nRunning inference on {name} set")
        results = run_inference(model, tokenizer, data, args)
        output_file = save_results(results, args.output_dir, name)
        output_files.append(output_file)

    print("\nDone. Output files:")
    for f in output_files:
        print(f"- {f}")


if __name__ == "__main__":
    main()
