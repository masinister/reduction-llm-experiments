import unsloth
from unsloth import FastLanguageModel
import argparse
import os
import torch
from datasets import load_dataset
from transformers import TextStreamer

from prompts import setup_tokenizer_with_template, build_conversation_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/finetuned_model")
    parser.add_argument("--csv", required=True, help="Path to CSV file containing dataset")
    parser.add_argument("--output_dir", default="./inference_outputs", help="Directory to save inference results")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    ds = load_dataset("csv", data_files=args.csv, split="train")

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_dir,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )

    print("Setting up tokenizer with Alpaca template...")
    tokenizer = setup_tokenizer_with_template(tokenizer)

    FastLanguageModel.for_inference(model)

    print(f"Running inference on {len(ds)} examples...")
    
    # Open output file for writing results
    output_file = os.path.join(args.output_dir, "inference_results.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            source = example.get("source", "")
            target = example.get("target", "")
            source_text = example.get("source_text", "")
            target_text = example.get("target_text", "")

            print(f"\n--- Example {i+1}/{len(ds)} ---")
            print(f"Source: {source}")
            print(f"Target: {target}")
            
            # Write to file as well
            f.write(f"\n--- Example {i+1}/{len(ds)} ---\n")
            f.write(f"Source: {source}\n")
            f.write(f"Target: {target}\n")
            f.write(f"Source text: {source_text[:100]}...\n")
            f.write(f"Target text: {target_text[:100]}...\n")

            conv = build_conversation_dict(
                source = source,
                target = target,
                source_text = source_text,
                target_text = target_text,
                reduction_full_text = None,
                include_assistant = False
            )

            prompt_text = tokenizer.apply_chat_template(
                conv,
                tokenize = False,
                add_generation_prompt = True
            )

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                               padding=True).to(model.device)

            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            print("--- Model output ---")
            f.write("--- Model output ---\n")
            
            # Capture the generated output
            with torch.no_grad():
                output = model.generate(
                    input_ids = inputs["input_ids"],
                    attention_mask = inputs.get("attention_mask", None),
                    max_new_tokens = args.max_new_tokens,
                    do_sample = False,
                    return_dict_in_generate = True,
                    output_scores = False,
                )
            
            # Decode the generated text (excluding the prompt)
            generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(generated_text)
            f.write(generated_text + "\n")
            print("--------------------")
            f.write("--------------------\n")
    
    print(f"\nInference results saved to: {output_file}")

if __name__ == "__main__":
    main()
