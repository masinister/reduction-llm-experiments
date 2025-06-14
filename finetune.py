import os
import argparse
import torch
import json
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on reduction dataset with LoRA and multi-GPU support")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model identifier or HF repo ID, e.g., llama-base or meta-llama/Llama-2-7b-hf")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with reductions")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned",
                        help="Where to save model checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device during training (set to 1 for memory efficiency)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size per device during evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients to simulate larger batch sizes (effective batch = batch_size * accumulation * num_gpus)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Reduced default epochs for memory efficiency")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Reduced max length for memory efficiency")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to a DeepSpeed config JSON for ZeRO or FP16")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (automatically set by DeepSpeed)")
    return parser.parse_args()

def load_and_prepare(args, tokenizer):
    csv_path = os.path.expanduser(args.csv_path)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    raw = load_dataset("csv", data_files=csv_path, split="train")
    # 80/10/10 split
    split1 = raw.train_test_split(test_size=0.2, seed=42)
    split2 = split1['train'].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({
        'train': split2['train'],
        'validation': split2['test'],
        'test': split1['test']
    })

    def tokenize_fn(example):
        prompt = (
            "### Instruction:\\nWrite a natural-language LaTeX reduction given source and target.\\n"
            "### Input:\\nSource: {src}\\nTarget: {tgt}\\n### Output:\\n"
        ).format(src=example['source_text'], tgt=example['target_text'])
        full = prompt + example['reduction_full_text']
        tok = tokenizer(full, truncation=True, max_length=args.max_length, padding=False)
        tok['labels'] = tok['input_ids'].copy()
        return tok

    tokenized = dataset.map(tokenize_fn, batched=False, remove_columns=dataset['train'].column_names)
    return tokenized

def run_inference(model, tokenizer, test_dataset, original_test_data, args):
    """Run inference on the test set and save results to JSON"""
    print("Starting inference on test set...")
    
    model.eval()
    results = []
    
    for i, (test_example, original_example) in enumerate(zip(test_dataset, original_test_data)):
        print(f"Processing test example {i+1}/{len(test_dataset)}")
        
        # Create the prompt (same format as training)
        prompt = (
            "### Instruction:\nWrite a natural-language LaTeX reduction given source and target.\n"
            "### Input:\nSource: {src}\nTarget: {tgt}\n### Output:\n"
        ).format(src=original_example['source_text'], tgt=original_example['target_text'])
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the prompt)
        generated_reduction = generated_text[len(prompt):].strip()
        
        # Store the result
        result = {
            "index": i,
            "source_text": original_example['source_text'],
            "target_text": original_example['target_text'],
            "ground_truth_reduction": original_example['reduction_full_text'],
            "generated_reduction": generated_reduction,
            "prompt": prompt
        }
        results.append(result)
    
    # Save results to JSON file
    output_file = os.path.join(args.output_dir, 'inference_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Inference results saved to: {output_file}")
    print(f"Generated predictions for {len(results)} test examples")
    
    return results

def setup_deepspeed_config(args):
    """Setup DeepSpeed configuration with ZeRO-3 if no config provided"""
    if args.deepspeed_config is None:
        # Use the default ZeRO-3 config
        default_config = "deepspeed-config.json"
        if os.path.exists(default_config):
            args.deepspeed_config = default_config
            print(f"Using default DeepSpeed config: {default_config}")
        else:
            print("Warning: No DeepSpeed config found. Training without DeepSpeed.")
    else:
        print(f"Using provided DeepSpeed config: {args.deepspeed_config}")
    
    return args

def main():
    args = parse_args()
    
    # Setup DeepSpeed configuration
    args = setup_deepspeed_config(args)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    print(f"Starting fine-tuning with the following arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure model is available locally: download from HF if needed
    if not os.path.isdir(args.model_name):
        print(f"Model path '{args.model_name}' not found locally. Downloading from Hugging Face Hub...")
        repo_id = args.model_name
        cache_dir = snapshot_download(repo_id=repo_id)
        args.model_name = cache_dir
        print(f"Downloaded to '{cache_dir}'")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
      # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")
    
    print("Model loaded successfully")
        
    print("Setting up LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("LoRA setup complete")

    print("Loading and preparing dataset...")
    tokenized_ds = load_and_prepare(args, tokenizer)
    print(f"Dataset loaded. Train: {len(tokenized_ds['train'])}, Validation: {len(tokenized_ds['validation'])}, Test: {len(tokenized_ds['test'])}")

    # Keep original test data for inference
    csv_path = os.path.expanduser(args.csv_path)
    raw = load_dataset("csv", data_files=csv_path, split="train")
    split1 = raw.train_test_split(test_size=0.2, seed=42)
    original_test_data = split1['test']

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_strategy='epoch',
        eval_strategy='epoch',
        save_total_limit=3,
        weight_decay=0.01,
        fp16=True,
        deepspeed=args.deepspeed_config,
        label_names=["labels"],
        # Additional memory optimizations
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        optim="adamw_torch_fused",  # More memory efficient optimizer
        max_grad_norm=1.0,
        # Reduce eval frequency to save memory
        eval_steps=500,
        logging_first_step=True
    )

    print("Creating trainer with DeepSpeed...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        data_collator=data_collator,
        processing_class=tokenizer
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, 'final'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final'))

    # Run inference on test set
    print("\n" + "="*50)
    print("TRAINING COMPLETE - STARTING INFERENCE")
    print("="*50)
    
    # Load the final trained model for inference
    final_model_path = os.path.join(args.output_dir, 'final')
    print(f"Loading final model from: {final_model_path}")
    
    # For inference, we need to load the model in inference mode
    inference_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    inference_model = get_peft_model(inference_model, peft_config)
    inference_model.load_adapter(final_model_path, adapter_name="default")
    
    # Run inference
    inference_results = run_inference(inference_model, tokenizer, tokenized_ds['test'], original_test_data, args)
    
    print(f"\nInference complete! Results saved to {os.path.join(args.output_dir, 'inference_results.json')}")

if __name__ == '__main__':
    main()
