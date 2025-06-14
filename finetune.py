import os
import argparse
import torch
import json

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import snapshot_download

def print_memory_usage():
    """Print current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print("CUDA not available")

class MemoryLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU mem this epoch: {allocated:.2f} GB")

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
                        help="Path to DeepSpeed config JSON. Options: deepspeed-config.json (balanced), deepspeed-config-nvme.json (max memory), deepspeed-config-profile.json (debugging)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (automatically set by DeepSpeed)")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", 
                        choices=["float16", "bfloat16", "float32"],                        help="Model data type for memory efficiency (default: bfloat16)")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Legacy option - DeepSpeed now handles CPU offloading automatically")
    return parser.parse_args()

def load_and_prepare(args, tokenizer):
    csv_path = os.path.expanduser(args.csv_path)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    raw = load_dataset("csv", data_files=csv_path, split="train")
    # 80/10/10 split
    split1 = raw.train_test_split(test_size=0.2, seed=42)
    split2 = split1['train'].train_test_split(test_size=0.1, seed=42)
    
    # Create both raw and tokenized datasets from the same splits
    raw_dataset = DatasetDict({
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

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=False, remove_columns=raw_dataset['train'].column_names)
    return tokenized_dataset, raw_dataset

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
        
        # Move inputs to the same device as the model
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
    print_memory_usage()

    
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    print("Loading model with DeepSpeed-optimized memory management...")
    
    # Determine the model dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    model_dtype = dtype_map[args.model_dtype]
    print(f"Using model dtype: {args.model_dtype}")
    
    # Load model with memory optimizations and DeepSpeed synergy
    model_kwargs = {
        "device_map": "auto",
        "offload_folder": "offload",
        "offload_state_dict": True,
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Warning: Model does not support gradient checkpointing")
    
    print("Model loaded successfully")
    
    print("Setting up LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("LoRA setup complete")
    
    print_memory_usage()
    
    print("Loading and preparing dataset...")
    tokenized_ds, raw_ds = load_and_prepare(args, tokenizer)
    print(f"Dataset loaded. Train: {len(tokenized_ds['train'])}, Validation: {len(tokenized_ds['validation'])}, Test: {len(tokenized_ds['test'])}")

    # Use the same test split for inference
    original_test_data = raw_ds['test']

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
        save_total_limit=2,  # Reduced from 3 to save disk space
        weight_decay=0.01,
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
        logging_first_step=True,
        # Additional memory savings
        load_best_model_at_end=False,  # Don't load best model to save memory
        metric_for_best_model=None,
        greater_is_better=None,
        # Reduce memory usage during evaluation
        prediction_loss_only=True,
        include_inputs_for_metrics=False
    )
    
    print("Creating trainer with DeepSpeed...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[MemoryLoggerCallback()]
    )

    print("Starting training...")
    trainer.train()
    
    print_memory_usage()
    
    # Save model with safetensors format
    trainer.save_model(os.path.join(args.output_dir, 'final'))
    # Also save the model directly with safetensors for compatibility
    model.save_pretrained(os.path.join(args.output_dir, 'final'), safe_serialization=True)
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final'))
    
    # Run inference on test set
    print("\n" + "="*50)
    print("TRAINING COMPLETE - STARTING INFERENCE")
    print("="*50)
    
    # Use the already trained model for inference (no need to reload)
    print("Using trained model for inference...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    inference_results = run_inference(model, tokenizer, tokenized_ds['test'], original_test_data, args)
    
    print(f"\nInference complete! Results saved to {os.path.join(args.output_dir, 'inference_results.json')}")

    # At the end of training script
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
