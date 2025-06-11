import os
import argparse
import torch
import gc
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from huggingface_hub import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on reduction dataset with LoRA and multi-GPU support")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model identifier or HF repo ID, e.g., llama-base or meta-llama/Llama-2-7b-hf")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with reductions")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned",
                        help="Where to save model checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Accumulate gradients to simulate larger batch sizes")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to a DeepSpeed config JSON for ZeRO or FP16")
    return parser.parse_args()

def load_and_prepare(args, tokenizer):
    # Expand the CSV path if it starts with ~
    csv_path = os.path.expanduser(args.csv_path)
    
    # Check if the CSV file exists
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
            "### Instruction:\nWrite a natural-language LaTeX reduction given source and target.\n"
            "### Input:\nSource: {src}\nTarget: {tgt}\n### Output:\n"
        ).format(src=example['source_text'], tgt=example['target_text'])
        full = prompt + example['reduction_full_text']
        tok = tokenizer(full, truncation=True, max_length=args.max_length, padding='max_length')
        tok['labels'] = tok['input_ids'].copy()
        return tok

    tokenized = dataset.map(tokenize_fn, batched=False, remove_columns=dataset['train'].column_names)
    return tokenized

def main():
    args = parse_args()
      # Clear any existing CUDA cache and reset GPU state
    if torch.cuda.is_available():
        # Force cleanup of any existing CUDA contexts
        try:
            # More aggressive cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()  # Clean up inter-process communication
            
            # Reset all devices
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            torch.cuda.set_device(0)  # Reset to device 0
            
            # Additional cleanup
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Warning: CUDA cleanup failed: {e}")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name()}")
            
            # Display GPU memory status
            for i in range(torch.cuda.device_count()):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"GPU {i} - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                except Exception as e:
                    print(f"GPU {i} - Unable to query memory: {e}")
    
    # Add delay to allow GPU state to settle
    import time
    time.sleep(2)
    
    print(f"Starting fine-tuning with the following arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure model is available locally: download from HF if needed
    if not os.path.isdir(args.model_name):
        print(f"Model path '{args.model_name}' not found locally. Downloading from Hugging Face Hub...")
        repo_id = args.model_name
        cache_dir = snapshot_download(repo_id=repo_id)
        args.model_name = cache_dir
        print(f"Downloaded to '{cache_dir}'")
    
    # Launch with torchrun for multi-GPU: torchrun --nproc_per_node=N fine_tune.py ...
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set padding token to EOS token")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory usage
        low_cpu_mem_usage=True,      # Enable memory-efficient loading
    )
    print("Model loaded successfully")
    
    # LoRA setup
    print("Setting up LoRA...")
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout)
    model = get_peft_model(model, peft_config)
    print("LoRA setup complete")

    print("Loading and preparing dataset...")
    tokenized_ds = load_and_prepare(args, tokenizer)
    print(f"Dataset loaded. Train: {len(tokenized_ds['train'])}, Validation: {len(tokenized_ds['validation'])}, Test: {len(tokenized_ds['test'])}")

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
        eval_strategy='epoch',  # Updated from evaluation_strategy
        save_total_limit=3,
        weight_decay=0.01,
        fp16=True,
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, 'final'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final'))

if __name__ == '__main__':
    main()
