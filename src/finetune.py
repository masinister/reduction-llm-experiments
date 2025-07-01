import os
import argparse
import json
import datetime
import random

import torch
import torch.distributed as dist
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import snapshot_download

class MemoryLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU mem this epoch: {peak:.2f} GB")


class AverageTrainLossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.epoch_train_losses = []
        self.current_epoch_losses = []

    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get('logs', {})
        if 'loss' in logs:
            self.current_epoch_losses.append(logs['loss'])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.current_epoch_losses:
            avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
            self.epoch_train_losses.append(avg_loss)
            print(f"Average training loss this epoch: {avg_loss:.4f}")
            self.current_epoch_losses = []


def print_memory_usage():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory — Allocated: {alloc:.2f} GB, Reserved: {resv:.2f} GB")
    else:
        print("CUDA not available")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./fine-tuned")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--model_dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from latest DCP checkpoint")
    p.add_argument("--cot", action="store_true", help="Use chain-of-thought data format")
    return p.parse_args()

def load_and_prepare(args, tokenizer):
    csv_path = os.path.expanduser(args.csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    raw = load_dataset("csv", data_files=csv_path, split="train")
    if len(raw) < 10:
        raise ValueError("Require at least 10 samples")

    random.seed(42)
    idx = list(range(len(raw)))
    random.shuffle(idx)
    val_idx, test_idx = idx[:5], idx[5:10]
    train_idx = idx[10:]

    os.makedirs(args.output_dir, exist_ok=True)

    held_info = {
        "dataset_size": len(raw),
        "csv_path": csv_path,
        "validation_indices": val_idx,
        "test_indices": test_idx,
        "train_indices": train_idx,
        "split_seed": 42,
        "split_timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "held_out_indices.json"), "w") as f:
        json.dump(held_info, f, indent=2)

    splits = DatasetDict({
        "train": raw.select(train_idx),
        "validation": raw.select(val_idx),
        "test": raw.select(test_idx),
    })

    def tokenize_fn(ex):
        # Check if we have chain-of-thought data
        if args.cot and 'chain_of_thought' in ex and ex['chain_of_thought'] and ex['chain_of_thought'].strip():
            # CoT format: include chain-of-thought reasoning before the reduction
            messages = [
                {
                    "role": "system",
                    "content": "Write a natural-language LaTeX reduction given source and target. Think step by step."
                },
                {
                    "role": "user", 
                    "content": f"Source: {ex['source_text']}\nTarget: {ex['target_text']}"
                },
                {
                    "role": "assistant",
                    "content": f"Let me think through this step by step:\n\n{ex['chain_of_thought']}\n\nTherefore, the reduction is:\n\n{ex['reduction_full_text']}"
                }
            ]
        else:
            # Standard format: direct reduction without chain-of-thought
            messages = [
                {
                    "role": "system",
                    "content": "Write a natural-language LaTeX reduction given source and target."
                },
                {
                    "role": "user", 
                    "content": f"Source: {ex['source_text']}\nTarget: {ex['target_text']}"
                },
                {
                    "role": "assistant",
                    "content": ex["reduction_full_text"]
                }
            ]
        
        # Use chat template to format the conversation
        try:
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False  # Don't add generation prompt for training
            )
        except Exception as e:
            # Fallback to manual formatting if tokenizer doesn't support chat templates
            print(f"Warning: Chat template not supported, falling back to manual formatting: {e}")
            formatted_text = (
                f"System: {messages[0]['content']}\n"
                f"User: {messages[1]['content']}\n"
                f"Assistant: {messages[2]['content']}"
            )
        
        # Tokenize the formatted text with max_length padding
        tok = tokenizer(
            formatted_text, 
            truncation=True, 
            max_length=args.max_length,
            padding="max_length",  # Pad to max_length for consistent tensor sizes
            return_tensors=None  # Return lists, not tensors
        )
        # Ensure labels are properly aligned and same length as input_ids
        tok["labels"] = tok["input_ids"].copy()
        return tok

    # Get column names before mapping to ensure we remove all raw text fields
    columns_to_remove = splits["train"].column_names
    tokenized_splits = splits.map(tokenize_fn, batched=False, remove_columns=columns_to_remove)
    
    # Debug: Print tokenization statistics
    train_lengths = [len(ex['input_ids']) for ex in tokenized_splits["train"]]
    print(f"Tokenization stats - Min: {min(train_lengths)}, Max: {max(train_lengths)}, Avg: {sum(train_lengths)/len(train_lengths):.1f}")
    print(f"All sequences should be exactly max_length ({args.max_length}): {all(l == args.max_length for l in train_lengths)}")
    
    return tokenized_splits


def main():
    args = parse_args()
    
    # torchrun handles distributed initialization automatically
    print_memory_usage()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(args.model_name):
        args.model_name = snapshot_download(repo_id=args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure padding side is left for causal LM training
    tokenizer.padding_side = "right"

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    # Configure quantization for Q-LoRA with optional CPU offload
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype_map[args.model_dtype],
        bnb_4bit_storage_dtype=dtype_map[args.model_dtype],
        llm_int8_threshold=6.0,
    )
    
    # Load the base model with QLoRA quantization enabled
    # CRITICAL: Load on CPU first to avoid DTensor issues with PEFT
    # device_map="auto" creates DTensors that break PEFT's compress_statistics
    model_kwargs = {
        "torch_dtype": dtype_map[args.model_dtype],
        "device_map": {"": "cpu"},  # Load entirely on CPU first
        "trust_remote_code": True,
        "quantization_config": bnb_config,
        "attn_implementation": "sdpa",
    }
    
    # Note: CPU offload not needed since we're loading on CPU initially
    # DDP will handle GPU distribution after PEFT injection
    
    print("Loading model on CPU to avoid DTensor issues with PEFT...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    # Enable gradient checkpointing BEFORE PEFT injection - critical for 70B + 4K context
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False
    
    # Debug check: Verify 4-bit quantization is working
    num_4bit = sum(1 for p in base_model.parameters() if p.dtype == torch.int8)
    print(f"Found {num_4bit} int8 layers (should be >0 for 4-bit quant).")

    # Apply PEFT on CPU where we have real Linear4bit modules (not DTensor proxies)
    print("Applying PEFT/LoRA on CPU...")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=None,
    )
    model = get_peft_model(base_model, peft_cfg)
    
    # DDP will automatically move the model to GPUs during training
    print("PEFT injection complete. DDP will handle GPU distribution.")
    
    if torch.cuda.device_count() > 1:
        print(f"Using DDP with {torch.cuda.device_count()} GPUs and 4-bit quantization")
        # Make sure all LoRA parameters require gradients
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad_(True)
        
        # Ensure the model is in training mode
        model.train()
        
        # Print trainable parameters for debugging
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        print(f"Trainable parameters: {len(trainable_params)}")
        print(f"First few trainable params: {trainable_params[:5]}")
    else:
        print("Using single GPU with 4-bit quantization")
        # Single GPU - just ensure training mode
        model.train()
    
    print(f"Model has {model.num_parameters():,} total parameters")
    print(f"Model has {model.num_parameters(only_trainable=True):,} trainable parameters")

    tokenized_ds = load_and_prepare(args, tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        weight_decay=0.01,
        label_names=["labels"],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=0.3,
        bf16=args.model_dtype == "bfloat16",
        fp16=args.model_dtype == "float16",
        tf32=True,
        # Simplified DDP settings - torchrun handles the rest
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        callbacks=[MemoryLoggerCallback(), AverageTrainLossCallback()],
    )
    # Debug: Check gradient setup after trainer initialization
    if torch.cuda.device_count() > 1:
        print("Post-trainer initialization gradient check:")
        grad_params = 0
        total_params = 0
        for name, param in trainer.model.named_parameters():
            total_params += 1
            if param.requires_grad:
                grad_params += 1
                if grad_params <= 3:  # Print first few
                    print(f"  {name}: requires_grad={param.requires_grad}, dtype={param.dtype}")
        print(f"  Total: {grad_params}/{total_params} parameters require gradients")
    
    if args.resume:
        try:
            print("Attempting to resume from last checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        except Exception as e:
            print(f"⚠️ Resume failed: {e}. Starting fresh.")
            trainer.train()
    else:
        trainer.train()
    
    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
