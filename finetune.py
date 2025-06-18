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

# Memory logging callback remains unchanged
class MemoryLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU mem this epoch: {peak:.2f} GB")


def print_memory_usage():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory — Allocated: {alloc:.2f} GB, Reserved: {resv:.2f} GB")
    else:
        print("CUDA not available")

def evaluate_on_test_set(model, tokenizer, raw_dataset, test_indices, args):
    """
    Evaluate the model on the test set and save outputs to JSON.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        raw_dataset: The original dataset (before tokenization)
        test_indices: List of test set indices
        args: Training arguments containing output_dir and max_length
    
    Returns:
        str: Path to the saved evaluation results JSON file
    """
    print(f"Evaluating model on {len(test_indices)} test samples...")
    
    # Set model to eval mode
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for idx in test_indices:
            sample = raw_dataset[idx]
            
            # Create the same prompt format used during training
            prompt = (
                "### Instruction:\nWrite a natural-language LaTeX reduction given source and target.\n"
                f"### Input:\nSource: {sample['source_text']}\nTarget: {sample['target_text']}\n### Output:\n"
            )
            
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            
            # Move to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Allow for reasonable response length
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode the generated text (removing the input prompt)
            prompt_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][prompt_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Store the result
            result = {
                "sample_index": idx,
                "source_text": sample["source_text"],
                "target_text": sample["target_text"],
                "ground_truth_reduction": sample["reduction_full_text"],
                "generated_reduction": generated_text.strip(),
                "prompt": prompt,
            }
            results.append(result)
            
            print(f"Processed sample {len(results)}/{len(test_indices)}")
    
    # Save results to JSON file
    eval_results_path = os.path.join(args.output_dir, "test_set_evaluation.json")
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_timestamp": datetime.datetime.now().isoformat(),
            "model_name": args.model_name,
            "test_set_size": len(test_indices),
            "test_indices": test_indices,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to: {eval_results_path}")
    return eval_results_path

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
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--model_dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from latest DCP checkpoint")
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
        prompt = (
            "### Instruction:\nWrite a natural-language LaTeX reduction given source and target.\n"
            f"### Input:\nSource: {ex['source_text']}\nTarget: {ex['target_text']}\n### Output:\n"
        )
        full = prompt + ex["reduction_full_text"]
        tok = tokenizer(full, truncation=True, max_length=args.max_length)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized_splits = splits.map(tokenize_fn, batched=False, remove_columns=splits["train"].column_names)
    return tokenized_splits, raw, test_idx


def main():
    args = parse_args()
    
    if torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        # Set the device for this process
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    print_memory_usage()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(args.model_name):
        args.model_name = snapshot_download(repo_id=args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    # Configure quantization for Q-LoRA (only for single GPU to avoid FSDP conflicts)
    use_quantization = torch.cuda.device_count() == 1
    bnb_config = None
    
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype_map[args.model_dtype],
            bnb_4bit_storage_dtype=dtype_map[args.model_dtype],
        )
        print("Using Q-LoRA with 4-bit quantization (single GPU mode)")
    else:
        print("Using full precision LoRA (multi-GPU FSDP mode - quantization disabled)")

    # Load the base model with FSDP-compatible settings
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.model_dtype],
        device_map=None,
        trust_remote_code=True,
        quantization_config=bnb_config,  # Only use quantization for single GPU
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False    # Apply LoRA directly to the base model - let TrainingArguments handle FSDP
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # Additional settings for better FSDP compatibility
        modules_to_save=None,  # Don't save additional modules to avoid conflicts
    )
    model = get_peft_model(base_model, peft_cfg)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    print(f"Model has {model.num_parameters():,} total parameters")
    print(f"Model has {model.num_parameters(only_trainable=True):,} trainable parameters")

    tokenized_ds, raw_dataset, test_indices = load_and_prepare(args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
        fsdp="full_shard auto_wrap" + (" offload" if args.cpu_offload else "") if torch.cuda.device_count() > 1 else "",
        fsdp_config={
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "false", 
            "use_orig_params": "true",  # Important for LoRA compatibility with FSDP
        } if torch.cuda.device_count() > 1 else {},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        callbacks=[MemoryLoggerCallback()],
    )
    
    if args.resume:
        try:
            print("Attempting to resume from last checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        except Exception as e:
            print(f"⚠️ Resume failed: {e}. Starting fresh.")
            trainer.train()
    else:
        trainer.train()
        
    # Evaluate model on test set after saving
    print("Evaluating model on held-out test set...")
    evaluate_on_test_set(model, tokenizer, raw_dataset, test_indices, args)
        
    print_memory_usage()

    print("Saving final checkpoint...")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        trainer.save_state()
        tokenizer.save_pretrained(args.output_dir)
        print("Final checkpoint saved to:", args.output_dir)

if __name__ == "__main__":
    main()
