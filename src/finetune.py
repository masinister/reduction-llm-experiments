"""
Fine-tuning script with LoRA, FSDP, and Sequence Parallelism.

Hybrid Parallelism Strategy (Clean Wrapping Order):
- REQUIRES multi-GPU setup for SP+FSDP hybrid training
- Step 1: Apply Sequence Parallelism to all normalization layers 
- Step 2: Explicitly wrap model with FSDP using SP-aware auto_wrap_policy
- SP-aware FSDP auto_wrap_policy excludes SP-modified modules
- Prevents PyTorch storage errors through proper module exclusion and clean wrapping order
"""

import os
import argparse
import json
import datetime
import random
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
try:
    # Try to import LlamaRMSNorm for type checking (optional)
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
except ImportError:
    LlamaRMSNorm = None
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import snapshot_download

# Global variable to track SP-modified modules for FSDP auto_wrap_policy
_sp_modified_modules = set()

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


class PerformanceLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.step_times = []
        self.step_start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
            
            # Log every 50 steps and provide running average
            if len(self.step_times) % 50 == 0:
                avg_time = sum(self.step_times[-50:]) / 50
                print(f"Average step time (last 50 steps): {avg_time:.3f}s")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.step_times:
            avg_time = sum(self.step_times) / len(self.step_times)
            print(f"Average step time this epoch: {avg_time:.3f}s")
            self.step_times = []


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
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--model_dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
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

def create_sp_aware_fsdp_policy(min_num_params=1e8):
    """
    Create a custom FSDP auto_wrap_policy that excludes SP-modified modules.
    
    This policy combines size-based wrapping with exclusion of modules that
    have been modified by Sequence Parallelism to avoid PyTorch storage errors.
    
    Args:
        min_num_params: Minimum number of parameters to wrap a module
        
    Returns:
        A custom auto_wrap_policy function
    """
    def sp_aware_policy(module, recurse, nonwrapped_numel):
        # Never wrap SP-modified modules 
        if module in _sp_modified_modules:
            return False
            
        # Use size-based policy for other modules
        return size_based_auto_wrap_policy(module, recurse, nonwrapped_numel, min_num_params)
    
    return sp_aware_policy


def main():
    args = parse_args()
    
    # Require multi-GPU setup for SP+FSDP hybrid training
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"SP+FSDP hybrid training requires at least 2 GPUs, found {torch.cuda.device_count()}")
    
    print(f"🚀 SP+FSDP hybrid training enabled for {torch.cuda.device_count()} GPUs")
    print("   → CLEAN WRAPPING ORDER: SP first, then FSDP")
    print("   → SP handles normalization layers across sequence dimension")
    print("   → FSDP provides parameter sharding with custom auto_wrap_policy")
    
    # Initialize distributed training
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", 
            timeout=datetime.timedelta(seconds=3600)  # 1 hour timeout
        )
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
    tokenizer.padding_side = "right"

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    # Load the base model for SP+FSDP hybrid training
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.model_dtype],
        device_map=None,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False

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
    
    # Initialize sequence parallelism
    print(f"Initializing sequence parallelism with {torch.cuda.device_count()} GPUs...")
    
    # Create device mesh across all GPUs for tensor parallelism
    tp_mesh = init_device_mesh("cuda", (torch.cuda.device_count(),))
    print(f"Device mesh created: {tp_mesh}")
    
    # Get the underlying model from PEFT wrapper
    base_model = model.get_base_model()
    
    # Build parallelization plan for all normalization layers
    parallelize_plan = {}
    
    # Clear the global tracker for SP-modified modules
    global _sp_modified_modules
    _sp_modified_modules.clear()
    
    # Find all normalization layers across the entire model
    for name, module in base_model.named_modules():
        # Match by name patterns (works for LLaMA RMSNorm)
        name_match = any(norm_pattern in name.lower() for norm_pattern in [
            "layer_norm", "layernorm", "rms_norm", "rmsnorm"
        ]) or (name.lower() == "norm" or name.endswith(".norm"))
        
        # Match by module type (fallback for any normalization layer)
        type_match = False
        module_type_name = type(module).__name__.lower()
        if any(norm_type in module_type_name for norm_type in ['norm', 'layernorm', 'rmsnorm']):
            type_match = True
        
        if name_match or type_match:
            parallelize_plan[name] = SequenceParallel()
            # Track this module for FSDP exclusion
            _sp_modified_modules.add(module)
    
    print(f"Found {len(parallelize_plan)} normalization layers for SP")
    if not parallelize_plan:
        raise RuntimeError("No normalization layers found for sequence parallelism")
    
    # Apply sequence parallelism to the base model
    parallelize_module(
        module=base_model,
        device_mesh=tp_mesh,
        parallelize_plan=parallelize_plan
    )
    print("✅ Sequence parallelism applied successfully")
    print(f"📋 Tracked {len(_sp_modified_modules)} SP-modified modules for FSDP exclusion")
    print_memory_usage()
    
    # Ensure all LoRA parameters require gradients for FSDP
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad_(True)
    
    model.train()
    
    # Print trainable parameters for debugging
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Model has {model.num_parameters():,} total parameters")
    print(f"Model has {model.num_parameters(only_trainable=True):,} trainable parameters")

    # CLEAN WRAPPING ORDER: Apply FSDP after sequence parallelism
    print("🔧 Applying FSDP wrapping with SP-aware auto_wrap_policy...")
    print(f"   → Excluding {len(_sp_modified_modules)} SP-modified modules from FSDP wrapping")
    
    # Create SP-aware FSDP auto_wrap_policy
    sp_aware_policy = create_sp_aware_fsdp_policy(min_num_params=1e8)
    
    # Explicitly wrap the model with FSDP after SP is applied
    model = FSDP(
        model,
        auto_wrap_policy=sp_aware_policy,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
    )
    print("✅ FSDP wrapping completed successfully")
    print_memory_usage()

    tokenized_ds = load_and_prepare(args, tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )

    # Configure training args WITHOUT FSDP (since we already wrapped the model)
    print("🎯 Configuring training arguments (FSDP already applied to model)")

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
        # No FSDP config needed - model is already wrapped
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        callbacks=[MemoryLoggerCallback(), AverageTrainLossCallback(), PerformanceLoggerCallback()],
    )
    
    # Debug: Check gradient setup after trainer initialization
    print("Post-trainer initialization gradient check (explicit SP+FSDP hybrid mode):")
    grad_params = 0
    total_params = 0
    for name, param in trainer.model.named_parameters():
        total_params += 1
        if param.requires_grad:
            grad_params += 1
            if grad_params <= 3:  # Print first few
                print(f"  {name}: requires_grad={param.requires_grad}, dtype={param.dtype}")
    print(f"  Total: {grad_params}/{total_params} parameters require gradients")
    print(f"  Model is FSDP-wrapped: {isinstance(trainer.model, FSDP)}")
    
    if args.resume:
        try:
            print("Attempting to resume from last checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        except Exception as e:
            print(f"⚠️ Resume failed: {e}. Starting fresh.")
            trainer.train()
    else:
        trainer.train()
    
    # Final barrier: ensure all operations are complete
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: Waiting for all ranks to complete...")
        try:
            dist.barrier()
            print(f"Rank {dist.get_rank()}: All operations completed successfully!")
            print(f"Rank {dist.get_rank()}: Cleaning up distributed process group...")
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Warning: Failed to cleanup process group: {e}")
        except Exception as e:
            print(f"Rank {dist.get_rank()}: Warning - final barrier failed: {e}")


if __name__ == "__main__":
    main()
