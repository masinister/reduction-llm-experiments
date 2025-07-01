"""
Fine-tuning script with support for LoRA, FSDP, and Sequence Parallelism.

Hybrid Parallelism Strategy:
- Sequence parallelism + FSDP enabled by default for multi-GPU setups
- Sequence parallelism handles longer sequences by distributing across sequence dimension
- FSDP provides parameter sharding for memory efficiency
- Use --disable_sequence_parallel to use FSDP only
- Use --disable_fsdp to use sequence parallelism only
- Single GPU uses quantization for memory efficiency
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
try:
    # Try to import LlamaRMSNorm for type checking (optional)
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
except ImportError:
    LlamaRMSNorm = None
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
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from latest DCP checkpoint")
    p.add_argument("--cot", action="store_true", help="Use chain-of-thought data format")
    p.add_argument("--disable_sequence_parallel", action="store_true", help="Disable sequence parallelism")
    p.add_argument("--disable_fsdp", action="store_true", help="Disable FSDP (useful for debugging)")
    p.add_argument("--sp_layers_limit", type=int, default=None, help="Limit SP to first N transformer layers (for testing)")
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
    
    # Enable sequence parallelism by default for multi-GPU setups
    args.sequence_parallel = torch.cuda.device_count() > 1 and not args.disable_sequence_parallel
    
    # Enable FSDP by default for multi-GPU setups (can be combined with sequence parallelism)
    args.use_fsdp = torch.cuda.device_count() > 1 and not args.disable_fsdp
    
    if args.sequence_parallel and args.use_fsdp:
        print(f"🚀 Sequence parallelism + FSDP enabled for {torch.cuda.device_count()} GPUs (hybrid approach)")
        print("   → SP handles long sequences by distributing across sequence dimension")
        print("   → FSDP provides parameter sharding for memory efficiency")
    elif args.sequence_parallel:
        print(f"🚀 Sequence parallelism enabled for {torch.cuda.device_count()} GPUs")
        print("   → Pure SP mode: distributing sequence processing across GPUs")
    elif args.use_fsdp:
        print(f"🚀 FSDP enabled for {torch.cuda.device_count()} GPUs")
        print("   → Parameter sharding without sequence parallelism")
    else:
        print("🔧 Single GPU or distributed training disabled")
    
    if torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            # Initialize with longer timeout to handle evaluation delays
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
    
    # Ensure padding side is left for causal LM training
    tokenizer.padding_side = "right"

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    # Configure quantization for Q-LoRA (only for single GPU)
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
    elif args.sequence_parallel and args.use_fsdp:
        print("Using sequence parallelism + FSDP (quantization disabled for compatibility)")
    elif args.sequence_parallel:
        print("Using sequence parallelism (quantization disabled for compatibility)")
    elif args.use_fsdp:
        print("Using FSDP (quantization disabled for compatibility)")
    else:
        print("Using full precision LoRA")

    # Load the base model with FSDP-compatible settings
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.model_dtype],
        device_map=None,
        trust_remote_code=True,
        quantization_config=bnb_config,
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
    
    # Initialize sequence parallelism if requested and multi-GPU available
    if args.sequence_parallel and torch.cuda.device_count() > 1:
        try:
            print(f"Initializing sequence parallelism with {torch.cuda.device_count()} GPUs...")
            
            # Create device mesh across all GPUs for tensor parallelism
            tp_mesh = init_device_mesh("cuda", (torch.cuda.device_count(),))
            print(f"Device mesh created: {tp_mesh}")
            
            # Get the underlying model from PEFT wrapper
            base_model = model.get_base_model()
            
            # Build comprehensive parallelization plan for all normalization layers
            parallelize_plan = {}
            
            # Use named_modules() to find all normalization layers across the entire model
            for name, module in base_model.named_modules():
                # Method 1: Match by name patterns (works for LLaMA RMSNorm)
                name_match = any(norm_pattern in name.lower() for norm_pattern in [
                    "layer_norm",      # matches input_layer_norm, post_attention_layer_norm, etc.
                    "layernorm", 
                    "rms_norm",
                    "rmsnorm"
                ]) or (name.lower() == "norm" or name.endswith(".norm"))  # final norm layer
                
                # Method 2: Match by module type (fallback for any normalization layer)
                type_match = False
                module_type_name = type(module).__name__.lower()
                if any(norm_type in module_type_name for norm_type in ['norm', 'layernorm', 'rmsnorm']):
                    type_match = True
                
                if name_match or type_match:
                    # Apply layer limit if specified
                    if args.sp_layers_limit is not None:
                        # Extract layer number from path like "model.layers.5.input_layer_norm"
                        if "layers." in name:
                            try:
                                layer_num = int(name.split("layers.")[1].split(".")[0])
                                if layer_num >= args.sp_layers_limit:
                                    continue  # Skip layers beyond limit
                            except (ValueError, IndexError):
                                pass  # If we can't parse layer number, include it anyway
                    
                    parallelize_plan[name] = SequenceParallel()
            
            if args.sp_layers_limit is not None:
                print(f"Layer limit applied: only processing first {args.sp_layers_limit} layers")
            
            print(f"Total normalization layers found: {len(parallelize_plan)}")
            if parallelize_plan:
                print(f"Applying sequence parallelism to {len(parallelize_plan)} normalization layers")
                
                # Apply sequence parallelism to the entire base model with comprehensive plan
                parallelize_module(
                    module=base_model,
                    device_mesh=tp_mesh,
                    parallelize_plan=parallelize_plan
                )
                print("✅ Sequence parallelism applied successfully to all normalization layers")
                
                # Log memory usage after SP setup
                print_memory_usage()
                
            else:
                print("No normalization layers found for sequence parallelism, skipping")
                args.sequence_parallel = False
                
        except Exception as e:
            print(f"Warning: Failed to initialize sequence parallelism: {e}")
            print("Falling back to standard multi-GPU training (FSDP)")
            args.sequence_parallel = False
    
    if torch.cuda.device_count() > 1 and args.use_fsdp:
        # Make sure all LoRA parameters require gradients (FSDP mode)
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
        
        if args.sequence_parallel:
            print("Model configured for sequence parallelism + FSDP hybrid approach")
        else:
            print("Model configured for FSDP only")
    elif args.sequence_parallel and torch.cuda.device_count() > 1:
        # Sequence parallelism only mode
        model.train()
        print("Model configured for sequence parallelism only")
    else:
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
        # Enable FSDP when requested (can be combined with sequence parallelism)
        fsdp="full_shard" if args.use_fsdp else "",
        fsdp_config={
            "use_orig_params": "true",
        } if args.use_fsdp else {},
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
    if torch.cuda.device_count() > 1 and args.use_fsdp:
        mode_desc = "FSDP + Sequence Parallel" if args.sequence_parallel else "FSDP only"
        print(f"Post-trainer initialization gradient check ({mode_desc} mode):")
        grad_params = 0
        total_params = 0
        for name, param in trainer.model.named_parameters():
            total_params += 1
            if param.requires_grad:
                grad_params += 1
                if grad_params <= 3:  # Print first few
                    print(f"  {name}: requires_grad={param.requires_grad}, dtype={param.dtype}")
        print(f"  Total: {grad_params}/{total_params} parameters require gradients")
    elif args.sequence_parallel and torch.cuda.device_count() > 1:
        print("Post-trainer initialization check (Sequence Parallel only mode):")
        print(f"  Model type: {type(trainer.model)}")
        print(f"  Device count: {torch.cuda.device_count()}")
    
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
