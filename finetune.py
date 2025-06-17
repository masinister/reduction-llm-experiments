import os
import argparse
import json
import datetime
import random

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import snapshot_download
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
import torch.distributed.checkpoint as dcp
from accelerate.utils import merge_fsdp_weights


# Memory logging callback remains unchanged
class MemoryLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU mem this epoch: {peak:.2f} GB")

# Stateful wrapper for model (and optimizer if used)
class AppState(Stateful):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict.get("model"),
            optim_state_dict=state_dict.get("optim", None),
        )

def print_memory_usage():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory — Allocated: {alloc:.2f} GB, Reserved: {resv:.2f} GB")
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
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "hybrid_shard", "no_shard"],
    )
    p.add_argument("--fsdp_mixed_precision", action="store_true", default=True)
    p.add_argument("--fsdp_activation_checkpointing", action="store_true", default=True)
    p.add_argument("--fsdp_auto_wrap", action="store_true", default=True)
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

    return splits.map(tokenize_fn, batched=False, remove_columns=splits["train"].column_names)


def main():
    args = parse_args()
    print_memory_usage()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(args.model_name):
        args.model_name = snapshot_download(repo_id=args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    # 1) Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.model_dtype],
        device_map=None,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # 2) Wrap in FSDP (if >1 GPU)
    fsdp_cfg = {
        "sharding_strategy":             args.fsdp_sharding_strategy,
        "mixed_precision":               args.fsdp_mixed_precision,
        "cpu_offload":                   args.cpu_offload,
        "activation_checkpointing":      args.fsdp_activation_checkpointing,
        "auto_wrap_policy":              transformer_auto_wrap_policy if args.fsdp_auto_wrap else None,
    }
    wrapped_model = (
        FSDP(base_model, **fsdp_cfg)
        if torch.cuda.device_count() > 1
        else base_model.to("cuda")
    )

    # 3) Now apply LoRA
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(wrapped_model, peft_cfg)

    tokenized_ds = load_and_prepare(args, tokenizer)
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
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        fsdp=args.fsdp_sharding_strategy if fsdp_cfg else None,
        fsdp_config=fsdp_cfg,
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
        ckpt_dir = os.path.join(args.output_dir, "dcp_ckpt")
        try:
            print(f"Resuming from {ckpt_dir}")
            app_state = AppState(trainer.model, trainer.optimizer)
            dcp.load({"app": app_state}, checkpoint_id=ckpt_dir)
        except Exception as e:
            print(f"⚠️ Resume failed: {e}. Starting fresh.")

    trainer.train()
    print_memory_usage()

    # DCP-based saving
    final_ckpt = os.path.join(args.output_dir, "dcp_ckpt")
    app_state = AppState(trainer.model, trainer.optimizer)
    print(f"Saving checkpoint via DCP to {final_ckpt}...")
    dcp.save({"app": app_state}, checkpoint_id=final_ckpt)

    print("Merging FSDP shards into a single model...")
    if trainer.args.local_rank == 0:
        merge_fsdp_weights(
            checkpoint_dir=os.path.join(args.output_dir, "dcp_ckpt"),
            output_path=os.path.join(args.output_dir, "merged"),
            safe_serialization=True,
            remove_checkpoint_dir=False
        )
    torch.distributed.barrier()
    print("✔️ Merged model available at:", os.path.join(args.output_dir, "merged"))

    print("Saving final model and tokenizer...")
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    if trainer.args.local_rank == 0:
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print("Done! Final model at:", final_dir)

if __name__ == "__main__":
    main()
