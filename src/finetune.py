import argparse
import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from prompts import setup_tokenizer_with_template, build_conversation_dict


def init_distributed():
    """Initialise torch.distributed if launched under torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and dist.is_available():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        rank = dist.get_rank()
    else:
        rank = 0

    return local_rank, world_size, rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def build_train_dataset(csv_path):
    ds = load_dataset("csv", data_files=csv_path, split="train")

    def make_convo(ex):
        conv = build_conversation_dict(
            source = ex.get("source",""),
            target = ex.get("target",""),
            source_text = ex.get("source_text",""),
            target_text = ex.get("target_text",""),
            reduction_full_text = ex.get("reduction_full_text",""),
            include_assistant = True
        )
        return {"conversations": conv}

    ds = ds.map(make_convo, remove_columns=ds.column_names)

    return ds

def format_dataset_with_template(ds, tokenizer):
    def apply_template(examples):
        texts = []
        for conv in examples["conversations"]:
            txt = tokenizer.apply_chat_template(
                conv,
                tokenize = False,
                add_generation_prompt = False
            )
            texts.append(txt)
        return {"text": texts}

    ds2 = ds.map(apply_template, batched=True, remove_columns=["conversations"])
    return ds2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    parser.add_argument("--csv", required=True, help="Path to karp.csv")
    parser.add_argument("--output_dir", default="outputs/karp_alpaca_lora")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()

    local_rank, world_size, rank = init_distributed()
    is_main_process = rank == 0

    def log(message):
        if is_main_process:
            print(message, flush=True)

    log("Loading dataset...")
    ds_raw = build_train_dataset(args.csv)

    log("Loading model and tokenizer...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    log("Setting up tokenizer with Alpaca template...")
    tokenizer = setup_tokenizer_with_template(tokenizer)

    log("Formatting dataset...")
    ds = format_dataset_with_template(ds_raw, tokenizer)

    log("Patching model with LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = [
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
        ],
        lora_alpha = args.lora_r,
        lora_dropout = 0.0,
        bias = "none",
        use_gradient_checkpointing = False,
        random_state = 3407,
        use_rslora = False,
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_train_epochs = args.num_train_epochs,
        logging_steps = 10,
        save_strategy = "epoch",
        fp16 = False,
        bf16 = False,
        evaluation_strategy = "no",
        learning_rate = 2e-4,
        weight_decay = 0.0,
        ddp_find_unused_parameters = False if world_size > 1 else None,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 1,
        packing = False,
        args = training_args,
    )

    log("Starting training...")
    trainer.train()

    if trainer.is_world_process_zero():
        log("Saving adapter + tokenizer...")
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        log(f"Model saved to {args.output_dir}")

    cleanup_distributed()
