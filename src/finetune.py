import argparse
import os
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import torch

from prompts import setup_tokenizer_with_template, build_conversation_dict

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

    print("Loading dataset...")
    ds_raw = build_train_dataset(args.csv)

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    print("Setting up tokenizer with Alpaca template...")
    tokenizer = setup_tokenizer_with_template(tokenizer)

    print("Formatting dataset...")
    ds = format_dataset_with_template(ds_raw, tokenizer)

    print("Patching model with LoRA...")
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

    print("Starting training...")
    trainer.train()

    print("Saving adapter + tokenizer...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")
