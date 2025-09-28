import argparse
import os
import torch
import torch.distributed as dist
from datasets import load_dataset

from prompts import setup_tokenizer_with_template, build_conversation_dict


def init_distributed():
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/finetuned_model")
    parser.add_argument("--csv", required=True, help="Path to CSV file containing dataset")
    parser.add_argument("--output_dir", default="./inference_outputs", help="Directory to save inference results")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    local_rank, world_size, rank = init_distributed()
    total_ranks = world_size

    def log(message, *, only_main=False):
        if only_main and rank != 0:
            return
        prefix = f"[rank {rank}] " if total_ranks > 1 else ""
        print(prefix + message, flush=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    log("Loading dataset...", only_main=True)
    ds = load_dataset("csv", data_files=args.csv, split="train")
    total_examples = len(ds)

    log("Loading model and tokenizer...", only_main=True)
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_dir,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    log("Setting up tokenizer with Alpaca template...", only_main=True)
    tokenizer = setup_tokenizer_with_template(tokenizer)

    FastLanguageModel.for_inference(model)
    model.eval()

    indices_for_rank = list(range(rank, total_examples, total_ranks))
    log(
        f"Processing {len(indices_for_rank)} of {total_examples} examples on this rank.",
        only_main=total_ranks == 1,
    )

    local_generations = []

    for example_index in indices_for_rank:
        example = ds[example_index]
        source = example.get("source", "")
        target = example.get("target", "")
        source_text = example.get("source_text", "")
        target_text = example.get("target_text", "")

        conv = build_conversation_dict(
            source = source,
            target = target,
            source_text = source_text,
            target_text = target_text,
            reduction_full_text = None,
            include_assistant = False
        )

        prompt_text = tokenizer.apply_chat_template(
            conv,
            tokenize = False,
            add_generation_prompt = True
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids = inputs["input_ids"],
                attention_mask = inputs.get("attention_mask", None),
                max_new_tokens = args.max_new_tokens,
                do_sample = False,
                return_dict_in_generate = True,
                output_scores = False,
            )

        generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        snippet = source_text[:100] + ("..." if len(source_text) > 100 else "")
        target_snippet = target_text[:100] + ("..." if len(target_text) > 100 else "")

        example_number = example_index + 1
        entry = (
            example_index,
            f"\n--- Example {example_number}/{total_examples} ---\n"
            f"Source: {source}\n"
            f"Target: {target}\n"
            f"Source text: {snippet}\n"
            f"Target text: {target_snippet}\n"
            "--- Model output ---\n"
            f"{generated_text}\n"
            "--------------------\n",
        )

        local_generations.append(entry)

    if total_ranks > 1:
        gathered = [None] * total_ranks if rank == 0 else None
        dist.gather_object(local_generations, gathered, dst=0)
        dist.barrier()

        if rank == 0:
            merged = []
            for chunk in gathered:
                if chunk:
                    merged.extend(chunk)

            merged.sort(key=lambda item: item[0])

            output_file = os.path.join(args.output_dir, "inference_results.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                for _, text in merged:
                    print(text, end="", flush=True)
                    f.write(text)

            log(f"\nInference results saved to: {output_file}", only_main=True)
    else:
        output_file = os.path.join(args.output_dir, "inference_results.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for _, text in sorted(local_generations, key=lambda item: item[0]):
                print(text, end="", flush=True)
                f.write(text)

        log(f"\nInference results saved to: {output_file}", only_main=True)

    cleanup_distributed()

if __name__ == "__main__":
    main()
