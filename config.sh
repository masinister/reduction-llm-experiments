#!/bin/bash

# Model configuration
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
JUDGE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"

# Data paths
CSV_PATH="~/data/karp.csv"

# Output directories
OUTPUT_DIR="./llama_finetune"
INFERENCE_OUTPUT="./inference_results"
EVAL_OUTPUT="./evaluation_results"

# Training parameters
BATCH_SIZE=4
GRAD_ACCUM=16
LEARNING_RATE=1e-5
EPOCHS=30
MAX_LENGTH=4096

# LoRA parameters
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Evaluation parameters
EVAL_MAX_LENGTH=4096
