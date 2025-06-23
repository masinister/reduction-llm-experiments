#!/bin/bash

# Model configuration
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
JUDGE_MODEL="meta-llama/Llama-3.3-70B-Instruct"

# Data paths
CSV_PATH="~/data/karp.csv"

# Output directories
OUTPUT_DIR="./llama_finetune"
INFERENCE_OUTPUT="./inference_results"
EVAL_OUTPUT="./evaluation_results"

# Training parameters
BATCH_SIZE=1
GRAD_ACCUM=16
LEARNING_RATE=1e-4
EPOCHS=30
MAX_LENGTH=2048

# LoRA parameters
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Evaluation parameters
EVAL_MAX_LENGTH=4096
