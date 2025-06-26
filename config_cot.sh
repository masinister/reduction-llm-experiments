#!/bin/bash

# Configuration for Chain-of-Thought (CoT) enhanced training
# Use this config with: ./run.sh config_cot.sh --cot

# Model configuration
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
JUDGE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"

# Data paths - Point to CoT-enhanced dataset
CSV_PATH="~/data/karp_cot.csv"  # Generated using run_cot_generation.sh

# Output directories with CoT suffix for clarity
OUTPUT_DIR="./llama_finetune_cot"
INFERENCE_OUTPUT="./inference_results_cot"
EVAL_OUTPUT="./evaluation_results_cot"

# Training parameters - Increased for CoT reasoning
BATCH_SIZE=2  # Smaller batch due to longer sequences
GRAD_ACCUM=32  # Increased accumulation to maintain effective batch size
LEARNING_RATE=5e-6  # Slightly lower for better reasoning training
EPOCHS=20  # Fewer epochs may suffice with enhanced data
MAX_LENGTH=8192  # Increased for reasoning chains

# LoRA parameters - Slightly higher for reasoning capabilities
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Evaluation parameters - Increased for reasoning evaluation
EVAL_MAX_LENGTH=8192
