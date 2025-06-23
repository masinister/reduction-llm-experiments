#!/bin/bash
#SBATCH --job-name=llama_inference
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --mem=256G
#SBATCH --time=12:00:00

# Usage: ./inference.sh BASE_MODEL CSV_PATH MODEL_PATH INFERENCE_OUTPUT MAX_LENGTH
# All parameters are required
# 
# BASE_MODEL: The base model name used in finetune.py (e.g., meta-llama/Llama-3.3-70B-Instruct)
# CSV_PATH: Path to the CSV dataset file
# MODEL_PATH: Path to the output directory from finetune.py (contains trainer_state.json and checkpoints)
# INFERENCE_OUTPUT: Directory to save inference results
# MAX_LENGTH: Maximum sequence length for tokenization (should match finetune.py value)

# Enable strict error handling
set -euxo pipefail

# Print job details
echo "===== Inference Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "================================="

# Load modules
module load cuda/12.6.3/5fe76nu
module load python/3.11.10
source ~/venvs/reductions/bin/activate

# Parameters from CLI (required)
BASE_MODEL=${1}
CSV_PATH=$(eval echo ${2})
MODEL_PATH=${3}
INFERENCE_OUTPUT=${4}
MAX_LENGTH=${5}

echo ""
echo "Base model: $BASE_MODEL"
echo "CSV path: $CSV_PATH"
echo "Model path (finetune.py output): $MODEL_PATH"
echo "Inference output dir: $INFERENCE_OUTPUT"
echo "Max length: $MAX_LENGTH"
echo ""

# Validate required files
if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: CSV file not found at $CSV_PATH"
  exit 1
fi

HELD_OUT_FILE="$MODEL_PATH/held_out_indices.json"
if [[ ! -f "$HELD_OUT_FILE" ]]; then
    echo "ERROR: held_out_indices.json not found at $HELD_OUT_FILE"
    echo "This should have been created during training."
    exit 1
fi
echo "Found held-out indices file."

# Check if trainer_state.json exists (indicates finetune.py output)
TRAINER_STATE_FILE="$MODEL_PATH/trainer_state.json"
if [[ ! -f "$TRAINER_STATE_FILE" ]]; then
    # Find the latest (largest) checkpoint directory
    LATEST_CKPT=$(find "$MODEL_PATH" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
    if [[ -z "$LATEST_CKPT" ]]; then
        echo "ERROR: No checkpoint directories found in $MODEL_PATH."
        exit 1
    fi
    TRAINER_STATE_FILE="$LATEST_CKPT/trainer_state.json"
    if [[ ! -f "$TRAINER_STATE_FILE" ]]; then
        echo "ERROR: trainer_state.json not found in latest checkpoint directory: $LATEST_CKPT."
        echo "This script requires models trained with finetune.py"
        echo ""
        echo "Debugging information:"
        echo "Contents of $MODEL_PATH:"
        ls -la "$MODEL_PATH" 2>/dev/null || echo "Directory $MODEL_PATH does not exist"
        echo ""
        echo "Looking for checkpoint directories:"
        find "$MODEL_PATH" -name "checkpoint-*" -type d 2>/dev/null || echo "No checkpoint directories found"
        echo ""
        echo "This usually means:"
        echo "1. Fine-tuning job hasn't completed yet"
        echo "2. Fine-tuning job failed"
        echo "3. Output directory path mismatch"
        echo ""
        exit 1
    fi
    echo "Found trainer state file in latest checkpoint directory: $TRAINER_STATE_FILE - confirmed finetune.py output."
else
    echo "Found trainer state file - confirmed finetune.py output."
fi

# Run inference with the new simplified approach
echo "Running inference with PEFT model from finetune.py output..."
python src/inference.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --csv_path "$CSV_PATH" \
    --output_dir "$INFERENCE_OUTPUT" \
    --device "auto" \
    --model_dtype "bfloat16" \
    --max_length "$MAX_LENGTH" \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --do_sample

echo ""
echo "✅ Inference completed at $(date)"
echo "Results saved to: $INFERENCE_OUTPUT/inference_results.csv"
