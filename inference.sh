#!/bin/bash
#SBATCH --job-name=llama_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --mem=256G
#SBATCH --time=2:00:00

# Enable strict error handling
set -euxo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job details
echo "===== Inference Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "================================="

# Load modules
module load python/3.11.10
module load cuda/12.4.0/3mdaov5
source ~/venvs/reductions/bin/activate

# Parameters from CLI or defaults
MODEL_NAME=${1:-"meta-llama/Llama-3.3-70B-Instruct"}
CSV_PATH=$(eval echo ${2:-"~/data/karp.csv"})
OUTPUT_DIR=${3:-"./llama_finetune"}
INFERENCE_OUTPUT=${4:-"./inference_results"}
TEST_SET=${5:-"test"}

echo ""
echo "Model name: $MODEL_NAME"
echo "CSV path: $CSV_PATH"
echo "Model output dir: $OUTPUT_DIR"
echo "Inference output dir: $INFERENCE_OUTPUT"
echo "Test set: $TEST_SET"
echo ""

# Validate required files
if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: CSV file not found at $CSV_PATH"
  exit 1
fi

HELD_OUT_FILE="$OUTPUT_DIR/held_out_indices.json"
if [[ ! -f "$HELD_OUT_FILE" ]]; then
    echo "ERROR: held_out_indices.json not found at $HELD_OUT_FILE"
    echo "This should have been created during training."
    exit 1
fi
echo "Found held-out indices file."

# Run inference
if [[ -d "$OUTPUT_DIR/merged" ]]; then
    echo "Running inference with merged model..."
    python inference.py \
        --model_path "$OUTPUT_DIR/merged" \
        --csv_path "$CSV_PATH" \
        --output_dir "$INFERENCE_OUTPUT" \
        --test_set "$TEST_SET" \
        --device "auto" \
        --model_dtype "bfloat16" \
        --max_new_tokens 512 \
        --temperature 0.7 \
        --do_sample

elif [[ -d "$OUTPUT_DIR/final" ]]; then
    echo "Merged model not found. Using adapters and merging on the fly..."
    python inference.py \
        --model_path "$OUTPUT_DIR/final" \
        --csv_path "$CSV_PATH" \
        --output_dir "$INFERENCE_OUTPUT" \
        --test_set "$TEST_SET" \
        --device "auto" \
        --model_dtype "bfloat16" \
        --max_new_tokens 512 \
        --temperature 0.7 \
        --do_sample \
        --merge_adapters

else
    echo "ERROR: No trained model found in $OUTPUT_DIR"
    echo "Expected to find either:"
    echo "  - $OUTPUT_DIR/merged"
    echo "  - $OUTPUT_DIR/final"
    exit 1
fi

echo ""
echo "✅ Inference completed at $(date)"
echo "Results saved to: $INFERENCE_OUTPUT/inference_results_${TEST_SET}.json"
