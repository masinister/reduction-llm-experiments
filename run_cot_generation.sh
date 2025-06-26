#!/bin/bash

# Simple wrapper for chain-of-thought generation
# Usage: ./run_cot_generation.sh [INPUT_CSV] [OUTPUT_CSV] [COT_MODEL] [MAX_LENGTH]
# All parameters are optional and have defaults

# Default parameters
DEFAULT_INPUT_CSV="~/data/karp.csv"
DEFAULT_OUTPUT_CSV="../data/karp_cot.csv"
DEFAULT_COT_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"
DEFAULT_MAX_LENGTH=32768

# Use provided parameters or defaults
INPUT_CSV=${1:-$DEFAULT_INPUT_CSV}
OUTPUT_CSV=${2:-$DEFAULT_OUTPUT_CSV}
COT_MODEL=${3:-$DEFAULT_COT_MODEL}
MAX_LENGTH=${4:-$DEFAULT_MAX_LENGTH}

echo "🧠 Chain-of-Thought Generation for Karp Dataset"
echo "=============================================="
echo "Input CSV: $INPUT_CSV"
echo "Output CSV: $OUTPUT_CSV"
echo "CoT Model: $COT_MODEL"
echo "Max Length: $MAX_LENGTH"
echo "=============================================="

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/cot_generation_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "📁 Log directory: $LOG_DIR"
echo ""

# Submit the CoT generation job
echo "🚀 Submitting chain-of-thought generation job..."
COT_JOB_ID=$(sbatch --parsable \
    --output="${LOG_DIR}/cot_generation_%j.out" \
    --error="${LOG_DIR}/cot_generation_%j.err" \
    scripts/generate_cot.sh "$INPUT_CSV" "$OUTPUT_CSV" "$COT_MODEL" "$MAX_LENGTH")

echo "CoT generation job ID: $COT_JOB_ID"
echo ""
echo "This job will:"
echo "  1. 📖 Load the original Karp dataset from $INPUT_CSV"
echo "  2. 🧠 Generate chain-of-thought reasoning for each reduction using $COT_MODEL"
echo "  3. 💾 Save the augmented dataset with CoT to $OUTPUT_CSV"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/cot_generation_${COT_JOB_ID}.out"
echo ""
echo "Expected output:"
echo "  - Original dataset with additional 'chain_of_thought' column"
echo "  - Detailed step-by-step reasoning for each reduction"
echo "  - Checkpoints saved every 10 examples for recovery"
echo ""
echo "Job logs will be saved to: ${LOG_DIR}/cot_generation_${COT_JOB_ID}.{out,err}"
