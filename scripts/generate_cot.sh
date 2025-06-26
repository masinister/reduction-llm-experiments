#!/bin/bash
#SBATCH --job-name=karp_cot_generation
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --mem=256G
#SBATCH --time=12:00:00

# Usage: ./generate_cot.sh INPUT_CSV OUTPUT_CSV COT_MODEL MAX_LENGTH [RESUME_FROM]
# All parameters except RESUME_FROM are required
# 
# INPUT_CSV: Path to the input karp.csv file
# OUTPUT_CSV: Path to save the augmented karp_cot.csv file
# COT_MODEL: Model to use for chain-of-thought generation (e.g., meta-llama/Llama-3.3-70B-Instruct)
# MAX_LENGTH: Maximum sequence length for CoT model
# RESUME_FROM: Optional - resume from specific index (for interrupted runs)

# Enable strict error handling
set -euxo pipefail

# Print job details
echo "===== CoT Generation Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "======================================"

# Load modules
module load cuda/12.6.3/5fe76nu
module load python/3.11.10
source ~/venvs/reductions/bin/activate

# Parameters from CLI (required)
INPUT_CSV=$(eval echo ${1})
OUTPUT_CSV=$(eval echo ${2})
COT_MODEL=${3}
MAX_LENGTH=${4}
RESUME_FROM=${5:-0}

echo ""
echo "Input CSV: $INPUT_CSV"
echo "Output CSV: $OUTPUT_CSV"
echo "CoT model: $COT_MODEL"
echo "Max length: $MAX_LENGTH"
echo "Resume from index: $RESUME_FROM"
echo ""

# Validate required files
if [[ ! -f "$INPUT_CSV" ]]; then
  echo "ERROR: Input CSV file not found at $INPUT_CSV"
  exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$OUTPUT_DIR"

echo "Starting chain-of-thought generation..."

# Run chain-of-thought generation
python src/generate_cot.py \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_CSV" \
    --cot_model "$COT_MODEL" \
    --max_new_tokens "$MAX_LENGTH" \
    --temperature 0.6 \
    --device auto \
    --model_dtype bfloat16 \
    --batch_size 1 \
    --resume_from "$RESUME_FROM"

echo ""
echo "✅ Chain-of-thought generation completed at $(date)"
echo "Results saved to: $OUTPUT_CSV"
echo "======================================"
