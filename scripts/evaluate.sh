#!/bin/bash
#SBATCH --job-name=llama_evaluate
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --mem=256G
#SBATCH --time=3:00:00

# Usage: ./evaluate.sh [INFERENCE_DIR] [JUDGE_MODEL] [OUTPUT_DIR] [MAX_LENGTH]
# All parameters are optional and have defaults
# 
# INFERENCE_DIR: Directory containing inference results CSV files (inference_results_test.csv and inference_results_validation.csv)
# JUDGE_MODEL: Model to use as judge (e.g., meta-llama/Llama-3.3-70B-Instruct)
# OUTPUT_DIR: Directory to save evaluation results
# MAX_LENGTH: Maximum sequence length for judge model

# Enable strict error handling
set -euxo pipefail

# Print job details
echo "===== Evaluation Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=================================="

# Load modules
module load cuda/12.6.3/5fe76nu
module load python/3.11.10
source ~/venvs/reductions/bin/activate

# Parameters from CLI or defaults
INFERENCE_DIR=${1:-"./inference_results"}
JUDGE_MODEL=${2:-"meta-llama/Llama-3.3-70B-Instruct"}
OUTPUT_DIR=${3:-"./evaluation_results"}
MAX_LENGTH=${4:-4096}

# Define the expected inference result file
INFERENCE_RESULTS="$INFERENCE_DIR/inference_results.csv"

echo ""
echo "Inference directory: $INFERENCE_DIR"
echo "Judge model: $JUDGE_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Max length: $MAX_LENGTH"
echo ""

# Check if inference results file exists
if [ ! -f "$INFERENCE_RESULTS" ]; then
    echo "❌ Error: Inference results file not found: $INFERENCE_RESULTS"
    echo "Make sure the inference job completed successfully."
    exit 1
fi

echo "📊 Starting LLM-as-a-judge evaluation for combined results..."

# Evaluate the combined results file
python src/evaluate.py \
    --inference_results "$INFERENCE_RESULTS" \
    --judge_model "$JUDGE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --max_length "$MAX_LENGTH" \
    --temperature 0.1 \
    --device auto \
    --model_dtype bfloat16 \
    --batch_size 1

echo ""
echo "✅ Evaluation completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "File generated: evaluation_inference_results.csv"

echo ""
echo "===== Evaluation Job Completed ====="