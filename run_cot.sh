#!/bin/bash

# SLURM submission wrapper for the pipeline with Chain-of-Thought mode
# Usage: ./run_cot.sh [CONFIG_FILE]
# Optional: Specify a custom config file, defaults to ./config_cot.sh

# Parse arguments
CONFIG_FILE=${1:-"./config_cot.sh"}

# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file '$CONFIG_FILE' not found!"
    echo "Please create a config file or specify a valid path."
    echo "Example: ./run_cot.sh /path/to/your/config_cot.sh"
    exit 1
fi

echo "📋 Loading configuration from: $CONFIG_FILE"
source "$CONFIG_FILE"

echo "🧠 Chain-of-thought mode: ENABLED"

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/run_cot_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "🚀 Submitting llama_pipeline job with CoT..."
echo "📁 Log directory: $LOG_DIR"

# Submit the pipeline script as a SLURM job with proper output redirection
PIPELINE_JOB_ID=$(sbatch --parsable \
    --job-name=llama_pipeline_cot \
    --output="${LOG_DIR}/pipeline_%j.out" \
    --error="${LOG_DIR}/pipeline_%j.err" \
    scripts/submit_pipeline.sh "$LOG_DIR" "$CONFIG_FILE" --cot)

echo "Pipeline job ID: $PIPELINE_JOB_ID"
echo ""
echo "Pipeline Components (Chain-of-Thought):"
echo "  1. 🔬 Fine-tuning: Trains model on reduction dataset with CoT"
echo "  2. 🔍 Inference: Generates reductions on both test and validation sets with CoT"
echo "  3. 📊 Evaluation: LLM-as-a-judge evaluates generated reductions for both sets"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/pipeline_${PIPELINE_JOB_ID}.out"
echo ""
echo "Pipeline logs will be saved to: ${LOG_DIR}/pipeline_${PIPELINE_JOB_ID}.{out,err}"
