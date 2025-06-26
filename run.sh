#!/bin/bash

# SLURM submission wrapper for the pipeline
# Usage: ./run.sh [CONFIG_FILE] [--cot]
# Optional: Specify a custom config file, defaults to ./config.sh
# Optional: Use --cot flag to enable chain-of-thought mode

# Parse arguments
CONFIG_FILE=${1:-"./config.sh"}
COT_FLAG=""

# Check if first argument is --cot (when no config file is specified)
if [[ "$1" == "--cot" ]]; then
    CONFIG_FILE="./config.sh"
    COT_FLAG="--cot"
# Check if second argument is --cot (when config file is specified)
elif [[ "$2" == "--cot" ]]; then
    COT_FLAG="--cot"
fi
# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file '$CONFIG_FILE' not found!"
    echo "Please create a config file or specify a valid path."
    echo "Example: ./run.sh /path/to/your/config.sh"
    exit 1
fi

echo "📋 Loading configuration from: $CONFIG_FILE"
source "$CONFIG_FILE"

if [[ -n "$COT_FLAG" ]]; then
    echo "🧠 Chain-of-thought mode: ENABLED"
else
    echo "🧠 Chain-of-thought mode: DISABLED"
fi

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "🚀 Submitting llama_pipeline job..."
echo "📁 Log directory: $LOG_DIR"

# Submit the pipeline script as a SLURM job with proper output redirection
PIPELINE_JOB_ID=$(sbatch --parsable \
    --job-name=llama_pipeline \
    --output="${LOG_DIR}/pipeline_%j.out" \
    --error="${LOG_DIR}/pipeline_%j.err" \
    scripts/submit_pipeline.sh "$LOG_DIR" "$CONFIG_FILE" $COT_FLAG)

echo "Pipeline job ID: $PIPELINE_JOB_ID"
echo ""
echo "Pipeline Components:"
echo "  1. 🔬 Fine-tuning: Trains model on reduction dataset"
echo "  2. 🔍 Inference: Generates reductions on both test and validation sets"
echo "  3. 📊 Evaluation: LLM-as-a-judge evaluates generated reductions for both sets"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/pipeline_${PIPELINE_JOB_ID}.out"
echo ""
echo "Pipeline logs will be saved to: ${LOG_DIR}/pipeline_${PIPELINE_JOB_ID}.{out,err}"
