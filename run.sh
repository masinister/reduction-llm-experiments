#!/bin/bash

# SLURM submission wrapper for the pipeline
# Usage: ./run.sh [MODEL_NAME] [CSV_PATH] [OUTPUT_DIR] [BATCH_SIZE] [GRAD_ACCUM] [LEARNING_RATE] [EPOCHS] [MAX_LENGTH] [INFERENCE_OUTPUT] [TEST_SET]

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "🚀 Submitting llama_pipeline job..."
echo "📁 Log directory: $LOG_DIR"

# Submit the run.sh script as a SLURM job with proper output redirection
PIPELINE_JOB_ID=$(sbatch --parsable \
    --job-name=llama_pipeline \
    --output="${LOG_DIR}/pipeline_%j.out" \
    --error="${LOG_DIR}/pipeline_%j.err" \
    scripts/submit_pipeline.sh "$LOG_DIR" "$@")

echo "Pipeline job ID: $PIPELINE_JOB_ID"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/pipeline_${PIPELINE_JOB_ID}.out"
echo ""
echo "Pipeline logs will be saved to: ${LOG_DIR}/pipeline_${PIPELINE_JOB_ID}.{out,err}"
