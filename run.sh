#!/bin/bash
#SBATCH --job-name=llama_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# Usage: ./run.sh [MODEL_NAME] [CSV_PATH] [OUTPUT_DIR] [BATCH_SIZE] [GRAD_ACCUM] [LEARNING_RATE] [EPOCHS] [MAX_LENGTH] [INFERENCE_OUTPUT] [TEST_SET]
# All parameters are optional and have defaults

# Create logs directory
mkdir -p logs

# Parameters from CLI or defaults
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
CSV_PATH=$(eval echo ${2:-"~/data/karp.csv"})
OUTPUT_DIR=${3:-"./llama_finetune"}
BATCH_SIZE=${4:-1}
GRAD_ACCUM=${5:-16}
LEARNING_RATE=${6:-2e-4}
EPOCHS=${7:-20}
MAX_LENGTH=${8:-2048}
INFERENCE_OUTPUT=${9:-"./inference_results"}
TEST_SET=${10:-"test"}

echo "==================== PIPELINE PARAMETERS ===================="
echo "Model name: $MODEL_NAME"
echo "CSV path: $CSV_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size per device: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "Learning rate: $LEARNING_RATE"
echo "Number of epochs: $EPOCHS"
echo "Max sequence length: $MAX_LENGTH"
echo "Inference output dir: $INFERENCE_OUTPUT"
echo "Test set: $TEST_SET"
echo "============================================================="

# Submit jobs with dependencies
echo "🚀 Submitting fine-tuning job..."
FINETUNE_JOB_ID=$(sbatch --parsable finetune.sh "$MODEL_NAME" "$CSV_PATH" "$OUTPUT_DIR" "$BATCH_SIZE" "$GRAD_ACCUM" "$LEARNING_RATE" "$EPOCHS" "$MAX_LENGTH")
echo "Fine-tuning job ID: $FINETUNE_JOB_ID"

echo "🔍 Submitting inference job (depends on fine-tuning)..."
INFERENCE_JOB_ID=$(sbatch --parsable --dependency=afterok:$FINETUNE_JOB_ID inference.sh "$MODEL_NAME" "$CSV_PATH" "$OUTPUT_DIR" "$INFERENCE_OUTPUT" "$TEST_SET" "$MAX_LENGTH")
echo "Inference job ID: $INFERENCE_JOB_ID"

echo ""
echo "🎉 Pipeline jobs submitted successfully!"
echo "========================================"
echo "Fine-tuning job ID: $FINETUNE_JOB_ID"
echo "Inference job ID:   $INFERENCE_JOB_ID"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/finetune_${FINETUNE_JOB_ID}.out"
echo "  tail -f logs/inference_${INFERENCE_JOB_ID}.out"
echo ""
echo "Expected outputs:"
echo "  - Model directory:     $OUTPUT_DIR"
echo "  - Inference results:   $INFERENCE_OUTPUT/inference_results_${TEST_SET}.json"
echo "========================================"
