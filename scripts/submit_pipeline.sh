#!/bin/bash

# Usage: ./submit_pipeline.sh LOG_DIR CONFIG_FILE
# Both LOG_DIR and CONFIG_FILE are required

# Get the log directory from the first parameter
LOG_DIR="$1"
CONFIG_FILE="$2"

# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

echo "📋 Loading configuration from: $CONFIG_FILE"
source "$CONFIG_FILE"

# If running as SLURM job, set up logging
if [ ! -z "$SLURM_JOB_ID" ]; then
    PIPELINE_LOG="${LOG_DIR}/pipeline_${SLURM_JOB_ID}"
    exec 1> >(tee "${PIPELINE_LOG}.out")
    exec 2> >(tee "${PIPELINE_LOG}.err")
    echo "=== SLURM Pipeline Job Started ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Log files: ${PIPELINE_LOG}.{out,err}"
    echo "================================="
fi

# Parameters from config file (with path expansion for CSV_PATH)
CSV_PATH=$(eval echo "$CSV_PATH")

echo "==================== PIPELINE PARAMETERS ===================="
echo "Model name: $MODEL_NAME"
echo "CSV path: $CSV_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size per device: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "Learning rate: $LEARNING_RATE"
echo "Number of epochs: $EPOCHS"
echo "Max sequence length: $MAX_LENGTH"
echo "LoRA rank: $LORA_R"
echo "LoRA alpha: $LORA_ALPHA"
echo "LoRA dropout: $LORA_DROPOUT"
echo "Inference output dir: $INFERENCE_OUTPUT"
echo "Judge model: $JUDGE_MODEL"
echo "Evaluation output dir: $EVAL_OUTPUT"
echo "============================================================="

# Submit jobs with dependencies
echo "🚀 Submitting fine-tuning job..."
FINETUNE_JOB_ID=$(sbatch --parsable --output="${LOG_DIR}/finetune_%j.out" --error="${LOG_DIR}/finetune_%j.err" scripts/finetune.sh "$MODEL_NAME" "$CSV_PATH" "$OUTPUT_DIR" "$BATCH_SIZE" "$GRAD_ACCUM" "$LEARNING_RATE" "$EPOCHS" "$MAX_LENGTH" "$LORA_R" "$LORA_ALPHA" "$LORA_DROPOUT")
echo "Fine-tuning job ID: $FINETUNE_JOB_ID"

echo "🔍 Submitting inference job (depends on fine-tuning)..."
INFERENCE_JOB_ID=$(sbatch --parsable --dependency=afterok:$FINETUNE_JOB_ID --output="${LOG_DIR}/inference_%j.out" --error="${LOG_DIR}/inference_%j.err" scripts/inference.sh "$MODEL_NAME" "$CSV_PATH" "$OUTPUT_DIR" "$INFERENCE_OUTPUT" "$MAX_LENGTH")
echo "Inference job ID: $INFERENCE_JOB_ID"
echo "  -> Dependency: afterok:$FINETUNE_JOB_ID (will run only if fine-tuning succeeds)"

echo "📊 Submitting evaluation job for both sets (depends on inference)..."
EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$INFERENCE_JOB_ID --output="${LOG_DIR}/evaluate_%j.out" --error="${LOG_DIR}/evaluate_%j.err" scripts/evaluate.sh "$INFERENCE_OUTPUT" "$JUDGE_MODEL" "$EVAL_OUTPUT" "$EVAL_MAX_LENGTH")
echo "Evaluation job ID: $EVAL_JOB_ID"
echo "  -> Dependency: afterok:$INFERENCE_JOB_ID (will run only if inference succeeds)"

echo ""
echo "🎉 Pipeline jobs submitted successfully!"
echo "========================================"
echo "Fine-tuning job ID: $FINETUNE_JOB_ID"
echo "Inference job ID:   $INFERENCE_JOB_ID"
echo "Evaluation job ID:  $EVAL_JOB_ID"
echo ""
echo "📁 Log directory created: $LOG_DIR"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/finetune_${FINETUNE_JOB_ID}.out"
echo "  tail -f ${LOG_DIR}/inference_${INFERENCE_JOB_ID}.out"
echo "  tail -f ${LOG_DIR}/evaluate_${EVAL_JOB_ID}.out"
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "  tail -f ${LOG_DIR}/pipeline_${SLURM_JOB_ID}.out"
fi
echo ""
echo "Expected outputs:"
echo "  - Model directory:     $OUTPUT_DIR"
echo "  - Inference results:   $INFERENCE_OUTPUT/inference_results.csv"
echo "  - Evaluation results:  $EVAL_OUTPUT/evaluation_inference_results.csv"
echo "  - Pipeline logs:       $LOG_DIR/"
echo "========================================"

# Final status message
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo ""
    echo "=== SLURM Pipeline Job Completed ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "All pipeline logs saved to: $LOG_DIR/"
    echo "==================================="
fi
