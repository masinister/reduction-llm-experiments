#!/usr/bin/env bash
#SBATCH --job-name=llama_full_pipeline
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --constraint=H200
#SBATCH --mem=400G
#SBATCH --time=24:00:00

# Usage: ./run.sh [MODEL_NAME] [CSV_PATH] [OUTPUT_DIR] [BATCH_SIZE] [GRAD_ACCUM] [LEARNING_RATE] [EPOCHS] [MAX_LENGTH] [INFERENCE_OUTPUT] [TEST_SET]
# All parameters are optional and have defaults
#
# MODEL_NAME: Base model to fine-tune (e.g., meta-llama/Llama-3.1-8B-Instruct) 
# CSV_PATH: Path to the CSV dataset file
# OUTPUT_DIR: Directory to save fine-tuned model (will contain checkpoints and trainer_state.json)
# BATCH_SIZE: Per-device batch size for training
# GRAD_ACCUM: Gradient accumulation steps
# LEARNING_RATE: Learning rate for training
# EPOCHS: Number of training epochs
# MAX_LENGTH: Maximum sequence length
# INFERENCE_OUTPUT: Directory to save inference results
# TEST_SET: Which set to run inference on (test, validation, or both)

# Enable strict error handling + job control
set -euxo pipefail
set -m  # enable job control so children belong to our process group

# Create logs directory
mkdir -p logs
echo "Logs directory created."

# SLURM info
echo "================ FULL PIPELINE JOB INFO ================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node list:   $SLURM_NODELIST"
echo "Submitted on: $(date)"
echo "========================================================="

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

echo "" 
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
echo ""

# ========================================
# PHASE 1: FINE-TUNING
# ========================================
echo ""
echo "🚀 PHASE 1: Starting fine-tuning..."
echo "========================================"

FINETUNE_START=$(date)
echo "Fine-tuning started at: $FINETUNE_START"

# Run finetune.sh script
echo "Calling finetune.sh with parameters..."
./finetune.sh "$MODEL_NAME" "$CSV_PATH" "$OUTPUT_DIR" "$BATCH_SIZE" "$GRAD_ACCUM" "$LEARNING_RATE" "$EPOCHS" "$MAX_LENGTH"

FINETUNE_END=$(date)
echo ""
echo "✅ Fine-tuning completed at: $FINETUNE_END"
echo "Fine-tuning duration: $FINETUNE_START to $FINETUNE_END"

# Validate fine-tuning outputs
HELD_OUT_FILE="$OUTPUT_DIR/held_out_indices.json"
if [[ ! -f "$HELD_OUT_FILE" ]]; then
    echo "ERROR: held_out_indices.json not found at $HELD_OUT_FILE"
    echo "Fine-tuning may have failed."
    exit 1
fi
echo "✅ Found held-out indices file."

TRAINER_STATE_FILE="$OUTPUT_DIR/trainer_state.json"
if [[ ! -f "$TRAINER_STATE_FILE" ]]; then
    echo "ERROR: trainer_state.json not found at $TRAINER_STATE_FILE"
    echo "Fine-tuning may have failed."
    exit 1
fi
echo "✅ Found trainer state file."

# Check for checkpoint directories
CHECKPOINT_DIRS=($(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || true))
if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: No checkpoint directories found in $OUTPUT_DIR"
    echo "Fine-tuning may have failed."
    exit 1
fi
echo "✅ Found checkpoint directories: ${CHECKPOINT_DIRS[@]##*/}"

# ========================================
# PHASE 2: INFERENCE
# ========================================
echo ""
echo "🔍 PHASE 2: Starting inference..."
echo "=================================="

INFERENCE_START=$(date)
echo "Inference started at: $INFERENCE_START"

# Create inference output directory
mkdir -p "$INFERENCE_OUTPUT"

# Run inference.sh script
echo "Calling inference.sh with parameters..."
./inference.sh "$MODEL_NAME" "$CSV_PATH" "$OUTPUT_DIR" "$INFERENCE_OUTPUT" "$TEST_SET"

INFERENCE_END=$(date)
echo ""
echo "✅ Inference completed at: $INFERENCE_END"
echo "Inference duration: $INFERENCE_START to $INFERENCE_END"

# ========================================
# PIPELINE COMPLETION SUMMARY
# ========================================
echo ""
echo "🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================"
echo "Fine-tuning started: $FINETUNE_START"
echo "Fine-tuning ended:   $FINETUNE_END"  
echo "Inference started:   $INFERENCE_START"
echo "Inference ended:     $INFERENCE_END"
echo ""
echo "Outputs:"
echo "  - Model directory:     $OUTPUT_DIR"
echo "  - Latest checkpoint:   $OUTPUT_DIR/checkpoint-*"
echo "  - Held-out indices:    $OUTPUT_DIR/held_out_indices.json"
echo "  - Inference results:   $INFERENCE_OUTPUT/inference_results_${TEST_SET}.json"
echo ""
echo "Scripts called:"
echo "  - finetune.sh: Fine-tuning with distributed training"
echo "  - inference.sh: Inference with PEFT model"
echo "========================================"
