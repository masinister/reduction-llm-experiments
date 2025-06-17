#!/bin/bash
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

# Enable strict error handling
set -euxo pipefail

# Create logs directory
mkdir -p logs
echo "Logs directory created."

# SLURM info
echo "================ FULL PIPELINE JOB INFO ================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node list:   $SLURM_NODELIST"
echo "Submitted on: $(date)"
echo "========================================================="

# Load required modules
echo "Loading required modules..."
module load python/3.11.10
module load cuda/12.4.0/3mdaov5
module load libaio/0.3.113/xtilfep

# Activate virtual environment
echo "Activating virtual environment..."
source ~/venvs/reductions/bin/activate

# Parameters from CLI or defaults
MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
CSV_PATH=$(eval echo ${2:-"~/data/karp.csv"})
OUTPUT_DIR=${3:-"./llama_finetune"}
BATCH_SIZE=${4:-1}
GRAD_ACCUM=${5:-16}
LEARNING_RATE=${6:-1e-4}
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

# Check GPU status
echo "GPU status:"
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

# Debug paths and environment
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not Set}"
echo "Working directory: $(pwd)"
echo "Files:"
ls -lh

# Validate required files
if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: CSV file not found at $CSV_PATH"
  exit 1
fi

if [[ ! -f "finetune.py" ]]; then
  echo "ERROR: finetune.py not found in working directory"
  exit 1
fi

if [[ ! -f "inference.py" ]]; then
  echo "ERROR: inference.py not found in working directory"
  exit 1
fi

echo "All required files found."

# Memory and compute environment optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export TORCH_CPP_LOG_LEVEL=ERROR
# Set master port for distributed training
export MASTER_PORT=29501

echo "Environment configured for FSDP fine-tuning."

# ========================================
# PHASE 1: FINE-TUNING
# ========================================
echo ""
echo "🚀 PHASE 1: Starting fine-tuning..."
echo "========================================"

FINETUNE_START=$(date)
echo "Fine-tuning started at: $FINETUNE_START"

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --master_port=$MASTER_PORT \
  finetune.py \
  --model_name "$MODEL_NAME" \
  --csv_path "$CSV_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$EPOCHS" \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length "$MAX_LENGTH" \
  --model_dtype bfloat16 \
  --cpu_offload \
  --fsdp_sharding_strategy "full_shard" \
  --fsdp_mixed_precision \
  --fsdp_activation_checkpointing \
  --fsdp_auto_wrap

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

if [[ ! -d "$OUTPUT_DIR/final" ]]; then
    echo "ERROR: Final model not found at $OUTPUT_DIR/final"
    echo "Fine-tuning may have failed."
    exit 1
fi
echo "✅ Found final model directory."

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

# Run inference - check for merged model first, then use adapters
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

INFERENCE_END=$(date)
echo ""
echo "✅ Inference completed at: $INFERENCE_END"
echo "Inference duration: $INFERENCE_START to $INFERENCE_END"

# ========================================
# FINAL SUMMARY
# ========================================
echo ""
echo "🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Pipeline started:   $FINETUNE_START"
echo "Fine-tuning ended:  $FINETUNE_END"
echo "Inference ended:    $INFERENCE_END"
echo ""
echo "📁 Output Locations:"
echo "  - Model directory:    $OUTPUT_DIR"
echo "  - Final model:        $OUTPUT_DIR/final"
echo "  - Inference results:  $INFERENCE_OUTPUT/inference_results_${TEST_SET}.json"
echo ""
echo "📊 To view results:"
echo "  cat $INFERENCE_OUTPUT/inference_results_${TEST_SET}.json"
echo ""
echo "🔄 To run inference again on different test set:"
echo "  ./inference.sh \"$MODEL_NAME\" \"$CSV_PATH\" \"$OUTPUT_DIR\" \"$INFERENCE_OUTPUT\" validation"
echo "=========================================="
