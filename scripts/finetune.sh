#!/usr/bin/env bash
#SBATCH --job-name=llama_finetune
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --constraint=H200
#SBATCH --mem=400G
#SBATCH --time=24:00:00

# Usage: ./finetune.sh MODEL_NAME CSV_PATH OUTPUT_DIR BATCH_SIZE GRAD_ACCUM LEARNING_RATE EPOCHS MAX_LENGTH LORA_R LORA_ALPHA LORA_DROPOUT [--cot]
# All parameters except --cot are required

# Enable strict error handling + job control
set -euxo pipefail
set -m  # enable job control so children belong to our process group

# SLURM info
echo "================ SLURM JOB INFO ================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node list:   $SLURM_NODELIST"
echo "Submitted on: $(date)"
echo "================================================"

# Load required modules
echo "Loading required modules..."
module load cuda/12.6.3/5fe76nu
module load python/3.11.10

# Activate virtual environment
echo "Activating virtual environment..."
source ~/venvs/reductions/bin/activate

# Parameters from CLI (required)
MODEL_NAME=${1}
CSV_PATH=$(eval echo ${2})
OUTPUT_DIR=${3}
BATCH_SIZE=${4}
GRAD_ACCUM=${5}
LEARNING_RATE=${6}
EPOCHS=${7}
MAX_LENGTH=${8}
LORA_R=${9}
LORA_ALPHA=${10}
LORA_DROPOUT=${11}

# Check for optional --cot flag
COT_FLAG=""
if [[ "${12}" == "--cot" ]]; then
    COT_FLAG="--cot"
    echo "Using chain-of-thought mode"
fi

# Set master port (can override via env)
export MASTER_PORT=${MASTER_PORT:-29501}

echo ""
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
echo "Master port: $MASTER_PORT"
if [[ -n "$COT_FLAG" ]]; then
    echo "Chain-of-thought mode: ENABLED"
else
    echo "Chain-of-thought mode: DISABLED"
fi
echo ""

# Pre-cleanup: kill stale processes listening on MASTER_PORT
if lsof -iTCP:${MASTER_PORT} -sTCP:LISTEN -t >/dev/null; then
  echo "⚠️  Port ${MASTER_PORT} in use; killing stale process(es)..."
  lsof -iTCP:${MASTER_PORT} -sTCP:LISTEN -t | xargs --no-run-if-empty kill -9 || true
fi

# Cleanup function to kill child processes on exit
cleanup() {
  echo "🧹 Cleaning up child processes..."
  pkill -P $$ || true
}
trap cleanup EXIT

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

if [[ ! -f "src/finetune.py" ]]; then
  echo "ERROR: src/finetune.py not found in working directory"
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

# NCCL timeout configurations to handle evaluation delays
export NCCL_TIMEOUT=3600  # 1 hour timeout for NCCL operations
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # 1 hour heartbeat timeout
export TORCH_NCCL_DESYNC_DEBUG=1  # Enable desync debugging

echo "Environment configured for FSDP fine-tuning."

# Training command
echo ""
echo "🚀 Starting training..."
accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision bf16 \
  --fsdp_sharding_strategy FULL_SHARD \
  --fsdp_auto_wrap_policy NO_WRAP \
  --fsdp_use_orig_params True \
  src/finetune.py \
    --model_name "$MODEL_NAME" \
    --csv_path "$CSV_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$EPOCHS" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --max_length "$MAX_LENGTH" \
    --model_dtype bfloat16 \
    --cpu_offload \
    $COT_FLAG

echo ""
echo "✅ Fine-tuning completed at $(date)"
echo "=================================================="
echo "Outputs:"
echo "  - Model directory: $OUTPUT_DIR"
echo "  - Latest checkpoint: $OUTPUT_DIR/checkpoint-*"
echo "  - Held-out indices: $OUTPUT_DIR/held_out_indices.json"
echo "=================================================="
echo "To run inference:"
echo "  ./scripts/inference.sh \"$MODEL_NAME\" \"$CSV_PATH\" \"$OUTPUT_DIR\" ./inference_results test"
