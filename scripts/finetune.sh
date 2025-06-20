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

# Usage: ./finetune.sh [MODEL_NAME] [CSV_PATH] [OUTPUT_DIR] [BATCH_SIZE] [GRAD_ACCUM] [LEARNING_RATE] [EPOCHS] [MAX_LENGTH]
# All parameters are optional and have defaults

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

# Parameters from CLI or defaults
MODEL_NAME=${1:-"meta-llama/Llama-3.3-70B-Instruct"}
CSV_PATH=$(eval echo ${2:-"~/data/karp.csv"})
OUTPUT_DIR=${3:-"./llama_finetune"}
BATCH_SIZE=${4:-1}
GRAD_ACCUM=${5:-16}
LEARNING_RATE=${6:-2e-4}
EPOCHS=${7:-20}
MAX_LENGTH=${8:-2048}

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
echo "Master port: $MASTER_PORT"
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
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --master_port=$MASTER_PORT \
  src/finetune.py \
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
  --cpu_offload

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
