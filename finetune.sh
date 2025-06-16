#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --constraint=H200
#SBATCH --mem=400G
#SBATCH --time=24:00:00

# Enable strict error handling
set -euxo pipefail

# Create logs directory
mkdir -p logs
echo "Logs directory created."

# SLURM info
echo "================ SLURM JOB INFO ================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node list:   $SLURM_NODELIST"
echo "Submitted on: $(date)"
echo "================================================"

# Load required modules
echo "Loading required modules..."
module load python/3.11.10
module load cuda/12.4.0/3mdaov5
module load libaio/0.3.113/xtilfep

# Activate virtual environment
echo "Activating virtual environment..."
source ~/venvs/reductions/bin/activate

# Check GPU status
echo "GPU status:"
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

# Debug paths and environment
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not Set}"
echo "Working directory: $(pwd)"
echo "Files:"
ls -lh

# Paths
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
CSV_PATH="$HOME/data/karp.csv"
OUTPUT_DIR="./llama_finetune"

# Expand tilde (~) and check CSV path
EXPANDED_CSV_PATH=$(eval echo "$CSV_PATH")
[[ -f "$EXPANDED_CSV_PATH" ]] || { echo "CSV file not found at $EXPANDED_CSV_PATH"; exit 1; }

# Check training script
[[ -f "finetune.py" ]] || { echo "finetune.py not found in working directory"; exit 1; }

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

echo "Environment configured for FSDP fine-tuning."

# Training command
echo ""
echo "Starting training..."
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  finetune.py \
  --model_name "$MODEL_NAME" \
  --csv_path "$EXPANDED_CSV_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 20 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 2048 \
  --model_dtype bfloat16 \
  --cpu_offload \
  --fsdp_sharding_strategy "full_shard" \
  --fsdp_mixed_precision \
  --fsdp_activation_checkpointing \
  --fsdp_auto_wrap

echo ""
echo "✅ Fine-tuning completed at $(date)"
echo "=================================================="
echo "Outputs:"
echo "  - Model directory: $OUTPUT_DIR"
echo "  - Final model:     $OUTPUT_DIR/final"
echo "=================================================="
echo "To run inference:"
echo "  ./inference.sh \"$MODEL_NAME\" \"$EXPANDED_CSV_PATH\" \"$OUTPUT_DIR\" ./inference_results test"
