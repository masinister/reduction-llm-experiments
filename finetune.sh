#!/bin/bash
#SBATCH --job-name=llama_finetune           # Job name
#SBATCH --output=logs/finetune_%j.out       # Standard output and error log
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=short                    # Partition (queue) name
#SBATCH --nodes=1                           # Run on a single node
#SBATCH --ntasks-per-node=1                 # Run a single task
#SBATCH --cpus-per-task=10                  # Number of CPU cores per task
#SBATCH --gres=gpu:2                        # Number of GPUs per node (max allowed)
#SBATCH --mem=400G                          # Total memory per node (increased for 70B model)
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec (1 day max for short)

# Enable strict error handling
set -e
set -x

echo "Starting job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Create log directory first (before SBATCH tries to write to it)
mkdir -p logs
echo "Created logs directory"

# Load modules if necessary (e.g., python, cuda)
echo "Loading Python module..."
module load python/3.11.10 || { echo "Failed to load Python module"; exit 1; }

echo "Loading CUDA module..."
module load cuda/12.4.0/3mdaov5 || { echo "Failed to load CUDA module"; exit 1; }

echo "Activating virtual environment..."
source ~/venvs/reductions/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Reset GPU state
echo "Resetting GPU state..."
nvidia-smi --gpu-reset || true
sleep 2

echo "Checking GPU status..."
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

# Create log directory
mkdir -p logs

echo "Checking Python availability..."
which python3 || echo "python3 not found"
which python || echo "python not found"

echo "Checking environment variables..."
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la

MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
CSV_PATH="~/data/karp.csv"
OUTPUT_DIR="./llama_finetune/"
DEEPSPEED_CONFIG="deepspeed-config.json"

echo "Checking if files exist..."
ls -la finetune.py || { echo "finetune.py not found"; exit 1; }
ls -la $DEEPSPEED_CONFIG || { echo "DeepSpeed config not found"; exit 1; }

echo "Expanding CSV path..."
EXPANDED_CSV_PATH=$(eval echo $CSV_PATH)
echo "CSV path: $EXPANDED_CSV_PATH"
ls -la $EXPANDED_CSV_PATH || { echo "CSV file not found at $EXPANDED_CSV_PATH"; exit 1; }

# Determine number of GPUs to use
if [ -n "$SLURM_GPUS_PER_NODE" ] && [ "$SLURM_GPUS_PER_NODE" != "" ]; then
    NUM_GPUS=$SLURM_GPUS_PER_NODE
elif [ -n "$SLURM_GPUS" ] && [ "$SLURM_GPUS" != "" ]; then
    NUM_GPUS=$SLURM_GPUS
else
    # Fallback: count available GPUs using multiple methods
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" = "" ]; then
        echo "Warning: nvidia-smi query failed, trying device count..."
        NUM_GPUS=$(nvidia-smi -L | wc -l)
        if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" = "" ]; then
            echo "Warning: Could not determine GPU count, defaulting to 2"
            NUM_GPUS=2
        fi
    fi
fi

echo "Using $NUM_GPUS GPUs for training"

# Set CUDA environment variables for cleanup
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Starting torchrun with $NUM_GPUS GPUs..."
torchrun --nproc_per_node=$NUM_GPUS \
  finetune.py \
  --model_name $MODEL_NAME \
  --csv_path $CSV_PATH \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_train_epochs 20 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 1024 \
  --deepspeed_config $DEEPSPEED_CONFIG
