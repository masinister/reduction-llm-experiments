#!/bin/bash
#SBATCH --job-name=llama_finetune           # Job name
#SBATCH --output=logs/finetune_%j.out       # Standard output and error log
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=short                   # Partition (queue) name
#SBATCH --nodes=1                           # Run on a single node
#SBATCH --ntasks-per-node=1                 # Run a single task
#SBATCH --cpus-per-task=10                  # Number of CPU cores per task
#SBATCH --gres=gpu:4                        # Number of GPUs per node
#SBATCH --constraint=H200                   # Specify GPU type
#SBATCH --mem=400G                          # Total memory per node
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

nvidia-smi -q
sleep 2

echo "Checking GPU status..."
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

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

echo "Checking DeepSpeed configuration..."
cat $DEEPSPEED_CONFIG | grep -q '"stage": 3' && echo "✓ ZeRO-3 configuration detected" || echo "⚠ Warning: ZeRO-3 not detected in config"

echo "Expanding CSV path..."
EXPANDED_CSV_PATH=$(eval echo $CSV_PATH)
echo "CSV path: $EXPANDED_CSV_PATH"
ls -la $EXPANDED_CSV_PATH || { echo "CSV file not found at $EXPANDED_CSV_PATH"; exit 1; }

# Set CUDA environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export NCCL_P2P_DISABLE=1  # Disable P2P for memory efficiency
export NCCL_IB_DISABLE=1   # Disable InfiniBand if causing issues
export CUDA_LAUNCH_BLOCKING=0  # Enable async CUDA operations
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling
export OMP_NUM_THREADS=8  # Limit OpenMP threads to reduce CPU memory

echo "Memory optimization environment variables set:"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"

echo "Starting DeepSpeed with optimized memory settings..."
echo "Command to be executed:"
echo "deepspeed finetune.py --model_name $MODEL_NAME --csv_path $CSV_PATH --output_dir $OUTPUT_DIR --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 1e-4 --num_train_epochs 3 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --max_length 2048 --model_dtype bfloat16 --cpu_offload --deepspeed_config $DEEPSPEED_CONFIG"
echo ""
deepspeed \
  finetune.py \
  --model_name $MODEL_NAME \
  --csv_path $CSV_PATH \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 2048 \
  --model_dtype bfloat16 \
  --cpu_offload \
  --deepspeed_config $DEEPSPEED_CONFIG

echo "Fine-tuning job completed at $(date)"
echo "Check the following files for results:"
echo "  - Training logs: $OUTPUT_DIR"
echo "  - Inference results: $OUTPUT_DIR/inference_results.json"
echo "  - Model checkpoints: $OUTPUT_DIR/final"

echo "Job finished successfully!"
