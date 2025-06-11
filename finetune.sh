#!/bin/bash
#SBATCH --job-name=llama_finetune           # Job name
#SBATCH --output=logs/finetune_%j.out       # Standard output and error log
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=short                   # Partition (queue) name
#SBATCH --nodes=1                           # Run on a single node
#SBATCH --ntasks-per-node=1                 # Run a single task
#SBATCH --cpus-per-task=10                  # Number of CPU cores per task
#SBATCH --gres=gpu:4                        # Number of GPUs per node
#SBATCH --mem=128G                          # Total memory per node (128GB)
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec

# Load modules if necessary (e.g., python, cuda)
module load python/3.11.10

module load cuda/12.4.0/3mdaov5

source ~/venvs/reductions/bin/activate

nvidia-smi

# Create log directory
mkdir -p logs

MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
CSV_PATH="~/data/karp.csv"
OUTPUT_DIR="llama_finetune/"
DEEPSPEED_CONFIG="deepspeed_config.json"

torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE \
  fine_tune.py \
  --model_name $MODEL_NAME \
  --csv_path $CSV_PATH \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_train_epochs 20 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 1024 \
  --deepspeed_config $DEEPSPEED_CONFIG \
