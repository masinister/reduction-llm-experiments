#!/bin/bash
#SBATCH --job-name=llm_inference_multi_gpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/inference_multi_gpu_%j.out
#SBATCH --error=logs/inference_multi_gpu_%j.err

# Example: Multi-GPU inference with tensor parallelism
# This example uses 4 GPUs with tensor parallelism
# Adjust --gres=gpu:N and tensor_parallel_size accordingly

set -euo pipefail

echo "===== Multi-GPU CoT Job ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Date: $(date)"
echo "===================================="

# Load modules and activate environment
module load cuda/12.6.3/5fe76nu
module load python/3.11.10
source ~/venvs/reduction-llm/bin/activate

# Set Ray temp directory (important for multi-GPU)
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# Make all 4 GPUs visible to the smoke test
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run smoke test with multiple GPUs available
python examples/reduction_cot_batch.py --toy --input_csv="~/data/karp.csv"

# Cleanup Ray temp directory
rm -rf "$RAY_TMPDIR"

echo ""
echo "✅ Multi-GPU job completed at $(date)"
