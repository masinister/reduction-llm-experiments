#!/bin/bash
#SBATCH --job-name=llm_inference_multi_gpu
#SBATCH --partition=gpu
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

echo "===== Multi-GPU Inference Job ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Date: $(date)"
echo "===================================="

# Load modules and activate environment
module load cuda/12.6.3/5fe76nu
module load python/3.11.10
source ~/venvs/reduction-llm/bin/activate

# Set Ray temp directory (important for multi-GPU)
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# Run inference with 4-GPU tensor parallelism
python -c "
from src.inference import Model

# Create model with 4-GPU tensor parallelism
# Ray backend is automatically used for multi-GPU
model = Model(
    model_id='meta-llama/Meta-Llama-3-70B-Instruct',  # Large model for multi-GPU
    tensor_parallel_size=4,  # Split across 4 GPUs
    temperature=0.7,
    max_tokens=2048,
    gpu_memory_utilization=0.95  # Use most of GPU memory
)

# Run inference
result = model.infer('Explain polynomial-time reductions in detail.')
print(f'Response: {result[\"text\"]}')
print(f'Tokens: {result[\"tokens\"]}')
print(f'Latency: {result[\"latency_s\"]:.2f}s')
"

# Cleanup Ray temp directory
rm -rf "$RAY_TMPDIR"

echo ""
echo "✅ Multi-GPU job completed at $(date)"
