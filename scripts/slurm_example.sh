#!/bin/bash
#SBATCH --job-name=llm_inference
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# Example SLURM job script for running inference
# Usage: sbatch slurm_example.sh

set -euo pipefail

echo "===== Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "======================="

# Load modules and activate environment
module load cuda/12.6.3/5fe76nu
module load python/3.11.10
source ~/venvs/reduction-llm/bin/activate

# Your inference code here
# Example: Run one of your scripts
python examples/reduction_batch.py \
    --input_csv data/input.csv \
    --output_csv results/output.csv \
    --temperature 0.7 \
    --max_tokens 2048

echo "✅ Job completed at $(date)"
