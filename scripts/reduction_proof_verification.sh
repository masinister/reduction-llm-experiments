#!/bin/bash
#SBATCH --job-name=proof_verification
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/proof_verification_%j.out
#SBATCH --error=logs/proof_verification_%j.err

# Structured proof verification for reductions
# Extracts claims, generates proofs, and verifies each claim
# Uses 4 GPUs with tensor parallelism for faster inference

set -euo pipefail

echo "===== Proof Verification Job ====="
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

# Make all 4 GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
# Model parameters come from config.ini
# Only specify paths and proof-specific settings here
INPUT_CSV="${INPUT_CSV:-~/data/karp.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-~/results/verified_proofs_$(date +%Y%m%d_%H%M%S).csv}"
NUM_CLAIMS="${NUM_CLAIMS:-5}"

# Run proof verification
# Model, temperature, and other inference params come from config.ini
python examples/reduction_proof_verification.py \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_CSV" \
    --num_claims $NUM_CLAIMS

# Cleanup Ray temp directory
rm -rf "$RAY_TMPDIR"

echo ""
echo "✅ Proof verification job completed at $(date)"
echo "Results saved to: $OUTPUT_CSV"
