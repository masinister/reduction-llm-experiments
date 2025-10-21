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

set -euo pipefail

echo "===== Proof Verification Job ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Date: $(date)"
echo "===================================="

module load cuda/12.6.3/5fe76nu
module load python/3.11.10
source ~/venvs/reduction-llm/bin/activate

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3

INPUT_CSV="${INPUT_CSV:-~/data/karp.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-~/results/verified_proofs_$(date +%Y%m%d_%H%M%S).csv}"
NUM_CLAIMS="${NUM_CLAIMS:-5}"

python examples/reduction_proof_verification.py \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_CSV" \
    --num_claims $NUM_CLAIMS

rm -rf "$RAY_TMPDIR"

echo ""
echo "✅ Proof verification job completed at $(date)"
echo "Results saved to: $OUTPUT_CSV"
