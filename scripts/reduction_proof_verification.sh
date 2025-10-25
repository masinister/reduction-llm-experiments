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
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Node: ${SLURM_NODELIST:-n/a}"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-n/a}"
echo "Date: $(date)"
echo "===================================="

# Cluster environment (adjust if needed)
module load cuda/12.6.3/5fe76nu || true
module load python/3.11.10 || true
if [[ -d ~/venvs/reduction-llm ]]; then
    source ~/venvs/reduction-llm/bin/activate
fi

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"

# Use all 4 GPUs by default (adjust for your cluster)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Inputs/outputs
INPUT_CSV="${INPUT_CSV:-~/data/karp.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-~/results/verified_proofs_$(date +%Y%m%d_%H%M%S).csv}"

set -x
python examples/reduction_refine.py \
    --csv "$INPUT_CSV" \
    --output-dir "$(dirname "$OUTPUT_CSV")" \
    --max-iters "${MAX_ITERS:-6}" \
    --use-real-model
set +x

rm -rf "$RAY_TMPDIR" || true

echo ""
echo "✅ Proof verification job completed at $(date)"
echo "Results saved to: $OUTPUT_CSV"

