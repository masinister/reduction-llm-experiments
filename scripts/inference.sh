#!/bin/bash
#SBATCH --job-name=karp_inference
#SBATCH --partition=short
#SBATCH --gres=gpu:4
#SBATCH --constraint=A100|H100|H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err

set -euo pipefail

source ./config.sh

module load "${MODULE_PYTHON}"
module load "${MODULE_CUDA}"

source "${VENV_PATH}/bin/activate"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  echo "SLURM_GPUS_ON_NODE not set; please request GPUs via --gres and re-submit." >&2
  exit 1
fi

GPUS_PER_NODE="${GPUS_PER_NODE%%,*}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
GPUS_PER_NODE="${GPUS_PER_NODE##*(}"
GPUS_PER_NODE="${GPUS_PER_NODE%%)*}"

if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${GPUS_PER_NODE}" -lt 1 ]]; then
  echo "Unable to parse GPU count from SLURM_GPUS_ON_NODE='${SLURM_GPUS_ON_NODE}'" >&2
  exit 1
fi

echo "Launching inference across ${GPUS_PER_NODE} GPU(s)" >&2

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" src/inference.py \
  --model_dir "${INFERENCE_MODEL}" \
  --csv "${TEST_CSV_PATH}" \
  --output_dir "${INFERENCE_OUTPUT_DIR}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_seq_length "${MAX_SEQ}"
