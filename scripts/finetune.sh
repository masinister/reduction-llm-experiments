#!/bin/bash
#SBATCH --job-name=karp_finetune
#SBATCH --partition=short
#SBATCH --gres=gpu:4
#SBATCH --constraint=A100|H100|H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err

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

# SLURM may return values like "gpu:4" or "1(x4)"; keep only the leading integer.
GPUS_PER_NODE="${GPUS_PER_NODE%%,*}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
GPUS_PER_NODE="${GPUS_PER_NODE##*(}"
GPUS_PER_NODE="${GPUS_PER_NODE%%)*}"

if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${GPUS_PER_NODE}" -lt 1 ]]; then
  echo "Unable to parse GPU count from SLURM_GPUS_ON_NODE='${SLURM_GPUS_ON_NODE}'" >&2
  exit 1
fi

echo "Launching fine-tuning across ${GPUS_PER_NODE} GPU(s)" >&2

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" src/finetune.py \
  --csv "${TRAIN_CSV_PATH}" \
  --model_name "${BASE_MODEL}" \
  --output_dir "${FINETUNED_MODEL}" \
  --max_seq_length "${MAX_SEQ}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --num_train_epochs "${EPOCHS}" \
  --lora_r "${LORA_R}"
