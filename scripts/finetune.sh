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

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES not set; request GPUs via --gres." >&2
  exit 1
fi

IFS=',' read -ra _gpus <<< "${CUDA_VISIBLE_DEVICES}"
GPUS_PER_NODE="${#_gpus[@]}"

if (( GPUS_PER_NODE < 1 )); then
  echo "No CUDA devices visible to this job." >&2
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
