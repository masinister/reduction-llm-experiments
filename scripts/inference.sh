#!/bin/bash
#SBATCH --job-name=karp_inference
#SBATCH --partition=short
#SBATCH --gres=gpu:1
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

mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}" logs

source "${VENV_PATH}/bin/activate"

python src/inference.py \
  --model_dir "${MODEL_DIR}" \
  --csv "${TEST_CSV_PATH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_seq_length "${MAX_SEQ}"
