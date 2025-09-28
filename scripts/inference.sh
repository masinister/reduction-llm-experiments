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

GPUS_PER_NODE="$(python - <<'PY'
import torch

count = torch.cuda.device_count()
if count == 0:
    raise SystemExit("No CUDA devices available for inference.")

print(count)
PY
)"

echo "Launching inference across ${GPUS_PER_NODE} visible GPU(s)" >&2

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" src/inference.py \
  --model_dir "${INFERENCE_MODEL}" \
  --csv "${TEST_CSV_PATH}" \
  --output_dir "${INFERENCE_OUTPUT_DIR}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_seq_length "${MAX_SEQ}"
