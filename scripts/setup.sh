#!/bin/bash
#SBATCH --job-name=unsloth_setup
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

set -euo pipefail

source ./config.sh

module load "${MODULE_PYTHON}"
module load "${MODULE_CUDA}"

mkdir -p data models logs "$(dirname "${FINETUNED_MODEL}")" "${INFERENCE_OUTPUT_DIR}"

rm -rf "${VENV_PATH}"
python -m venv "${VENV_PATH}"

source "${VENV_PATH}/bin/activate"
pip install --upgrade pip
pip install --upgrade build setuptools wheel

# Extract CUDA version from MODULE_CUDA (e.g., "cuda12.6/toolkit/12.6.2" -> "126")
CUDA_VERSION=$(echo "${MODULE_CUDA}" | sed -n 's/cuda\([0-9]*\)\.\([0-9]*\).*/\1\2/p')
echo "Detected CUDA version: ${CUDA_VERSION}"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
pip install unsloth