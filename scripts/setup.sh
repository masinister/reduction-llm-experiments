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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d .) 
pip install unsloth