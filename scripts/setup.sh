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

mkdir -p data models logs outputs

rm -rf "${VENV_PATH}"
python -m venv "${VENV_PATH}"

source "${VENV_PATH}/bin/activate"
pip install --upgrade pip
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install --no-build-isolation "unsloth[cu126-ampere-torch280]@git+https://github.com/unslothai/unsloth.git"

