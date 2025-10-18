#!/bin/bash
#SBATCH --job-name=setup_reduction_llm
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# One-time setup script for SLURM cluster
# Run as: sbatch scripts/slurm_setup.sh
# Or directly: ./scripts/slurm_setup.sh (if you prefer interactive)

set -euo pipefail

echo "Setting up reduction-llm environment on SLURM cluster..."
echo ""

# Load modules (adjust versions to match your cluster)
echo "Loading modules..."
module load cuda/12.6.3/5fe76nu
module load python/3.11.10

# Create virtualenv
echo "Creating virtualenv at ~/venvs/reduction-llm..."
python3 -m venv ~/venvs/reduction-llm
source ~/venvs/reduction-llm/bin/activate

# This matches bootstrap.sh exactly:

# 1) Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# 2) Install torch first (with CUDA 12.6 support)
echo "Installing PyTorch with CUDA 12.6 support..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3) Install the package in editable mode
echo "Installing reduction-llm package..."
# SLURM_SUBMIT_DIR is set to the directory where sbatch was run
python3 -m pip install -e "${SLURM_SUBMIT_DIR:-.}"

echo ""
echo "✅ Setup complete at $(date)!"
echo ""
echo "To use in future jobs, add to your SLURM script:"
echo "  module load cuda/12.6.3/5fe76nu"
echo "  module load python/3.11.10"
echo "  source ~/venvs/reduction-llm/bin/activate"
