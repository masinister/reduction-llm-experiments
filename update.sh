#!/bin/bash
#SBATCH --job-name=update                 # Job name
#SBATCH --output=logs/update_%j.out       # STDOUT/ERR log
#SBATCH --error=logs/update_%j.err
#SBATCH --partition=short                 # Partition (queue)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=01:00:00

# 1) Load modules
module load cuda/12.6.3/5fe76nu
module load python/3.11.10

# 2) Create & activate virtualenv
python3 -m venv ~/venvs/reductions
source ~/venvs/reductions/bin/activate

# 3) Upgrade pip & install torch first (so pip-compile can import it)
pip install --upgrade pip
pip install \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --find-links https://download.pytorch.org/whl/cu126/torch_stable.html

# 4) Install pip-tools
pip install pip-tools

# 6) Run pip-compile to generate fully pinned requirements.txt
pip-compile requirements.in \
  --output-file=requirements.txt \
  --upgrade \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --find-links https://download.pytorch.org/whl/cu126/torch_stable.html

# 7) Install from the newly generated requirements.txt
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --find-links https://download.pytorch.org/whl/cu126/torch_stable.html

echo "✅ requirements.txt regenerated and environment installed."
