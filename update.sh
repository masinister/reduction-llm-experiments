#!/bin/bash
#SBATCH --job-name=update                 # Job name
#SBATCH --output=logs/update_%j.out       # Standard output and error log
#SBATCH --error=logs/update_%j.err
#SBATCH --partition=short                 # Partition (queue) name
#SBATCH --nodes=1                         # Run on a single node
#SBATCH --ntasks-per-node=1               # Run a single task
#SBATCH --cpus-per-task=10                # Number of CPU cores per task
#SBATCH --mem=10G                         # Total memory per node
#SBATCH --time=01:00:00                   # Time limit hrs:min:sec

# Load modules
module load cuda/12.6.3/5fe76nu
module load python/3.11.10

# Create & activate virtual env
python3 -m venv ~/venvs/reductions
echo "Activating virtual environment..."
source ~/venvs/reductions/bin/activate

# Upgrade pip and install pip-tools
pip install --upgrade pip
pip install pip-tools

# Generate top-level requirements.in
cat > requirements.in << 'EOF'
# Use CUDA-enabled wheels for PyTorch
torch[cuda126]
torchvision[cuda126]

# Core FSDP + QLoRA libs
xformers
transformers
datasets
peft
accelerate
huggingface_hub
bitsandbytes
trl

# Utilities
evaluate
pandas
numpy
EOF

# Compile to fully pinned requirements.txt, pulling from both PyPI and your CUDA index
pip-compile requirements.in \
  --output-file=requirements.txt \
  --upgrade \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --find-links https://download.pytorch.org/whl/cu126/torch_stable.html

# Install everything from the generated requirements.txt
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --find-links https://download.pytorch.org/whl/cu126/torch_stable.html

echo "✅ Environment updated with fully pinned dependencies."
