#!/bin/bash
#SBATCH --job-name=update                 # Job name
#SBATCH --output=logs/update_%j.out       # Standard output and error log
#SBATCH --error=logs/update_%j.err
#SBATCH --partition=short                 # Partition (queue) name
#SBATCH --nodes=1                         # Run on a single node
#SBATCH --ntasks-per-node=1               # Run a single task
#SBATCH --cpus-per-task=10                # Number of CPU cores per task
#SBATCH --mem=10G                         # Total memory per node (increased for 70B model)
#SBATCH --time=01:00:00                   # Time limit hrs:min:sec (1 day max for short)


# Load modules if necessary (e.g., python, cuda)
module load cuda/12.6.3/5fe76nu
module load python/3.11.10

python3 -m venv ~/venvs/reductions

echo "Activating virtual environment..."
source ~/venvs/reductions/bin/activate

pip install --upgrade pip

pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --find-links https://download.pytorch.org/whl/cu126/torch_stable.html