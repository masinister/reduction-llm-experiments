#!/usr/bin/env bash
set -euo pipefail

echo "Installing Python dependencies inside the container..."

# 1) Upgrade pip
python3 -m pip install --upgrade pip

# 2) Install torch first (with CUDA 12.6 support to match Dockerfile)
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3) Install the package in editable mode
python3 -m pip install -e /workspace

echo "Package 'reduction-llm' installed successfully."
