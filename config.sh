# central config (sourced by scripts/*.sh)

export MODULE_PYTHON="python/3.11.12/7vjddo3"
export MODULE_CUDA="cuda/12.1/toolkit/12.1.1"

export VENV_PATH="$HOME/venvs/unsloth"
export REQ_FILE="./requirements.txt"

export TRAIN_CSV_PATH="./karp.csv"
export TEST_CSV_PATH="./karp.csv"
export MODEL_NAME="unsloth/Meta-Llama-3.1-8B-bnb-4bit"

export MODEL_DIR="./models/llama_4bit"
export OUTPUT_DIR="./outputs"

export MAX_SEQ="2048"
export BATCH_SIZE="1"
export GRAD_ACC="4"
export EPOCHS="3"
export LORA_R="16"

export MAX_NEW_TOKENS="512"