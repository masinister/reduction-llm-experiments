# central config (sourced by scripts/*.sh)

export MODULE_PYTHON="python/3.11.12"
export MODULE_CUDA="cuda12.6/toolkit/12.6.2"

export VENV_PATH="$HOME/venvs/unsloth"
export REQ_FILE="./requirements.txt"

export TRAIN_CSV_PATH="data/karp.csv"
export TEST_CSV_PATH="data/karp.csv"

# Model paths (can be HuggingFace identifiers or local directories)
export BASE_MODEL="unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit"  # Model to be fine-tuned
export FINETUNED_MODEL="./models/finetuned_model"
export INFERENCE_MODEL="unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit"
# export INFERENCE_MODEL="./models/finetuned_model"

# Output directory for model inference content/results
export INFERENCE_OUTPUT_DIR="./inference_outputs"

export MAX_SEQ="2048"
export BATCH_SIZE="1"
export GRAD_ACC="4"
export EPOCHS="3"
export LORA_R="16"

export MAX_NEW_TOKENS="512"
