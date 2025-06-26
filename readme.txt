# LLM Reduction Experiments Pipeline

This pipeline implements a complete machine learning workflow for training and evaluating models on computational complexity reductions.

## Pipeline Components

1. **Fine-tuning** (`src/finetune.py`, `scripts/finetune.sh`): 
   - Trains a large language model (default: Llama-3.3-70B) on reduction datasets
   - Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
   - Automatically splits data into train/validation/test sets
   - Supports chain-of-thought (CoT) format for reasoning-enhanced training

2. **Inference** (`src/inference.py`, `scripts/inference.sh`):
   - Generates reductions using the fine-tuned model
   - Runs on held-out test and validation sets
   - Saves results in CSV format for evaluation
   - Supports CoT format for step-by-step reasoning generation

3. **Evaluation** (`src/evaluate.py`, `scripts/evaluate.sh`):
   - Uses LLM-as-a-judge approach to evaluate generated reductions
   - Compares generated reductions against ground truth
   - Scores on correctness, completeness, clarity, and efficiency
   - Provides ACCEPT/PARTIAL/REJECT decisions with confidence scores
   - Adapts evaluation criteria for CoT vs. standard format

4. **Chain-of-Thought Generation** (`src/generate_cot.py`, `scripts/generate_cot.sh`):
   - Generates synthetic reasoning steps for existing reduction datasets
   - Creates CoT-enhanced datasets for improved training
   - Uses large language models to produce step-by-step reasoning

## Usage

### Standard Mode (Direct Reduction)

Basic usage:
```bash
./run.sh
```

### Chain-of-Thought Mode (Reasoning + Reduction)

To use chain-of-thought enhanced training and inference:
```bash
./run.sh --cot
```

With custom config:
```bash
./run.sh config_cot.sh --cot
```

### Generating CoT Data

To create chain-of-thought enhanced datasets:
```bash
./run_cot_generation.sh ~/data/karp.csv data/karp_cot.csv
```

Then use the enhanced dataset with CoT mode:
```bash
# Update CSV_PATH in config_cot.sh to point to karp_cot.csv, or use:
CSV_PATH="data/karp_cot.csv" ./run.sh --cot
```

### Manual Component Execution

You can also run individual components with CoT support:

```bash
# Fine-tuning with CoT
scripts/finetune.sh MODEL_NAME CSV_PATH OUTPUT_DIR BATCH_SIZE GRAD_ACCUM LEARNING_RATE EPOCHS MAX_LENGTH LORA_R LORA_ALPHA LORA_DROPOUT --cot

# Inference with CoT  
scripts/inference.sh BASE_MODEL CSV_PATH MODEL_PATH INFERENCE_OUTPUT MAX_LENGTH --cot

# Evaluation with CoT
scripts/evaluate.sh INFERENCE_DIR JUDGE_MODEL OUTPUT_DIR MAX_LENGTH --cot
```

### Parameters:
- `MODEL_NAME`: Base model (default: "meta-llama/Llama-3.3-70B-Instruct")
- `CSV_PATH`: Path to dataset CSV (default: "~/data/karp.csv")
- `OUTPUT_DIR`: Fine-tuning output directory (default: "./llama_finetune")
- `BATCH_SIZE`: Batch size per device (default: 4)
- `GRAD_ACCUM`: Gradient accumulation steps (default: 16)
- `LEARNING_RATE`: Learning rate (default: 1e-5)
- `EPOCHS`: Number of training epochs (default: 30)
- `MAX_LENGTH`: Maximum sequence length (default: 4096)
- `INFERENCE_OUTPUT`: Inference results directory (default: "./inference_results")
- `JUDGE_MODEL`: Model for evaluation (default: "nvidia/Llama-3_3-Nemotron-Super-49B-v1")
- `EVAL_OUTPUT`: Evaluation results directory (default: "./evaluation_results")
- `--cot`: Optional flag to enable chain-of-thought mode

## Chain-of-Thought Format

When using `--cot`, the pipeline expects and generates data in the following format:

**Training Format (CoT):**
```
System: Write a natural-language LaTeX reduction given source and target. Think step by step.
User: Source: [source_problem]
      Target: [target_problem]
Assistant: Let me think through this step by step:

[chain_of_thought_reasoning]

Therefore, the reduction is:

[reduction_full_text]
```

**Standard Format:**
```
System: Write a natural-language LaTeX reduction given source and target.
User: Source: [source_problem]
      Target: [target_problem]  
Assistant: [reduction_full_text]
```

## Dataset Requirements

**Standard Dataset Format:**
Your CSV file must contain these columns:
- `source_text`: Description of the source computational problem
- `target_text`: Description of the target computational problem
- `reduction_full_text`: The complete reduction from source to target

**Chain-of-Thought Dataset Format:**
When using `--cot`, your CSV should additionally contain:
- `chain_of_thought`: Step-by-step reasoning leading to the reduction

You can generate CoT data using the provided script:
```bash
./run_cot_generation.sh ~/data/karp.csv data/karp_cot.csv
```

## Configuration Files

**Standard Configuration:** `config.sh`
- Optimized for direct reduction training
- Default parameters: batch_size=4, max_length=4096

**CoT Configuration:** `config_cot.sh`  
- Optimized for reasoning-enhanced training
- Adjusted parameters: batch_size=2, max_length=8192, higher LoRA rank
- Points to CoT dataset by default

## Output Structure

```
llama_finetune/              # Fine-tuned model
├── checkpoint-*/            # Training checkpoints
├── held_out_indices.json   # Test/validation splits
└── trainer_state.json      # Training logs

inference_results/           # Inference outputs
└── inference_results.csv   # Combined test and validation results

evaluation_results/          # Evaluation results
└── evaluation_inference_results.csv  # LLM-as-a-judge evaluation results

data/                       # Generated CoT data (optional)
└── karp_cot.csv           # Chain-of-thought enhanced dataset

logs/                       # SLURM job logs
└── run_YYYYMMDD_HHMMSS/   # Timestamped log directory
    ├── pipeline_*.{out,err}
    ├── finetune_*.{out,err}
    ├── inference_*.{out,err}
    └── evaluate_*.{out,err}
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Monitor pipeline progress
tail -f logs/run_*/pipeline_*.out

# Monitor individual components
tail -f logs/run_*/finetune_*.out
tail -f logs/run_*/inference_*.out
tail -f logs/run_*/evaluate_*.out
```

## Complete Workflow Examples

### Standard Workflow
```bash
# 1. Run with default configuration
./run.sh

# 2. Monitor progress
squeue -u $USER
tail -f logs/run_*/pipeline_*.out

# 3. Check results
ls ./llama_finetune/              # Model checkpoints
ls ./inference_results/           # Generated reductions  
ls ./evaluation_results/          # LLM judge evaluations
```

### Chain-of-Thought Workflow
```bash
# 1. Generate CoT data (one-time setup)
./run_cot_generation.sh ~/data/karp.csv data/karp_cot.csv

# 2. Run CoT pipeline with optimized config
./run.sh config_cot.sh --cot

# 3. Compare with standard results
./run.sh config.sh               # Standard mode
./run.sh config_cot.sh --cot     # CoT mode
```

### Individual Component Testing
```bash
# Test fine-tuning only (with all required parameters)
scripts/finetune.sh "meta-llama/Llama-3.3-70B-Instruct" \
    "~/data/karp.csv" "./test_output" 2 16 1e-5 5 4096 8 16 0.1 --cot

# Test inference only (after training)
scripts/inference.sh "meta-llama/Llama-3.3-70B-Instruct" \
    "~/data/karp.csv" "./test_output" "./test_inference" 4096 --cot

# Test evaluation only (after inference)  
scripts/evaluate.sh "./test_inference" \
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1" "./test_eval" 4096 --cot
```