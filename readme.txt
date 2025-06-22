# LLM Reduction Experiments Pipeline

This pipeline implements a complete machine learning workflow for training and evaluating models on computational complexity reductions.

## Pipeline Components

1. **Fine-tuning** (`src/finetune.py`, `scripts/finetune.sh`): 
   - Trains a large language model (default: Llama-3.3-70B) on reduction datasets
   - Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
   - Automatically splits data into train/validation/test sets

2. **Inference** (`src/inference.py`, `scripts/inference.sh`):
   - Generates reductions using the fine-tuned model
   - Runs on held-out test and validation sets
   - Saves results in CSV format for evaluation

3. **Evaluation** (`src/evaluate.py`, `scripts/evaluate.sh`):
   - Uses LLM-as-a-judge approach to evaluate generated reductions
   - Compares generated reductions against ground truth
   - Scores on correctness, completeness, clarity, and efficiency
   - Provides ACCEPT/PARTIAL/REJECT decisions with confidence scores

## Usage

Basic usage:
```bash
./run.sh
```

Full parameter specification:
```bash
./run.sh [MODEL_NAME] [CSV_PATH] [OUTPUT_DIR] [BATCH_SIZE] [GRAD_ACCUM] [LEARNING_RATE] [EPOCHS] [MAX_LENGTH] [INFERENCE_OUTPUT] [TEST_SET] [JUDGE_MODEL] [EVAL_OUTPUT]
```

### Parameters:
- `MODEL_NAME`: Base model (default: "meta-llama/Llama-3.3-70B-Instruct")
- `CSV_PATH`: Path to dataset CSV (default: "~/data/karp.csv")
- `OUTPUT_DIR`: Fine-tuning output directory (default: "./llama_finetune")
- `BATCH_SIZE`: Batch size per device (default: 1)
- `GRAD_ACCUM`: Gradient accumulation steps (default: 16)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `EPOCHS`: Number of training epochs (default: 30)
- `MAX_LENGTH`: Maximum sequence length (default: 2048)
- `INFERENCE_OUTPUT`: Inference results directory (default: "./inference_results")
- `TEST_SET`: Which sets to evaluate ("test", "validation", or "both", default: "both")
- `JUDGE_MODEL`: Model for evaluation (default: "meta-llama/Llama-3.3-70B-Instruct")
- `EVAL_OUTPUT`: Evaluation results directory (default: "./evaluation_results")

## Output Structure

```
llama_finetune/              # Fine-tuned model
├── checkpoint-*/            # Training checkpoints
├── held_out_indices.json   # Test/validation splits
└── trainer_state.json      # Training logs

inference_results/           # Inference outputs
├── inference_results_test.csv
└── inference_results_validation.csv

evaluation_results/          # Evaluation results
├── evaluation_inference_results_test.csv
├── evaluation_inference_results_validation.csv  
├── evaluation_summary_inference_results_test.json
└── evaluation_summary_inference_results_validation.json

logs/                       # SLURM job logs
└── run_YYYYMMDD_HHMMSS/   # Timestamped log directory
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