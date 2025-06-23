# LLM Pipeline Configuration

This pipeline now uses a centralized configuration system for easier parameter management.

## Quick Start

1. **Use default configuration:**
   ```bash
   ./run.sh
   ```

2. **Use custom configuration:**
   ```bash
   ./run.sh /path/to/your/config.sh
   ```

## Configuration Files

### Default Configuration
- **File:** `config.sh`
- Contains all default parameters for the pipeline
- Automatically loaded if no config file is specified

### Example Custom Configuration
- **File:** `config.example.sh`
- Shows how to create a custom configuration
- Copy and modify this file for your experiments

## Configuration Parameters

### Model Settings
- `MODEL_NAME`: The base model to fine-tune
- `JUDGE_MODEL`: Model used for evaluation

### Data Paths
- `CSV_PATH`: Path to your training dataset

### Output Directories
- `OUTPUT_DIR`: Where to save the fine-tuned model
- `INFERENCE_OUTPUT`: Where to save inference results
- `EVAL_OUTPUT`: Where to save evaluation results

### Training Parameters
- `BATCH_SIZE`: Batch size per device
- `GRAD_ACCUM`: Gradient accumulation steps
- `LEARNING_RATE`: Learning rate for training
- `EPOCHS`: Number of training epochs
- `MAX_LENGTH`: Maximum sequence length for training

### Evaluation Parameters
- `EVAL_MAX_LENGTH`: Maximum sequence length for evaluation

## Creating Custom Configurations

1. Copy the example config:
   ```bash
   cp config.example.sh my_experiment.sh
   ```

2. Edit the parameters in `my_experiment.sh`

3. Run with your custom config:
   ```bash
   ./run.sh my_experiment.sh
   ```

## Benefits

- **Centralized:** All parameters in one place
- **Reusable:** Easy to share and version control configurations
- **Flexible:** Override defaults without modifying core scripts
- **Simple:** No complex command-line argument parsing
