# GPT-2 ARC Neural Reasoning Model

This project implements a neural reasoning model based on the GPT-2 architecture to solve tasks from the Abstraction and Reasoning Corpus (ARC) challenge.

## Features

- **Data Handling**: Utilizes a custom `ArcDataset` class for handling and preprocessing ARC data.
- **Model Architecture**: Implements a `GPT2ARC` model leveraging the pre-trained GPT-2 architecture.
- **Training**: Includes a `train.py` script for training the model using PyTorch Lightning, with support for logging, checkpointing, and hyperparameter tuning via Optuna.
- **Evaluation**: Provides an `evaluate.py` script to assess the trained model on test datasets.
- **Testing**: Comprehensive test suite using `pytest` to ensure model and data integrity.
- **Experiment Tracking**: Integrates experiment tracking and results collection mechanisms for better experiment management.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/arc-neural-reasoning-model.git
cd arc-neural-reasoning-model
pip install -e .
```

**Note**: If you plan to use experiment tracking with Weights & Biases (W&B), ensure you have a W&B account and have logged in:

```bash
wandb login
```

For development, install the extra dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Training the Model

The `train.py` script is the main entry point for training the GPT-2 ARC Neural Reasoning Model. It leverages PyTorch Lightning for streamlined training, incorporates hyperparameter tuning with Optuna, and integrates experiment tracking and result collection.

#### **Prerequisites**

- Ensure you have the necessary data available. You can use the standard ARC dataset or synthetic data.

#### **Basic Training Command**

Run the training loop for a specified number of epochs with default hyperparameters:

```bash
python src/training/train.py --max-epochs 10
```

#### **Common Command-Line Arguments**

- `--use-optuna`: Enable loading the best hyperparameters from an Optuna study.
- `--optuna-study-name`: Name of the Optuna study to load (default: `gpt2_arc_optimization`).
- `--optuna-storage`: Storage URL for the Optuna study (default: `sqlite:///optuna_results.db`).
- `--n-embd`: Embedding dimension (default: `64`).
- `--n-head`: Number of attention heads (default: `4`).
- `--n-layer`: Number of transformer layers (default: `4`).
- `--batch-size`: Batch size for training (default: `16`).
- `--learning-rate`: Learning rate (default: `1e-4`).
- `--max-epochs`: **(Required)** Maximum number of training epochs.
- `--use-gpu`: Utilize GPU for training if available.
- `--no-logging`: Disable logging.
- `--no-checkpointing`: Disable model checkpointing.
- `--no-progress-bar`: Disable progress bar during training.
- `--fast-dev-run`: Execute a fast development test run.
- `--model_checkpoint`: Path to a model checkpoint to resume training.
- `--project`: W&B project name for experiment tracking (default: `gpt2-arc`).
- `--results-dir`: Directory to save results (default: `./results`).
- `--run-name`: Name of the run for saving results (default: `default_run`).
- `--use-synthetic-data`: Use synthetic data for training.
- `--synthetic-data-path`: Path to the synthetic data directory.
- `--log-level`: Logging level (default: `INFO`).

#### **Example Commands**

1. **Training with Optuna Hyperparameters**

    ```bash
    python src/training/train.py --max-epochs 20 --use-optuna --optuna-study-name my_study --optuna-storage sqlite:///my_optuna.db
    ```

2. **Training with Synthetic Data and GPU**

    ```bash
    python src/training/train.py --max-epochs 15 --use-synthetic-data --synthetic-data-path path/to/synthetic_data --use-gpu
    ```

3. **Resuming Training from a Checkpoint**

    ```bash
    python src/training/train.py --max-epochs 25 --model_checkpoint path/to/checkpoint.ckpt
    ```

#### **Best Practices**

- **Start with Default Settings**: Begin with default configurations to ensure the setup is correct.
- **Use Virtual Environments**: Manage dependencies using virtual environments to avoid conflicts.
    ```bash
    python -m venv env
    source env/bin/activate
    ```
- **Monitor Training with TensorBoard**: Launch TensorBoard to visualize training progress.
    ```bash
    tensorboard --logdir=runs
    ```
- **Experiment with Hyperparameters**: Utilize Optuna for hyperparameter tuning to optimize model performance.
- **Leverage Checkpointing**: Enable checkpointing to save model states and resume training if interrupted.
- **Handle Resources Appropriately**: Ensure GPU resources are available and adjust `batch-size` based on GPU memory.

### Evaluating the Model

The `evaluate.py` script assesses the performance of the GPT-2 ARC Neural Reasoning Model on a test dataset from the Abstraction and Reasoning Corpus (ARC).

#### **Prerequisites**

- **Model Checkpoint**: Ensure you have a trained model checkpoint file (`.ckpt`).
- **Test Data**: Access to the ARC test dataset compatible with the `ARCDataset` class.
- **Weights & Biases (W&B)**: *(Optional)* For experiment tracking, ensure you have a W&B account and have logged in using `wandb login`.

#### **Basic Evaluation Command**

Run the following command to evaluate the model:

```bash
python src/evaluate.py --model_checkpoint path/to/model_checkpoint.ckpt --batch_size 32
```

#### **Common Command-Line Arguments**

- `--model_checkpoint` (**Required**): Path to the model checkpoint file.
- `--batch_size`: Batch size for evaluation (default: `32`).
- `--output_dir`: Directory to save evaluation results (default: `./evaluation_results`).
- `--log-level`: Logging level (`DEBUG`, `INFO`, etc.; default: `INFO`).
- `--wandb_project`: W&B project name for experiment tracking (default: `arc-evaluation`).
- `--wandb_run_name`: W&B run name *(optional)*.

#### **Example Command**

```bash
python src/evaluate.py \
  --model_checkpoint path/to/model_checkpoint.ckpt \
  --batch_size 64 \
  --output_dir ./my_eval_results \
  --log-level DEBUG \
  --wandb_project my_project \
  --wandb_run_name evaluation_run_01
```

#### **Best Practices**

- **Consistent Environment**: Use the same Python environment and dependencies as used during training to ensure compatibility.
- **Verify Checkpoint Integrity**: Ensure the model checkpoint is not corrupted and matches the model architecture.
- **Utilize W&B for Tracking**: Leverage W&B to monitor evaluation metrics in real-time and maintain experiment logs.
- **Organize Evaluation Results**: Structure your `output_dir` to systematically store and access evaluation results for easy reference.

## Running Tests

To run the tests, use the following command:

```
pytest -v
```

This will run all tests and display the results, including test coverage.

## Contributing

[Add contribution guidelines here]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
