import typer
from enum import Enum
from typing import Optional, List
from gpt2_arc.src.optimization.optimizer import run_optimization
import random
import torch
import numpy as np
import os
import sys
import logging
from argparse import Namespace
from gpt2_arc.src.optimization.utils.logging_config import configure_logging



# Create app with underscore style 
app = typer.Typer(
    help="Optimize hyperparameters for the ARC Neural Reasoning Model",
    options_metavar="[--option_name=VALUE]"  # Use underscore style in help text
)

# Reuse enums from train_cli for consistency
class MatmulPrecision(str, Enum):
    HIGHEST = "highest"
    HIGH = "high"
    MEDIUM = "medium"

class Accelerator(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"

class GrokfastType(str, Enum):
    EMA = "ema"
    MA = "ma"

def validate_mamba_ratio(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise typer.BadParameter("mamba_ratio must be between 0.0 and 1.0")
    return value

def validate_splits(train_split: float, val_split: float, test_split: float) -> None:
    if not abs(train_split + val_split + test_split - 1.0) < 1e-6:
        raise typer.BadParameter("Split proportions must sum to 1.0")

@app.command()
def optimize(
    # Optimization specific parameters
    n_trials: int = typer.Option(10, "--n_trials", help="Number of trials for optimization"),
    n_jobs: int = typer.Option(1, "--n_jobs", help="Number of parallel jobs. -1 means using all available cores"),
    study_name: str = typer.Option("gpt2_arc_optimization_v3", "--study_name", help="Name of the Optuna study"),
    storage: str = typer.Option("sqlite:///optuna_results.db", "--storage", help="Storage path for Optuna results"),
    random_seed: int = typer.Option(42, "--random_seed", help="Random seed for reproducibility"),
    
    # Hyperparameter ranges
    batch_size_min: int = typer.Option(1, "--batch_size_min", help="Minimum value for batch_size"),
    batch_size_max: int = typer.Option(32, "--batch_size_max", help="Maximum value for batch_size"),
    learning_rate_min: float = typer.Option(1e-5, "--learning_rate_min", help="Minimum value for learning_rate"),
    learning_rate_max: float = typer.Option(1e-2, "--learning_rate_max", help="Maximum value for learning_rate"),
    n_head_min: int = typer.Option(1, "--n_head_min", help="Minimum value for n_head"),
    n_head_max: int = typer.Option(1, "--n_head_max", help="Maximum value for n_head"),
    n_head_exp_min: int = typer.Option(1, "--n_head_exp_min", help="Minimum exponent for n_head (2^x)"),
    n_head_exp_max: int = typer.Option(4, "--n_head_exp_max", help="Maximum exponent for n_head (2^x)"),
    n_embd_max: int = typer.Option(1, "--n_embd_max", help="Maximum value for n_embd"),
    n_embd_multiplier_min: int = typer.Option(1, "--n_embd_multiplier_min", help="Minimum multiplier for n_embd"),
    n_embd_multiplier_max: int = typer.Option(4, "--n_embd_multiplier_max", help="Maximum multiplier for n_embd"),
    n_layer_min: int = typer.Option(1, "--n_layer_min", help="Minimum value for n_layer"),
    n_layer_max: int = typer.Option(4, "--n_layer_max", help="Maximum value for n_layer"),
    max_epochs_min: int = typer.Option(1, "--max_epochs_min", help="Minimum value for max_epochs"),
    max_epochs_max: int = typer.Option(10, "--max_epochs_max", help="Maximum value for max_epochs"),
    dropout_min: float = typer.Option(0.0, "--dropout_min", help="Minimum value for dropout"),
    dropout_max: float = typer.Option(0.5, "--dropout_max", help="Maximum value for dropout"),
    dropout_step: float = typer.Option(0.1, "--dropout_step", help="Step size for dropout"),
    
    # Mamba specific ranges
    mamba_ratio_min: float = typer.Option(1.0, "--mamba_ratio_min", help="Minimum value for mamba_ratio"),
    mamba_ratio_max: float = typer.Option(8.0, "--mamba_ratio_max", help="Maximum value for mamba_ratio"),
    mamba_ratio_step: float = typer.Option(0.25, "--mamba_ratio_step", help="Step size for mamba_ratio"),
    d_state_min: int = typer.Option(1, "--d_state_min", help="Minimum value for d_state"),
    d_state_max: int = typer.Option(16, "--d_state_max", help="Maximum value for d_state"),
    d_conv_min: int = typer.Option(1, "--d_conv_min", help="Minimum value for d_conv"),
    d_conv_max: int = typer.Option(4, "--d_conv_max", help="Maximum value for d_conv"),
    mamba_depth_min: int = typer.Option(1, "--mamba_depth_min", help="Minimum value for mamba_depth"),
    mamba_depth_max: int = typer.Option(4, "--mamba_depth_max", help="Maximum value for mamba_depth"),
    mamba_expand_min: int = typer.Option(2, "--mamba_expand_min", help="Minimum value for mamba_expand"),
    mamba_expand_max: int = typer.Option(4, "--mamba_expand_max", help="Maximum value for mamba_expand"),
    
    # Grokfast ranges
    grokfast_alpha_min: float = typer.Option(0.9, "--grokfast_alpha_min", help="Minimum value for grokfast_alpha"),
    grokfast_alpha_max: float = typer.Option(0.99, "--grokfast_alpha_max", help="Maximum value for grokfast_alpha"),
    grokfast_lamb_min: float = typer.Option(1.0, "--grokfast_lamb_min", help="Minimum value for grokfast_lamb"),
    grokfast_lamb_max: float = typer.Option(3.0, "--grokfast_lamb_max", help="Maximum value for grokfast_lamb"),
    grokfast_window_size_min: int = typer.Option(50, "--grokfast_window_size_min", help="Minimum value for grokfast_window_size"),
    grokfast_window_size_max: int = typer.Option(200, "--grokfast_window_size_max", help="Maximum value for grokfast_window_size"),
    grokfast_type_choices: List[str] = typer.Option(["ema", "ma"], "--grokfast_type_choices", help="List of Grokfast types to consider during tuning"),
    
    # Data configuration
    train_split: float = typer.Option(0.8, "--train_split", help="Proportion of data for training"),
    val_split: float = typer.Option(0.1, "--val_split", help="Proportion of data for validation"),
    test_split: float = typer.Option(0.1, "--test_split", help="Proportion of data for testing"),
    max_train_samples: Optional[int] = typer.Option(None, "--max_train_samples", help="Maximum number of training samples to use"),
    
    # Training configuration
    val_check_interval: float = typer.Option(0.01, "--val_check_interval", 
        help=(
            "How often to perform validation. "
            "If a float, represents the fraction of an epoch (e.g., 0.5 for halfway through each epoch). "
            "If an integer, represents the number of training steps."
        )
    ),
    include_pad_in_loss: bool = typer.Option(
        True, 
        "--include_pad_in_loss",
        help="Whether to include the padding class in the loss calculation"
    ),
    model_checkpoint: Optional[str] = typer.Option(None, "--model_checkpoint", help="Path to model checkpoint to resume optimization from"),
    
    # Hardware and performance settings
    use_gpu: bool = typer.Option(False, "--use_gpu", help="Use GPU for training if available"),
    accelerator: Accelerator = typer.Option(Accelerator.GPU, "--accelerator", help="Accelerator to use for training"),
    matmul_precision: MatmulPrecision = typer.Option(MatmulPrecision.MEDIUM, "--matmul_precision", help="Matrix multiplication precision"),
    num_workers: Optional[int] = typer.Option(None, "--num_workers", help="Number of worker threads for DataLoader"),
    prefetch_factor: int = typer.Option(2, "--prefetch_factor", help="Number of batches to prefetch per worker"),
    
    # Feature flags
    fast_dev_run: bool = typer.Option(False, "--fast_dev_run", help="Run a fast development test"),
    use_grokfast: bool = typer.Option(False, "--use_grokfast", help="Enable Grokfast gradient filtering"),
    use_synthetic_data: bool = typer.Option(False, "--use_synthetic_data", help="Use synthetic data for training"),
    enable_symbol_freq: bool = typer.Option(False, "--enable_symbol_freq", help="Enable symbol frequency calculation"),
    no_persistent_workers: bool = typer.Option(False, "--no_persistent_workers", help="Disable persistent workers in DataLoader"),
    no_pin_memory: bool = typer.Option(False, "--no_pin_memory", help="Disable pin_memory in DataLoader"),
    no_progress_bar: bool = typer.Option(False, "--no_progress_bar", help="Disable progress bar"),
    
    # Paths and logging
    synthetic_data_path: str = typer.Option("", "--synthetic_data_path", help="Path to synthetic data for training"),
    log_level: str = typer.Option("INFO", "--log_level", help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    project: str = typer.Option("gpt2-arc-optimization", "--project", help="Project name for experiment tracking"),
):
    """
    Run hyperparameter optimization for the ARC Neural Reasoning Model.
    
    This command provides a comprehensive interface for optimizing model hyperparameters
    using Optuna, with support for parallel trials and various search space configurations.
    """
    # Configure logging first, before any other operations
    configure_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured successfully")
    # Validate split proportions
    validate_splits(train_split, val_split, test_split)

    # Validate val_check_interval
    if val_check_interval <= 0:
        typer.echo("The --val_check_interval must be a positive number.", err=True)
        raise typer.Exit(code=1)
    
    # Set up logging
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level_num,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level_num)
    
    # Handle storage path
    if not storage.startswith("sqlite:///"):
        if os.path.isabs(storage):
            storage = f"sqlite:////{storage}"
        else:
            storage = f"sqlite:///{os.path.abspath(storage)}"
    
    logger.debug(f"Optuna storage URL set to: {storage}")
    
    # Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    logger.debug(f"Random seed set to: {random_seed}")
    
    # Create args namespace for compatibility
    args = Namespace(**locals())
    
    # Log parsed arguments for debugging
    logger.debug(f"Parsed arguments: {vars(args)}")
    
    # Run optimization
    run_optimization(
        n_trials=n_trials,
        storage_name=storage,
        n_jobs=n_jobs,
        args=args,
        study_name=study_name
    )

if __name__ == "__main__":
    app()