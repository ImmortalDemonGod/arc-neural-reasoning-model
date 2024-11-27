import typer
from enum import Enum
from typing import Optional
import train
from argparse import Namespace

# Create app with underscore style 
app = typer.Typer(
    help="Train the ARC Neural Reasoning Model",
    options_metavar="[--option_name=VALUE]"  # Use underscore style in help text
)

# Enums remain the same since they're Python classes
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

def validate_splits(ctx: typer.Context, splits: tuple[float, float, float]) -> tuple[float, float, float]:
    train_split, val_split, test_split = splits
    if not abs(sum(splits) - 1.0) < 1e-6:
        raise typer.BadParameter("Split proportions must sum to 1.0")
    return splits

@app.command()
def train_model(
    # Required parameters
    max_epochs: int = typer.Option(..., "--max_epochs", help="Maximum number of epochs"),

    # Model architecture
    n_embd: int = typer.Option(12, "--n_embd", help="Embedding size for the model"),
    n_head: int = typer.Option(1, "--n_head", help="Number of attention heads"),
    n_layer: int = typer.Option(4, "--n_layer", help="Number of total layers (transformer and mamba)"),
    mamba_ratio: float = typer.Option(
        1.0,
        "--mamba_ratio",
        callback=validate_mamba_ratio,
        help="Proportion of Mamba layers (0.0-1.0)"
    ),
    dropout: float = typer.Option(0.05, "--dropout", help="Dropout rate"),
    d_state: int = typer.Option(4, "--d_state", help="Mamba state dimension"),
    d_conv: int = typer.Option(1, "--d_conv", help="Mamba convolution dimension"),
    mamba_depth: int = typer.Option(1, "--mamba_depth", help="Depth of each Mamba layer"),
    mamba_expand: int = typer.Option(2, "--mamba_expand", help="Expand factor for each Mamba layer"),

    # Training configuration
    batch_size: int = typer.Option(16, "--batch_size", help="Training batch size"),
    learning_rate: float = typer.Option(1e-4, "--learning_rate", help="Learning rate"),
    val_check_interval: float = typer.Option(0.01, "--val_check_interval", help="Validation check interval as fraction of epoch or steps"),
    
    # Data loading
    num_workers: int = typer.Option(4, "--num_workers", help="Number of worker threads for DataLoader"),
    prefetch_factor: int = typer.Option(2, "--prefetch_factor", help="Number of batches to prefetch per worker"),
    no_persistent_workers: bool = typer.Option(False, "--no_persistent_workers", help="Disable persistent workers in DataLoader"),
    pin_memory: bool = typer.Option(True, "--pin_memory", help="Enable pin_memory in DataLoader"),
    max_train_samples: Optional[int] = typer.Option(None, "--max_train_samples", help="Maximum number of training samples to use"),
    
    # Hardware and performance
    accelerator: Accelerator = typer.Option(
        Accelerator.GPU,
        "--accelerator",
        help="Accelerator to use for training"
    ),
    matmul_precision: MatmulPrecision = typer.Option(
        MatmulPrecision.MEDIUM,
        "--matmul_precision",
        help="Matrix multiplication precision"
    ),
    use_gpu: bool = typer.Option(False, "--use_gpu", help="Use GPU for training if available"),

    # Development options 
    use_profiler: bool = typer.Option(False, "--use_profiler", help="Enable the custom profiler"),
    fast_dev_run: bool = typer.Option(False, "--fast_dev_run", help="Run a fast development test"),

    # Grokfast configuration
    use_grokfast: bool = typer.Option(False, "--use_grokfast", help="Enable Grokfast for gradient filtering"),
    grokfast_type: GrokfastType = typer.Option(
        GrokfastType.EMA,
        "--grokfast_type",
        help="Type of Grokfast filter to use"
    ),
    grokfast_alpha: float = typer.Option(0.98, "--grokfast_alpha", help="Alpha parameter for Grokfast-EMA"),
    grokfast_lamb: float = typer.Option(2.0, "--grokfast_lamb", help="Lambda parameter for Grokfast filters"),
    grokfast_window_size: int = typer.Option(100, "--grokfast_window_size", help="Window size for Grokfast-MA"),

    # Data configuration
    use_synthetic_data: bool = typer.Option(False, "--use_synthetic_data", help="Use synthetic data for training"),
    synthetic_data_path: Optional[str] = typer.Option(None, "--synthetic_data_path", help="Path to synthetic data directory"),
    train_split: float = typer.Option(0.8, "--train_split", help="Proportion of data for training"),
    val_split: float = typer.Option(0.1, "--val_split", help="Proportion of data for validation"),
    test_split: float = typer.Option(0.1, "--test_split", help="Proportion of data for testing"),
    enable_symbol_freq: bool = typer.Option(False, "--enable_symbol_freq", help="Enable symbol frequency calculation"),

    # Loss and accuracy configuration
    include_pad_in_loss: bool = typer.Option(True, "--include_pad_in_loss", help="Include padding class in loss calculation"),
    include_pad_in_accuracy: bool = typer.Option(True, "--include_pad_in_accuracy", help="Include padding class in accuracy calculations"),

    # Logging and checkpointing
    log_level: str = typer.Option("INFO", "--log_level", help="Logging level"),
    no_logging: bool = typer.Option(False, "--no_logging", help="Disable logging"),
    no_checkpointing: bool = typer.Option(False, "--no_checkpointing", help="Disable checkpointing"),
    no_progress_bar: bool = typer.Option(False, "--no_progress_bar", help="Disable progress bar"),
    model_checkpoint: Optional[str] = typer.Option(None, "--model_checkpoint", help="Path to model checkpoint to resume training"),
    results_dir: str = typer.Option("./results", "--results_dir", help="Directory to save results"),
    run_name: str = typer.Option("default_run", "--run_name", help="Name of the run for saving results"),
    project: str = typer.Option("gpt2-arc", "--project", help="W&B project name"),

    # Profiling options
    profiler_dirpath: str = typer.Option("./profiler_logs", "--profiler_dirpath", help="Directory for profiler output"),
    profiler_filename: str = typer.Option("profile", "--profiler_filename", help="Filename for profiler output"),

    # Optuna configuration
    use_optuna: bool = typer.Option(False, "--use_optuna", help="Use best hyperparameters from Optuna study"),
    optuna_storage: str = typer.Option(
        "sqlite:///optuna_results.db",
        "--optuna_storage",
        help="Storage URL for the Optuna study"
    ),
    optuna_study_name: Optional[str] = typer.Option(
        None,
        "--optuna_study_name", 
        help="Name of the Optuna study to load"
    ),
):
    """
    Train the ARC Neural Reasoning Model with the specified configuration.
    
    This command provides a comprehensive interface for training the model with various
    configuration options for model architecture, training process, data loading,
    and performance optimization.
    """
    # Validate split proportions
    validate_splits(None, (train_split, val_split, test_split))
    
    # Create args namespace to maintain compatibility with existing train.py
    args = Namespace(
        max_epochs=max_epochs,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        mamba_ratio=mamba_ratio,
        dropout=dropout,
        d_state=d_state,
        d_conv=d_conv,
        mamba_depth=mamba_depth,
        mamba_expand=mamba_expand,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_check_interval=val_check_interval,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        no_persistent_workers=no_persistent_workers,
        pin_memory=pin_memory,
        max_train_samples=max_train_samples,
        accelerator=accelerator.value,
        matmul_precision=matmul_precision.value,
        use_gpu=use_gpu,
        use_grokfast=use_grokfast,
        grokfast_type=grokfast_type.value,
        grokfast_alpha=grokfast_alpha,
        grokfast_lamb=grokfast_lamb,
        grokfast_window_size=grokfast_window_size,
        use_synthetic_data=use_synthetic_data,
        synthetic_data_path=synthetic_data_path,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        enable_symbol_freq=enable_symbol_freq,
        include_pad_in_loss=include_pad_in_loss,
        include_pad_in_accuracy=include_pad_in_accuracy,
        log_level=log_level,
        no_logging=no_logging,
        no_checkpointing=no_checkpointing,
        no_progress_bar=no_progress_bar,
        model_checkpoint=model_checkpoint,
        results_dir=results_dir,
        run_name=run_name,
        project=project,
        use_profiler=use_profiler,
        fast_dev_run=fast_dev_run,
        profiler_dirpath=profiler_dirpath,
        profiler_filename=profiler_filename,
        use_optuna=use_optuna,
        optuna_storage=optuna_storage,
        optuna_study_name=optuna_study_name,
    )
    
    # Call the existing train.main() function
    train.main(args)

if __name__ == "__main__":
    app()