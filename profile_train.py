
import argparse
import logging
import os
import sys

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader

# Import your project's modules
# Adjust the import paths based on your project structure
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.trainer import ARCTrainer, get_num_workers
from gpt2_arc.src.utils.results_collector import ResultsCollector
from gpt2_arc.src.utils.grokfast_callback import GrokfastCallback
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs; adjust as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_minimal_config(args):
    """
    Set up a minimal configuration for profiling purposes.
    """
    # Define minimal model configuration
    model_config = ModelConfig(
        n_embd=8,               # Minimal embedding size
        n_head=1,               # Single attention head
        n_layer=2,              # Two layers (Transformer or Mamba)
        dropout=0.1,            # Reduced dropout
        mamba_ratio=0.5,        # Balanced between Transformer and Mamba layers
        d_state=4,
        d_conv=2,
        mamba_depth=1,
        mamba_expand=2
    )

    # Define minimal training configuration
    training_config = TrainingConfig(
        batch_size=4,                        # Small batch size
        learning_rate=1e-3,                  # Higher learning rate for faster convergence
        max_epochs=1,                        # Single epoch for quick profiling
        use_gpu=args.use_gpu,                # Utilize GPU if available
        log_level="INFO",                    # Set appropriate logging level
        use_synthetic_data=True,             # Use synthetic data for speed
        synthetic_data_path=args.synthetic_data_path,  # Path to synthetic data
        include_pad_in_loss=False,           # Optional: exclude padding from loss
        include_pad_in_accuracy=False,       # Optional: exclude padding from accuracy
        num_workers=2,                       # Reduced number of workers
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True if args.use_gpu else False,
        use_grokfast=False,                  # Disable Grokfast for minimal setup
        symbol_freq=None,
        balance_symbols=False,
        balancing_method="none"
    )

    # Combine into a single Config object
    config = Config(model=model_config, training=training_config)
    return config


def prepare_datasets(config, args):
    """
    Prepare training, validation, and test datasets.
    """
    logger.info("Preparing datasets for profiling.")

    # Initialize synthetic training dataset
    train_dataset = ARCDataset(
        data_source=config.training.synthetic_data_path,
        is_test=False,
        max_samples=100,  # Limit number of samples for profiling
        num_symbols=config.training.num_symbols if hasattr(config.training, 'num_symbols') else 11,
        pad_symbol_idx=config.training.pad_symbol_idx if hasattr(config.training, 'pad_symbol_idx') else 10,
        symbol_freq=config.training.symbol_freq
    )

    # Initialize synthetic validation dataset
    val_dataset = ARCDataset(
        data_source=config.training.synthetic_data_path,
        is_test=True,  # Using same synthetic data for simplicity
        max_samples=50,
        num_symbols=config.training.num_symbols if hasattr(config.training, 'num_symbols') else 11,
        pad_symbol_idx=config.training.pad_symbol_idx if hasattr(config.training, 'pad_symbol_idx') else 10,
        symbol_freq=config.training.symbol_freq
    )

    # Initialize synthetic test dataset
    test_dataset = ARCDataset(
        data_source=config.training.synthetic_data_path,
        is_test=True,
        max_samples=50,
        num_symbols=config.training.num_symbols if hasattr(config.training, 'num_symbols') else 11,
        pad_symbol_idx=config.training.pad_symbol_idx if hasattr(config.training, 'pad_symbol_idx') else 10,
        symbol_freq=config.training.symbol_freq
    )

    return train_dataset, val_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Profile the ARC Neural Reasoning Model with Minimal Parameters")
    parser.add_argument(
        "--use_profiler",
        action="store_true",
        help="Enable the profiler for detailed performance analysis."
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Conduct a fast development run with a single batch and epoch."
    )
    parser.add_argument(
        "--synthetic_data_path",
        type=str,
        required=True,
        help="Path to the synthetic data directory or file."
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for profiling if available."
    )
    parser.add_argument(
        "--profiler_dirpath",
        type=str,
        default="./profiler_logs",
        help="Directory path for profiler output files."
    )
    parser.add_argument(
        "--profiler_filename",
        type=str,
        default="profile",
        help="Filename for profiler output."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/profiling_experiment",
        help="Directory for TensorBoard logs."
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",  # Default to 'gpu'; can override to 'cpu' if needed
        choices=["cpu", "gpu", "tpu"],
        help="Accelerator to use for profiling: 'cpu', 'gpu', or 'tpu'. Defaults to 'gpu'."
    )
    
    args = parser.parse_args()

    logger.info("Starting profiling with minimal parameters.")

    # Set up minimal configuration
    config = set_minimal_config(args)
    logger.debug(f"Minimal configuration set: {config}")

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(config, args)

    # Initialize ResultsCollector
    results_collector = ResultsCollector(config)

    # Initialize ExperimentTracker
    tracker = ExperimentTracker(config, project="gpt2-arc-profiling")

    # Initialize the model
    model = GPT2ARC(
        config=config,
        num_classes=11,  # Adjust based on your synthetic data
        symbol_freq=config.training.symbol_freq,
        pad_symbol_idx=config.training.pad_symbol_idx
    )

    # Initialize ARCTrainer
    trainer_module = ARCTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        args=args,
        results_collector=results_collector,
        test_dataset=test_dataset
    )

    # Set Hyperparameters
    trainer_module.log_hyperparameters()

    # Initialize TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f"profiling_experiment_{results_collector.experiment_id}"
    )

    # Initialize Callbacks
    callbacks = []

    # Add ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="profiling-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    callbacks.append(checkpoint_callback)

    # Determine profiler activities based on the selected accelerator
    if args.accelerator.lower() == "cpu":
        profiler_activities = [ProfilerActivity.CPU]
    elif args.accelerator.lower() == "gpu":
        profiler_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    elif args.accelerator.lower() == "tpu":
        profiler_activities = [ProfilerActivity.CPU]  # TPU-specific activities can be added if supported
    else:
        profiler_activities = []

    # Debug log for selected activities
    logger.debug(f"Selected profiler activities based on accelerator '{args.accelerator}': {profiler_activities}")

    # Initialize Profiler with conditional activities
    profiler = PyTorchProfiler(
        dirpath=args.profiler_dirpath,
        filename=args.profiler_filename,
        activities=profiler_activities,  # Set activities based on accelerator
        record_shapes=True,
        with_stack=True
    ) if args.use_profiler and profiler_activities else None

    # Initialize PyTorch Lightning Trainer with profiler
    pl_trainer = Trainer(
        max_epochs=config.training.max_epochs,
        logger=tb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        precision=16,  # Use mixed precision
        accelerator=args.accelerator,  # Use the newly added argument
        devices=1 if (args.use_gpu and torch.cuda.is_available()) or args.accelerator == 'cpu' else None,
        profiler=profiler,
        val_check_interval=0.1  # Frequent validation for profiling
    )

    # Start Training
    logger.info("Starting training for profiling.")
    pl_trainer.fit(trainer_module)

    # Run Testing
    logger.info("Starting testing after profiling.")
    pl_trainer.test(trainer_module, dataloaders=DataLoader(test_dataset, batch_size=config.training.batch_size))

    # Finish Experiment Tracking
    if 'tracker' in locals():
        tracker.finish()

    logger.info("Profiling completed successfully.")


if __name__ == "__main__":
    main()
