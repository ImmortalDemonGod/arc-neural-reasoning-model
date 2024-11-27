# gpt2_arc/src/training/train.py
import argparse
import logging

logger = logging.getLogger(__name__)
from typing import Optional
import sys
import logging
import os
import datetime
import arckit
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_config_manager import ConfigurationManager

# Define the base directory for the arc-neural-reasoning-model
arc_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Add the root directory of the project to the PYTHONPATH
project_root = arc_model_dir
sys.path.insert(0, project_root)

import pytorch_lightning as pl
#import torch.autograd.profiler as profiler
from pytorch_lightning.callbacks import ModelCheckpoint

from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config
from gpt2_arc.src.training.trainer import ARCTrainer, get_num_workers
logger = logging.getLogger(__name__)
logger.debug(f"Imported get_num_workers from training_helpers: {get_num_workers}")
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
from gpt2_arc.src.utils.results_collector import ResultsCollector
from typing import List, Dict, Any
from gpt2_arc.src.training.data_manager import DataManager  # Add this import
logger = logging.getLogger(__name__)

class ConfigSavingModelCheckpoint(ModelCheckpoint):
    def __init__(self, config: Config, trial_num: str = 'NA', task_id: str = 'NA', iter_num: str = 'NA', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.trial_num = trial_num
        self.task_id = task_id
        self.iter_num = iter_num
        self.timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")  # e.g., 20240308T153045

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        # Add custom metadata to the checkpoint
        checkpoint['model_config'] = self.config.model.__dict__
        checkpoint['trial_num'] = self.trial_num
        checkpoint['task_id'] = self.task_id
        checkpoint['iter_num'] = self.iter_num
        checkpoint['timestamp'] = self.timestamp

        # Add the current epoch to the checkpoint
        checkpoint['epoch'] = trainer.current_epoch

        super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def format_checkpoint_name(self, metrics: dict) -> str:
        """
        Override the method to include custom placeholders in the filename.
        """
        return self.filename.format(
            trial_num=self.trial_num,
            task_id=self.task_id,
            iter_num=self.iter_num,
            val_loss=metrics.get("val_loss", 0.0),
            epoch=metrics.get("epoch", 0),
            timestamp=self.timestamp
        )


def main(args: argparse.Namespace) -> None:
    
    # Initialize configuration
    config_manager = ConfigurationManager(args)
    config_manager.setup_logging()  # Setup logging first
    config = config_manager.create_initial_config()

    # Validate configuration including Mamba layers
    config_manager.validate_configuration(config)


    logger.debug(f"Command line arguments: {args}")


        
    logger.info("Starting main function")
    logger.debug("Initializing PyTorch Lightning Trainer")
    logger.debug(f"Command line arguments: {args}")

    trainer = None  # Initialize trainer to None

    try:

        logger.debug(f"Configuration: {config}")

        # Initialize DataManager
        data_manager = DataManager(config, args)

        # Load all datasets
        train_data, val_data, test_data = data_manager.load_all_datasets()
        # Log the source of each dataset
        logger.info(f"Training dataset source: {'synthetic data' if args.use_synthetic_data else 'official ARC data'}")
        logger.info("Validation dataset source: official ARC data")
        logger.info("Test dataset source: official ARC data")
        
        # Log the number of samples in each dataset
        logger.debug(f"Number of training samples: {len(train_data)}")
        logger.debug(f"Number of validation samples: {len(val_data)}")
        logger.debug(f"Number of test samples: {len(test_data)}")

        # Calculate and update symbol frequencies if enabled
        if args.enable_symbol_freq:
            symbol_freq = train_data.get_symbol_frequencies()
            symbol_freq_dict = {i: float(freq) for i, freq in enumerate(symbol_freq)}
            config = config_manager.update_config_with_symbol_freq(config, symbol_freq_dict)


        # Set the number of classes based on TrainingConfig
        num_classes = config.training.num_classes
        logger.info(f"Number of classes set to: {num_classes}")

        # Ensure test_data is not None
        assert test_data is not None, "Test dataset is None after loading."

        # Initialize model
        logger.info("Initializing model")
        model = GPT2ARC(
            config=config,
            num_classes=config.training.num_classes,  # Use num_classes from config
            pad_symbol_idx=config.training.pad_symbol_idx
        )
        logger.debug(f"Model initialized with config: {config}")



        # Load the checkpoint if specified
        if args.model_checkpoint:
            logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
            config, state_dict = config_manager.load_checkpoint_config(args.model_checkpoint)
            
            model = GPT2ARC(
                config=config,
                num_classes=config.training.num_classes,
                symbol_freq=config.training.symbol_freq,
                pad_symbol_idx=config.training.pad_symbol_idx
            )
            model.load_state_dict(state_dict)
            logger.debug(f"Loaded model configuration with num_classes={config.training.num_classes}")


        # Initialize experiment tracker
        tracker = ExperimentTracker(config, project=args.project)

        results_collector = ResultsCollector(config)
        logger.info("Initializing ARCTrainer")
        trainer = ARCTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            config=config,
            args=args,
            results_collector=results_collector,
            test_dataset=test_data
        )
        trainer.log_hyperparameters()

        logger.info("Setting up PyTorch Lightning trainer")

        # Get profiler configuration
        profiler = config_manager.get_profiler_config()

        # Get callbacks
        callbacks = config_manager.get_callbacks(config, args.model_checkpoint)

        # Get TensorBoard logger
        tb_logger = config_manager.get_tensorboard_logger(trainer.results_collector.experiment_id)
        
        # Get accelerator config
        
        accelerator_config = config_manager.get_accelerator_config()

        # Later when creating pl_trainer:
        pl_trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=tb_logger,
            callbacks=callbacks if callbacks else None,
            enable_checkpointing=not args.no_checkpointing,
            enable_progress_bar=not args.no_progress_bar,
            fast_dev_run=args.fast_dev_run,
            gradient_clip_val=1.0,
            precision=16,
            **accelerator_config,  # Unpack accelerator configuration
            profiler=profiler,
            val_check_interval=args.val_check_interval
        )

        if tb_logger:
            trainer.results_collector.set_tensorboard_log_path(tb_logger.log_dir)
            logger.debug(f"TensorBoard log path set in results collector: {tb_logger.log_dir}")

        # Log initial memory usage
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved()} bytes")

        logger.info("Starting model training")
        logger.debug("Training parameters: ")
        logger.debug(f"Batch size: {config.training.batch_size}, Learning rate: {config.training.learning_rate}, Max epochs: {config.training.max_epochs}")

        # Train the model
        logger.info("Starting model training")
        # Update the fit call to exclude DataLoaders
        pl_trainer.fit(trainer)

        # Log memory usage after training
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"CUDA memory allocated after training: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"CUDA memory reserved after training: {torch.cuda.memory_reserved()} bytes")

        logger.info("Model training completed successfully.")

        # Get data loaders
        _, _, test_loader = data_manager.create_data_loaders(train_data, val_data, test_data)

        # After training, run test
        logger.info("Starting model evaluation on test dataset.")
        logger.info("Running model evaluation")
        logger.debug("Preparing to run Trainer.test()")
        test_results = pl_trainer.test(model=trainer, dataloaders=test_loader)
        if test_results:
            avg_test_loss = sum(result['avg_test_loss'] for result in test_results) / len(test_results)
            avg_test_accuracy = sum(result['avg_test_accuracy'] for result in test_results) / len(test_results)
            avg_test_diff_accuracy = sum(result['avg_test_diff_accuracy'] for result in test_results) / len(test_results)

            logger.info(f"Test results - Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.4f}, Diff Accuracy: {avg_test_diff_accuracy:.4f}")

            results = {
                "avg_test_loss": avg_test_loss,
                "avg_test_accuracy": avg_test_accuracy,
                "avg_test_diff_accuracy": avg_test_diff_accuracy,
            }

            # Add task-specific results
            for result in test_results:
                for key, value in result.items():
                    if key.endswith('_test_accuracy') or key.endswith('_test_diff_accuracy'):
                        results[key] = value

            trainer.results_collector.set_test_results(results)

        trainer.results_collector.set_final_metrics({
            "best_val_loss": trainer.best_val_loss,
            "best_epoch": trainer.best_epoch,
            "final_test_loss": avg_test_loss,
            "final_test_accuracy": avg_test_accuracy,
            "final_test_diff_accuracy": avg_test_diff_accuracy
        })

        # Save the final model with configuration
        logger.info("Saving final model with configuration")
        model_path = f"final_model_{trainer.results_collector.experiment_id}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'state_dict': trainer.model.state_dict(),
            'model_config': trainer.config.model.__dict__,
            'training_config': trainer.config.training.__dict__,
            'pad_symbol_idx': trainer.config.training.pad_symbol_idx,
            'symbol_freq': trainer.config.training.symbol_freq
        }, model_path)
        trainer.results_collector.set_checkpoint_path(model_path)
        logger.debug(f"Model and configuration saved to: {model_path}")

        # Save results
        logger.info("Saving experiment results")
        os.makedirs("results", exist_ok=True)
        results_path = f"results/experiment_{trainer.results_collector.experiment_id}.json"
        trainer.results_collector.save_to_json(results_path)
        logger.debug(f"Results saved to: {results_path}")

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            logger.error("CUDA out of memory error occurred.")
            logger.error("Consider reducing the batch size or model complexity.")
            raise RuntimeError("CUDA out of memory error occurred.")
        else:
            logger.error(f"A runtime error occurred: {str(e)}", exc_info=True)
            raise RuntimeError(f"A runtime error occurred: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        sys.exit(1)  # Exit the program after logging the error
    finally:
        if 'tracker' in locals():
            tracker.finish()

    if trainer is not None:
        # ... proceed with training ...
        pass
    else:
        logger.error("Trainer was not initialized. Exiting the training loop.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ARC Neural Reasoning Model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_profiler", action="store_true", help="Enable the custom profiler")
    group.add_argument("--fast_dev_run", action="store_true", help="Run a fast development test")
    
    parser.add_argument(
        "--optuna_study_name",
        type=str,
        default=None,
        help="Name of the Optuna study to load. If not provided and only one study exists in storage, it will be used automatically."
    )
    parser.add_argument("--optuna_storage", type=str, default="sqlite:///optuna_results.db", help="Storage URL for the Optuna study")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to load. Use None to load all samples."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,  # Increased from None/1 to 4
        help="Number of worker threads for DataLoader. Increasing this can speed up data loading."
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker."
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_true",
        help="Disable persistent workers in DataLoader."
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable pin_memory in DataLoader for faster GPU data transfer."
    )
    parser.set_defaults(pin_memory=True)  # Enable by default if using GPU
    parser.add_argument("--n_embd", type=int, default=12, help="Embedding size for the model.")
    parser.add_argument("--n_head", type=int, default=1, help="Number of attention heads for profiling")
    parser.add_argument(
        "--n_layer",
        type=int,
        default=4,  # Aumentar el valor predeterminado para mayor flexibilidad
        help="Número total de capas (transformer y mamba). Mayor número permite combinaciones más flexibles."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for profiling")  # Increased from 1 to 16
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, required=True, help="Maximum number of epochs")
    def valid_mamba_ratio(value: str) -> float:
        try:
            fvalue = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not a valid value for mamba_ratio. It must be a float between 0.0 and 1.0.")
        if fvalue < 0.0 or fvalue > 1.0:
            raise argparse.ArgumentTypeError(f"mamba_ratio must be between 0.0 and 1.0. Provided value: {fvalue}")
        return fvalue

    parser.add_argument(
        "--mamba_ratio",
        type=valid_mamba_ratio,  # Use custom validation function
        default=1.0,
        help="Proportion of Mamba layers relative to the total number of Transformer layers. Must be between 0.0 and 1.0."
    )

    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--d_state", type=int, default=4, help="Mamba state dimension")
    parser.add_argument("--d_conv", type=int, default=1, help="Mamba convolution dimension")
    parser.add_argument("--mamba_depth", type=int, default=1, help="Depth of each Mamba layer")
    parser.add_argument("--mamba_expand", type=int, default=2, help="Expand factor for each Mamba layer")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--use_grokfast", action="store_true", help="Enable Grokfast for gradient filtering.")
    parser.add_argument(
        "--include_pad_in_loss",
        dest="include_pad_in_loss",
        action="store_true",
        help="Include the padding class in the loss calculation."
    )
    parser.add_argument(
        "--no_include_pad_in_loss",
        dest="include_pad_in_loss",
        action="store_false",
        help="Exclude the padding class from the loss calculation."
    )
    parser.set_defaults(include_pad_in_loss=True)
    parser.add_argument(
        "--grokfast_type",
        type=str,
        default="ema",
        choices=["ema", "ma"],
        help="Type of Grokfast filter to use: 'ema' or 'ma'."
    )
    parser.add_argument(
        "--grokfast_alpha",
        type=float,
        default=0.98,
        help="Alpha parameter for Grokfast-EMA."
    )
    parser.add_argument(
        "--grokfast_lamb",
        type=float,
        default=2.0,
        help="Lambda parameter for Grokfast filters."
    )
    parser.add_argument(
        "--grokfast_window_size",
        type=int,
        default=100,
        help="Window size for Grokfast-MA."
    )
    parser.add_argument("--no_logging", action="store_true", help="Disable logging")
    parser.add_argument("--no_checkpointing", action="store_true", help="Disable checkpointing")
    parser.add_argument("--no_progress-bar", action="store_true", help="Disable progress bar")
    parser.add_argument("--model_checkpoint", type=str, help="Path to the model checkpoint to resume training")
    parser.add_argument("--project", type=str, default="gpt2-arc", help="W&B project name")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.01,
        help=(
            "How often to perform validation. "
            "If a float, represents the fraction of an epoch (e.g., 0.5 for halfway through each epoch). "
            "If an integer, represents the number of training steps."
        )
    )
    parser.add_argument(
        "--enable_symbol_freq",
        action="store_true",
        help="Enable the calculation of symbol frequencies."
    )
    parser.set_defaults(enable_symbol_freq=False)
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--run_name", type=str, default="default_run", help="Name of the run for saving results")
    parser.add_argument("--use_synthetic_data", action="store_true", help="Use synthetic data for training")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--val_split", type=float, default=0.1, help="Proportion of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1, help="Proportion of data to use for testing")
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="medium",
        choices=["highest", "high", "medium"],
        help="Set the internal precision of float32 matrix multiplications. Options: 'highest', 'high', 'medium'. Defaults to 'medium'."
    )
    parser.add_argument("--synthetic_data_path", type=str, help="Path to synthetic data directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--use_optuna", action="store_true", help="Use best hyperparameters from Optuna study")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "tpu"],
        help="Accelerator to use for training: 'cpu', 'gpu', or 'tpu'. Defaults to 'gpu'."
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
        "--include_pad_in_accuracy",
        type=lambda x: (str(x).lower() in ['true', '1', 't', 'y', 'yes']),
        default=True,
        help="Whether to include the padding class in accuracy calculations. (True/False)"
    )
    
    args = parser.parse_args()


    main(args)

