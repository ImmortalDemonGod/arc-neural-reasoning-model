# gpt2_arc/src/training/train.py
import argparse
import multiprocessing
import sys
import logging
import os
import json
from unittest.mock import MagicMock, patch, patch
import optuna
import arckit
import numpy as np
import torch
from lightning.pytorch.profilers import AdvancedProfiler
from torch.utils.data import DataLoader, WeightedRandomSampler

# Define the base directory for the arc-neural-reasoning-model
arc_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Add the root directory of the project to the PYTHONPATH
project_root = arc_model_dir
sys.path.insert(0, project_root)

import pytorch_lightning as pl
import torch.autograd.profiler as profiler
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
from gpt2_arc.src.utils.results_collector import ResultsCollector

def get_num_workers():
    try:
        return multiprocessing.cpu_count() // 2  # Use half of the available CPUs
    except NotImplementedError:
        return 4  # Default fallback
logger = logging.getLogger(__name__)

class ConfigSavingModelCheckpoint(ModelCheckpoint):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        checkpoint['model_config'] = self.config.model.__dict__

def main(args):
    # Set logging level
    log_level = getattr(logging, args.log_level.upper() if hasattr(args, 'log_level') else 'DEBUG', logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    profiler = AdvancedProfiler(
        dirpath=args.profiler_dirpath,
        filename=args.profiler_filename,
        line_count_restriction=args.line_count_restriction,
        dump_stats=args.dump_stats
    ) if args.use_profiler else None
    
    logger.setLevel(logging.DEBUG)  # Ensure logger is set to DEBUG
    
    logger.info("Starting main function")
    logger.debug(f"Command line arguments: {args}")

    trainer = None  # Initialize trainer to None

    try:
        if args.use_optuna:
            logger.info("Loading best hyperparameters from Optuna study")
            study = optuna.load_study(study_name=args.optuna_study_name, storage=args.optuna_storage)
            best_params = study.best_params
            logger.debug(f"Loaded best parameters: {best_params}")
            
            n_head = 2 ** best_params['n_head_exp']
            n_embd = n_head * best_params['n_embd_multiplier']
            n_embd = 2 ** int(np.log2(n_embd))
            
            model_config = ModelConfig(
                n_embd=n_embd,
                n_head=n_head,
                n_layer=best_params['n_layer'],
                dropout=best_params['dropout'],
                use_gpu=args.use_gpu,
                log_level=args.log_level,
                use_synthetic_data=args.use_synthetic_data,
                synthetic_data_path=args.synthetic_data_path
            )
            training_config = TrainingConfig(
                batch_size=best_params['batch_size'],
                learning_rate=best_params['learning_rate'],
                max_epochs=args.max_epochs  # Always use the user-provided max_epochs
            )
        else:
            logger.info("Using provided or default hyperparameters")
            model_config = ModelConfig(
                n_embd=args.n_embd,
                n_head=args.n_head,
                n_layer=args.n_layer,
                mamba_ratio=args.mamba_ratio,
                d_state=args.d_state,
                d_conv=args.d_conv,
                dropout=args.dropout
            )
            training_config = TrainingConfig(
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_epochs=args.max_epochs,
                use_gpu=args.use_gpu,
                log_level=args.log_level,
                use_synthetic_data=args.use_synthetic_data,
                synthetic_data_path=args.synthetic_data_path
            )
        
        config = Config(model=model_config, training=training_config)
        logger.debug(f"Configuration: {config}")

        # Load data
        logger.info("Loading data")
        if args.use_synthetic_data:
            if not args.synthetic_data_path:
                raise ValueError("Synthetic data path not provided")
            logger.info(f"Loading synthetic data from {args.synthetic_data_path}")
            synthetic_files = os.listdir(args.synthetic_data_path)
            logger.debug(f"Total files in synthetic data path: {len(synthetic_files)}")
            logger.debug(f"Sample files: {synthetic_files[:5]}... (total {len(synthetic_files)})")
            train_data = ARCDataset(args.synthetic_data_path)
            synthetic_files = os.listdir(args.synthetic_data_path)
            logger.debug(f"Listing files in synthetic data path for validation: {synthetic_files[:5]}... (total {len(synthetic_files)})")
            val_data = ARCDataset(args.synthetic_data_path, is_test=True)
        else:
            logger.info("Loading ARC dataset")
            train_set, eval_set = arckit.load_data()
            train_data = ARCDataset(train_set)
            val_data = ARCDataset(eval_set)

        # Access dataset statistics
        train_grid_stats = train_data.get_grid_size_stats()
        train_symbol_freq = train_data.get_symbol_frequencies()

        val_grid_stats = val_data.get_grid_size_stats()
        val_symbol_freq = val_data.get_symbol_frequencies()

        # Convert train_symbol_freq from numpy array to dictionary with string keys
        train_symbol_freq_dict = {str(idx): float(freq) for idx, freq in enumerate(train_symbol_freq)}

        # Convert train_symbol_freq from numpy array to dictionary with string keys
        train_symbol_freq_dict = {str(idx): float(freq) for idx, freq in enumerate(train_symbol_freq)}
        
        # Update the TrainingConfig with the symbol_freq dictionary
        training_config.symbol_freq = train_symbol_freq_dict

        logger.info(f"Train Grid Size Stats: {train_grid_stats}")
        logger.info(f"Train Symbol Frequencies: {train_symbol_freq_dict}")
        logger.info(f"Validation Grid Size Stats: {val_grid_stats}")
        logger.info(f"Validation Symbol Frequencies: {val_symbol_freq}")

        # Initialize experiment tracker
        tracker = ExperimentTracker(config, project=args.project)

        # Log dataset statistics to ExperimentTracker
        tracker.log_metric("train_max_grid_height", train_grid_stats.get("max_height", 0))
        tracker.log_metric("train_max_grid_width", train_grid_stats.get("max_width", 0))
        tracker.log_metric("train_symbol_frequencies", train_symbol_freq)

        tracker.log_metric("val_max_grid_height", val_grid_stats.get("max_height", 0))
        tracker.log_metric("val_max_grid_width", val_grid_stats.get("max_width", 0))
        tracker.log_metric("val_symbol_frequencies", val_symbol_freq)

        # Example: Adjust model configuration based on grid size stats
        max_grid_height = max(train_grid_stats.get("max_height", 30), val_grid_stats.get("max_height", 30))
        max_grid_width = max(train_grid_stats.get("max_width", 30), val_grid_stats.get("max_width", 30))
        logger.debug(f"Adjusted max grid size - Height: {max_grid_height}, Width: {max_grid_width}")

        # Set the number of classes
        num_classes = 10
        logger.info(f"Number of classes set to: {num_classes}")

        num_train_samples = train_data.get_num_samples()
        num_val_samples = val_data.get_num_samples()
        logger.info(f"Number of training examples: {num_train_samples}")
        logger.info(f"Number of validation examples: {num_val_samples}")
        
        if num_train_samples == 0 or num_val_samples == 0:
            logger.error("The dataset is empty. Please check the synthetic data path or dataset contents.")
            return

        logger.debug(f"Train data size: {train_data.get_num_samples()}, Validation data size: {val_data.get_num_samples()}")

        # Set the number of classes
        num_classes = 10
        logger.info(f"Number of classes set to: {num_classes}")

        # Create DataLoader instances
        logger.info("Creating DataLoader instances")
        # Create DataLoader instances
        logger.info("Creating DataLoader instances")
        if config.training.balance_symbols:
            if config.training.balancing_method == "weighting":
                # Compute class weights (inverse of frequencies)
                class_weights = 1.0 / torch.tensor(train_symbol_freq, dtype=torch.float)
                # Removed WeightedRandomSampler as it is not appropriate for multi-class samples
                train_loader = DataLoader(
                    train_data,
                    batch_size=config.training.batch_size,
                    num_workers=get_num_workers(),
                    shuffle=True,  # Enable shuffle
                    pin_memory=True if args.use_gpu else False,
                    prefetch_factor=config.training.prefetch_factor,
                    persistent_workers=config.training.persistent_workers
                )
                logger.debug("Class weights applied in loss function. WeightedRandomSampler removed.")
            elif config.training.balancing_method == "oversampling":
                # Placeholder for oversampling implementation
                logger.info("Oversampling method selected, but not yet implemented.")
                # Implement oversampling logic here if desired
                train_loader = DataLoader(
                    train_data,
                    batch_size=config.training.batch_size,
                    num_workers=get_num_workers(),
                    shuffle=True,  # Enable shuffle if not using a sampler
                    pin_memory=True if args.use_gpu else False,
                    prefetch_factor=config.training.prefetch_factor,
                    persistent_workers=config.training.persistent_workers
                )
            else:
                logger.warning(f"Unknown balancing method: {config.training.balancing_method}. Skipping balancing.")
                train_loader = DataLoader(
                    train_data,
                    batch_size=config.training.batch_size,
                    num_workers=get_num_workers(),
                    shuffle=True,  # Enable shuffle
                    pin_memory=True if args.use_gpu else False,
                    prefetch_factor=config.training.prefetch_factor,
                    persistent_workers=config.training.persistent_workers
                )
        else:
            train_loader = DataLoader(
                train_data,
                batch_size=config.training.batch_size,
                num_workers=get_num_workers(),
                shuffle=True,  # Enable shuffle
                pin_memory=True if args.use_gpu else False,
                prefetch_factor=config.training.prefetch_factor,
                persistent_workers=config.training.persistent_workers
            )
        val_loader = DataLoader(
            val_data,
            batch_size=config.training.batch_size,
            num_workers=get_num_workers(),
            pin_memory=True if args.use_gpu else False,
            prefetch_factor=config.training.prefetch_factor,
            persistent_workers=config.training.persistent_workers
        )
        logger.debug(f"DataLoaders created with batch size {args.batch_size}")

        # Initialize model
        logger.info("Initializing model")
        model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=train_symbol_freq_dict)
        logger.debug(f"Model initialized with config: {model_config}")

        # Load the checkpoint if specified
        if args.model_checkpoint:
            logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
            checkpoint = torch.load(args.model_checkpoint)
            if 'model_config' in checkpoint:
                model_config = ModelConfig(**checkpoint['model_config'])
                model = GPT2ARC(config=model_config)
            model.load_state_dict(checkpoint['state_dict'])

        # Initialize results collector
        results_collector = ResultsCollector(config)

        # Initialize experiment tracker
        tracker = ExperimentTracker(config, project=args.project)

        logger.debug("Initializing ExperimentTracker")
        tracker = ExperimentTracker(config, project=args.project)

        logger.debug("Initializing ARCTrainer")
        trainer = ARCTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            config=config
        )
        trainer.log_hyperparameters()

        # Set up PyTorch Lightning trainer
        logger.info("Setting up PyTorch Lightning trainer")
        callbacks = []
        if not args.no_checkpointing:
            checkpoint_callback = ConfigSavingModelCheckpoint(
                config=config,
                dirpath="checkpoints",
                filename="arc_model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)

        if not args.no_logging:
            tb_logger = TensorBoardLogger(
                save_dir="runs",
                name=f"experiment_{trainer.results_collector.experiment_id}"
            )
            logger.debug(f"TensorBoard logger initialized. Log dir: {tb_logger.log_dir}")
        else:
            tb_logger = False
            logger.debug("Logging is disabled")

        pl_trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=tb_logger,
            callbacks=callbacks if callbacks else None,
            enable_checkpointing=not args.no_checkpointing,
            enable_progress_bar=not args.no_progress_bar,
            fast_dev_run=True,
            gradient_clip_val=1.0,
            accelerator='gpu' if args.use_gpu and torch.cuda.is_available() else 'cpu',
            devices=1,
            reload_dataloaders_every_n_epochs=1,
            profiler=profiler
        )

        if tb_logger:
            trainer.results_collector.set_tensorboard_log_path(tb_logger.log_dir)
            logger.debug(f"TensorBoard log path set in results collector: {tb_logger.log_dir}")

        # Log initial memory usage
        logger.info(f"Initial memory usage: {torch.cuda.memory_allocated()} bytes")

        # Train the model
        logger.info("Starting model training")
        pl_trainer.fit(trainer, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Log memory usage after training
        logger.info(f"Memory usage after training: {torch.cuda.memory_allocated()} bytes")

        # After training, run test
        logger.info("Running model evaluation")
        test_results = pl_trainer.test(trainer)
        if test_results:
            avg_test_loss = sum(result['avg_test_loss'] for result in test_results) / len(test_results)
            avg_test_accuracy = sum(result['avg_test_accuracy'] for result in test_results) / len(test_results)
            avg_test_diff_accuracy = sum(result['avg_test_diff_accuracy'] for result in test_results) / len(test_results)

            logger.info(f"Test results - Loss: {avg_test_loss}, Accuracy: {avg_test_accuracy}, Diff Accuracy: {avg_test_diff_accuracy}")

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
            "final_test_accuracy": avg_test_accuracy
        })

        # Save the final model with configuration
        logger.info("Saving final model with configuration")
        model_path = f"final_model_{trainer.results_collector.experiment_id}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'state_dict': trainer.model.state_dict(),
            'model_config': trainer.config.model.__dict__
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
    parser.add_argument("--use-optuna", action="store_true", help="Use best hyperparameters from Optuna study")
    parser.add_argument("--optuna-study-name", type=str, default="gpt2_arc_optimization", help="Name of the Optuna study to load")
    parser.add_argument("--optuna-storage", type=str, default="sqlite:///optuna_results.db", help="Storage URL for the Optuna study")
    parser.add_argument("--n-embd", type=int, default=4, help="Embedding dimension for profiling")
    parser.add_argument("--n-head", type=int, default=1, help="Number of attention heads for profiling")
    parser.add_argument("--n-layer", type=int, default=1, help="Number of transformer layers for profiling")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, required=True, help="Maximum number of epochs")
    parser.add_argument("--mamba-ratio", type=int, default=0, help="Number of Mamba layers per Transformer layer")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--d-state", type=int, default=4, help="Mamba state dimension")
    parser.add_argument("--d-conv", type=int, default=1, help="Mamba convolution dimension")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--no-logging", action="store_true", help="Disable logging")
    parser.add_argument("--no-checkpointing", action="store_true", help="Disable checkpointing")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable progress bar")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run a fast development test")
    parser.add_argument("--model_checkpoint", type=str, help="Path to the model checkpoint to resume training")
    parser.add_argument("--project", type=str, default="gpt2-arc", help="W&B project name")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--run-name", type=str, default="default_run", help="Name of the run for saving results")
    parser.add_argument("--use-synthetic-data", action="store_true", help="Use synthetic data for training")
    parser.add_argument("--synthetic-data-path", type=str, help="Path to synthetic data directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--use-profiler", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument(
        "--profiler-dirpath",
        type=str,
        default="./profiler_logs",
        help="Directory path for profiler output files."
    )
    parser.add_argument(
        "--profiler-filename",
        type=str,
        default="profile",
        help="Filename for profiler output."
    )
    parser.add_argument(
        "--line-count-restriction",
        type=float,
        default=1.0,
        help="Restriction on the number of lines in profiler summary."
    )
    parser.add_argument(
        "--dump-stats",
        action="store_true",
        help="Whether to dump raw profiler statistics."
    )
    
    args = parser.parse_args()
    main(args)

