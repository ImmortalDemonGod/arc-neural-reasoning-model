# gpt2_arc/src/training/train.py
import argparse
import sys
import logging
import os
import json
from unittest.mock import MagicMock, patch, patch
import optuna
import arckit
import numpy as np
import torch

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

# Set up logging
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
            train_data = ARCDataset(args.synthetic_data_path)
            val_data = ARCDataset(args.synthetic_data_path, is_test=True)
        else:
            logger.info("Loading ARC dataset")
            train_set, eval_set = arckit.load_data()
            train_data = ARCDataset(train_set)
            val_data = ARCDataset(eval_set)
        logger.debug(f"Train data size: {len(train_data)}, Validation data size: {len(val_data)}")

        # Set the number of classes
        num_classes = 10
        logger.info(f"Number of classes set to: {num_classes}")

        # Create DataLoader instances
        logger.info("Creating DataLoader instances")
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=int(args.batch_size), num_workers=7)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=int(args.batch_size), num_workers=7)
        logger.debug(f"DataLoaders created with batch size {args.batch_size}")

        # Initialize model
        logger.info("Initializing model")
        model = GPT2ARC(config=model_config, num_classes=num_classes)
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
            fast_dev_run=args.fast_dev_run,
            gradient_clip_val=1.0,
            accelerator='gpu' if args.use_gpu and torch.cuda.is_available() else 'cpu',
            devices=1
        )

        if tb_logger:
            trainer.results_collector.set_tensorboard_log_path(tb_logger.log_dir)
            logger.debug(f"TensorBoard log path set in results collector: {tb_logger.log_dir}")

        # Train the model
        logger.info("Starting model training")
        with profiler.profile(record_shapes=True, profile_memory=True) as prof:
            pl_trainer.fit(trainer)

        # After training, print profiler results
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

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
        raise
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
    parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, required=True, help="Maximum number of epochs")
    parser.add_argument("--mamba-ratio", type=int, default=0, help="Number of Mamba layers per Transformer layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--d-state", type=int, default=16, help="Mamba state dimension")
    parser.add_argument("--d-conv", type=int, default=4, help="Mamba convolution dimension")
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
    
    args = parser.parse_args()
    main(args)

# Synthetic data tests have been moved to gpt2_arc/tests/test_synthetic_data.py
