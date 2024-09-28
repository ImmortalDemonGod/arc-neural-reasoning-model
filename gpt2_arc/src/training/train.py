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

# Define the base directory for the arc-neural-reasoning-model
arc_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Add the root directory of the project to the PYTHONPATH
project_root = arc_model_dir
sys.path.insert(0, project_root)

import pytorch_lightning as pl
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
    log_level = getattr(logging, args.log_level.upper() if hasattr(args, 'log_level') else 'INFO', logging.DEBUG)
    logger.setLevel(log_level)
    
    logger.info("Starting main function")
    logger.debug(f"Command line arguments: {args}")

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
                n_layer=best_params['n_layer']
            )
            training_config = TrainingConfig(
                batch_size=best_params['batch_size'],
                learning_rate=best_params['learning_rate'],
                max_epochs=args.max_epochs  # Always use the user-provided max_epochs
            )
        else:
            logger.info("Using provided or default hyperparameters")
            model_config = ModelConfig(n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)
            training_config = TrainingConfig(batch_size=args.batch_size, learning_rate=args.learning_rate, max_epochs=args.max_epochs)
        
        config = Config(model=model_config, training=training_config)
        logger.debug(f"Configuration: {config}")

        # Load data
        logger.info("Loading data")
        if args.use_synthetic_data:
            if not args.synthetic_data_path:
                raise ValueError("Synthetic data path not provided")
            train_data = ARCDataset(args.synthetic_data_path)
            val_data = ARCDataset(args.synthetic_data_path, is_test=True)
        else:
            train_set, eval_set = arckit.load_data()
            train_data = ARCDataset(train_set)
            val_data = ARCDataset(eval_set)
        logger.debug(f"Train data size: {len(train_data)}, Validation data size: {len(val_data)}")

        # Create DataLoader instances
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=int(args.batch_size), num_workers=7)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=int(args.batch_size), num_workers=7)

        # Initialize model
        logger.info("Initializing model")
        
        # Determine the number of classes from the dataset
        # Function to find the maximum label value in the dataset
        def find_max_label(task_set):
            max_label = 0
            for task in task_set.tasks:
                for sample in task.train + task.test:
                    input_grid, output_grid = sample
                    max_label = max(max_label, np.max(input_grid), np.max(output_grid))
            return max_label

        # Determine the number of classes from the datasets
        max_label_train = find_max_label(train_set)
        max_label_val = find_max_label(eval_set)
        max_label_overall = max(max_label_train, max_label_val)
        num_classes = max_label_overall + 1  # Add 1 because labels start from 0

        logger.info(f"Determined num_classes: {num_classes}")
        
        model = GPT2ARC(config=model_config, num_classes=num_classes)
        logger.debug(f"Model structure: {model}")

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

        # Initialize trainer
        logger.info("Initializing trainer")
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
            print(f"DEBUG: TensorBoard logger initialized. Log dir: {tb_logger.log_dir}")
        else:
            tb_logger = False
            print("DEBUG: Logging is disabled")

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
            print(f"DEBUG: TensorBoard log path set in results collector: {tb_logger.log_dir}")

        # Train the model
        logger.info("Starting model training")
        pl_trainer.fit(trainer)

        # After training, run test
        logger.info("Running model evaluation")
        test_results = pl_trainer.test(trainer)
        if test_results:
            avg_test_loss = test_results[0]['test_loss']
        
            # Calculate average test accuracy
            accuracy_keys = [key for key in test_results[0].keys() if key.endswith('_test_accuracy')]
            if accuracy_keys:
                avg_test_accuracy = sum(test_results[0][key] for key in accuracy_keys) / len(accuracy_keys)
            else:
                avg_test_accuracy = None  # or some default value
        
            test_diff_accuracy = test_results[0].get('test_diff_accuracy')
        
            # Logging results
            log_message = f"Test results - Loss: {avg_test_loss}"
            if avg_test_accuracy is not None:
                log_message += f", Avg Accuracy: {avg_test_accuracy}"
            if test_diff_accuracy is not None:
                log_message += f", Diff Accuracy: {test_diff_accuracy}"
            logger.info(log_message)
        
            # Collecting results
            results = {
                "test_loss": avg_test_loss,
            }
            if avg_test_accuracy is not None:
                results["avg_test_accuracy"] = avg_test_accuracy
            if test_diff_accuracy is not None:
                results["test_diff_accuracy"] = test_diff_accuracy
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

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        if 'tracker' in locals():
            tracker.log_metric("training_interrupted", 1)
            tracker.log_metric("error_message", str(e))
        raise
    finally:
        if 'tracker' in locals():
            tracker.finish()

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
    
    args = parser.parse_args()
    main(args)

# Synthetic data tests have been moved to gpt2_arc/tests/test_synthetic_data.py
