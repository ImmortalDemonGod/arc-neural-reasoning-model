# gpt2_arc/src/training/train.py
import argparse
import sys
import logging
import os
import json
from unittest.mock import MagicMock
import optuna
import arckit

# Add the root directory of the project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

#print("Current PYTHONPATH:", sys.path)

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
import json
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
from gpt2_arc.src.utils.results_collector import ResultsCollector
import optuna
import os

# Set up logging
logger = logging.getLogger(__name__)

def main(args):
    logger.info("Starting main function")
    logger.debug(f"Command line arguments: {args}")

    try:
        if args.use_optuna:
            logger.info("Loading best hyperparameters from Optuna study")
            study = optuna.load_study(study_name=args.optuna_study_name, storage=args.optuna_storage)
            best_params = study.best_params
            logger.debug(f"Loaded best parameters: {best_params}")
            
            model_config = ModelConfig(
                n_embd=best_params.get("n_embd", 768),
                n_head=best_params.get("n_head", 12),
                n_layer=best_params.get("n_layer", 12)
            )
            training_config = TrainingConfig(
                batch_size=best_params.get("batch_size", 32),
                learning_rate=best_params.get("learning_rate", 1e-4),
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
        train_set, eval_set = arckit.load_data()
        train_data = ARCDataset(train_set)
        val_data = ARCDataset(eval_set)
        logger.debug(f"Train data size: {len(train_data)}, Validation data size: {len(val_data)}")

        # Load model configuration from JSON file
        config_path = f"results/experiment_{args.model_checkpoint.split('_')[-1].replace('.pth', '.json')}"
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        model_config = ModelConfig(
            n_embd=config_data['config']['model']['n_embd'],
            n_head=config_data['config']['model']['n_head'],
            n_layer=config_data['config']['model']['n_layer'],
            dropout=config_data['config']['model']['dropout']
        )

        # Initialize model
        logger.info("Initializing model")
        model = GPT2ARC(config=model_config)
        logger.debug(f"Model structure: {model}")

        # Load the checkpoint if specified
        if args.model_checkpoint:
            logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
            checkpoint = torch.load(args.model_checkpoint)
            model.load_state_dict(checkpoint)

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
            checkpoint_callback = ModelCheckpoint(
                dirpath="checkpoints",
                filename="arc_model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)

        tb_logger = False if args.no_logging else TensorBoardLogger("tb_logs", name="arc_model")

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

        # Train the model
        logger.info("Starting model training")
        pl_trainer.fit(trainer)

        # After training, run test
        logger.info("Running model evaluation")
        test_results = pl_trainer.test(trainer)
        if test_results:
            avg_test_loss = test_results[0]['test_loss']
            avg_test_accuracy = test_results[0]['test_accuracy']
            logger.info(f"Test results - Loss: {avg_test_loss}, Accuracy: {avg_test_accuracy}")
            trainer.results_collector.set_test_results({
                "test_loss": avg_test_loss,
                "test_accuracy": avg_test_accuracy
            })

        trainer.results_collector.set_final_metrics({
            "best_val_loss": trainer.best_val_loss,
            "best_epoch": trainer.best_epoch,
            "final_test_loss": avg_test_loss,
            "final_test_accuracy": avg_test_accuracy
        })

        # Save the final model
        logger.info("Saving final model")
        model_path = f"final_model_{trainer.results_collector.experiment_id}.pth"
        torch.save(trainer.model.state_dict(), model_path)
        trainer.results_collector.set_checkpoint_path(model_path)
        logger.debug(f"Model saved to: {model_path}")

        # Save results
        logger.info("Saving experiment results")
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
    parser.add_argument("--n-embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n-layer", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
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
    
    args = parser.parse_args()
    main(args)

