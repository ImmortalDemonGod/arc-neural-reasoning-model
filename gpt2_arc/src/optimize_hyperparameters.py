# gpt2_arc/src/optimize_hyperparameters.py
import argparse  # Import argparse for command-line argument parsing
import optuna
import logging
import sys
import os
import torch
import pytorch_lightning as pl
import psutil
import gc

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.data.arc_dataset import ARCDataset
import arckit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def objective(trial):
    logger.info(f"Starting trial {trial.number}")
    log_memory_usage()
    
    # Suggest hyperparameters using ranges from command-line arguments
    model_config = ModelConfig(
        n_embd=trial.suggest_int("n_embd", args.n_embd_min, args.n_embd_max),
        n_head=trial.suggest_int("n_head", args.n_head_min, args.n_head_max),
        n_layer=trial.suggest_int("n_layer", args.n_layer_min, args.n_layer_max),
        dropout=trial.suggest_float("dropout", args.dropout_min, args.dropout_max)
    )
    optimizer_name = trial.suggest_categorical("optimizer", args.optimizers)
    
    training_config = TrainingConfig(
        optimizer_name=optimizer_name,
        batch_size=trial.suggest_int("batch_size", args.batch_size_min, args.batch_size_max),
        learning_rate=trial.suggest_float("learning_rate", args.learning_rate_min, args.learning_rate_max, log=True),
        max_epochs=trial.suggest_int("max_epochs", args.max_epochs_min, args.max_epochs_max)
    )
    config = Config(model=model_config, training=training_config)
    
    logger.info(f"Trial {trial.number} config: {config}")
    
    try:
        # Load data
        train_set, eval_set = arckit.load_data()
        train_data = ARCDataset(train_set)
        val_data = ARCDataset(eval_set)
        logger.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
        
        # Create model and trainer
        model = GPT2ARC(config.model)
        arc_trainer = ARCTrainer(model, train_data, val_data, config)

        # Set up PyTorch Lightning trainer
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=False,  # Disable logging to keep things simple
            enable_checkpointing=False,  # Disable checkpointing for simplicity
            accumulate_grad_batches=4,  # Add gradient accumulation
        )

        # Train and evaluate
        logger.info(f"Starting training for trial {trial.number}")
        log_memory_usage()
        trainer.fit(arc_trainer)
        
        # Test the model
        log_memory_usage()
        test_results = trainer.test(arc_trainer)
        val_accuracy = test_results[0]['test_accuracy']  # Assuming test_accuracy is logged

        logger.info(f"Trial {trial.number} completed with validation accuracy: {val_accuracy}")
        log_memory_usage()
        
        # Clean up to free memory
        del trainer, arc_trainer, model
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage()
        
        return val_accuracy
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        log_memory_usage()
        raise optuna.exceptions.TrialPruned()

def run_optimization(n_trials=100, storage_name="sqlite:///optuna_results.db"):
    study_name = "gpt2_arc_optimization"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize"
    )
    
    logger.info(f"Starting optimization with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Optimization completed")
    if study.best_trial:
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")
        logger.info("Best hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("No successful trials found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for GPT2ARC model.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials for optimization.")
    parser.add_argument("--optimizers", nargs='+', default=["Adam", "SGD"], help="List of optimizers to try.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_results.db", help="Storage path for Optuna results.")
    
    parser.add_argument("--n_embd_min", type=int, default=32, help="Minimum value for n_embd.")
    parser.add_argument("--n_embd_max", type=int, default=128, help="Maximum value for n_embd.")
    parser.add_argument("--n_head_min", type=int, default=2, help="Minimum value for n_head.")
    parser.add_argument("--n_head_max", type=int, default=4, help="Maximum value for n_head.")
    parser.add_argument("--n_layer_min", type=int, default=1, help="Minimum value for n_layer.")
    parser.add_argument("--n_layer_max", type=int, default=3, help="Maximum value for n_layer.")
    parser.add_argument("--batch_size_min", type=int, default=16, help="Minimum value for batch_size.")
    parser.add_argument("--batch_size_max", type=int, default=64, help="Maximum value for batch_size.")
    parser.add_argument("--learning_rate_min", type=float, default=1e-5, help="Minimum value for learning_rate.")
    parser.add_argument("--learning_rate_max", type=float, default=1e-2, help="Maximum value for learning_rate.")
    parser.add_argument("--max_epochs_min", type=int, default=5, help="Minimum value for max_epochs.")
    parser.add_argument("--max_epochs_max", type=int, default=20, help="Maximum value for max_epochs.")
    parser.add_argument("--dropout_min", type=float, default=0.0, help="Minimum value for dropout.")
    parser.add_argument("--dropout_max", type=float, default=0.5, help="Maximum value for dropout.")
    
    args = parser.parse_args()
    
    run_optimization(n_trials=args.n_trials, storage_name=args.storage)
