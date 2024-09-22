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
    
    # Suggest hyperparameters
    model_config = ModelConfig(
        n_embd=trial.suggest_int("n_embd", 32, 128),
        n_head=trial.suggest_int("n_head", 2, 4),
        n_layer=trial.suggest_int("n_layer", 1, 3)
    )
    training_config = TrainingConfig(
        batch_size=trial.suggest_int("batch_size", 16, 64),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        max_epochs=trial.suggest_int("max_epochs", 5, 20)
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
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_results.db", help="Storage path for Optuna results.")
    
    args = parser.parse_args()
    
    run_optimization(n_trials=args.n_trials, storage_name=args.storage)
