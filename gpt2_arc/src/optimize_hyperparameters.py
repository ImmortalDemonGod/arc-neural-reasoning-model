import optuna
import logging
import sys
import os
import torch
import pytorch_lightning as pl

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

def objective(trial):
    logger.info(f"Starting trial {trial.number}")
    
    # Suggest hyperparameters
    model_config = ModelConfig(
        n_embd=trial.suggest_int("n_embd", 64, 256),
        n_head=trial.suggest_int("n_head", 2, 8),
        n_layer=trial.suggest_int("n_layer", 1, 6)
    )
    training_config = TrainingConfig(
        batch_size=trial.suggest_int("batch_size", 16, 128),
        learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        max_epochs=trial.suggest_int("max_epochs", 5, 50)
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
            gpus=1 if torch.cuda.is_available() else None,
            logger=False,  # Disable logging to keep things simple
            enable_checkpointing=False,  # Disable checkpointing for simplicity
        )

        # Train and evaluate
        logger.info(f"Starting training for trial {trial.number}")
        trainer.fit(arc_trainer)
        
        # Test the model
        test_results = trainer.test(arc_trainer)
        val_accuracy = test_results[0]['test_accuracy']  # Assuming test_accuracy is logged

        logger.info(f"Trial {trial.number} completed with validation accuracy: {val_accuracy}")
        return val_accuracy
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def run_optimization(n_trials=100):
    study_name = "gpt2_arc_optimization"
    storage_name = "sqlite:///optuna_results.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize"
    )
    
    logger.info(f"Starting optimization with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Optimization completed")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    run_optimization(n_trials=10)  # Start with a small number of trials for testing
