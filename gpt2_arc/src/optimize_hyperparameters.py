# gpt2_arc/src/optimize_hyperparameters.py
import argparse
import optuna
import logging
import sys
import os
import torch
import pytorch_lightning as pl
import numpy as np
from optuna.pruners import MedianPruner
from pytorch_lightning.loggers import TensorBoardLogger

class CustomPruningCallback(pl.Callback):
    def __init__(self, trial, monitor="val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.data.arc_dataset import ARCDataset
import arckit

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def validate_hyperparameters(n_embd, n_head, n_layer):
    """Validate that hyperparameters meet necessary constraints."""
    logger.debug(f"Validating hyperparameters: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}")
    assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
    assert n_embd >= n_head, f"n_embd ({n_embd}) must be greater than or equal to n_head ({n_head})"
    assert n_layer > 0, f"n_layer ({n_layer}) must be positive"
    logger.debug("Hyperparameters validated successfully")
    return True

def objective(trial):
    logger.info(f"Starting trial {trial.number}")

    try:
        # Suggest n_head as a power of 2
        n_head_exp = trial.suggest_int("n_head_exp", args.n_head_exp_min, args.n_head_exp_max)
        n_head = 2 ** n_head_exp
        logger.debug(f"Suggested n_head: {n_head} (2^{n_head_exp})")

        # Suggest n_embd as a multiple of n_head and ensure it's a power of 2
        n_embd_multiplier = trial.suggest_int("n_embd_multiplier", args.n_embd_multiplier_min, args.n_embd_multiplier_max)
        n_embd = n_head * n_embd_multiplier
        n_embd = 2 ** int(np.log2(n_embd))
        logger.debug(f"Adjusted n_embd: {n_embd}")

        # Suggest n_layer
        n_layer = trial.suggest_int("n_layer", args.n_layer_min, args.n_layer_max)
        logger.debug(f"Suggested n_layer: {n_layer}")

        # Validate hyperparameters
        validate_hyperparameters(n_embd, n_head, n_layer)

        model_config = ModelConfig(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer
        )
        logger.debug(f"Model config: {model_config}")

        # Suggest training hyperparameters
        batch_size = trial.suggest_int("batch_size", args.batch_size_min, args.batch_size_max)
        learning_rate = trial.suggest_float("learning_rate", args.learning_rate_min, args.learning_rate_max, log=True)
        max_epochs = trial.suggest_int("max_epochs", args.max_epochs_min, args.max_epochs_max)

        training_config = TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs
        )

        config = Config(model=model_config, training=training_config)
        logger.debug(f"Full config: {config}")

        # Load data
        logger.debug("Loading data")
        train_set, eval_set = arckit.load_data()
        train_data = ARCDataset(train_set)
        val_data = ARCDataset(eval_set)
        logger.debug(f"Data loaded. Train set size: {len(train_data)}, Validation set size: {len(val_data)}")

        # Create model and trainer
        logger.debug("Creating model and trainer")
        model = GPT2ARC(config.model)
        arc_trainer = ARCTrainer(model, train_data, val_data, config)

        # Set up PyTorch Lightning trainer with custom pruning callback
        pruning_callback = CustomPruningCallback(trial, monitor="val_loss")
        experiment_id = f"optuna_trial_{trial.number}"
        tb_logger = TensorBoardLogger(save_dir="runs", name=f"experiment_{experiment_id}")
        print(f"DEBUG: Optuna trial TensorBoard logger initialized. Log dir: {tb_logger.log_dir}")
        
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            callbacks=[pruning_callback],
            logger=tb_logger,
            enable_checkpointing=False,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
        )
        print(f"DEBUG: Trainer created for Optuna trial with TensorBoard logger")
        logger.debug(f"Trainer created with config: {trainer.state}")

        # Train and evaluate
        logger.debug("Starting training")
        trainer.fit(arc_trainer)

        # Get the best validation loss
        best_val_loss = trainer.callback_metrics.get("val_loss").item()
        logger.info(f"Trial {trial.number} completed. Best validation loss: {best_val_loss}")

        return best_val_loss

    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}", exc_info=True)
        raise optuna.exceptions.TrialPruned()

def run_optimization(n_trials=100, storage_name="sqlite:///optuna_results.db"):
    study_name = "gpt2_arc_optimization"

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner
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

    parser.add_argument("--n_head_exp_min", type=int, default=1, help="Minimum exponent for n_head (2^x)")
    parser.add_argument("--n_head_exp_max", type=int, default=3, help="Maximum exponent for n_head (2^x)")
    parser.add_argument("--n_embd_multiplier_min", type=int, default=4, help="Minimum multiplier for n_embd")
    parser.add_argument("--n_embd_multiplier_max", type=int, default=16, help="Maximum multiplier for n_embd")
    parser.add_argument("--n_layer_min", type=int, default=1, help="Minimum value for n_layer")
    parser.add_argument("--n_layer_max", type=int, default=6, help="Maximum value for n_layer")
    parser.add_argument("--batch_size_min", type=int, default=8, help="Minimum value for batch_size")
    parser.add_argument("--batch_size_max", type=int, default=32, help="Maximum value for batch_size")
    parser.add_argument("--learning_rate_min", type=float, default=1e-5, help="Minimum value for learning_rate")
    parser.add_argument("--learning_rate_max", type=float, default=1e-2, help="Maximum value for learning_rate")
    parser.add_argument("--max_epochs_min", type=int, default=5, help="Minimum value for max_epochs")
    parser.add_argument("--max_epochs_max", type=int, default=20, help="Maximum value for max_epochs")

    args = parser.parse_args()

    run_optimization(n_trials=args.n_trials, storage_name=args.storage)
