# gpt2_arc/src/training/train.py
import argparse
import logging
from gpt2_arc.src.optimize_hyperparameters import run_optimization
import optuna
import numpy as np
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig

# Set up logging
logger = logging.getLogger(__name__)

def main(args):
    if args.use_optuna:
        logger.info("Loading best hyperparameters from Optuna study")
        study = optuna.load_study(study_name="gpt2_arc_optimization", storage=args.optuna_storage)
        best_params = study.best_params
        logger.debug(f"Best parameters from Optuna: {best_params}")

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
            max_epochs=best_params['max_epochs']
        )
        config = Config(model=model_config, training=training_config)
        logger.info(f"Using Optuna optimized configuration: {config}")
    else:
        logger.info("Using default or provided configuration")
        # Use default or provided configuration

    # Rest of your main function code here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ARC Neural Reasoning Model")
    parser.add_argument("--use_optuna", action="store_true", help="Use Optuna optimized hyperparameters")
    parser.add_argument("--optuna_storage", type=str, default="sqlite:///optuna_results.db", help="Optuna storage path")
    # Add other arguments as needed
    args = parser.parse_args()
    main(args)
