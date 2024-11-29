# gpt2_arc/src/optimization/optimizer.py
import argparse
import logging
import optuna
from functools import partial
import torch

from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler

from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.optimization.trial_manager import objective
from gpt2_arc.src.training.utils.data_manager import DataManager
from typing import Optional

logger = logging.getLogger(__name__)


def create_pruner_and_sampler(n_trials: int):
    # Create pruner and sampler based on the number of trials
    if n_trials < 10:
        n_startup_trials = 1
    else:
        n_startup_trials = 5

    pruner = PercentilePruner(
        percentile=25,
        n_startup_trials=n_startup_trials,
        n_warmup_steps=2,
        interval_steps=1
    )
    sampler = TPESampler(n_startup_trials=5)
    return pruner, sampler


def initialize_config_and_data(args: Optional[argparse.Namespace]):
    # If args doesn't exist, create it
    if args is None:
        args = argparse.Namespace()
    
    # Ensure required attributes exist
    if not hasattr(args, 'project'):
        args.project = "gpt2-arc-optimization"  # Set a default project name

    # Rest of the function stays the same
    model_config = ModelConfig()
    training_config = TrainingConfig()
    config = Config(model=model_config, training=training_config)
    data_manager_instance = DataManager(config=config, args=args)
    
    all_synthetic_data = None
    if args.use_synthetic_data:
        all_synthetic_data = data_manager_instance.load_and_split_synthetic_data(args, config)
        if all_synthetic_data:
            logger.info(f"Synthetic data loaded with {len(all_synthetic_data['train_dataset'])} samples.")

    return config, data_manager_instance, all_synthetic_data


def adjust_n_jobs(n_jobs: int, args: Optional[argparse.Namespace]) -> int:
    # Adjust the number of jobs based on GPU availability
    if args.use_gpu:
        available_gpus = torch.cuda.device_count()
        if available_gpus > 1:
            n_jobs = max(n_jobs, available_gpus)
        else:
            n_jobs = 1  # Limit to 1 to prevent memory issues
    return n_jobs


def execute_study(study, objective_partial, n_trials, n_jobs):
    # Execute the optimization study
    logger.info(f"Starting optimization with {n_trials} trials using {n_jobs} parallel jobs")
    logger.info("Data Splitting Ratios - Train: 80%, Validation: 10%, Test: 10%")

    study.optimize(objective_partial, n_trials=n_trials, n_jobs=n_jobs)
    logger.info("Optimization completed")


def log_best_trial_summary(best_trial):
    """Logs the summary of the best trial."""
    best_model_summary = best_trial.user_attrs.get("model_summary")
    if best_model_summary:
        logger.info("Model summary for the best trial:")
        logger.info(best_model_summary)
    else:
        logger.debug("No model summary found for the best trial")


def log_best_trial_details(best_trial):
    """Logs detailed information about the best trial."""
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best value: {best_trial.value}")

    # Set additional user attributes
    best_trial.set_user_attr("mamba_ratio", best_trial.params.get("mamba_ratio"))
    best_trial.set_user_attr("d_state", best_trial.params.get("d_state"))
    best_trial.set_user_attr("d_conv", best_trial.params.get("d_conv"))

    logger.info("Best Mamba metrics:")
    mamba_metrics = ['mamba_forward_pass_time', 'mamba_params', 'mamba_params_ratio']
    for metric in mamba_metrics:
        value = best_trial.user_attrs.get(metric)
        if value is not None:
            logger.info(f"  {metric}: {value}")

    logger.info("Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")


def handle_best_trial(study):
    """Handles the best trial after the optimization study."""
    if not study.trials:
        logger.info("No trials have been completed.")
        return

    # Filter for completed trials only
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
        logger.info("No trials have been completed successfully.")
        return

    # Get best trial among completed ones
    best_trial = min(completed_trials, key=lambda t: t.value)

    if best_trial.state == optuna.trial.TrialState.COMPLETE:
        logger.debug("Best trial found, attempting to retrieve model summary")

        # Log model summary if available
        best_model_summary = best_trial.user_attrs.get("model_summary")
        if best_model_summary:
            logger.info("Model summary for best trial:")
            logger.info(best_model_summary)
        else:
            logger.debug("No model summary found for best trial")

        # Log trial details
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value: {best_trial.value}")

        # Set additional user attributes
        for param in ["mamba_ratio", "d_state", "d_conv"]:
            if param in best_trial.params:
                best_trial.set_user_attr(param, best_trial.params[param])

        # Log mamba metrics
        logger.info("Best mamba metrics:")
        mamba_metrics = ['mamba_forward_pass_time', 'mamba_params', 'mamba_params_ratio']
        for metric in mamba_metrics:
            value = best_trial.user_attrs.get(metric)
            if value is not None:
                logger.info(f" {metric}: {value}")

        # Log best hyperparameters
        logger.info("Best hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f" {key}: {value}")
    else:
        logger.warning(f"Best trial {best_trial.number} is not in COMPLETE state")


def run_optimization(
    n_trials: int = 100,
    storage_name: str = "sqlite:///optuna_results.db",
    n_jobs: int = -1,
    args: Optional[argparse.Namespace] = None,
    study_name: str = "gpt2_arc_optimization_v2"
) -> None:
    pruner, sampler = create_pruner_and_sampler(n_trials)
    config, data_manager_instance, all_synthetic_data = initialize_config_and_data(args)

    # Create a partial objective function that includes all_synthetic_data
    objective_partial = partial(objective, args=args, all_synthetic_data=all_synthetic_data)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=sampler
    )

    n_jobs = adjust_n_jobs(n_jobs, args)
    execute_study(study, objective_partial, n_trials, n_jobs)
    handle_best_trial(study)