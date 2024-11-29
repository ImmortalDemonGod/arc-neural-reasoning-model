# gpt2_arc/src/optimization/trial_manager.py

import argparse
import gc
import optuna
import torch
import pytorch_lightning as pl
import numpy as np
import multiprocessing
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gpt2_arc.src.training.trainer import ARCTrainer, NanLossPruningCallback
from gpt2_arc.src.training.utils.training_config_manager import ModelConfigSaver
from gpt2_arc.src.utils.results_collector import ResultsCollector
from gpt2_arc.src.optimization.callbacks.tracking import BestEpochTrackerCallback
from gpt2_arc.src.optimization.callbacks.pruning import CustomPruningCallback
from gpt2_arc.src.optimization.utils.config_utils import initialize_config
from gpt2_arc.src.optimization.utils.data_utils import load_datasets, calculate_symbol_frequencies
from gpt2_arc.src.optimization.utils.hyperparameter_utils import suggest_hyperparameters
from gpt2_arc.src.optimization.utils.config_utils import validate_and_adjust_hyperparameters
from gpt2_arc.src.optimization.utils.memory_utils import estimate_model_memory, check_memory_constraints
from gpt2_arc.src.optimization.utils.model_utils import create_model, generate_model_summary
from gpt2_arc.src.optimization.utils.training_utils import train_and_evaluate
import logging
logger = logging.getLogger(__name__)

logger.debug("Trial manager module loaded")

def objective(trial: optuna.trial.Trial, args: argparse.Namespace, all_synthetic_data: Optional[Dict[str, Any]]) -> float:
    # Initialize variables used in finally clause
    model = None
    trainer = None
    arc_trainer = None
    
    logger.info(f"Starting trial {trial.number}")
    logger.debug("=== Trial Initialization ===")
    #print("=== Trial Initialization ===")

    try:
        # STAGE 1: Configuration
        try:
            config = initialize_config()
            logger.info(f"Trial {trial.number}: Config initialized successfully")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed during config initialization")
            raise

        # STAGE 2: Dataset Loading
        try:
            train_data, val_data, test_data = load_datasets(trial, args, config, all_synthetic_data)
            logger.info(f"Trial {trial.number}: Datasets loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed during dataset loading")
            raise

        # STAGE 3: Symbol Frequencies
        try:
            symbol_freq_dict, balance_symbols, balancing_method = calculate_symbol_frequencies(args, config, train_data)
            logger.info(f"Trial {trial.number}: Symbol frequencies calculated - {len(symbol_freq_dict)} symbols")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed during symbol frequency calculation")
            raise

        # STAGE 4: Hyperparameters
        try:
            hyperparameters = suggest_hyperparameters(trial, args, config)
            validate_and_adjust_hyperparameters(hyperparameters)
            logger.info(f"Trial {trial.number}: Hyperparameters suggested and validated")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed during hyperparameter setup")
            raise

        # STAGE 5: Memory Check
        try:
            estimated_memory, available_memory = estimate_model_memory(hyperparameters, args, config)
            check_memory_constraints(estimated_memory, available_memory, trial)
            logger.info(f"Trial {trial.number}: Memory check passed - {estimated_memory:.2f}GB / {available_memory:.2f}GB")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed during memory check")
            raise

        # STAGE 6: Model Creation
        try:
            model = create_model(args, config, symbol_freq_dict, hyperparameters)
            logger.info(f"Trial {trial.number}: Model created successfully")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed during model creation")
            raise

        # STAGE 7: Training Setup
        try:
            results_collector = ResultsCollector(config)
            arc_trainer = ARCTrainer(
                model=model,
                train_dataset=train_data,
                val_dataset=val_data,
                config=config,
                args=args,
                results_collector=results_collector
            )
            trainer = setup_trainer(args, config, trial, arc_trainer)
            logger.info(f"Trial {trial.number}: Training setup completed")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed during training setup")
            raise

        # STAGE 8: Training and Evaluation
        try:
            logger.info(f"Trial {trial.number}: Starting training")
            best_val_loss = train_and_evaluate(trainer, arc_trainer, test_data, config, trial)
            logger.info(f"Trial {trial.number}: Training completed with best_val_loss={best_val_loss}")
            return best_val_loss
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed during training/evaluation")
            raise

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error("Full traceback:", exc_info=True)
        # Re-raise as TrialPruned to handle the failure gracefully
        raise optuna.exceptions.TrialPruned(f"Trial failed: {str(e)}")
    
    finally:
        logger.debug(f"=== Cleanup for Trial {trial.number} ===")
        cleanup_resources(model, trainer, arc_trainer, trial)

# Helper Functions

def setup_trainer(args, config, trial, arc_trainer):
    # Callbacks
    callbacks = []
    pruning_callback = CustomPruningCallback(trial, monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    nan_loss_pruning_callback = NanLossPruningCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/trial_{trial.number}",
        filename=f"{'tuning-' if args.model_checkpoint else ''}step_{{step}}-val_loss_{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    model_config_saver = ModelConfigSaver(config)
    best_epoch_tracker = BestEpochTrackerCallback()
    callbacks.extend([
        pruning_callback,
        early_stop_callback,
        nan_loss_pruning_callback,
        checkpoint_callback,
        model_config_saver,
        best_epoch_tracker
    ])

    # Logger
    tb_logger = TensorBoardLogger(save_dir="runs", name=f"experiment_optuna_trial_{trial.number}")

    # Accelerator settings
    accelerator, devices, strategy = determine_accelerator_settings(args)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=1.0,
        val_check_interval=args.val_check_interval,
        precision=16,
        enable_checkpointing=True,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        enable_progress_bar=not args.no_progress_bar
    )
    logger.debug(f"Trainer created with configuration: {trainer.state}")
    return trainer

def determine_accelerator_settings(args):
    if args.accelerator == "tpu":
        return 'tpu', 'xla:1', 'tpu_spawn'
    elif args.accelerator == "gpu":
        if torch.cuda.is_available():
            return 'gpu', 1, 'auto'
        else:
            return 'cpu', 1, 'auto'
    else:
        return 'cpu', 1, 'auto'

def handle_runtime_error(e, trial):
    if 'CUDA out of memory' in str(e):
        logger.error(f"Trial {trial.number}: CUDA out of memory error.")
        trial.set_user_attr('failed_reason', 'CUDA out of memory')
        raise optuna.exceptions.TrialPruned()
    else:
        logger.error(f"Trial {trial.number}: A runtime error occurred: {str(e)}", exc_info=True)
        raise RuntimeError(f"Trial {trial.number}: A runtime error occurred: {str(e)}")

def handle_generic_exception(e, trial):
    if "symbol_freq" in str(e):
        logger.error(f"Trial {trial.number}: 'symbol_freq' is missing or invalid.", exc_info=True)
        raise optuna.exceptions.TrialPruned(f"Trial {trial.number}: 'symbol_freq' is missing or invalid.")
    else:
        logger.error(f"Trial {trial.number}: An unexpected error occurred: {str(e)}", exc_info=True)
        raise optuna.exceptions.TrialPruned(f"Trial {trial.number}: An unexpected error occurred: {str(e)}")

def cleanup_resources(model=None, trainer=None, arc_trainer=None, trial=None):
    """
    Clean up resources after trial completion or failure.
    All parameters are optional to handle cases where initialization failed.
    """
    try:
        if trial is not None:
            logger.debug(f"Cleaning up after trial {trial.number}")
        
        if model is not None:
            del model
        if trainer is not None:
            del trainer
        if arc_trainer is not None:
            del arc_trainer
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if trial is not None:
            logger.debug(f"Cleanup completed for trial {trial.number}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")