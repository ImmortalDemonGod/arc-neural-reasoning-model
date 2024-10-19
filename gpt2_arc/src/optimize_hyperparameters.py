# gpt2_arc/src/optimize_hyperparameters.py
import argparse
import multiprocessing
import random
import optuna
import logging
import sys
import os
import torch
from torch.utils.data import DataLoader
import gc
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.model_summary import ModelSummary
from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
#from gpt2_arc.src.utils.training_helpers import get_num_workers
from gpt2_arc.src.training.trainer import NanLossPruningCallback
from functools import partial  # Import partial
from gpt2_arc.src.training.train import load_dataset, load_and_split_synthetic_data
from gpt2_arc.src.training.train import ModelConfigSaver
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.utils.results_collector import ResultsCollector

class BestEpochTrackerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get("val_loss")
        if current_val_loss is not None:
            if not hasattr(self, 'best_val_loss') or current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_epoch = trainer.current_epoch
                logger.debug(f"New best_val_loss: {self.best_val_loss} at epoch {self.best_epoch}")
from gpt2_arc.src.utils.model_memory_estimator import (
    calculate_params,
    estimate_memory_usage,
    get_available_memory,
    can_fit_model
)

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

# Define the base directory for the arc-neural-reasoning-model
arc_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Add the project root to the Python path
project_root = arc_model_dir
sys.path.insert(0, project_root)

from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.data.arc_dataset import ARCDataset
import arckit
from gpt2_arc.src.utils.performance_metrics import calculate_mamba_efficiency



def validate_hyperparameters(n_embd, n_head, n_layer, mamba_ratio, d_state, d_conv, dropout):
    """Validate that hyperparameters meet necessary constraints."""
    logger.debug(f"Validating hyperparameters: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, "
                 f"mamba_ratio={mamba_ratio}, d_state={d_state}, d_conv={d_conv}, dropout={dropout}")
    assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
    assert n_embd >= n_head, f"n_embd ({n_embd}) must be greater than or equal to n_head ({n_head})"
    assert n_layer > 0, f"n_layer ({n_layer}) must be positive"
    assert d_state > 0, f"d_state ({d_state}) must be positive"
    assert d_conv > 0, f"d_conv ({d_conv}) must be positive"
    assert 0.0 <= dropout <= 1.0, f"dropout ({dropout}) must be between 0.0 and 1.0"
    logger.debug("Hyperparameters validated successfully")
    return True





def objective(trial, args, train_data, val_data, test_data):
    model = None
    trainer = None
    arc_trainer = None
    logger.info(f"Starting trial {trial.number}")
    try:
        # Initialize fixed hyperparameters dictionary
        fixed_hyperparams = {}

        # Initialize config and symbol_freq_dict
        model_config = ModelConfig()
        training_config = TrainingConfig()
        config = Config(model=model_config, training=training_config)
        symbol_freq_dict = {}

        # Load hyperparameters from checkpoint if provided
        if args.use_synthetic_data:
            logger.info("Using synthetic data for training, validation, and testing.")
            all_synthetic_data = load_and_split_synthetic_data(args, config)
            train_data = load_dataset(args, config, dataset_type='train', all_synthetic_data=all_synthetic_data)
            val_data = load_dataset(args, config, dataset_type='val', all_synthetic_data=all_synthetic_data)
            test_data = load_dataset(args, config, dataset_type='test', all_synthetic_data=all_synthetic_data)
        else:
            logger.info("Using official ARC datasets for training, validation, and testing.")
            train_data = load_dataset(args, config, dataset_type='train')
            val_data = load_dataset(args, config, dataset_type='val')
            test_data = load_dataset(args, config, dataset_type='test')

        # Create configuration
        model_config = ModelConfig()
        training_config = TrainingConfig()
        config = Config(model=model_config, training=training_config)


        if args.fast_dev_run:
            # Disable symbol frequency balancing for fast development run
            symbol_freq_dict = {}
            balance_symbols = False
            balancing_method = "none"
            logger.debug("fast_dev_run is enabled. Disabling symbol frequency balancing.")
        else:
            if args.enable_symbol_freq:
                # Calculate Symbol Frequencies
                if args.use_synthetic_data:
                    logger.debug("Calculating symbol frequencies for synthetic training set")
                    symbol_freq = train_data.get_symbol_frequencies()
                else:
                    logger.debug("Calculating symbol frequencies for ARC training set")
                    symbol_freq = train_data.get_symbol_frequencies()

                logger.debug(f"Computed symbol frequencies: {symbol_freq}")

                # Directly copy symbol_freq to symbol_freq_dict
                # Ensure symbol_freq_dict is a dictionary
                if isinstance(symbol_freq, np.ndarray):
                    # If symbol_freq is a NumPy array, convert it to a dictionary
                    symbol_freq_dict = {i: float(freq) for i, freq in enumerate(symbol_freq)}
                    logger.debug("Converted symbol_freq from NumPy array to dictionary.")
                elif isinstance(symbol_freq, dict):
                    symbol_freq_dict = symbol_freq.copy()
                    logger.debug("Copied symbol_freq as a dictionary.")
                else:
                    raise TypeError(f"Unexpected type for symbol_freq: {type(symbol_freq)}. Expected dict or np.ndarray.")

                # Assert that symbol_freq_dict is indeed a dictionary
                assert isinstance(symbol_freq_dict, dict), f"symbol_freq_dict must be a dict, but got {type(symbol_freq_dict)}."

                # Remove the padding symbol from symbol_freq_dict
                pad_symbol_idx = config.training.pad_symbol_idx
                symbol_freq_dict.pop(pad_symbol_idx, None)
                logger.debug(f"Removed pad_symbol_idx ({pad_symbol_idx}) from symbol_freq_dict. New length: {len(symbol_freq_dict)}")

                # Debugging: Check keys and their types
                logger.debug(f"Keys in symbol_freq_dict after popping padding symbol: {list(symbol_freq_dict.keys())}")
                logger.debug(f"Types of keys in symbol_freq_dict: {set(type(k) for k in symbol_freq_dict.keys())}")

                # Ensure the length of symbol_freq_dict matches num_classes - 1
                assert len(symbol_freq_dict) == config.training.num_classes - 1, (
                    f"Length of symbol_freq_dict ({len(symbol_freq_dict)}) does not match num_classes minus padding ({config.training.num_classes - 1})."
                )
                balance_symbols = True
                balancing_method = "weighting"
            else:
                symbol_freq_dict = {}
                logger.debug("Symbol frequency calculation is disabled. Using empty symbol_freq_dict.")
                balance_symbols = False
                balancing_method = "none"
        include_pad_in_loss = args.include_pad_in_loss
        torch.set_float32_matmul_precision(args.matmul_precision)
        logger.info(f"Trial {trial.number}: Set float32 matmul precision to: {args.matmul_precision}")

        if not args.model_checkpoint:
            # Only suggest hyperparameters that are not fixed by the checkpoint
            # Suggest n_head exponent and calculate n_head
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

            # Suggest Mamba-specific hyperparameters
            mamba_ratio = trial.suggest_float("mamba_ratio", args.mamba_ratio_min, args.mamba_ratio_max, step=args.mamba_ratio_step)
            d_state = trial.suggest_int("d_state", args.d_state_min, args.d_state_max)
            d_conv = trial.suggest_int("d_conv_min", args.d_conv_min, args.d_conv_max)

            # Suggest dropout rate
            dropout = trial.suggest_float("dropout", args.dropout_min, args.dropout_max, step=args.dropout_step)
            mamba_depth = trial.suggest_int("mamba_depth", args.mamba_depth_min, args.mamba_depth_max)
            logger.debug(f"Suggested mamba_depth: {mamba_depth}")

            mamba_expand = trial.suggest_int("mamba_expand", args.mamba_expand_min, args.mamba_expand_max)
            logger.debug(f"Suggested mamba_expand: {mamba_expand}")

            logger.debug(f"Using suggested hyperparameters: n_head={n_head}, n_embd={n_embd}, "
                         f"n_layer={n_layer}, mamba_ratio={mamba_ratio}, d_state={d_state}, "
                         f"d_conv={d_conv}, dropout={dropout}, mamba_depth={mamba_depth}, "
                         f"mamba_expand={mamba_expand}")

            # Ensure the Config uses the fixed hyperparameters
            config = Config(model=model_config, training=training_config)


        # Use grokfast based on command line argument
        use_grokfast = args.use_grokfast

        if use_grokfast:
            # Suggest Grokfast type based on command-line choices
            grokfast_type = trial.suggest_categorical("grokfast_type", args.grokfast_type_choices)

            # Suggest Grokfast alpha within specified range
            grokfast_alpha = trial.suggest_float("grokfast_alpha", args.grokfast_alpha_min, args.grokfast_alpha_max)

            # Suggest Grokfast lambda within specified range
            grokfast_lamb = trial.suggest_float("grokfast_lamb", args.grokfast_lamb_min, args.grokfast_lamb_max)

            # If using 'ma', suggest window_size within specified range
            if grokfast_type == "ma":
                grokfast_window_size = trial.suggest_int("grokfast_window_size", args.grokfast_window_size_min, args.grokfast_window_size_max)
            else:
                grokfast_window_size = None
        else:
            grokfast_type = None
            grokfast_alpha = None
            grokfast_lamb = None
            grokfast_window_size = None
        batch_size = trial.suggest_int("batch_size", args.batch_size_min, args.batch_size_max)
        learning_rate = trial.suggest_float("learning_rate", args.learning_rate_min, args.learning_rate_max, log=True)
        max_epochs = trial.suggest_int("max_epochs", args.max_epochs_min, args.max_epochs_max)
        
        # If a checkpoint is used, set fixed values and do not suggest architecture-related hyperparameters
        if args.model_checkpoint:
            n_head = model_config.n_head
            n_embd = model_config.n_embd
            n_layer = model_config.n_layer
            mamba_ratio = model_config.mamba_ratio
            d_state = model_config.d_state
            d_conv = model_config.d_conv
            dropout = model_config.dropout
            mamba_depth = model_config.mamba_depth
            mamba_expand = model_config.mamba_expand

        # Validate hyperparameters
        validate_hyperparameters(n_embd, n_head, n_layer, mamba_ratio, d_state, d_conv, dropout)

        # Check if the model will fit in memory
        # Adjust the total number of layers to include Mamba layers
        total_mamba_layers = int(n_layer * mamba_ratio)
        total_layers = n_layer + total_mamba_layers

        # Recalculate total parameters based on total_layers
        total_params = calculate_params(
            n_layers=total_layers,
            n_heads=n_head,
            d_model=n_embd,
            mamba_ratio=mamba_ratio,
            d_state=d_state,
            d_conv=d_conv,
            mamba_depth=mamba_depth,
            mamba_expand=mamba_expand
        )
        # Improve memory estimation by considering additional factors like optimizer state and activation memory
        safety_margin = 0.1  # 10% safety margin
        estimated_memory = estimate_memory_usage(
            total_params=total_params,
            batch_size=batch_size,
            height=30,  # Adjust as necessary based on your data
            width=30,   # Adjust as necessary
            d_model=n_embd
        )
        available_memory = get_available_memory()
        estimated_memory *= (1 + safety_margin)

        logger.debug(f"Trial {trial.number}: Estimated memory usage: {estimated_memory:.2f} GB")
        logger.debug(f"Trial {trial.number}: Available memory: {available_memory:.2f} GB")

        # Prune the trial if estimated memory exceeds 80% of available memory
        if not can_fit_model(estimated_memory, available_memory * 0.8):
            logger.warning(f"Trial {trial.number}: Model too large for available memory. Skipping.")
            raise optuna.exceptions.TrialPruned()

        logger.debug(f"Suggested dropout rate: {dropout}")

        # Calculate Symbol Frequencies only if enabled
        if args.enable_symbol_freq:
            try:
                if args.use_synthetic_data:
                    logger.debug("Calculating symbol frequencies for synthetic training set")
                    symbol_freq = train_data.get_symbol_frequencies()
                else:
                    logger.debug("Calculating symbol frequencies for ARC training set")
                    symbol_freq = train_data.get_symbol_frequencies()

                logger.debug(f"Computed symbol frequencies: {symbol_freq}")

                if isinstance(symbol_freq, np.ndarray):
                    symbol_freq_dict = {i: float(freq) for i, freq in enumerate(symbol_freq)}
                    logger.debug("Converted symbol_freq from NumPy array to dictionary.")
                elif isinstance(symbol_freq, dict):
                    symbol_freq_dict = symbol_freq.copy()
                    logger.debug("Copied symbol_freq as a dictionary.")
                else:
                    raise TypeError(f"Unexpected type for symbol_freq: {type(symbol_freq)}. Expected dict or np.ndarray.")

                pad_symbol_idx = config.training.pad_symbol_idx
                symbol_freq_dict.pop(pad_symbol_idx, None)
                logger.debug(f"Removed pad_symbol_idx ({pad_symbol_idx}) from symbol_freq_dict. New length: {len(symbol_freq_dict)}")

                assert isinstance(symbol_freq_dict, dict), f"symbol_freq_dict must be a dict, but got {type(symbol_freq_dict)}."
                assert len(symbol_freq_dict) == config.training.num_classes - 1, (
                    f"Length of symbol_freq_dict ({len(symbol_freq_dict)}) does not match num_classes minus padding ({config.training.num_classes - 1})."
                )

            except Exception as e:
                logger.error(f"Symbol frequency calculation failed: {str(e)}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Symbol frequency calculation failed: {str(e)}")
        else:
            symbol_freq_dict = {}
            logger.debug("Symbol frequency calculation is disabled. Using empty symbol_freq_dict.")

        if args.model_checkpoint:
            # Use fixed hyperparameters from the checkpoint
            model_config = ModelConfig(**fixed_hyperparams)
            training_config = TrainingConfig(
                num_classes=config.model.num_classes,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                use_grokfast=use_grokfast,
                grokfast_type=grokfast_type,
                grokfast_alpha=grokfast_alpha,
                grokfast_lamb=grokfast_lamb,
                grokfast_window_size=grokfast_window_size,
                include_pad_in_loss=include_pad_in_loss,
                symbol_freq=symbol_freq_dict,
                balance_symbols=balance_symbols,
                balancing_method="weighting" if balance_symbols else "none"
            )
        else:
            # Use suggested hyperparameters
            model_config = ModelConfig(
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                dropout=dropout,
                mamba_ratio=mamba_ratio,
                d_state=d_state,
                d_conv=d_conv,
                mamba_depth=mamba_depth,
                mamba_expand=mamba_expand
            )
            training_config = TrainingConfig(
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                use_grokfast=use_grokfast,
                grokfast_type=grokfast_type,
                grokfast_alpha=grokfast_alpha,
                grokfast_lamb=grokfast_lamb,
                grokfast_window_size=grokfast_window_size,
                include_pad_in_loss=include_pad_in_loss,
                symbol_freq=symbol_freq_dict,
                balance_symbols=balance_symbols,
                balancing_method=balancing_method,
                num_workers=args.num_workers if args.num_workers is not None else multiprocessing.cpu_count(),
                prefetch_factor=args.prefetch_factor,
                persistent_workers=not args.no_persistent_workers,
                pin_memory=not args.no_pin_memory
            )

        config = Config(model=model_config, training=training_config)
        config.estimated_memory = estimated_memory
        config.available_memory = available_memory
        logger.debug(f"Suggested Mamba parameters - mamba_ratio: {mamba_ratio}, d_state: {d_state}, d_conv: {d_conv}")
        trial.set_user_attr("mamba_ratio", mamba_ratio)
        trial.set_user_attr("d_state", d_state)
        trial.set_user_attr("d_conv", d_conv)
        trial.set_user_attr("mamba_depth", mamba_depth)
        trial.set_user_attr("mamba_expand", mamba_expand)
        logger.debug(f"Full config: {config}")

        # Instantiate the ModelConfigSaver callback with the current config
        model_config_saver = ModelConfigSaver(config)


        # Calculate Symbol Frequencies
        if args.use_synthetic_data:
            logger.debug("Calculating symbol frequencies for synthetic training set")
            symbol_freq = train_data.get_symbol_frequencies()
        else:
            logger.debug("Calculating symbol frequencies for ARC training set")
            symbol_freq = train_data.get_symbol_frequencies()

        logger.debug(f"Computed symbol frequencies: {symbol_freq}")


        # Create model and trainer
        logger.debug("Creating model and trainer")
        num_classes = config.training.num_classes
        # Instantiate the GPT2ARC model with the constructed Config
        if args.model_checkpoint:
            # Load the checkpoint
            logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
            checkpoint = torch.load(args.model_checkpoint, map_location="cpu")

            # Extract model configuration from the checkpoint
            if 'model_config' in checkpoint:
                model_config_dict = checkpoint['model_config']
                model_config = ModelConfig(**model_config_dict)
                logger.debug("Extracted model configuration from checkpoint.")
            else:
                logger.error("Model configuration not found in checkpoint. Cannot proceed.")
                raise ValueError("Model configuration not found in checkpoint.")

            # Set hyperparameters from the extracted model_config
            n_head = model_config.n_head
            n_embd = model_config.n_embd
            n_layer = model_config.n_layer
            mamba_ratio = model_config.mamba_ratio
            d_state = model_config.d_state
            d_conv = model_config.d_conv
            dropout = model_config.dropout
            mamba_depth = model_config.mamba_depth
            mamba_expand = model_config.mamba_expand

            # Validate hyperparameters
            validate_hyperparameters(n_embd, n_head, n_layer, mamba_ratio, d_state, d_conv, dropout)

            # Reconstruct the config with the extracted model_config and training_config
            config = Config(model=model_config, training=training_config)

            # Instantiate the model with the exact configuration used during training
            model = GPT2ARC(config=config, num_classes=model_config.num_classes, symbol_freq=symbol_freq_dict, pad_symbol_idx=config.training.pad_symbol_idx)

            # Load the state_dict from the checkpoint
            if 'state_dict' in checkpoint:
                # Remove the "model." prefix from state dict keys
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            else:
                state_dict = checkpoint  # Adjust based on how you saved your model

            # Load the state_dict into the model with strict=False
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                logger.debug(f"Successfully loaded state_dict from checkpoint: {args.model_checkpoint}")
                if missing_keys:
                    logger.warning(f"Missing keys when loading state_dict: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")

                # Handle specific missing keys if necessary
                if "loss_fn.weight" in missing_keys:
                    logger.debug("'loss_fn.weight' not found in state_dict. Initializing with default weights.")
                    default_weights = torch.ones(config.model.num_classes)
                    model.loss_fn.weight = default_weights
                    logger.debug(f"'loss_fn.weight' initialized with weights: {default_weights}")
            except RuntimeError as e:
                logger.error(f"Error loading state_dict: {e}")
                raise e
            
        else:
            model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=symbol_freq_dict)
        
        # Generate model summary
        print("DEBUG: Attempting to generate model summary")
        try:
            model_summary = str(ModelSummary(model, max_depth=-1))
            print("DEBUG: Model summary generated successfully")
        except Exception as e:
            print(f"DEBUG: Error generating model summary - {str(e)}")
            model_summary = "Error generating model summary"

        # Save model summary to trial user attributes
        print("DEBUG: Attempting to save model summary to trial user attributes")
        try:
            trial.set_user_attr("model_summary", model_summary)
            print("DEBUG: Model summary saved to trial user attributes")
        except Exception as e:
            print(f"DEBUG: Error saving model summary to trial - {str(e)}")

        print("DEBUG: Model summary:")
        print(model_summary)

        # Calculate Mamba efficiency metrics
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug("Calculating Mamba efficiency metrics")
        sample_input = torch.randn(1, 1, 6, 6).to(device)
        model.to(device)
        mamba_metrics = calculate_mamba_efficiency(model, sample_input)
        for key, value in mamba_metrics.items():
            trial.set_user_attr(key, value)
            logger.debug(f"Mamba metric - {key}: {value}")

        # Initialize the ResultsCollector
        results_collector = ResultsCollector(config)

        arc_trainer = ARCTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            config=config,
            args=args,  # Add this line to pass args
            results_collector=results_collector  # Pass ResultsCollector to ARCTrainer
        )

        # Set up PyTorch Lightning trainer with custom pruning callback
        if args.no_progress_bar:
            logger.info("Disabling progress bar.")
        else:
            logger.info("Enabling progress bar.")
        pruning_callback = CustomPruningCallback(trial, monitor="val_loss")
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
        # Determine accelerator parameters based on the --accelerator argument
        if args.accelerator == "tpu":
            accelerator = 'tpu'
            devices = 'xla:1'  # Use 'xla:8' for TPU v3-8 pods
            strategy = 'tpu_spawn'  # Recommended strategy for TPU
        elif args.accelerator == "gpu":
            if torch.cuda.is_available():
                accelerator = 'gpu'
                devices = 1
            else:
                accelerator = 'cpu'
                devices = 1
            strategy = 'auto'  # Changed from None to 'auto'
        else:
            accelerator = 'cpu'
            devices = 1
            strategy = 'auto'  # Changed from None to 'auto'
            
        nan_loss_pruning_callback = NanLossPruningCallback()
        #callbacks.append(nan_loss_pruning_callback)
        logger.info("NanLossPruningCallback added to the training callbacks.")
        experiment_id = f"optuna_trial_{trial.number}"
        tb_logger = TensorBoardLogger(save_dir="runs", name=f"experiment_{experiment_id}")
        print(f"DEBUG: Optuna trial TensorBoard logger initialized. Log dir: {tb_logger.log_dir}")
        
        # Extract trial number
        trial_num = trial.number

        # Define task_id (assuming a single task; modify as needed for multiple tasks)
        task_id = "main_task"  # Replace with dynamic task identification if necessary

        # Define iter_num (e.g., based on trial.number or another tracking mechanism)
        iter_num = 1  # Initialize to 1; increment as needed within your optimization loop

        # Initialize the checkpoint callback with descriptive filename
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/trial_{trial.number}",
            filename=f"{'tuning-' if args.model_checkpoint else ''}epoch_{{epoch:02d}}-val_loss_{{val_loss:.4f}}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        logger.info("Standard ModelCheckpoint callback added to the training callbacks.")

        # Initialize the BestEpochTrackerCallback
        best_epoch_tracker = BestEpochTrackerCallback()

        # Initialize PyTorch Lightning Trainer with the checkpoint callback
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            callbacks=[pruning_callback, early_stop_callback, nan_loss_pruning_callback, checkpoint_callback, model_config_saver, best_epoch_tracker],
            logger=tb_logger,
            gradient_clip_val=1.0,    # Add gradient clipping
            val_check_interval=args.val_check_interval,  # Added line
            precision=16,             # Enable Automatic Mixed Precision
            enable_checkpointing=True,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            enable_progress_bar=not args.no_progress_bar
        )
        print("DEBUG: Trainer created for Optuna trial with TensorBoard logger")
        logger.debug(f"Trainer created with config: {trainer.state}")

        # Ensure model is in train mode before training
        model.train()
        logger.debug("Model set to train mode before training")

        # Enhanced Logging: Log model mode before training
        logger.info("Before training:")
        for name, module in model.named_modules():
            logger.debug(f"{name}: {'train' if module.training else 'eval'}")

        # Retrieve the best validation loss from the ModelCheckpoint callback
        if checkpoint_callback.best_model_score is not None:
            best_val_loss = checkpoint_callback.best_model_score.item()
            logger.info(f"Trial {trial.number}: Best validation loss: {best_val_loss}")
        else:
            logger.warning(f"Trial {trial.number}: No checkpoints were saved. Assigning a high validation loss.")
            best_val_loss = float('inf')
            return best_val_loss  # This will assign a poor score but won't raise a pruning exception
        logger.info(f"Trial {trial.number} completed. Best validation loss: {best_val_loss}")
        logger.debug("Starting training")
        trainer.fit(arc_trainer)

        # Enhanced Logging: Log model mode after training
        logger.info("After training:")
        for name, module in model.named_modules():
            logger.debug(f"{name}: {'train' if module.training else 'eval'}")

        # Define DataLoader for test data
        test_loader = DataLoader(
            test_data,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )

        # Evaluate the model on test data
        logger.info("Evaluating model on test dataset.")
        test_results = trainer.test(model=arc_trainer, dataloaders=test_loader)

        # Process test results
        if test_results:
            avg_test_loss = sum(result['test_loss'] for result in test_results) / len(test_results)
            avg_test_accuracy = sum(result['test_accuracy'] for result in test_results) / len(test_results)
            avg_test_diff_accuracy = sum(result['test_diff_accuracy'] for result in test_results) / len(test_results)

            logger.info(f"Test results - Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.4f}, Diff Accuracy: {avg_test_diff_accuracy:.4f}")

            # Update final metrics with actual test results
            arc_trainer.results_collector.set_final_metrics({
                "best_val_loss": best_val_loss,
                "best_epoch": trainer.current_epoch,
                "final_test_loss": avg_test_loss,
                "final_test_accuracy": avg_test_accuracy,
                "final_test_diff_accuracy": avg_test_diff_accuracy
            })

        return best_val_loss

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            logger.error(f"Trial {trial.number}: CUDA out of memory error.")
            logger.error("Pruning trial and suggesting to adjust hyperparameters.")
            trial.set_user_attr('failed_reason', 'CUDA out of memory')
            raise optuna.exceptions.TrialPruned()
        else:
            logger.error(f"Trial {trial.number}: A runtime error occurred: {str(e)}", exc_info=True)
            raise RuntimeError(f"Trial {trial.number}: A runtime error occurred: {str(e)}")
    except Exception as e:
        # Improved exception handling for symbol frequency issues
        if "symbol_freq" in str(e):
            logger.error(f"Trial {trial.number}: 'symbol_freq' is missing or invalid. Ensure it is calculated and passed correctly.", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Trial {trial.number}: 'symbol_freq' is missing or invalid.")
        else:
            logger.error(f"Trial {trial.number}: An unexpected error occurred: {str(e)}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Trial {trial.number}: An unexpected error occurred: {str(e)}")
    finally:
        # Ensure Proper Cleanup Between Trials
        logger.debug(f"Cleaning up after trial {trial.number}")
        if model is not None:
            del model
        if trainer is not None:
            del trainer
        if arc_trainer is not None:
            del arc_trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug(f"Cleanup completed for trial {trial.number}")
        gc.collect()
        torch.cuda.empty_cache()



from functools import partial

def run_optimization(n_trials=100, storage_name="sqlite:///optuna_results.db", n_jobs=-1, args=None, study_name="gpt2_arc_optimization_v2"):

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

    # Initialize configuration
    model_config = ModelConfig()
    training_config = TrainingConfig()
    config = Config(model=model_config, training=training_config)

    logger.info("Loading datasets once before optimization...")
    if args.use_synthetic_data:
        logger.info("Using synthetic data for training, validation, and testing.")
        all_synthetic_data = load_and_split_synthetic_data(args, config)
        train_data = load_dataset(args, config, dataset_type='train', all_synthetic_data=all_synthetic_data)
        val_data = load_dataset(args, config, dataset_type='val', all_synthetic_data=all_synthetic_data)
        test_data = load_dataset(args, config, dataset_type='test', all_synthetic_data=all_synthetic_data)
    else:
        logger.info("Using official ARC datasets for training, validation, and testing.")
        train_data = load_dataset(args, config, dataset_type='train')
        val_data = load_dataset(args, config, dataset_type='val')
        test_data = load_dataset(args, config, dataset_type='test')

    # Create a partial objective function that includes preloaded datasets
    objective_partial = partial(objective, args=args, train_data=train_data, val_data=val_data, test_data=test_data)
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=sampler
    )

    logger.info(f"Starting optimization with {n_trials} trials using {n_jobs} parallel jobs")
    logger.info(f"Data Splitting Ratios - Train: 80%, Validation: 10%, Test: 10%")
    if args.use_gpu:
        available_gpus = torch.cuda.device_count()
        if available_gpus > 1:
            n_jobs = max(n_jobs, available_gpus)
        else:
            n_jobs = 1  # Limit to 1 to prevent memory issues
    study.optimize(objective_partial, n_trials=n_trials, n_jobs=n_jobs)  # Use the partial function

    logger.info("Optimization completed")

    if study.best_trial and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
        print("DEBUG: Best trial found, attempting to retrieve model summary")
        best_model_summary = study.best_trial.user_attrs.get("model_summary")
        if best_model_summary:
            print("DEBUG: Model summary retrieved successfully")
            logger.info("Model summary for the best trial:")
            logger.info(best_model_summary)
        else:
            print("DEBUG: No model summary found for the best trial")
    else:
        logger.warning("No successful trials found. Please check the trial configurations and constraints.")
        if study.best_trial:
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best value: {study.best_trial.value}")
            
            best_trial = study.best_trial
            best_trial.set_user_attr("mamba_ratio", best_trial.params.get("mamba_ratio"))
            best_trial.set_user_attr("d_state", best_trial.params.get("d_state"))
            best_trial.set_user_attr("d_conv", best_trial.params.get("d_conv"))
    
            logger.info("Best Mamba metrics:")
            for key in ['mamba_forward_pass_time', 'mamba_params', 'mamba_params_ratio']:
                value = study.best_trial.user_attrs.get(key)
                if value is not None:
                    logger.info(f"  {key}: {value}")
            
            logger.info("Best hyperparameters:")
            for key, value in study.best_trial.params.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("No trials have been completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for GPT2ARC model.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials for optimization.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs. -1 means using all available cores.")
    parser.add_argument("--batch_size_min", type=int, default=1, help="Minimum value for batch_size.")
    parser.add_argument("--batch_size_max", type=int, default=1, help="Maximum value for batch_size.")
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a fast development test."
    )
    parser.add_argument(
        "--use_grokfast",
        action="store_true",
        help="Enable Grokfast for gradient filtering."
    )
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_results.db", help="Storage path for Optuna results.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help=(
            "How often to perform validation. "
            "If a float, represents the fraction of an epoch (e.g., 0.5 for halfway through each epoch). "
            "If an integer, represents the number of training steps."
        )
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint to resume optimization from."
    )
    parser.add_argument(
        "--include_pad_in_loss",
        type=lambda x: (str(x).lower() in ['true', '1', 't', 'y', 'yes']),
        default=True,
        help="Whether to include the padding class in the loss calculation. (True/False)"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="gpt2_arc_optimization_v3",
        help="Name of the Optuna study."
    )

    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--val_split", type=float, default=0.1, help="Proportion of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1, help="Proportion of data to use for testing")
    parser.add_argument("--n_embd_max", type=int, default=1, help="Maximum value for n_embd")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker threads for DataLoader. If not set, uses configuration default (total CPU count)."
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker."
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_true",
        help="Disable persistent workers in DataLoader."
    )
    parser.add_argument(
        "--no_pin_memory",
        action="store_true",
        help="Disable pin_memory in DataLoader."
    )
    parser.add_argument("--n_head_min", type=int, default=1, help="Minimum value for n_head")
    parser.add_argument("--n_head_max", type=int, default=1, help="Maximum value for n_head")
    parser.add_argument("--n_head_exp_min", type=int, default=1, help="Minimum exponent for n_head (2^x)")
    parser.add_argument("--n_head_exp_max", type=int, default=1, help="Maximum exponent for n_head (2^x)")
    parser.add_argument("--n_embd_multiplier_min", type=int, default=1, help="Minimum multiplier for n_embd")
    parser.add_argument("--n_embd_multiplier_max", type=int, default=1, help="Maximum multiplier for n_embd")
    parser.add_argument("--n_layer_min", type=int, default=1, help="Minimum value for n_layer")
    parser.add_argument("--n_layer_max", type=int, default=1, help="Maximum value for n_layer")
    parser.add_argument("--learning_rate_min", type=float, default=1e-5, help="Minimum value for learning_rate")
    parser.add_argument("--learning_rate_max", type=float, default=1e-2, help="Maximum value for learning_rate")
    parser.add_argument("--max_epochs_min", type=int, default=1, help="Minimum value for max_epochs")
    parser.add_argument("--max_epochs_max", type=int, default=10, help="Maximum value for max_epochs")

    parser.add_argument("--mamba_ratio_min", type=float, default=1.0, help="Minimum value for mamba_ratio")
    parser.add_argument("--mamba_ratio_max", type=float, default=8.0, help="Maximum value for mamba_ratio")
    parser.add_argument("--mamba_ratio_step", type=float, default=0.25, help="Step size for mamba_ratio")
    parser.add_argument("--d_state_min", type=int, default=1, help="Minimum value for d_state")
    parser.add_argument("--d_state_max", type=int, default=1, help="Maximum value for d_state")
    parser.add_argument("--d_conv_min", type=int, default=1, help="Minimum value for d_conv")
    parser.add_argument("--d_conv_max", type=int, default=1, help="Maximum value for d_conv")

    parser.add_argument("--dropout_min", type=float, default=0.0, help="Minimum value for dropout")
    parser.add_argument("--mamba_depth_min", type=int, default=1, help="Minimum value for mamba_depth")
    parser.add_argument("--mamba_depth_max", type=int, default=1, help="Maximum value for mamba_depth")
    parser.add_argument("--mamba_expand_min", type=int, default=2, help="Minimum value for mamba_expand")
    parser.add_argument("--mamba_expand_max", type=int, default=2, help="Maximum value for mamba_expand")
    parser.add_argument(
        "--enable_symbol_freq",
        action="store_true",
        help="Enable the calculation of symbol frequencies."
    )
    parser.set_defaults(enable_symbol_freq=False)
    parser.add_argument("--dropout_max", type=float, default=0.5, help="Maximum value for dropout")
    parser.add_argument("--dropout_step", type=float, default=0.1, help="Step size for dropout")
    parser.add_argument("--use_gpu", action="store_true", help="Flag to indicate whether to use GPU for training.")
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Disable the progress bar during training."
    )
    parser.add_argument("--use_synthetic_data", action="store_true", help="Flag to indicate whether to use synthetic data for training.")
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="medium",
        choices=["highest", "high", "medium"],
        help="Set the internal precision of float32 matrix multiplications for optimization trials. Options: 'highest', 'high', 'medium'. Defaults to 'medium'."
    )
    parser.add_argument("--synthetic_data_path", type=str, default="", help="Path to synthetic data for training.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "tpu"],
        help="Accelerator to use for training: 'cpu', 'gpu', or 'tpu'. Defaults to 'gpu'."
    )

    # Grokfast parameter ranges
    parser.add_argument("--grokfast_alpha_min", type=float, default=0.9, help="Minimum value for grokfast_alpha.")
    parser.add_argument("--grokfast_alpha_max", type=float, default=0.99, help="Maximum value for grokfast_alpha.")
    parser.add_argument("--grokfast_lamb_min", type=float, default=1.0, help="Minimum value for grokfast_lamb.")
    parser.add_argument("--grokfast_lamb_max", type=float, default=3.0, help="Maximum value for grokfast_lamb.")
    parser.add_argument("--grokfast_window_size_min", type=int, default=50, help="Minimum value for grokfast_window_size.")
    parser.add_argument("--grokfast_window_size_max", type=int, default=200, help="Maximum value for grokfast_window_size.")
    parser.add_argument("--grokfast_type_choices", type=str, nargs='+', default=["ema", "ma"], choices=["ema", "ma"], help="List of Grokfast types to consider during tuning.")


    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Log parsed arguments for debugging
    logger.debug(f"Parsed arguments: {vars(args)}")
    logger.setLevel(log_level)

    # Ensure the storage_name has the correct SQLite prefix and handle relative paths
    import os  # Ensure os is imported at the top of the file

    if not args.storage.startswith("sqlite:///"):
        if os.path.isabs(args.storage):
            args.storage = f"sqlite:////{args.storage}"
        else:
            args.storage = f"sqlite:///{os.path.abspath(args.storage)}"
    
    logger.debug(f"Optuna storage URL set to: {args.storage}")
    
    # Validate val_check_interval
    if args.val_check_interval <= 0:
        logger.error("The --val_check_interval must be a positive number.")
        sys.exit(1)

    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    logger.debug(f"Random seed set to: {args.random_seed}")

    run_optimization(
        n_trials=args.n_trials,
        storage_name=args.storage,
        n_jobs=args.n_jobs,
        args=args,
        study_name=args.study_name
    )

