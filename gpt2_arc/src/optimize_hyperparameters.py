# gpt2_arc/src/optimize_hyperparameters.py
import argparse
import optuna
import logging
import sys
import os
import torch
import gc
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.model_summary import ModelSummary
from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from gpt2_arc.src.training.trainer import NanLossPruningCallback
from gpt2_arc.src.training.train import ModelConfigSaver

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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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


def calculate_symbol_freq(dataset):
    """Calculate the frequency of each symbol in the dataset."""
    symbol_counts = {}
    total_symbols = 0
    for input_tensor, output_tensor, task_id in dataset:
        # Assuming symbols are represented as integers in the tensors
        input_symbols = input_tensor.flatten().tolist()
        output_symbols = output_tensor.flatten().tolist()
        symbols = input_symbols + output_symbols
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            total_symbols += 1
    
    if total_symbols == 0:
        raise ValueError("The dataset contains no symbols to calculate frequencies.")
    
    # Calculate normalized frequencies
    symbol_freq = {symbol: count / total_symbols for symbol, count in symbol_counts.items()}
    return symbol_freq



def objective(trial, args):
    model = None
    trainer = None
    arc_trainer = None
    logger.info(f"Starting trial {trial.number}")
    try:
        # Set float32 matrix multiplication precision
        torch.set_float32_matmul_precision(args.matmul_precision)
        logger.info(f"Trial {trial.number}: Set float32 matmul precision to: {args.matmul_precision}")
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


        validate_hyperparameters(n_embd, n_head, n_layer, mamba_ratio, d_state, d_conv, dropout)

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

        # Ensure hyperparameters are within the new limits
        n_head = min(n_head, 2 ** args.n_head_exp_max)
        n_embd = min(n_embd, 2 ** int(np.log2(args.n_embd_multiplier_max * n_head)))
        n_layer = min(n_layer, args.n_layer_max)
        mamba_ratio = min(mamba_ratio, args.mamba_ratio_max)
        d_state = min(d_state, args.d_state_max)
        d_conv = min(d_conv, args.d_conv_max)
        batch_size = min(batch_size, args.batch_size_max)
        # Optionally, log the clamped values
        logger.debug(f"Clamped hyperparameters: n_head={n_head}, n_embd={n_embd}, n_layer={n_layer}, \
            mamba_ratio={mamba_ratio}, d_state={d_state}, d_conv={d_conv}, batch_size={batch_size}")

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
        estimated_memory = estimate_memory_usage(
            total_params=total_params,
            batch_size=batch_size,
            height=30,  # Adjust as necessary based on your data
            width=30,   # Adjust as necessary
            d_model=n_embd
        )
        available_memory = get_available_memory()

        logger.debug(f"Trial {trial.number}: Estimated memory usage: {estimated_memory:.2f} GB")
        logger.debug(f"Trial {trial.number}: Available memory: {available_memory:.2f} GB")

        # Prune the trial if estimated memory exceeds 80% of available memory
        if not can_fit_model(estimated_memory, available_memory * 0.8):
            logger.warning(f"Trial {trial.number}: Model too large for available memory. Skipping.")
            raise optuna.exceptions.TrialPruned()

        logger.debug(f"Suggested dropout rate: {dropout}")

        model_config = ModelConfig(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            mamba_ratio=mamba_ratio,
            d_state=d_state,
            d_conv=d_conv
        )
        logger.debug(f"Model config: {model_config}")

        # Create TrainingConfig with Grokfast parameters from args
        include_pad_in_loss = args.include_pad_in_loss
        training_config = TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            use_grokfast=use_grokfast,  # Use the fixed value from args
            grokfast_type=grokfast_type,
            grokfast_alpha=grokfast_alpha,
            grokfast_lamb=grokfast_lamb,
            grokfast_window_size=grokfast_window_size,
            include_pad_in_loss=include_pad_in_loss
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

        # Load data
        if args.use_synthetic_data:
            if not args.synthetic_data_path:
                raise ValueError("Synthetic data path not provided")
            logger.info(f"Loading synthetic data from {args.synthetic_data_path}")
            train_data = ARCDataset(
                data_source=args.synthetic_data_path,
                is_test=False,
                num_symbols=config.model.n_embd
            )
            val_data = ARCDataset(
                data_source=args.synthetic_data_path,
                is_test=True,
                num_symbols=config.model.n_embd
            )
            logger.debug(f"Synthetic training data size: {len(train_data)}")
            logger.debug(f"Synthetic validation data size: {len(val_data)}")
        else:
            logger.info("Loading ARC dataset")
            train_set, eval_set = arckit.load_data()
            train_data = ARCDataset(train_set)
            val_data = ARCDataset(eval_set)
            logger.debug(f"ARC training data size: {len(train_data)}")
            logger.debug(f"ARC validation data size: {len(val_data)}")

        # Calculate Symbol Frequencies
        if args.use_synthetic_data:
            logger.debug("Calculating symbol frequencies for synthetic training set")
            symbol_freq = train_data.get_symbol_frequencies()
        else:
            logger.debug("Calculating symbol frequencies for ARC training set")
            symbol_freq = train_data.get_symbol_frequencies()

        logger.debug(f"Computed symbol frequencies: {symbol_freq}")

        # Convert symbol_freq from NumPy array to dictionary
        symbol_freq_dict = {str(i): float(freq) for i, freq in enumerate(symbol_freq)}
        logger.debug(f"Converted symbol frequencies to dictionary: {symbol_freq_dict}")

        # Assign the converted symbol_freq to the training configuration
        config.training.symbol_freq = symbol_freq_dict

        # Validate that symbol_freq_dict is not empty
        if not symbol_freq_dict:
            logger.error("symbol_freq_dict is empty. Cannot proceed with balance_symbols=True and balancing_method='weighting'.")
            raise ValueError("symbol_freq must be provided and non-empty when balance_symbols is True and balancing_method is 'weighting'.")

        # Create model and trainer
        logger.debug("Creating model and trainer")
        num_classes = 10  # Set this to the appropriate number of classes for your task
        model = GPT2ARC(config, num_classes=num_classes, symbol_freq=symbol_freq_dict)
        
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

        arc_trainer = ARCTrainer(model, train_data, val_data, config)

        # Set up PyTorch Lightning trainer with custom pruning callback
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
            dirpath="checkpoints",
            filename="trial_{trial_num}-epoch_{epoch:02d}-val_loss_{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        logger.info("Standard ModelCheckpoint callback added to the training callbacks.")

        # Initialize PyTorch Lightning Trainer with the checkpoint callback
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            callbacks=[pruning_callback, early_stop_callback, nan_loss_pruning_callback, checkpoint_callback, model_config_saver],
            logger=tb_logger,
            gradient_clip_val=1.0,    # Add gradient clipping
            precision=16,             # Enable Automatic Mixed Precision
            enable_checkpointing=True,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
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

        # Train and evaluate
        logger.debug("Starting training")
        trainer.fit(arc_trainer)

        # Enhanced Logging: Log model mode after training
        logger.info("After training:")
        for name, module in model.named_modules():
            logger.debug(f"{name}: {'train' if module.training else 'eval'}")

        # Update iter_num if needed (e.g., if multiple iterations per trial)
        iter_num += 1

        # Retrieve the best validation loss for optimization
        best_val_loss = trainer.callback_metrics.get("val_loss").item()
        logger.info(f"Trial {trial.number} completed. Best validation loss: {best_val_loss}")

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
        if "symbol_freq" in str(e):
            logger.error(f"Trial {trial.number}: 'symbol_freq' is missing. Ensure it is calculated and passed correctly.", exc_info=True)
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



from functools import partial

def run_optimization(n_trials=100, storage_name="sqlite:///optuna_results.db", n_jobs=-1, args=None, study_name="gpt2_arc_optimization_v2"):

    pruner = PercentilePruner(percentile=25, n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    sampler = TPESampler(n_startup_trials=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=sampler
    )

    logger.info(f"Starting optimization with {n_trials} trials using {n_jobs} parallel jobs")
    study.optimize(partial(objective, args=args), n_trials=n_trials, n_jobs=n_jobs)

    logger.info("Optimization completed")

    if study.best_trial:
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
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")
        
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
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for GPT2ARC model.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials for optimization.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_results.db", help="Storage path for Optuna results.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs. -1 means using all available cores.")
    parser.add_argument(
        "--study_name",
        type=str,
        default="gpt2_arc_optimization_v2",
        help="Name of the Optuna study."
    )

    parser.add_argument("--n_embd_min", type=int, default=4, help="Minimum value for n_embd")
    parser.add_argument("--n_embd_max", type=int, default=8, help="Maximum value for n_embd")
    parser.add_argument("--n_head_min", type=int, default=2, help="Minimum value for n_head")
    parser.add_argument("--n_head_max", type=int, default=16, help="Maximum value for n_head")
    parser.add_argument("--n_head_exp_min", type=int, default=1, help="Minimum exponent for n_head (2^x)")
    parser.add_argument("--n_head_exp_max", type=int, default=3, help="Maximum exponent for n_head (2^x)")
    parser.add_argument("--n_embd_multiplier_min", type=int, default=1, help="Minimum multiplier for n_embd")
    parser.add_argument("--n_embd_multiplier_max", type=int, default=2, help="Maximum multiplier for n_embd")
    parser.add_argument("--n_layer_min", type=int, default=4, help="Minimum value for n_layer")
    parser.add_argument("--n_layer_max", type=int, default=8, help="Maximum value for n_layer")
    parser.add_argument("--batch_size_min", type=int, default=32, help="Minimum value for batch_size")
    parser.add_argument("--batch_size_max", type=int, default=128, help="Maximum value for batch_size")
    parser.add_argument("--learning_rate_min", type=float, default=1e-5, help="Minimum value for learning_rate")
    parser.add_argument("--learning_rate_max", type=float, default=1e-2, help="Maximum value for learning_rate")
    parser.add_argument("--max_epochs_min", type=int, default=1, help="Minimum value for max_epochs")
    parser.add_argument("--max_epochs_max", type=int, default=20, help="Maximum value for max_epochs")

    parser.add_argument("--mamba_ratio_min", type=float, default=0.0, help="Minimum value for mamba_ratio")
    parser.add_argument("--mamba_ratio_max", type=float, default=8.0, help="Maximum value for mamba_ratio")
    parser.add_argument("--mamba_ratio_step", type=float, default=0.25, help="Step size for mamba_ratio")
    parser.add_argument("--d_state_min", type=int, default=1, help="Minimum value for d_state")
    parser.add_argument("--d_state_max", type=int, default=2, help="Maximum value for d_state")
    parser.add_argument("--d_conv_min", type=int, default=1, help="Minimum value for d_conv")
    parser.add_argument("--d_conv_max", type=int, default=2, help="Maximum value for d_conv")

    parser.add_argument("--dropout_min", type=float, default=0.0, help="Minimum value for dropout")
    parser.add_argument("--mamba_depth_min", type=int, default=1, help="Minimum value for mamba_depth")
    parser.add_argument("--mamba_depth_max", type=int, default=3, help="Maximum value for mamba_depth")
    parser.add_argument("--mamba_expand_min", type=int, default=2, help="Minimum value for mamba_expand")
    parser.add_argument("--mamba_expand_max", type=int, default=4, help="Maximum value for mamba_expand")
    parser.add_argument("--dropout_max", type=float, default=0.5, help="Maximum value for dropout")
    parser.add_argument("--dropout_step", type=float, default=0.1, help="Step size for dropout")
    parser.add_argument("--use_gpu", action="store_true", help="Flag to indicate whether to use GPU for training.")
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

    # Ensure the storage_name has the correct SQLite prefix and handle relative paths
    import os  # Ensure os is imported at the top of the file

    if not args.storage.startswith("sqlite:///"):
        if os.path.isabs(args.storage):
            args.storage = f"sqlite:////{args.storage}"
        else:
            args.storage = f"sqlite:///{os.path.abspath(args.storage)}"
    
    logger.debug(f"Optuna storage URL set to: {args.storage}")
    run_optimization(
        n_trials=args.n_trials,
        storage_name=args.storage,
        n_jobs=args.n_jobs,
        args=args,
        study_name=args.study_name
    )
