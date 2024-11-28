import argparse
import logging
import random
import sys
import os
import torch
import numpy as np

from gpt2_arc.src.optimization.optimizer import run_optimization

def main():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for GPT2ARC model.")
    # (All the argument definitions go here)
    args = parser.parse_args()
    
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for GPT2ARC model.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples to load. Use None to load all samples.")
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
        default=0.01,
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

if __name__ == "__main__":
    main()
