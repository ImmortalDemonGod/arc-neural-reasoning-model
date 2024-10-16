# gpt2_arc/src/training/train.py
import argparse
import multiprocessing
import sys
import logging
import os
import json
import datetime
from unittest.mock import MagicMock, patch
import optuna
import arckit
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from lightning.pytorch.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import Callback
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader, WeightedRandomSampler

# Define the base directory for the arc-neural-reasoning-model
arc_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Add the root directory of the project to the PYTHONPATH
project_root = arc_model_dir
sys.path.insert(0, project_root)

import pytorch_lightning as pl
#import torch.autograd.profiler as profiler
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
from gpt2_arc.src.utils.results_collector import ResultsCollector
from gpt2_arc.src.utils import GrokfastCallback

def get_num_workers():
    try:
        return multiprocessing.cpu_count() // 2  # Use half of the available CPUs
    except NotImplementedError:
        return 4  # Default fallback
logger = logging.getLogger(__name__)

class ConfigSavingModelCheckpoint(ModelCheckpoint):
    def __init__(self, config, trial_num='NA', task_id='NA', iter_num='NA', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.trial_num = trial_num
        self.task_id = task_id
        self.iter_num = iter_num
        self.timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")  # e.g., 20240308T153045

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Add custom metadata to the checkpoint
        checkpoint['model_config'] = self.config.model.__dict__
        checkpoint['trial_num'] = self.trial_num
        checkpoint['task_id'] = self.task_id
        checkpoint['iter_num'] = self.iter_num
        checkpoint['timestamp'] = self.timestamp

        # Add the current epoch to the checkpoint
        checkpoint['epoch'] = trainer.current_epoch

        super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def format_checkpoint_name(self, metrics):
        """
        Override the method to include custom placeholders in the filename.
        """
        return self.filename.format(
            trial_num=self.trial_num,
            task_id=self.task_id,
            iter_num=self.iter_num,
            val_loss=metrics.get("val_loss", 0.0),
            epoch=metrics.get("epoch", 0),
            timestamp=self.timestamp
        )

class ModelConfigSaver(Callback):
    def __init__(self, config):
        """
        Initialize the ModelConfigSaver callback with the current configuration.

        Args:
            config (Config): The configuration object containing model parameters.
        """
        super().__init__()
        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Override the checkpoint saving to include the model configuration.

        Args:
            trainer (pl.Trainer): The Trainer instance.
            pl_module (pl.LightningModule): The LightningModule being trained.
            checkpoint (dict): The checkpoint dictionary to be modified.
        """
        checkpoint['model_config'] = self.config.model.__dict__

        
def load_train_dataset(args, config):
    """
    Load the training dataset based on the provided arguments and configuration.
    
    Args:
        args: Parsed command-line arguments.
        config: Configuration object containing model and training settings.
    
    Returns:
        ARCDataset: Loaded training dataset.
    """
    if args.use_synthetic_data:
        if not args.synthetic_data_path:
            raise ValueError("Synthetic data path not provided")
        logger.info(f"Loading synthetic training data from {args.synthetic_data_path}")
        return ARCDataset(args.synthetic_data_path)
    else:
        logger.info("Loading ARC training dataset")
        train_set, _ = arckit.load_data()
        return ARCDataset(
            train_set, 
            num_symbols=11, 
            pad_symbol_idx=config.training.pad_symbol_idx
        )

def load_val_dataset(args, config):
    """
    Load the validation dataset based on the provided arguments and configuration.
    
    Args:
        args: Parsed command-line arguments.
        config: Configuration object containing model and training settings.
    
    Returns:
        ARCDataset: Loaded validation dataset.
    """
    try:
        if args.use_synthetic_data:
            if not args.synthetic_data_path:
                raise ValueError("Synthetic data path not provided")
            logger.info(f"Loading synthetic validation data from {args.synthetic_data_path}")
            return ARCDataset(args.synthetic_data_path, is_test=True)
        else:
            logger.info("Loading ARC validation dataset")
            _, eval_set = arckit.load_data()
            return ARCDataset(
                eval_set, 
                num_symbols=11, 
                pad_symbol_idx=config.training.pad_symbol_idx
            )
    except Exception as e:
        logger.error(f"Failed to load validation dataset: {e}", exc_info=True)
        raise


def main(args):
    # Set float32 matrix multiplication precision
    torch.set_float32_matmul_precision(args.matmul_precision)
    logger.info(f"Set float32 matmul precision to: {args.matmul_precision}")
    log_level = getattr(logging, args.log_level.upper() if hasattr(args, 'log_level') else 'DEBUG', logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    profiler = PyTorchProfiler(
        dirpath=args.profiler_dirpath,
        filename=args.profiler_filename,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Include CUDA activities
        record_shapes=True,
        with_stack=True  # Enable stack tracing
    ) if args.use_profiler else None
    
    logger.setLevel(logging.DEBUG)  # Ensure logger is set to DEBUG
    
    logger.info("Starting main function")
    logger.debug(f"Command line arguments: {args}")

    trainer = None  # Initialize trainer to None

    try:
        if args.use_optuna:
            logger.info("Loading best hyperparameters from Optuna study")
            study_name = args.optuna_study_name

            if study_name is None:
                # Retrieve all study summaries from the storage
                study_summaries = optuna.get_all_study_summaries(storage=args.optuna_storage)
                study_names = [summary.study_name for summary in study_summaries]
                
                if len(study_names) == 1:
                    study_name = study_names[0]
                    logger.info(f"Automatically selected the only available study: {study_name}")
                elif len(study_names) == 0:
                    logger.error("No studies found in the specified Optuna storage.")
                    sys.exit(1)
                else:
                    logger.error("Multiple studies found in the specified Optuna storage. Please specify the study name using --optuna-study-name.")
                    sys.exit(1)

            study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)
            best_params = study.best_params
            logger.debug(f"Loaded best parameters: {best_params}")
            
            n_head = 2 ** best_params['n_head_exp']
            n_embd = n_head * best_params['n_embd_multiplier']
            n_embd = 2 ** int(np.log2(n_embd))
            model_config = ModelConfig(
                n_embd=n_embd,
                n_head=n_head,
                n_layer=best_params['n_layer'],
                dropout=best_params['dropout']
            )
            training_config = TrainingConfig(
                batch_size=best_params['batch_size'],
                learning_rate=best_params['learning_rate'],
                max_epochs=args.max_epochs,
                use_gpu=args.use_gpu,
                log_level=args.log_level,
                use_synthetic_data=args.use_synthetic_data,
                synthetic_data_path=args.synthetic_data_path,
            )
            training_config = TrainingConfig(
                batch_size=best_params['batch_size'],
                learning_rate=best_params['learning_rate'],
                max_epochs=args.max_epochs,  # Always use the user-provided max_epochs
            )
        else:
            logger.info("Using provided or default hyperparameters")
            model_config = ModelConfig(
                n_embd=args.n_embd,
                n_head=args.n_head,
                n_layer=args.n_layer,
                mamba_ratio=args.mamba_ratio,
                d_state=args.d_state,
                d_conv=args.d_conv,
                dropout=args.dropout,
                mamba_depth=args.mamba_depth,
                mamba_expand=args.mamba_expand
            )
            training_config = TrainingConfig(
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_epochs=args.max_epochs,
                use_gpu=args.use_gpu,
                log_level=args.log_level,
                use_synthetic_data=args.use_synthetic_data,
                synthetic_data_path=args.synthetic_data_path,
                use_grokfast=args.use_grokfast,
                grokfast_type=args.grokfast_type,
                grokfast_alpha=args.grokfast_alpha,
                grokfast_lamb=args.grokfast_lamb,
                grokfast_window_size=args.grokfast_window_size,
            )
        
        config = Config(model=model_config, training=training_config)
        logger.debug(f"Configuration: {config}")

        # Load data
        logger.info("Loading data")
        logger.info("Loading training and validation datasets in parallel to optimize performance using separate processes")

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit parallel tasks for loading datasets
            future_train = executor.submit(load_train_dataset, args, config)
            future_val = executor.submit(load_val_dataset, args, config)
            
            # Retrieve the loaded datasets
            try:
                train_data = future_train.result()
                val_data = future_val.result()
                logger.info("Successfully loaded training and validation datasets in parallel")
            except Exception as e:
                logger.error(f"Error occurred while loading datasets in parallel: {e}", exc_info=True)
                raise e

        if args.disable_symbol_freq:
            balance_symbols = False
            balancing_method = "none"
            symbol_freq_dict = {}
            logger.debug("Symbol frequency calculation is disabled. Using empty symbol_freq_dict.")
        else:
            if args.use_synthetic_data:
                logger.debug("Calculating symbol frequencies for synthetic training set")
                symbol_freq = train_data.get_symbol_frequencies()
            else:
                logger.debug("Calculating symbol frequencies for ARC training set")
                symbol_freq = train_data.get_symbol_frequencies()
            
            symbol_freq_dict = {i: float(freq) for i, freq in enumerate(symbol_freq)}
            pad_symbol_idx = config.training.pad_symbol_idx
            symbol_freq_dict.pop(pad_symbol_idx, None)
            logger.debug(f"Removed pad_symbol_idx ({pad_symbol_idx}) from symbol_freq_dict. New length: {len(symbol_freq_dict)}")
            
            assert len(symbol_freq_dict) == config.training.num_classes - 1, (
                f"Length of symbol_freq_dict ({len(symbol_freq_dict)}) does not match num_classes minus padding ({config.training.num_classes - 1})."
            )
            balance_symbols = True
            balancing_method = "weighting"

        training_config = TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            use_grokfast=args.use_grokfast,
            grokfast_type=args.grokfast_type,
            grokfast_alpha=args.grokfast_alpha,
            grokfast_lamb=args.grokfast_lamb,
            grokfast_window_size=args.grokfast_window_size,
            include_pad_in_loss=args.include_pad_in_loss,
            symbol_freq=symbol_freq_dict,
            balance_symbols=balance_symbols,
            balancing_method=balancing_method
        )
        config = Config(model=model_config, training=training_config)

        # Calculate symbol frequencies if not disabled
        if not args.disable_symbol_freq:
            if args.use_synthetic_data:
                logger.debug("Calculating symbol frequencies for synthetic training set")
                train_symbol_freq = train_data.get_symbol_frequencies()
            else:
                logger.debug("Calculating symbol frequencies for ARC training set")
                train_symbol_freq = train_data.get_symbol_frequencies()
        else:
            train_symbol_freq = {}

        # Set the number of classes based on TrainingConfig
        num_classes = config.training.num_classes
        logger.info(f"Number of classes set to: {num_classes}")
        logger.info("Creating DataLoader instances")
        # Create DataLoader instances
        logger.info("Creating DataLoader instances")
        if config.training.balance_symbols:
            if config.training.balancing_method == "weighting":
                # Compute class weights (inverse of frequencies)
                class_weights = 1.0 / torch.tensor(train_symbol_freq, dtype=torch.float)
                # Removed WeightedRandomSampler as it is not appropriate for multi-class samples
                train_loader = DataLoader(
                    train_data,
                    batch_size=config.training.batch_size,
                    num_workers=get_num_workers(),
                    shuffle=True,  # Enable shuffle
                    pin_memory=True if args.use_gpu else False,
                    prefetch_factor=config.training.prefetch_factor,
                    persistent_workers=config.training.persistent_workers
                )
                logger.debug("Class weights applied in loss function. WeightedRandomSampler removed.")
            elif config.training.balancing_method == "oversampling":
                # Placeholder for oversampling implementation
                logger.info("Oversampling method selected, but not yet implemented.")
                # Implement oversampling logic here if desired
                train_loader = DataLoader(
                    train_data,
                    batch_size=config.training.batch_size,
                    num_workers=get_num_workers(),
                    shuffle=True,  # Enable shuffle if not using a sampler
                    pin_memory=True if args.use_gpu else False,
                    prefetch_factor=config.training.prefetch_factor,
                    persistent_workers=config.training.persistent_workers
                )
            else:
                logger.warning(f"Unknown balancing method: {config.training.balancing_method}. Skipping balancing.")
                train_loader = DataLoader(
                    train_data,
                    batch_size=config.training.batch_size,
                    num_workers=get_num_workers(),
                    shuffle=True,  # Enable shuffle
                    pin_memory=True if args.use_gpu else False,
                    prefetch_factor=config.training.prefetch_factor,
                    persistent_workers=config.training.persistent_workers
                )
        else:
            train_loader = DataLoader(
                train_data,
                batch_size=config.training.batch_size,
                num_workers=get_num_workers(),
                shuffle=True,  # Enable shuffle
                pin_memory=True if args.use_gpu else False,
                prefetch_factor=config.training.prefetch_factor,
                persistent_workers=config.training.persistent_workers
            )
        val_loader = DataLoader(
            val_data,
            batch_size=config.training.batch_size,
            num_workers=get_num_workers(),
            pin_memory=True if args.use_gpu else False,
            prefetch_factor=config.training.prefetch_factor,
            persistent_workers=config.training.persistent_workers
        )
        logger.debug(f"DataLoaders created with batch size {args.batch_size}")

        # Initialize model
        logger.info("Initializing model")
        model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=symbol_freq_dict)
        logger.debug(f"Model initialized with config: {model_config}")

        # Load the checkpoint if specified
        if args.model_checkpoint:
            logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
            checkpoint = torch.load(args.model_checkpoint)
            if 'model_config' in checkpoint and 'training_config' in checkpoint:
                model_config = ModelConfig(**checkpoint['model_config'])
                training_config = TrainingConfig(**checkpoint['training_config'])
                config = Config(model=model_config, training=training_config)
                num_classes = config.training.num_classes
                symbol_freq_dict = config.training.symbol_freq
                model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=symbol_freq_dict)
                logger.debug(f"Loaded TrainingConfig with num_classes={num_classes} from checkpoint")
            else:
                logger.error("Checkpoint missing 'model_config' or 'training_config'.")
                raise KeyError("Checkpoint must contain 'model_config' and 'training_config'.")
            model.load_state_dict(checkpoint['state_dict'])

        # Initialize results collector
        results_collector = ResultsCollector(config)

        # Initialize experiment tracker
        tracker = ExperimentTracker(config, project=args.project)

        logger.debug("Initializing ARCTrainer")
        trainer = ARCTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            config=config
        )
        trainer.log_hyperparameters()

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
        
        # Initialize callbacks list                                                                                  
        callbacks = []

        # Initialize GrokfastCallback if enabled
        if config.training.use_grokfast:
            grokfast_callback = GrokfastCallback(
                filter_type=config.training.grokfast_type,  # 'ema' or 'ma'
                alpha=config.training.grokfast_alpha,
                lamb=config.training.grokfast_lamb,
                window_size=config.training.grokfast_window_size if config.training.grokfast_type == 'ma' else 100,  # default for ma
                warmup=True,
                trigger=False
            )
            callbacks.append(grokfast_callback)
            logger.info("GrokfastCallback added to the training callbacks.")
        else:
            logger.info("Grokfast is disabled; no callback added.")

        # Add the standard ModelCheckpoint callback
        if not args.no_checkpointing:
            checkpoint_callback = ModelCheckpoint(
                dirpath="checkpoints",
                filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)

            # Instantiate and add the ModelConfigSaver callback
            model_config_saver = ModelConfigSaver(config)
            callbacks.append(model_config_saver)
            logger.info("ModelConfigSaver callback added to the training callbacks.")



        logger.info("Setting up PyTorch Lightning trainer")

        # Define trial_num, task_id, and iter_num
        trial_num = 0  # Initialize to 0 or another appropriate default
        task_id = "default_task"  # Replace with dynamic task identification if necessary
        iter_num = 1  # Initialize to 1; increment as needed within your training loop

        # Removed the custom ConfigSavingModelCheckpoint as it's not needed

        if not args.no_logging:
            tb_logger = TensorBoardLogger(
                save_dir="runs",
                name=f"experiment_{trainer.results_collector.experiment_id}"
            )
            logger.debug(f"TensorBoard logger initialized. Log dir: {tb_logger.log_dir}")
        else:
            tb_logger = False
            logger.debug("Logging is disabled")

        pl_trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=tb_logger,
            callbacks=callbacks if callbacks else None,  # This now includes ModelCheckpoint
            enable_checkpointing=not args.no_checkpointing,
            enable_progress_bar=not args.no_progress_bar,
            fast_dev_run=args.fast_dev_run,  # Use the command-line argument
            gradient_clip_val=1.0,    # Add gradient clipping
            precision=16,             # Enable Automatic Mixed Precision
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            profiler=profiler
        )

        if tb_logger:
            trainer.results_collector.set_tensorboard_log_path(tb_logger.log_dir)
            logger.debug(f"TensorBoard log path set in results collector: {tb_logger.log_dir}")

        # Log initial memory usage
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved()} bytes")

        # Train the model
        logger.info("Starting model training")
        pl_trainer.fit(trainer, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Log memory usage after training
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"CUDA memory allocated after training: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"CUDA memory reserved after training: {torch.cuda.memory_reserved()} bytes")

        # After training, run test
        logger.info("Running model evaluation")
        test_results = pl_trainer.test(trainer)
        if test_results:
            avg_test_loss = sum(result['avg_test_loss'] for result in test_results) / len(test_results)
            avg_test_accuracy = sum(result['avg_test_accuracy'] for result in test_results) / len(test_results)
            avg_test_diff_accuracy = sum(result['avg_test_diff_accuracy'] for result in test_results) / len(test_results)

            logger.info(f"Test results - Loss: {avg_test_loss}, Accuracy: {avg_test_accuracy}, Diff Accuracy: {avg_test_diff_accuracy}")

            results = {
                "avg_test_loss": avg_test_loss,
                "avg_acc_with_pad": avg_test_accuracy,
                "avg_acc_without_pad": avg_test_diff_accuracy,
            }

            # Add task-specific results
            for result in test_results:
                for key, value in result.items():
                    if key.endswith('_test_accuracy') or key.endswith('_test_diff_accuracy'):
                        results[key] = value

            trainer.results_collector.set_test_results(results)

        trainer.results_collector.set_final_metrics({
            "best_val_loss": trainer.best_val_loss,
            "best_epoch": trainer.best_epoch,
            "final_test_loss": avg_test_loss,
            "final_test_accuracy": avg_test_accuracy
        })

        # Save the final model with configuration
        logger.info("Saving final model with configuration")
        model_path = f"final_model_{trainer.results_collector.experiment_id}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'state_dict': trainer.model.state_dict(),
            'model_config': trainer.config.model.__dict__,
            'training_config': trainer.config.training.__dict__
        }, model_path)
        trainer.results_collector.set_checkpoint_path(model_path)
        logger.debug(f"Model and configuration saved to: {model_path}")

        # Save results
        logger.info("Saving experiment results")
        os.makedirs("results", exist_ok=True)
        results_path = f"results/experiment_{trainer.results_collector.experiment_id}.json"
        trainer.results_collector.save_to_json(results_path)
        logger.debug(f"Results saved to: {results_path}")

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            logger.error("CUDA out of memory error occurred.")
            logger.error("Consider reducing the batch size or model complexity.")
            raise RuntimeError("CUDA out of memory error occurred.")
        else:
            logger.error(f"A runtime error occurred: {str(e)}", exc_info=True)
            raise RuntimeError(f"A runtime error occurred: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        sys.exit(1)  # Exit the program after logging the error
    finally:
        if 'tracker' in locals():
            tracker.finish()

    if trainer is not None:
        # ... proceed with training ...
        pass
    else:
        logger.error("Trainer was not initialized. Exiting the training loop.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ARC Neural Reasoning Model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_profiler", action="store_true", help="Enable the custom profiler")
    group.add_argument("--fast_dev_run", action="store_true", help="Run a fast development test")
    
    parser.add_argument(
        "--optuna_study_name",
        type=str,
        default=None,
        help="Name of the Optuna study to load. If not provided and only one study exists in storage, it will be used automatically."
    )
    parser.add_argument("--optuna_storage", type=str, default="sqlite:///optuna_results.db", help="Storage URL for the Optuna study")
    parser.add_argument("--n_embd", type=int, default=4, help="Embedding dimension for profiling")
    parser.add_argument("--n_head", type=int, default=1, help="Number of attention heads for profiling")
    parser.add_argument("--n_layer", type=int, default=1, help="Number of transformer layers for profiling")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for profiling")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, required=True, help="Maximum number of epochs")
    parser.add_argument("--mamba_ratio", type=float, default=0.0, help="Mamba ratio (float value)")

    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--d_state", type=int, default=4, help="Mamba state dimension")
    parser.add_argument("--d_conv", type=int, default=1, help="Mamba convolution dimension")
    parser.add_argument("--mamba_depth", type=int, default=1, help="Depth of each Mamba layer")
    parser.add_argument("--mamba_expand", type=int, default=2, help="Expand factor for each Mamba layer")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--use_grokfast", action="store_true", help="Enable Grokfast for gradient filtering.")
    parser.add_argument(
        "--grokfast_type",
        type=str,
        default="ema",
        choices=["ema", "ma"],
        help="Type of Grokfast filter to use: 'ema' or 'ma'."
    )
    parser.add_argument(
        "--grokfast_alpha",
        type=float,
        default=0.98,
        help="Alpha parameter for Grokfast-EMA."
    )
    parser.add_argument(
        "--grokfast_lamb",
        type=float,
        default=2.0,
        help="Lambda parameter for Grokfast filters."
    )
    parser.add_argument(
        "--grokfast_window_size",
        type=int,
        default=100,
        help="Window size for Grokfast-MA."
    )
    parser.add_argument("--no_logging", action="store_true", help="Disable logging")
    parser.add_argument("--no_checkpointing", action="store_true", help="Disable checkpointing")
    parser.add_argument("--no_progress-bar", action="store_true", help="Disable progress bar")
    parser.add_argument("--model_checkpoint", type=str, help="Path to the model checkpoint to resume training")
    parser.add_argument("--project", type=str, default="gpt2-arc", help="W&B project name")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--run_name", type=str, default="default_run", help="Name of the run for saving results")
    parser.add_argument("--use_synthetic_data", action="store_true", help="Use synthetic data for training")
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="medium",
        choices=["highest", "high", "medium"],
        help="Set the internal precision of float32 matrix multiplications. Options: 'highest', 'high', 'medium'. Defaults to 'medium'."
    )
    parser.add_argument("--synthetic_data_path", type=str, help="Path to synthetic data directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--use_optuna", action="store_true", help="Use best hyperparameters from Optuna study")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "tpu"],
        help="Accelerator to use for training: 'cpu', 'gpu', or 'tpu'. Defaults to 'gpu'."
    )
    parser.add_argument(
        "--profiler_dirpath",
        type=str,
        default="./profiler_logs",
        help="Directory path for profiler output files."
    )
    parser.add_argument(
        "--profiler_filename",
        type=str,
        default="profile",
        help="Filename for profiler output."
    )
    
    args = parser.parse_args()

    # Validate mamba_ratio
    if args.mamba_ratio < 0.0:
        logger.error("Invalid value for --mamba_ratio: must be non-negative.")
        sys.exit(1)
    main(args)

