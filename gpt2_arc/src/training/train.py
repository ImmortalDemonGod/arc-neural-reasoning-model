# gpt2_arc/src/training/train.py
import argparse
from typing import Optional
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
from gpt2_arc.src.training.trainer import ARCTrainer, get_num_workers
logger = logging.getLogger(__name__)
logger.debug(f"Imported get_num_workers from training_helpers: {get_num_workers}")
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
from gpt2_arc.src.utils.results_collector import ResultsCollector
from gpt2_arc.src.utils import GrokfastCallback

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

        
def load_dataset(args, config, dataset_type='train', all_synthetic_data=None):
    """
    Load the specified dataset based on the provided arguments and configuration.

    Args:
        args: Parsed command-line arguments.
        config: Configuration object containing model and training settings.
        dataset_type (str): Type of dataset to load ('train', 'val', 'test').
        all_synthetic_data: Loaded synthetic data splits (if using synthetic data).

    Returns:
        Dataset: Loaded dataset (ARCDataset or Subset of ARCDataset).
    """
    if args.use_synthetic_data:
        if all_synthetic_data is None:
            raise ValueError("Synthetic data not loaded")
        dataset = all_synthetic_data['dataset']
        indices = all_synthetic_data[f'{dataset_type}_indices']
        logger.info(f"Using synthetic {dataset_type} dataset with {len(indices)} samples")
        return dataset
    else:
        logger.info(f"Loading ARC {dataset_type} dataset")
        train_set, eval_set = arckit.load_data()

        if dataset_type == 'train':
            data_source = train_set
        else:
            # Extract samples from eval_set
            samples = []
            for task in eval_set.tasks:
                # Combine train and test examples from each task
                for ex in task.train:
                    samples.append({'input': ex[0], 'output': ex[1], 'task_id': task.id})
                for ex in task.test:
                    samples.append({'input': ex[0], 'output': ex[1], 'task_id': task.id})

            total_samples = len(samples)
            val_size = int(total_samples * args.val_split / (args.val_split + args.test_split))
            test_size = total_samples - val_size

            # Shuffle samples
            random.shuffle(samples)

            # Split samples into validation and test sets
            val_samples = samples[:val_size]
            test_samples = samples[val_size:]

            if dataset_type == 'val':
                data_source = val_samples
            elif dataset_type == 'test':
                data_source = test_samples
            else:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")

        dataset = ARCDataset(
            data_source=data_source,
            is_test=(dataset_type == 'test'),
            num_symbols=config.training.num_symbols,
            pad_symbol_idx=config.training.pad_symbol_idx,
            symbol_freq=config.training.symbol_freq if args.enable_symbol_freq else None
        )
    return dataset

import random

def load_and_split_synthetic_data(args, config):
    """
    Load synthetic data using ARCDataset and split it into train, val, and test sets.

    Args:
        args: Parsed command-line arguments.
        config: Configuration object containing model and training settings.

    Returns:
        dict: A dictionary containing 'dataset', 'train_indices', 'val_indices', and 'test_indices'.
    """
    logger.info(f"Loading synthetic data from {args.synthetic_data_path}")
    # Initialize ARCDataset with the synthetic data directory
    synthetic_dataset = ARCDataset(
        data_source=args.synthetic_data_path,
        is_test=False,
        num_symbols=config.training.num_symbols,
        pad_symbol_idx=config.training.pad_symbol_idx,
        symbol_freq=config.training.symbol_freq if args.enable_symbol_freq else None
    )

    total_samples = len(synthetic_dataset)
    logger.info(f"Total synthetic samples loaded: {total_samples}")

    # Shuffle the indices
    indices = list(range(total_samples))
    random.shuffle(indices)

    # Calculate split indices
    train_end = int(args.train_split * total_samples)
    val_end = train_end + int(args.val_split * total_samples)

    # Split the indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    logger.debug(
        f"Synthetic data split into "
        f"{len(train_indices)} training, "
        f"{len(val_indices)} validation, and "
        f"{len(test_indices)} test samples"
    )

    return {
        'dataset': synthetic_dataset,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }


def main(args):
    if args.use_synthetic_data and not args.synthetic_data_path:
        raise ValueError("--synthetic_data_path must be provided when using synthetic data.")

    total_split = args.train_split + args.val_split + args.test_split
    if not abs(total_split - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test splits must sum to 1.0")
    torch.set_float32_matmul_precision(args.matmul_precision)
    logger.info(f"Set float32 matmul precision to: {args.matmul_precision}")
    logger.debug(f"Command line arguments: {args}")
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
    logger.debug("Initializing PyTorch Lightning Trainer")
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
                dropout=best_params['dropout'],
                num_workers=args.num_workers if args.num_workers is not None else multiprocessing.cpu_count(),
                prefetch_factor=args.prefetch_factor,
                persistent_workers=not args.no_persistent_workers,
                pin_memory=not args.no_pin_memory,
            )
            training_config = TrainingConfig(
                batch_size=best_params['batch_size'],
                learning_rate=best_params['learning_rate'],
                max_epochs=args.max_epochs,
                use_gpu=args.use_gpu,
                log_level=args.log_level,
                use_synthetic_data=args.use_synthetic_data,
                synthetic_data_path=args.synthetic_data_path,
                include_pad_in_loss=args.include_pad_in_loss,
                include_pad_in_accuracy=args.include_pad_in_accuracy
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
                dropout=args.dropout,
                mamba_ratio=args.mamba_ratio,
                d_state=args.d_state,
                d_conv=args.d_conv,
                mamba_depth=args.mamba_depth,
                mamba_expand=args.mamba_expand,
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

        # Validate split ratios sum to 1.0
        total_split = args.train_split + args.val_split + args.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")

        if args.use_synthetic_data:
            # Load and split synthetic data
            all_synthetic_data = load_and_split_synthetic_data(args, config)
        else:
            all_synthetic_data = None

        try:
            # Load datasets
            logger.info("Loading datasets")
            train_data = load_dataset(args, config, dataset_type='train', all_synthetic_data=all_synthetic_data)
            val_data = load_dataset(args, config, dataset_type='val', all_synthetic_data=all_synthetic_data)
            test_data = load_dataset(args, config, dataset_type='test', all_synthetic_data=all_synthetic_data)
            
            # Debugging: Log the number of samples loaded
            logger.debug(f"Loaded {len(train_data)} training samples.")
            logger.debug(f"Loaded {len(val_data)} validation samples.")
            logger.debug(f"Loaded {len(test_data)} test samples.")
            
            # Additional Assertion (Optional)
            assert test_data is not None, "Test dataset is None after loading."
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise  # Re-raise the exception after logging

        if args.enable_symbol_freq:
            logger.debug("Calculating symbol frequencies as it is enabled.")
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
        else:
            symbol_freq_dict = {}
            logger.debug("Symbol frequency calculation is disabled. Using empty symbol_freq_dict.")
            balance_symbols = False
            balancing_method = "none"

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

        # Calculate symbol frequencies if enabled
        if args.enable_symbol_freq:
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
        from torch.utils.data import Subset

        if args.use_synthetic_data:
            train_loader = DataLoader(
                train_data,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory if torch.cuda.is_available() else False,
                prefetch_factor=config.training.prefetch_factor,
                persistent_workers=config.training.persistent_workers,
                collate_fn=ARCDataset.collate_fn
            )

            val_loader = DataLoader(
                val_data,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=get_num_workers(config.training),
                pin_memory=config.training.pin_memory if args.use_gpu else False,
                prefetch_factor=config.training.prefetch_factor,
                persistent_workers=config.training.persistent_workers,
                collate_fn=ARCDataset.collate_fn
            )

            test_loader = DataLoader(
                test_data,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=get_num_workers(config.training),
                pin_memory=config.training.pin_memory if args.use_gpu else False,
                prefetch_factor=config.training.prefetch_factor,
                persistent_workers=config.training.persistent_workers,
                collate_fn=ARCDataset.collate_fn
            )
        else:
            # Create Training DataLoader
            logger.debug("Creating Training DataLoader")
            train_loader = DataLoader(
                train_data,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=get_num_workers(config.training),
                pin_memory=config.training.pin_memory if args.use_gpu else False,
                prefetch_factor=config.training.prefetch_factor,
                persistent_workers=config.training.persistent_workers,
                collate_fn=ARCDataset.collate_fn
            )
            logger.debug("Created Training DataLoader")

            # Create Validation DataLoader
            logger.debug("Creating Validation DataLoader")
            val_loader = DataLoader(
                val_data,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=get_num_workers(config.training),
                pin_memory=config.training.pin_memory if args.use_gpu else False,
                prefetch_factor=config.training.prefetch_factor,
                persistent_workers=config.training.persistent_workers,
                collate_fn=ARCDataset.collate_fn
            )
            logger.debug("Created Validation DataLoader")

            # Create Test DataLoader
            logger.debug("Creating Test DataLoader")
            test_loader = DataLoader(
                test_data,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=get_num_workers(config.training),
                pin_memory=config.training.pin_memory if args.use_gpu else False,
                prefetch_factor=config.training.prefetch_factor,
                persistent_workers=config.training.persistent_workers,
                collate_fn=ARCDataset.collate_fn
            )
            logger.debug("Created Test DataLoader")

        # Ensure test_data is not None
        assert test_data is not None, "Test dataset is None after loading."

        # Initialize model
        logger.info("Initializing model")
        model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=symbol_freq_dict, pad_symbol_idx=config.training.pad_symbol_idx)
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
                model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=symbol_freq_dict, pad_symbol_idx=config.training.pad_symbol_idx)
                logger.debug(f"Loaded TrainingConfig with num_classes={num_classes} from checkpoint")
            else:
                logger.error("Checkpoint missing 'model_config' or 'training_config'.")
                raise KeyError("Checkpoint must contain 'model_config' and 'training_config'.")
            model.load_state_dict(checkpoint['state_dict'])

        # Log dataset source information
        if args.use_synthetic_data:
            logger.info("Using synthetic data for training, validation, and testing.")
        else:
            logger.info("Using official ARC datasets for training, validation, and testing.")
        results_collector = ResultsCollector(config)

        # Initialize experiment tracker
        tracker = ExperimentTracker(config, project=args.project)


        logger.info("Initializing ARCTrainer")
        trainer = ARCTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            config=config,
            args=args,
            results_collector=results_collector,
            test_dataset=test_data
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
            checkpoint_filename = f"{'resume-' if args.model_checkpoint else ''}checkpoint-{{epoch:02d}}-{{val_loss:.4f}}"
            checkpoint_callback = ModelCheckpoint(
                dirpath="checkpoints",
                filename=checkpoint_filename,
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
            profiler=profiler,
            val_check_interval=args.val_check_interval  # Added line
        )

        if tb_logger:
            trainer.results_collector.set_tensorboard_log_path(tb_logger.log_dir)
            logger.debug(f"TensorBoard log path set in results collector: {tb_logger.log_dir}")

        # Log initial memory usage
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved()} bytes")

        logger.info("Starting model training")
        logger.debug("Training parameters: ")
        logger.debug(f"Batch size: {config.training.batch_size}, Learning rate: {config.training.learning_rate}, Max epochs: {config.training.max_epochs}")

        # Train the model
        logger.info("Starting model training")
        # Update the fit call to exclude DataLoaders
        pl_trainer.fit(trainer)

        # Log memory usage after training
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"CUDA memory allocated after training: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"CUDA memory reserved after training: {torch.cuda.memory_reserved()} bytes")

        logger.info("Model training completed successfully.")

        # After training, run test
        logger.info("Starting model evaluation on test dataset.")
        logger.info("Running model evaluation")
        logger.debug("Preparing to run Trainer.test()")
        test_results = pl_trainer.test(model=trainer, dataloaders=test_loader)
        if test_results:
            avg_test_loss = sum(result['avg_test_loss'] for result in test_results) / len(test_results)
            avg_test_accuracy = sum(result['avg_test_accuracy'] for result in test_results) / len(test_results)
            avg_test_diff_accuracy = sum(result['avg_test_diff_accuracy'] for result in test_results) / len(test_results)

            logger.info(f"Test results - Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.4f}, Diff Accuracy: {avg_test_diff_accuracy:.4f}")

            results = {
                "avg_test_loss": avg_test_loss,
                "avg_test_accuracy": avg_test_accuracy,
                "avg_test_diff_accuracy": avg_test_diff_accuracy,
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
            "final_test_accuracy": avg_test_accuracy,
            "final_test_diff_accuracy": avg_test_diff_accuracy
        })

        # Save the final model with configuration
        logger.info("Saving final model with configuration")
        model_path = f"final_model_{trainer.results_collector.experiment_id}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'state_dict': trainer.model.state_dict(),
            'model_config': trainer.config.model.__dict__,
            'training_config': trainer.config.training.__dict__,
            'pad_symbol_idx': trainer.config.training.pad_symbol_idx,
            'symbol_freq': trainer.config.training.symbol_freq
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,  # Increased from None/1 to 4
        help="Number of worker threads for DataLoader. Increasing this can speed up data loading."
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
        "--pin_memory",
        action="store_true",
        help="Enable pin_memory in DataLoader for faster GPU data transfer."
    )
    parser.set_defaults(pin_memory=True)  # Enable by default if using GPU
    parser.add_argument("--n_embd", type=int, default=12, help="Embedding size for the model.")
    parser.add_argument("--n_head", type=int, default=1, help="Number of attention heads for profiling")
    parser.add_argument("--n_layer", type=int, default=1, help="Number of transformer layers for profiling")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for profiling")  # Increased from 1 to 16
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
        "--include_pad_in_loss",
        dest="include_pad_in_loss",
        action="store_true",
        help="Include the padding class in the loss calculation."
    )
    parser.add_argument(
        "--no_include_pad_in_loss",
        dest="include_pad_in_loss",
        action="store_false",
        help="Exclude the padding class from the loss calculation."
    )
    parser.set_defaults(include_pad_in_loss=True)
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
        "--enable_symbol_freq",
        action="store_true",
        help="Enable the calculation of symbol frequencies."
    )
    parser.set_defaults(enable_symbol_freq=False)
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--run_name", type=str, default="default_run", help="Name of the run for saving results")
    parser.add_argument("--use_synthetic_data", action="store_true", help="Use synthetic data for training")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--val_split", type=float, default=0.1, help="Proportion of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1, help="Proportion of data to use for testing")
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
    parser.add_argument(
        "--include_pad_in_accuracy",
        type=lambda x: (str(x).lower() in ['true', '1', 't', 'y', 'yes']),
        default=True,
        help="Whether to include the padding class in accuracy calculations. (True/False)"
    )
    
    args = parser.parse_args()

    # Validate mamba_ratio
    if args.mamba_ratio < 0.0:
        logger.error("Invalid value for --mamba_ratio: must be non-negative.")
        sys.exit(1)
    # Validate the val_check_interval
    if args.val_check_interval <= 0:
        logger.error("The --val_check_interval must be a positive number.")
        sys.exit(1)

    main(args)

