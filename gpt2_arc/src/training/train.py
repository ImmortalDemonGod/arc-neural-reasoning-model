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
from lightning.pytorch.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader, WeightedRandomSampler

# Define the base directory for the arc-neural-reasoning-model
arc_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Add the root directory of the project to the PYTHONPATH
project_root = arc_model_dir
sys.path.insert(0, project_root)


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

def main(args):
    # Import project-specific modules after logging configuration
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from gpt2_arc.src.data.arc_dataset import ARCDataset
    from gpt2_arc.src.models.gpt2 import GPT2ARC
    from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
    from gpt2_arc.src.training.trainer import ARCTrainer
    from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
    from gpt2_arc.src.utils.results_collector import ResultsCollector
    from gpt2_arc.src.utils import GrokfastCallback
    from gpt2_arc.src.training.callbacks import ModelConfigSaver
    if args.n_embd % args.n_head != 0:
        adjusted_n_embd = ((args.n_embd + args.n_head - 1) // args.n_head) * args.n_head
        logger.warning(
            f"n_embd ({args.n_embd}) is not divisible by n_head ({args.n_head}). "
            f"Adjusting n_embd to the nearest higher multiple: {adjusted_n_embd}."
        )
        args.n_embd = adjusted_n_embd
    torch.set_float32_matmul_precision(args.matmul_precision)
    logger.info(f"Set float32 matmul precision to: {args.matmul_precision}")
    # Determine logging level based on --debug flag
    # Set log_level to DEBUG if --debug or --fast-dev-run is used
    if args.debug or args.fast_dev_run:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,  # Direct logs to stdout
        force=True          # Override existing logging configurations
    )
    logging.getLogger().setLevel(log_level)  # Ensure root logger is set to log_level

    # Optional: Add a StreamHandler to the specific logger
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    profiler = PyTorchProfiler(
        dirpath=args.profiler_dirpath,
        filename=args.profiler_filename,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Include CUDA activities
        record_shapes=True,
        with_stack=True  # Enable stack tracing
    ) if args.use_profiler else None
    
    
    # Add a print statement to confirm that main(args) is being executed
    print("INFO: Starting main function")
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
            
            # Override best_params with user-passed arguments if they are provided
            hyperparams = [
                'n_embd',
                'n_head',
                'n_layer',
                'dropout',
                'batch_size',
                'learning_rate',
                'mamba_ratio',
                'd_state',
                'd_conv',
                'mamba_depth',
                'mamba_expand',
            ]

            for param in hyperparams:
                arg_value = getattr(args, param.replace('-', '_'), None)
                default_value = parser.get_default(param.replace('-', '_'))
                if arg_value is not None and arg_value != default_value:
                    best_params[param] = arg_value

            n_head_exp = best_params.get('n_head_exp', 2)
            n_head = 2 ** n_head_exp

            n_embd_multiplier = best_params.get('n_embd_multiplier', 4)
            n_embd = n_head * n_embd_multiplier
            n_embd = 2 ** int(np.log2(n_embd))

            # Ensure that n_embd is set; if not, use args.n_embd
            if not n_embd:
                n_embd = args.n_embd if args.n_embd else 208  # Default to 208 if both are unavailable

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
                synthetic_data_path=args.synthetic_data_path
            )
        else:
            logger.info("Using provided or default hyperparameters")
            # Ensure that n_embd and n_head are not None

            model_config = ModelConfig(
                n_embd=args.n_embd,
                n_head=args.n_head,
                n_layer=args.n_layer,
                mamba_ratio=args.mamba_ratio,
                d_state=args.d_state,
                d_conv=args.d_conv,
                dropout=args.dropout,
                mamba_depth=args.mamba_depth if args.mamba_depth is not None else 1,
                mamba_expand=args.mamba_expand if args.mamba_expand is not None else 2
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
                grokfast_window_size=args.grokfast_window_size
            )
        
        config = Config(model=model_config, training=training_config)
        logger.debug(f"Configuration: {config}")

        logger.info("Loading data")
        if args.use_synthetic_data:
            if not args.synthetic_data_path:
                raise ValueError("Synthetic data path not provided")
            logger.info(f"Loading synthetic data from {args.synthetic_data_path}")
            synthetic_files = os.listdir(args.synthetic_data_path)
            logger.debug(f"Total files in synthetic data path: {len(synthetic_files)}")
            logger.debug(f"Sample files: {synthetic_files[:5]}... (total {len(synthetic_files)})")
            train_data = ARCDataset(
                data_source=args.synthetic_data_path,
                use_cache=not args.no_cache  # Pass the use_cache flag
            )
            synthetic_files = os.listdir(args.synthetic_data_path)
            logger.debug(f"Listing files in synthetic data path for validation: {synthetic_files[:5]}... (total {len(synthetic_files)})")
            val_data = ARCDataset(
                data_source=args.synthetic_data_path,
                is_test=True,
                use_cache=not args.no_cache  # Pass the use_cache flag
            )
            logger.info(f"Number of training examples: {len(train_data)}")
            logger.info(f"Number of validation examples: {len(val_data)}")
        else:
            logger.info("Loading ARC dataset")
            train_set, eval_set = arckit.load_data()
            logger.info(f"Number of tasks in train_set: {len(train_set.tasks)}")
            for task in train_set.tasks[:1]:  # Checking the first task
                logger.debug(f"Task ID: {task.id}")
                logger.debug(f"Number of train samples: {len(task.train)}")
                if task.train:
                    sample_input, sample_output = task.train[0]
                    logger.debug(f"Sample input type: {type(sample_input)}, value: {sample_input}")
                    logger.debug(f"Sample output type: {type(sample_output)}, value: {sample_output}")
                else:
                    logger.warning(f"Task {task.id} has no training samples.")

            logger.info(f"Number of tasks in eval_set: {len(eval_set.tasks)}")
            for task in eval_set.tasks[:5]:  # Inspect the first 5 tasks
                logger.info(f"Task {task.id} - Train samples: {len(task.train)}, Test samples: {len(task.test)}")
            logger.info(f"Number of tasks in eval_set: {len(eval_set.tasks)}")
            num_symbols = 10  # Adjust num_symbols as needed based on your dataset

            train_data = ARCDataset(
                train_set,
                is_test=False,
                num_symbols=num_symbols,
                debug=True,
                use_cache=not args.no_cache  # Pass the use_cache flag
            )
            val_data = ARCDataset(
                eval_set,
                is_test=True,
                num_symbols=num_symbols,
                debug=True,
                use_cache=not args.no_cache  # Pass the use_cache flag
            )
            logger.info(f"Number of training examples: {len(train_data)}")
            logger.info(f"Number of validation examples: {len(val_data)}")

        # Access dataset statistics
        train_grid_stats = train_data.get_grid_size_stats()
        train_symbol_freq = train_data.get_symbol_frequencies()

        val_grid_stats = val_data.get_grid_size_stats()
        val_symbol_freq = val_data.get_symbol_frequencies()

        # Convert train_symbol_freq from numpy array to dictionary with string keys
        train_symbol_freq_dict = {str(idx): float(freq) for idx, freq in enumerate(train_symbol_freq)}

        # Update the TrainingConfig with the symbol_freq dictionary
        training_config.symbol_freq = train_symbol_freq_dict

        logger.info(f"Train Grid Size Stats: {train_grid_stats}")
        logger.info(f"Train Symbol Frequencies: {train_symbol_freq_dict}")
        logger.info(f"Validation Grid Size Stats: {val_grid_stats}")
        logger.info(f"Validation Symbol Frequencies: {val_symbol_freq}")

        # Initialize experiment tracker
        tracker = ExperimentTracker(config, project=args.project)

        # Log dataset statistics to ExperimentTracker
        tracker.log_metric("train_max_grid_height", train_grid_stats.get("max_height", 0))
        tracker.log_metric("train_max_grid_width", train_grid_stats.get("max_width", 0))
        tracker.log_metric("train_symbol_frequencies", train_symbol_freq)

        tracker.log_metric("val_max_grid_height", val_grid_stats.get("max_height", 0))
        tracker.log_metric("val_max_grid_width", val_grid_stats.get("max_width", 0))
        tracker.log_metric("val_symbol_frequencies", val_symbol_freq)

        # Example: Adjust model configuration based on grid size stats
        max_grid_height = max(train_grid_stats.get("max_height", 30), val_grid_stats.get("max_height", 30))
        max_grid_width = max(train_grid_stats.get("max_width", 30), val_grid_stats.get("max_width", 30))
        logger.debug(f"Adjusted max grid size - Height: {max_grid_height}, Width: {max_grid_width}")

        # Set the number of classes
        num_classes = 10
        logger.info(f"Number of classes set to: {num_classes}")

        num_train_samples = train_data.get_num_samples()
        num_val_samples = val_data.get_num_samples()
        logger.info(f"Number of training examples: {num_train_samples}")
        logger.info(f"Number of validation examples: {num_val_samples}")
        
        if num_train_samples == 0 or num_val_samples == 0:
            logger.error("The dataset is empty. Please check the synthetic data path or dataset contents.")
            return

        logger.debug(f"Train data size: {train_data.get_num_samples()}, Validation data size: {val_data.get_num_samples()}")

        # Set the number of classes
        num_classes = 10
        logger.info(f"Number of classes set to: {num_classes}")

        # Create DataLoader instances with appropriate settings
        logger.info("Creating DataLoader instances")
        train_loader = DataLoader(
            train_data,
            batch_size=config.training.batch_size,
            num_workers=get_num_workers(),
            shuffle=True,  # Enable shuffle
            pin_memory=True if args.use_gpu else False,
            prefetch_factor=config.training.prefetch_factor,
            persistent_workers=config.training.persistent_workers,
            collate_fn=ARCDataset.collate_fn  # Ensure the custom collate_fn is used
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config.training.batch_size,
            num_workers=get_num_workers(),
            shuffle=False,  # Typically, shuffle is False for validation
            pin_memory=True if args.use_gpu else False,
            prefetch_factor=config.training.prefetch_factor,
            persistent_workers=config.training.persistent_workers,
            collate_fn=ARCDataset.collate_fn
        )
        logger.debug(f"DataLoaders created with batch size {args.batch_size}")

        # Ensure symbol_freq is non-empty
        if not train_symbol_freq_dict:
            logger.error("Training symbol frequencies are empty. Cannot initialize GPT2ARC.")
            sys.exit(1)

        # Initialize model with symbol_freq
        logger.info("Initializing model")
        model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=train_symbol_freq_dict)
        logger.debug(f"Model initialized with config: {model_config}")

        # Load the checkpoint if specified
        if args.model_checkpoint:
            logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
            checkpoint = torch.load(args.model_checkpoint)
            if 'model_config' in checkpoint:
                model_config = ModelConfig(**checkpoint['model_config'])
                model = GPT2ARC(config=model_config, num_classes=10)
            model.load_state_dict(checkpoint['state_dict'])

        # Initialize results collector
        results_collector = ResultsCollector(config)

        # Initialize experiment tracker
        tracker = ExperimentTracker(config, project=args.project)

        logger.debug("Initializing ExperimentTracker")
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
            "final_test_accuracy": avg_test_accuracy
        })

        # Save the final model with configuration
        logger.info("Saving final model with configuration")
        model_path = f"final_model_{trainer.results_collector.experiment_id}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'state_dict': trainer.model.state_dict(),
            'model_config': trainer.config.model.__dict__
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
            print("ERROR: CUDA out of memory error occurred.")  # Fallback print
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
    group.add_argument("--use-profiler", action="store_true", help="Enable the custom profiler")
    group.add_argument("--fast-dev-run", action="store_true", help="Run a fast development test")
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching of the dataset"
    )
    parser.add_argument(
        "--optuna_study_name",
        type=str,
        default=None,
        help="Name of the Optuna study to load. If not provided and only one study exists in storage, it will be used automatically."
    )
    parser.add_argument("--optuna_storage", type=str, default="sqlite:///optuna_results.db", help="Storage URL for the Optuna study")
    parser.add_argument("--n_embd", type=int, default=208, help="Embedding dimension. Must be divisible by n_head. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of transformer layers. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--max_epochs", type=int, required=True, help="Maximum number of epochs")
    parser.add_argument("--mamba_ratio", type=float, default=1.0, help="Mamba ratio (float value). Overrides Optuna's suggested value if provided.")

    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--d_state", type=int, default=64, help="Mamba state dimension. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--d_conv", type=int, default=128, help="Mamba convolution dimension. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--mamba_depth", type=int, default=1, help="Depth of each Mamba layer. Overrides Optuna's suggested value if provided.")
    parser.add_argument("--mamba_expand", type=int, default=2, help="Expand factor for each Mamba layer. Overrides Optuna's suggested value if provided.")
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
    parser.add_argument("--no_progress_bar", action="store_true", help="Disable progress bar")
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

    main(args)

