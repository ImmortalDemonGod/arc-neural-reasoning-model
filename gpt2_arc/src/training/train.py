import logging
import sys
import os
import datetime
import torch
from training_config_manager import ConfigurationManager
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
from gpt2_arc.src.utils.results_collector import ResultsCollector
from gpt2_arc.src.training.data_manager import DataManager

logger = logging.getLogger(__name__)

class ConfigSavingModelCheckpoint(ModelCheckpoint):
    def __init__(self, config: Config, trial_num: str = 'NA', task_id: str = 'NA', iter_num: str = 'NA', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.trial_num = trial_num
        self.task_id = task_id
        self.iter_num = iter_num
        self.timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")  # e.g., 20240308T153045

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        # Add custom metadata to the checkpoint
        checkpoint['model_config'] = self.config.model.__dict__
        checkpoint['trial_num'] = self.trial_num
        checkpoint['task_id'] = self.task_id
        checkpoint['iter_num'] = self.iter_num
        checkpoint['timestamp'] = self.timestamp

        # Add the current epoch to the checkpoint
        checkpoint['epoch'] = trainer.current_epoch

        super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def format_checkpoint_name(self, metrics: dict) -> str:
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


def main(args) -> None:
    """
    Main training function.
    
    Args:
        args: Arguments from train_cli.py with all training configuration options
    """
    # Initialize configuration
    config_manager = ConfigurationManager(args)
    config_manager.setup_logging()  # Setup logging first
    config = config_manager.create_initial_config()

    # Validate configuration including Mamba layers
    config_manager.validate_configuration(config)

    logger.debug(f"Command line arguments: {args}")
    
    logger.info("Starting main function")
    logger.debug("Initializing PyTorch Lightning Trainer")

    trainer = None  # Initialize trainer to None

    try:
        logger.debug(f"Configuration: {config}")

        # Initialize DataManager
        data_manager = DataManager(config, args)

        # Load all datasets
        train_data, val_data, test_data = data_manager.load_all_datasets()
        
        # Log the source of each dataset
        logger.info(f"Training dataset source: {'synthetic data' if args.use_synthetic_data else 'official ARC data'}")
        logger.info("Validation dataset source: official ARC data")
        logger.info("Test dataset source: official ARC data")
        
        # Log the number of samples in each dataset
        logger.debug(f"Number of training samples: {len(train_data)}")
        logger.debug(f"Number of validation samples: {len(val_data)}")
        logger.debug(f"Number of test samples: {len(test_data)}")

        # Calculate and update symbol frequencies if enabled
        if args.enable_symbol_freq:
            symbol_freq = train_data.get_symbol_frequencies()
            symbol_freq_dict = {i: float(freq) for i, freq in enumerate(symbol_freq)}
            config = config_manager.update_config_with_symbol_freq(config, symbol_freq_dict)

        # Set the number of classes based on TrainingConfig
        num_classes = config.training.num_classes
        logger.info(f"Number of classes set to: {num_classes}")

        # Ensure test_data is not None
        assert test_data is not None, "Test dataset is None after loading."

        # Initialize model
        logger.info("Initializing model")
        model = GPT2ARC(
            config=config,
            num_classes=config.training.num_classes,  # Use num_classes from config
            pad_symbol_idx=config.training.pad_symbol_idx
        )
        logger.debug(f"Model initialized with config: {config}")

        # Load the checkpoint if specified
        if args.model_checkpoint:
            logger.info(f"Loading model from checkpoint: {args.model_checkpoint}")
            config, state_dict = config_manager.load_checkpoint_config(args.model_checkpoint)
            
            model = GPT2ARC(
                config=config,
                num_classes=config.training.num_classes,
                symbol_freq=config.training.symbol_freq,
                pad_symbol_idx=config.training.pad_symbol_idx
            )
            model.load_state_dict(state_dict)
            logger.debug(f"Loaded model configuration with num_classes={config.training.num_classes}")

        # Initialize experiment tracker
        tracker = ExperimentTracker(config, project=args.project)

        results_collector = ResultsCollector(config)
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

        logger.info("Setting up PyTorch Lightning trainer")

        # Get profiler configuration
        profiler = config_manager.get_profiler_config()

        # Get callbacks
        callbacks = config_manager.get_callbacks(config, args.model_checkpoint)

        # Get TensorBoard logger
        tb_logger = config_manager.get_tensorboard_logger(trainer.results_collector.experiment_id)
        
        # Get accelerator config
        accelerator_config = config_manager.get_accelerator_config()

        # Create PyTorch Lightning trainer
        pl_trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=tb_logger,
            callbacks=callbacks if callbacks else None,
            enable_checkpointing=not args.no_checkpointing,
            enable_progress_bar=not args.no_progress_bar,
            fast_dev_run=args.fast_dev_run,
            gradient_clip_val=1.0,
            precision=16,
            **accelerator_config,  # Unpack accelerator configuration
            profiler=profiler,
            val_check_interval=args.val_check_interval
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
        pl_trainer.fit(trainer)

        # Log memory usage after training
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"CUDA memory allocated after training: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"CUDA memory reserved after training: {torch.cuda.memory_reserved()} bytes")

        logger.info("Model training completed successfully.")

        # Get data loaders
        _, _, test_loader = data_manager.create_data_loaders(train_data, val_data, test_data)

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

# Note: __main__ block removed as entry point is now train_cli.py