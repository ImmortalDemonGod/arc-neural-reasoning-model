# gpt2_arc/src/training/train.py
import logging
import sys
import os
import datetime
import torch
from training_config_manager import ConfigurationManager
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from gpt2_arc.src.config import Config
from gpt2_arc.src.training.data_manager import DataManager
from gpt2_arc.src.training.training_manager import TrainingManager

logger = logging.getLogger(__name__)

class ConfigSavingModelCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint that saves configuration with the model."""
    
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

    try:
        logger.debug(f"Configuration: {config}")

        # Initialize DataManager and load datasets
        data_manager = DataManager(config, args)
        train_data, val_data, test_data = data_manager.load_all_datasets()
        
        # Update symbol frequencies if enabled
        if args.enable_symbol_freq:
            symbol_freq = train_data.get_symbol_frequencies()
            symbol_freq_dict = {i: float(freq) for i, freq in enumerate(symbol_freq)}
            config = config_manager.update_config_with_symbol_freq(config, symbol_freq_dict)

        # Initialize TrainingManager
        training_manager = TrainingManager(config, args)
        
        # Initialize and load model
        model = training_manager.initialize_model()
        model = training_manager.load_checkpoint(model)

        # Get training components from config_manager
        profiler = config_manager.get_profiler_config()
        callbacks = config_manager.get_callbacks(config, args.model_checkpoint)
        tb_logger = config_manager.get_tensorboard_logger(training_manager.results_collector.experiment_id)
        accelerator_config = config_manager.get_accelerator_config()

        # Create trainer configuration
        trainer_config = {
            'max_epochs': config.training.max_epochs,
            'logger': tb_logger,
            'callbacks': callbacks if callbacks else None,
            'enable_checkpointing': not args.no_checkpointing,
            'enable_progress_bar': not args.no_progress_bar,
            'fast_dev_run': args.fast_dev_run,
            'gradient_clip_val': 1.0,
            'precision': 16,
            'profiler': profiler,
            'val_check_interval': args.val_check_interval,
            **accelerator_config,  # Unpack accelerator configuration
        }

        # Set up training components
        training_manager.setup_training(model, train_data, val_data, test_data, trainer_config)

        # Log GPU memory usage if applicable
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved()} bytes")

        # Execute training
        training_manager.train_model()

        # Log final GPU memory usage if applicable
        if args.use_gpu and torch.cuda.is_available():
            logger.info(f"CUDA memory allocated after training: {torch.cuda.memory_allocated()} bytes")
            logger.info(f"CUDA memory reserved after training: {torch.cuda.memory_reserved()} bytes")

        # Create test data loader and run evaluation
        _, _, test_loader = data_manager.create_data_loaders(train_data, val_data, test_data)
        test_metrics = training_manager.test_model(test_loader)

        logger.info(f"Test Metrics: {test_metrics}\n")

        # Save model and results
        model_path = training_manager.save_model()
        results_path = training_manager.save_results()
        
        logger.info("Training completed successfully")
        logger.debug(f"Model saved to: {model_path}")
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
        if 'training_manager' in locals():
            training_manager.cleanup()

# Note: __main__ block removed as entry point is now train_cli.py