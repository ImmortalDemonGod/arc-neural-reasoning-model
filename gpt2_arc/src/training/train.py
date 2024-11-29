# gpt2_arc/src/training/train.py
import logging
import sys
import os
from gpt2_arc.src.training.utils.training_config_manager import ConfigurationManager
from gpt2_arc.src.training.utils.data_manager import DataManager
from gpt2_arc.src.training.utils.training_manager import TrainingManager

logger = logging.getLogger(__name__)

def main(args) -> None:
    """
    Main training function with enhanced logging.
    
    Args:
        args: Arguments from train_cli.py with all training configuration options
    """
    # Initialize configuration
    config_manager = ConfigurationManager(args)
    config_manager.setup_logging()
    config = config_manager.create_initial_config()
    config_manager.validate_configuration(config)

    logger.debug(f"Command line arguments: {args}")
    logger.info("Starting main function")
    logger.info("=== Training Pipeline Start ===")

    try:
        logger.info("Step 1: Configuration and Data Loading")
        logger.debug(f"Configuration: {config}")

        # Initialize DataManager and load datasets
        logger.info("Initializing DataManager...")
        data_manager = DataManager(config, args)
        logger.info("Loading datasets...")
        train_data, val_data, test_data = data_manager.load_all_datasets()
        logger.info(f"Datasets loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} samples")
        
        # Update symbol frequencies if enabled
        if args.enable_symbol_freq:
            logger.info("Updating symbol frequencies...")
            symbol_freq = train_data.get_symbol_frequencies()
            symbol_freq_dict = {i: float(freq) for i, freq in enumerate(symbol_freq)}
            config = config_manager.update_config_with_symbol_freq(config, symbol_freq_dict)
            logger.info("Symbol frequencies updated")

        logger.info("Step 2: Model Initialization")
        logger.info("Initializing TrainingManager...")
        training_manager = TrainingManager(config, args)
        
        logger.info("Initializing model...")
        model = training_manager.initialize_model()
        if args.model_checkpoint:
            logger.info(f"Loading checkpoint from: {args.model_checkpoint}")
        model = training_manager.load_checkpoint(model)
        logger.info("Model initialization complete")

        logger.info("Step 3: Training Setup")
        logger.info("Setting up training components...")
        
        # Get training components from configuration manager
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
            **accelerator_config,
        }
        logger.debug(f"Trainer config created: {trainer_config}")

        # Setup and execute training
        logger.info("Step 4: Starting Training")
        training_manager.setup_training(model, train_data, val_data, test_data, trainer_config)
        training_manager.train_model()
        logger.info("Training completed")

        logger.info("Step 5: Evaluation")
        _, _, test_loader = data_manager.create_data_loaders(train_data, val_data, test_data)
        test_metrics = training_manager.test_model(test_loader)
        logger.info(f"Test Metrics: {test_metrics}")

        logger.info("Step 6: Saving Results")
        model_path = training_manager.save_model()
        results_path = training_manager.save_results()
        
        logger.info("=== Training Pipeline Complete ===")
        logger.debug(f"Model saved to: {model_path}")
        logger.debug(f"Results saved to: {results_path}")

    except Exception as e:
        logger.error("Training failed", exc_info=True)
        raise
    finally:
        if 'training_manager' in locals():
            logger.info("Cleaning up...")
            training_manager.cleanup()
            logger.info("Cleanup complete")

# Note: __main__ block removed as entry point is now train_cli.py