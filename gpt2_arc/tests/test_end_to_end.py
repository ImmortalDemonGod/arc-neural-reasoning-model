# gpt2_arc/tests/test_end_to_end.py
import pytest
import torch
import numpy as np
from src.data.arc_dataset import ARCDataset
from src.models.gpt2 import GPT2ARC
from src.training.trainer import ARCTrainer
from src.config import Config, ModelConfig, TrainingConfig
import pytorch_lightning as pl
import logging
import os
from pytest import approx

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def arc_data_path():
    # Adjust this path to the location of your ARC dataset JSON file
    return "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/syntheticARC/tasks/1c786137.json"

@pytest.mark.timeout(600)  # 10 minutes timeout
def test_end_to_end(arc_data_path):
    logger.debug("Starting end-to-end test")

    try:
        # Check if the ARC dataset file exists
        if not os.path.exists(arc_data_path):
            pytest.skip(f"ARC dataset file not found at {arc_data_path}")

        # Load and inspect the JSON file
        import json
        with open(arc_data_path, 'r') as f:
            raw_data = json.load(f)
        
        logger.debug(f"Type of data: {type(raw_data)}")
        if isinstance(raw_data, list):
            logger.debug(f"Number of items: {len(raw_data)}")
            if raw_data:
                logger.debug(f"Keys in the first item: {list(raw_data[0].keys()) if isinstance(raw_data[0], dict) else 'Not a dictionary'}")
        elif isinstance(raw_data, dict):
            logger.debug(f"Keys in the JSON file: {list(raw_data.keys())}")
        else:
            logger.debug(f"Unexpected data type: {type(raw_data)}")

        # Create datasets
        logger.debug("Creating train and validation datasets")
        full_dataset = ARCDataset(arc_data_path)
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        logger.debug(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

        # Initialize model
        logger.debug("Initializing model")
        model_config = ModelConfig(n_embd=128, n_head=4, n_layer=2)
        model = GPT2ARC(model_config)
        logger.debug(f"Model initialized with config: {model_config}")

        # Initialize trainer
        logger.debug("Initializing trainer")
        config = Config(model=model_config, training=TrainingConfig(batch_size=32, learning_rate=1e-4, max_epochs=10))
        trainer = ARCTrainer(model, train_dataset, val_dataset, config)
        logger.debug(f"Trainer initialized with config: {config}")

        # Create PyTorch Lightning trainer
        logger.debug("Creating PyTorch Lightning trainer")
        pl_trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )
        logger.debug("PyTorch Lightning trainer created")

        # Train model
        logger.debug("Starting model training")
        pl_trainer.fit(trainer)
        logger.debug("Model training completed")

        # Check that loss decreased
        train_losses = trainer.train_losses
        logger.debug(f"Training losses: {train_losses}")
        assert train_losses[-1] < train_losses[0], f"Training loss did not decrease. Initial loss: {train_losses[0]}, Final loss: {train_losses[-1]}"
        
        # Check that loss decreased consistently
        assert all(train_losses[i] > train_losses[i+1] for i in range(len(train_losses)-1)), "Training loss did not consistently decrease"

        # Evaluate model
        logger.debug("Starting model evaluation")
        val_results = pl_trainer.test(trainer, verbose=False)
        logger.debug(f"Validation results: {val_results}")
        
        # Check that validation accuracy improved
        assert val_results[0]['test_accuracy'] > 0.2, f"Validation accuracy did not improve. Final accuracy: {val_results[0]['test_accuracy']}"

        # Check that validation accuracy is within expected range
        assert 0.2 < val_results[0]['test_accuracy'] < 0.9, f"Validation accuracy {val_results[0]['test_accuracy']} is outside expected range (0.2, 0.9)"

        logger.debug(f"Final training loss: {train_losses[-1]:.4f}")
        logger.debug(f"Validation accuracy: {val_results[0]['test_accuracy']:.4f}")

        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params > 0, "Model has no parameters"
        assert trainable_params > 0, "Model has no trainable parameters"
        assert trainable_params == total_params, "Not all parameters are trainable"

        logger.debug(f"Total parameters: {total_params}")
        logger.debug(f"Trainable parameters: {trainable_params}")

        logger.debug("End-to-end test completed successfully")
    except Exception as e:
        logger.error(f"End-to-end test failed with error: {str(e)}")
        raise
