import pytest
import torch
import numpy as np
from src.data.arc_dataset import ARCDataset
from src.models.gpt2 import GPT2ARC
from src.training.trainer import ARCTrainer
from src.config import Config, ModelConfig, TrainingConfig
import pytorch_lightning as pl
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def real_data_sample():
    logger.debug("Creating real dataset sample")
    data = [
        {
            "train": [
                {
                    "input": np.array([
                        [0, 0, 0, 0, 0, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 8, 8, 8, 8, 0],
                        [0, 0, 0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [0, 0, 0, 0, 0, 0],
                        [0, 8, 8, 8, 8, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]
                    ])
                }
            ],
            "test": [
                {
                    "input": np.array([
                        [0, 0, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0, 0],
                        [0, 5, 5, 5, 5, 0],
                        [0, 5, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [0, 0, 0, 0, 0, 0],
                        [0, 5, 5, 5, 5, 0],
                        [0, 5, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]
                    ])
                }
            ]
        }
    ]
    logger.debug(f"Real dataset sample created with {len(data)} tasks")
    return data

def test_end_to_end(real_data_sample):
    logger.debug("Starting end-to-end test")

    # Create datasets
    logger.debug("Creating train and validation datasets")
    train_dataset = ARCDataset(real_data_sample)
    val_dataset = ARCDataset(real_data_sample, is_test=True)
    logger.debug(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Initialize model
    logger.debug("Initializing model")
    model_config = ModelConfig(n_embd=128, n_head=4, n_layer=2)
    model = GPT2ARC(model_config)
    logger.debug(f"Model initialized with config: {model_config}")

    # Initialize trainer
    logger.debug("Initializing trainer")
    config = Config(model=model_config, training=TrainingConfig(batch_size=2, learning_rate=1e-3, max_epochs=5))
    trainer = ARCTrainer(model, train_dataset, val_dataset, config)
    logger.debug(f"Trainer initialized with config: {config}")

    # Create PyTorch Lightning trainer
    logger.debug("Creating PyTorch Lightning trainer")
    pl_trainer = pl.Trainer(
        max_epochs=5,
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

    # Evaluate model
    logger.debug("Starting model evaluation")
    val_results = pl_trainer.test(trainer, verbose=False)
    logger.debug(f"Validation results: {val_results}")
    
    # Check that validation accuracy improved
    assert val_results[0]['test_accuracy'] > 0.5, f"Validation accuracy did not improve. Final accuracy: {val_results[0]['test_accuracy']}"

    logger.debug(f"Final training loss: {train_losses[-1]:.4f}")
    logger.debug(f"Validation accuracy: {val_results[0]['test_accuracy']:.4f}")

    logger.debug("End-to-end test completed successfully")
