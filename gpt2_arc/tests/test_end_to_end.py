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

import arckit

@pytest.mark.timeout(600)  # 10 minutes timeout
def test_end_to_end():
    logger.debug("Starting end-to-end test")

    try:
        # Load data using arckit
        logger.debug("Loading data using arckit")
        train_set, eval_set = arckit.load_data()
        
        # Create datasets using ARCDataset
        logger.debug("Creating train and validation datasets")
        full_dataset = ARCDataset(train_set, is_test=False)
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        logger.debug(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

        # Create a custom collate function to handle the data format
        def collate_fn(batch):
            inputs = [item[0].float() for item in batch]  # Convert to float
            outputs = [item[1].float() for item in batch]  # Convert to float
            logger.debug(f"Batch input dtypes before stack: {[item[0].dtype for item in batch]}")
            logger.debug(f"Batch output dtypes before stack: {[item[1].dtype for item in batch]}")

            # Inputs and outputs are already tensors, so we just need to stack them
            input_stack = torch.stack(inputs)
            output_stack = torch.stack(outputs)

            # Create a dummy attention mask (all ones)
            attention_mask = torch.ones(input_stack.size(0), input_stack.size(2) * input_stack.size(3))

            logger.debug(f"Collate function input dtype: {input_stack.dtype}")
            return input_stack, attention_mask, output_stack

        # Initialize model
        logger.debug("Initializing model")
        model_config = ModelConfig(n_embd=128, n_head=4, n_layer=2)
        model = GPT2ARC(model_config)
        logger.debug(f"Model initialized with config: {model_config}")

        # Initialize trainer
        logger.debug("Initializing trainer")
        config = Config(model=model_config, training=TrainingConfig(batch_size=32, learning_rate=1e-4, max_epochs=3))
        trainer = ARCTrainer(model, train_dataset, val_dataset, config)
        trainer.train_dataloader = lambda: torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, collate_fn=collate_fn)
        trainer.val_dataloader = lambda: torch.utils.data.DataLoader(val_dataset, batch_size=config.training.batch_size, collate_fn=collate_fn)
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
