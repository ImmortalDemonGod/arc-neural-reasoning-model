# gpt2_arc/tests/test_end_to_end.py
import pytest
import torch
import numpy as np
from src.data.arc_dataset import ARCDataset
from src.models.gpt2 import GPT2ARC
from src.training.trainer import ARCTrainer
from src.config import Config, ModelConfig, TrainingConfig
import pytorch_lightning as pl
import time
import logging
import os
from thop import profile, clever_format  # Import THOP
from pytest import approx

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def arc_data_path():
    # Adjust this path to the location of your ARC dataset JSON file
    return "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/syntheticARC/tasks/1c786137.json"

import arckit

def test_end_to_end():
    logger.debug("Starting end-to-end test")

    try:
        # Load data using arckit
        logger.debug("Loading data using arckit")
        train_set, eval_set = arckit.load_data()
        
        # Create datasets using ARCDataset
        logger.debug("Creating train and validation datasets")
        full_dataset = ARCDataset(train_set, is_test=False)
        # Use a smaller subset of the dataset
        subset_size = int(0.1 * len(full_dataset))  # Use 10% of the dataset
        train_dataset, _ = torch.utils.data.random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
        val_dataset, _ = torch.utils.data.random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
        logger.debug(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

        # Create a custom collate function to handle the data format
        def collate_fn(batch):
            inputs = [item[0].to(torch.float32) for item in batch]  # Convert to float32
            outputs = [item[1].to(torch.float32) for item in batch]  # Convert to float32
            logger.debug(f"Batch input dtypes before stack: {[item[0].dtype for item in batch]}")
            logger.debug(f"Batch output dtypes before stack: {[item[1].dtype for item in batch]}")

            # Inputs and outputs are already tensors, so we just need to stack them
            input_stack = torch.stack(inputs)
            output_stack = torch.stack(outputs)

            # Log data types after stacking
            logger.debug(f"Collate function input_stack dtype: {input_stack.dtype}")
            logger.debug(f"Collate function output_stack dtype: {output_stack.dtype}")

            # Create a dummy attention mask (all ones)
            attention_mask = torch.ones(input_stack.size(0), input_stack.size(2) * input_stack.size(3), dtype=torch.float32)

            logger.debug(f"Collate function attention_mask dtype: {attention_mask.dtype}")
            return input_stack, attention_mask, output_stack
            logger.debug(f"Batch output dtypes before stack: {[item[1].dtype for item in batch]}")

            # Inputs and outputs are already tensors, so we just need to stack them
            input_stack = torch.stack(inputs)
            output_stack = torch.stack(outputs)

            # Create a dummy attention mask (all ones)
            attention_mask = torch.ones(input_stack.size(0), input_stack.size(2) * input_stack.size(3), dtype=torch.float32)

            logger.debug(f"Collate function input dtype: {input_stack.dtype}")
            return input_stack, attention_mask, output_stack

        # Initialize model
        logger.debug("Initializing model")
        model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)  # Use smaller model configuration
        model = GPT2ARC(model_config).to(torch.float32)
        logger.debug(f"Model initialized with config: {model_config}")

        # # THOP Profiling - Commented out due to TypeError with MPS Tensors
        # logger.debug("Profiling model with THOP")
        # dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)  # Example input shape
        # macs, params = profile(model, inputs=(dummy_input,))
        # macs, params = clever_format([macs, params], "%.3f")
        # logger.info(f"MACs: {macs}, Parameters: {params}")

        # Initialize trainer
        logger.debug("Initializing trainer")
        config = Config(model=model_config, training=TrainingConfig(batch_size=32, learning_rate=1e-4, max_epochs=2))  # Reduce epochs to 2
        trainer = ARCTrainer(model, train_dataset, val_dataset, config)
        trainer.train_dataloader = lambda: torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, collate_fn=collate_fn)
        trainer.val_dataloader = lambda: torch.utils.data.DataLoader(val_dataset, batch_size=config.training.batch_size, collate_fn=collate_fn)
        trainer.test_dataloader = lambda: torch.utils.data.DataLoader(val_dataset, batch_size=config.training.batch_size, collate_fn=collate_fn)
        logger.debug(f"Trainer initialized with config: {config}")

        # Create PyTorch Lightning trainer
        logger.debug("Creating PyTorch Lightning trainer")
        # Measure training time
        start_time = time.time()
        
        pl_trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )
        logger.debug("PyTorch Lightning trainer created")

        # Evaluate model before training to get initial accuracy
        logger.info("Evaluating model before training")
        initial_val_results = pl_trainer.test(trainer, verbose=False)
        initial_accuracy = initial_val_results[0]['test_accuracy']
        initial_loss = initial_val_results[0]['test_loss']
        logger.info(f"Initial validation accuracy: {initial_accuracy}, Initial loss: {initial_loss}")
        print(f"Initial validation accuracy: {initial_accuracy}, Initial loss: {initial_loss}")
        logger.debug("Starting model training")
        pl_trainer.fit(trainer)
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Total training time: {training_time:.2f} seconds")
        logger.debug("Model training completed")

        # Check that loss decreased
        train_losses = trainer.train_losses
        logger.info(f"Training losses: {train_losses}")
        assert train_losses[-1] < train_losses[0], f"Training loss did not decrease. Initial loss: {train_losses[0]}, Final loss: {train_losses[-1]}"
        
        # Check that the final loss is lower than the initial loss
        assert train_losses[-1] < train_losses[0], "Final training loss should be lower than initial loss"

        # Check that the average loss per epoch decreases
        epoch_losses = [sum(train_losses[i:i+33])/33 for i in range(0, len(train_losses), 33)]
        assert all(epoch_losses[i] > epoch_losses[i+1] for i in range(len(epoch_losses)-1)), "Average training loss per epoch did not consistently decrease"

        # Evaluate model after training
        logger.debug("Evaluating model after training")
        final_val_results = pl_trainer.test(trainer, verbose=False)
        final_accuracy = final_val_results[0]['test_accuracy']
        final_loss = final_val_results[0]['test_loss']
        logger.info(f"Final validation accuracy: {final_accuracy}, Final loss: {final_loss}")
        print(f"Final validation accuracy: {final_accuracy}, Final loss: {final_loss}")

        # Check that validation accuracy improved
        assert final_accuracy > initial_accuracy, f"Validation accuracy did not improve. Initial accuracy: {initial_accuracy}, Final accuracy: {final_accuracy}"

        logger.info(f"Final training loss: {train_losses[-1]:.4f}")
        logger.info(f"Validation accuracy: {final_accuracy:.4f}")

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
