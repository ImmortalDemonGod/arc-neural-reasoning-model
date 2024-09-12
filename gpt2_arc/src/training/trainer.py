# gpt2_arc/src/training/trainer.py
import pytorch_lightning as pl
import torch
import logging
from torch import nn, optim
import time
from typing import Any, Dict, Optional
from collections import deque
from torch.optim.lr_scheduler import LambdaLR
from ..config import Config
from ..utils.helpers import differential_pixel_accuracy
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ARCTrainer(pl.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, config: Config):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.batch_size = config.training.batch_size
        self.lr = config.training.learning_rate
        self.train_losses = []
        self.logged_metrics = {}

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        
        logger.debug(f"Training step - Batch type: {type(batch)}, length: {len(batch)}")
        logger.debug(f"Batch[0] shape: {batch[0].shape}, Batch[1] shape: {batch[1].shape}")

        if isinstance(batch, tuple):
            input_ids, attention_mask, labels = batch
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].long()
        elif isinstance(batch, list) and len(batch) == 2:
            input_ids, labels = batch
            attention_mask = None
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Content: {batch}")

        # Ensure tensors are float32
        input_ids = input_ids.to(torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.float32)
        labels = labels.long()  # Ensure labels are of type Long
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log("train_loss", loss)
        self.train_losses.append(loss.item())
        end_time = time.time()
        batch_time = end_time - start_time
        logger.info(f"Batch {batch_idx} training time: {batch_time:.4f} seconds")
        
        return loss

    def validation_step(self, batch, batch_idx):
        logger.debug(f"Validation step - Batch type: {type(batch)}, length: {len(batch)}")
        logger.debug(f"Batch[0] shape: {batch[0].shape}, Batch[1] shape: {batch[1].shape}")

        if isinstance(batch, tuple):
            logger.debug(f"Batch is a tuple with {len(batch)} elements")
            for i, item in enumerate(batch):
                logger.debug(f"Tuple element {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
            input_ids, attention_mask, labels = batch
        elif isinstance(batch, dict):
            logger.debug(f"Batch is a dictionary with keys: {batch.keys()}")
            for key, value in batch.items():
                logger.debug(f"Dict item '{key}': type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
        elif isinstance(batch, list) and len(batch) == 2:
            input_ids, labels = batch
            attention_mask = None
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Content: {batch}")

        if input_ids is None or labels is None:
            raise ValueError(f"Missing required batch components. Batch content: {batch}")

        # Ensure tensors are float32
        input_ids = input_ids.to(torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.float32)
        labels = labels.long()
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log("val_loss", loss)
        self.logged_metrics["val_loss"] = loss.item()

    def test_step(self, batch, batch_idx):
        logger.debug(f"Test step - Batch type: {type(batch)}, length: {len(batch)}")
        logger.debug(f"Batch[0] shape: {batch[0].shape}, Batch[1] shape: {batch[1].shape}")

        if isinstance(batch, tuple):
            input_ids, attention_mask, labels = batch
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].long()
        elif isinstance(batch, list) and len(batch) == 2:
            input_ids, labels = batch
            attention_mask = None
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Content: {batch}")

        # Ensure tensors are float32
        input_ids = input_ids.to(torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.float32)
        labels = labels.long()  # Ensure labels are of type Long
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        B, T, C = outputs.size()  # Define B and C
        outputs = outputs.view(B, -1, C)
        predictions = torch.argmax(outputs, dim=-1)
        labels = labels.view(B, -1)
        accuracy = (predictions == labels).float().mean()
        self.log('test_accuracy', accuracy)
        self.log('test_loss', loss)  # Log the test loss
        # Calculate differential pixel accuracy
        differential_accuracy, _, _ = differential_pixel_accuracy(input_ids, labels, predictions)

        # Log all metrics
        self.log('test_accuracy', accuracy)
        self.log('test_loss', loss)
        self.log('differential_pixel_accuracy', differential_accuracy)

        return {
            "test_accuracy": accuracy,
            "test_loss": loss,
            "differential_pixel_accuracy": differential_accuracy
        }

    def on_save_checkpoint(self, checkpoint):
        config_dict = {
            'n_embd': self.config.model.n_embd,
            'n_head': self.config.model.n_head,
            'n_layer': self.config.model.n_layer,
            'dropout': self.config.model.dropout
        }
        checkpoint['config'] = config_dict
        logger.debug(f"Model configuration added to checkpoint: {config_dict}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95),
            'name': 'learning_rate',
        }
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)

    def compute_loss(self, outputs, labels):
        return nn.CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)
