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
        
        print(f"Validation step - Batch type: {type(batch)}")
        print(f"Validation step - Batch content: {batch}")

        print(f"Validation step - Batch type: {type(batch)}")
        print(f"Validation step - Batch content: {batch}")

        if isinstance(batch, tuple):
            input_ids, attention_mask, labels = batch
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].long()
        elif isinstance(batch, list) and len(batch) == 1:
            input_ids = batch[0]
            attention_mask = None
            labels = None
            raise ValueError("Batch must be either a tuple or a dictionary")

        # Ensure tensors are float32
        input_ids = input_ids.to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log("train_loss", loss)
        self.train_losses.append(loss.item())
        end_time = time.time()
        batch_time = end_time - start_time
        logger.info(f"Batch {batch_idx} training time: {batch_time:.4f} seconds")
        
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            input_ids, attention_mask, labels = batch
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
        else:
            raise ValueError("Batch must be either a tuple or a dictionary")
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log("val_loss", loss)
        self.logged_metrics["val_loss"] = loss.item()

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        B, T, C = outputs.size()  # Define B and C
        outputs = outputs.view(B, -1, C)
        predictions = torch.argmax(outputs, dim=-1)
        labels = labels.view(B, -1)
        accuracy = (predictions == labels).float().mean()
        self.log('test_accuracy', accuracy)
        self.log('test_loss', loss)  # Log the test loss
        return {"test_accuracy": accuracy}

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
