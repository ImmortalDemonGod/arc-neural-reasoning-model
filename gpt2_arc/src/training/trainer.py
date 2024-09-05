# gpt2_arc/src/training/trainer.py
import pytorch_lightning as pl
import torch
import logging
from torch import nn, optim
from typing import Any
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
        loss = self.compute_loss(outputs, labels)
        logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}: Training loss = {loss.item()}")
        if isinstance(batch, tuple):
            input_ids, attention_mask, labels = batch
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
        else:
            raise ValueError("Batch must be either a tuple or a dictionary")

        # Ensure tensors
        input_ids = (
            input_ids.long()
            if not isinstance(input_ids, torch.LongTensor)
            else input_ids
        )
        attention_mask = (
            attention_mask.float()
            if not isinstance(attention_mask, torch.FloatTensor)
            else attention_mask
        )
        labels = labels.long() if not isinstance(labels, torch.LongTensor) else labels
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log("train_loss", loss)
        self.train_losses.append(loss.item())
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
        if isinstance(batch, tuple):
            input_ids, attention_mask, labels = batch
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
        else:
            raise ValueError("Batch must be either a tuple or a dictionary")
        outputs = self(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=-1)
        accuracy = (predictions == labels).float().mean()
        self.log('test_accuracy', accuracy)
        return {"test_accuracy": accuracy}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def compute_loss(self, outputs, labels):
        return nn.CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)
