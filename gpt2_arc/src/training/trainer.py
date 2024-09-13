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
        self.test_outputs = []  # Store test outputs for aggregation

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        
        logger.debug(f"Training step - Batch type: {type(batch)}, length: {len(batch)}")
        if isinstance(batch, dict):
            logger.debug(f"Batch input_ids shape: {batch['input_ids'].shape}, Batch labels shape: {batch['labels'].shape}")
        else:
            logger.debug(f"Batch[0] shape: {batch[0].shape}, Batch[1] shape: {batch[1].shape}")

        if isinstance(batch, tuple) and len(batch) == 3:
            input_ids, attention_mask, labels = batch
            task_ids = None  # We don't have task_ids in this case
        elif isinstance(batch, tuple) and len(batch) == 4:
            input_ids, attention_mask, labels, task_ids = batch
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            task_ids = batch.get("task_ids")
        else:
            logger.error(f"Unexpected batch format: {type(batch)}. Content: {batch}")
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
        logger.debug(f"Batch content: {batch}")
        
        if isinstance(batch, dict):
            logger.debug(f"Batch['input_ids'] shape: {batch['input_ids'].shape}, Batch['labels'] shape: {batch['labels'].shape}")
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            task_ids = batch.get("task_ids")
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            logger.debug("Processing list/tuple batch with 3 elements")
            input_ids, attention_mask, labels = batch
            task_ids = None
        elif isinstance(batch, (list, tuple)) and len(batch) == 4:
            logger.debug("Processing list/tuple batch with 4 elements")
            input_ids, attention_mask, labels, task_ids = batch
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Content: {batch}")

        # Ensure tensors are float32
        input_ids = input_ids.to(torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.float32)
        if isinstance(labels, torch.Tensor):
            labels = labels.long()
        
        outputs = self(input_ids, attention_mask)
        loss = self.compute_loss(outputs, labels)
        self.log("val_loss", loss)
        self.logged_metrics["val_loss"] = loss.item()

    def test_step(self, batch, batch_index):
        logger.debug(f"Test step - Batch type: {type(batch)}, length: {len(batch)}")

        if isinstance(batch, list) and len(batch) == 3:
            inputs, outputs, task_ids = batch
            attention_mask = None
        elif isinstance(batch, tuple) and len(batch) == 3:
            inputs, outputs, task_ids = batch
            attention_mask = None
        elif isinstance(batch, tuple) and len(batch) == 4:
            inputs, attention_mask, outputs, task_ids = batch
            logger.debug(f"Task IDs in batch: {task_ids}")
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Content: {batch}")

        # Ensure inputs and outputs are the correct type
        inputs = inputs.float()
        outputs = outputs.long()

        # Create a dummy attention mask if it's None
        if attention_mask is None:
            attention_mask = torch.ones(inputs.size(0), inputs.size(2) * inputs.size(3), dtype=torch.float32, device=inputs.device)

        model_outputs = self(inputs, attention_mask)
        loss = self.compute_loss(model_outputs, outputs)
        
        B, T, C = model_outputs.size()
        model_outputs = model_outputs.view(B, -1, C)
        predictions = torch.argmax(model_outputs, dim=-1)
        outputs = outputs.view(B, -1)
        
        accuracy = (predictions == outputs).float().mean()
        diff_accuracy, _, _ = differential_pixel_accuracy(inputs, outputs, predictions)

        # Log overall metrics
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        self.log('test_diff_accuracy', diff_accuracy)

        # Log task-specific metrics
        for i, task_id in enumerate(task_ids):
            self.log(f'{task_id}_test_loss', loss[i] if loss.dim() > 0 else loss)
            self.log(f'{task_id}_test_accuracy', accuracy[i] if accuracy.dim() > 0 else accuracy)
            self.log(f'{task_id}_test_diff_accuracy', diff_accuracy[i] if isinstance(diff_accuracy, torch.Tensor) and diff_accuracy.dim() > 0 else diff_accuracy)

        logger.debug(f"Batch {batch_index} - Task IDs: {task_ids}")
        logger.debug(f"Batch {batch_index} - Loss: {loss.item()}, Accuracy: {accuracy.item()}")

        result = {
            'loss': loss,
            'accuracy': accuracy,
            'task_ids': task_ids
        }

        self.test_outputs.append(result)  # Store the result
        return result

    def on_test_epoch_end(self):
        all_task_ids = []
        all_losses = []
        all_accuracies = []

        logger.debug("Aggregating test results from test_step outputs.")

        for output in self.test_outputs:
            batch_task_ids = output['task_ids']
            if isinstance(batch_task_ids, torch.Tensor):
                batch_task_ids = batch_task_ids.tolist()

            all_task_ids.extend(batch_task_ids)
            all_losses.extend([output['loss'].item()] * len(batch_task_ids))
            all_accuracies.extend([output['accuracy'].item()] * len(batch_task_ids))

            logger.debug(f"Batch task IDs: {batch_task_ids}")
            logger.debug(f"Batch losses: {output['loss'].item()}, accuracies: {output['accuracy'].item()}")

        self.test_results = []
        for task_id, loss, accuracy in zip(all_task_ids, all_losses, all_accuracies):
            self.test_results.append({
                'task_id': task_id,
                'test_loss': loss,
                'test_accuracy': accuracy
            })

        logger.debug(f"Aggregated test results: {self.test_results}")
        self.test_outputs.clear()  # Clear outputs after processing

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
