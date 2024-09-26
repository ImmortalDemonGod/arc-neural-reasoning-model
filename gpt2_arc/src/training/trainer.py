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
from ..utils.results_collector import ResultsCollector
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
import os

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
        self.test_results = []  # Initialize test results for storing test outcomes
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.results_collector = ResultsCollector(config)
        self.writer = SummaryWriter(f"runs/experiment_{self.results_collector.experiment_id}")

    def get_tensorboard_logger(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                return logger.experiment
        print("DEBUG: No TensorBoardLogger found in trainer.loggers")
        return None

    def training_step(self, batch, batch_idx):
        logger.debug(f"Training step - Batch type: {type(batch)}, length: {len(batch)}")
        
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, labels = batch[:2]
            task_ids = batch[2] if len(batch) > 2 else None
        elif isinstance(batch, dict):
            inputs = batch.get("input_ids")
            labels = batch.get("labels")
            task_ids = batch.get("task_ids")
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Content: {batch}")

        # Ensure inputs and labels are the correct type
        inputs = inputs.float()
        labels = labels.long()

        outputs = self(inputs)
        loss = self.compute_loss(outputs, labels)
        
        if hasattr(self, 'log'):
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.results_collector.update_train_metrics(self.current_epoch, {"loss": loss.item()})
        
        tb_logger = self.get_tensorboard_logger()
        if tb_logger:
            tb_logger.add_scalar('train/loss', loss.item(), self.global_step)
            print(f"DEBUG: Logged training loss: {loss.item()} at step {self.global_step}")
        else:
            print(f"DEBUG: Failed to log training loss. No TensorBoard logger available.")
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logger.debug(f"Validation step - Batch type: {type(batch)}, length: {len(batch)}")
        
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                logger.error(f"Missing inputs or labels in batch. Inputs: {batch[0] if len(batch) > 0 else None}, Labels: {batch[1] if len(batch) > 1 else None}")
                raise ValueError("Batch must contain inputs and labels.")
            inputs, labels = batch[:2]
            task_ids = batch[2] if len(batch) > 2 else None
        elif isinstance(batch, dict):
            inputs = batch.get("input_ids")
            labels = batch.get("labels")
            task_ids = batch.get("task_ids")
            if inputs is None or labels is None:
                logger.error(f"Missing inputs or labels in batch. Inputs: {inputs}, Labels: {labels}")
                raise ValueError("Batch must contain inputs and labels.")
        else:
            logger.error(f"Unexpected batch format: {type(batch)}. Content: {batch}")
            raise ValueError(f"Unexpected batch format: {type(batch)}. Content: {batch}")

        # Ensure inputs and labels are the correct type
        inputs = inputs.float()
        labels = labels.long()

        outputs = self(inputs)
        loss = self.compute_loss(outputs, labels)
        
        if hasattr(self, 'log'):
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.logged_metrics["val_loss"] = loss.item()
        self.results_collector.update_val_metrics(self.current_epoch, {"loss": loss.item()})
        
        try:
            self.writer.add_scalar('val/loss', loss.item(), self.current_epoch)
            print(f"DEBUG: Logged validation loss: {loss.item()} for epoch {self.current_epoch}")
        except Exception as e:
            print(f"DEBUG: Error logging validation step: {str(e)}")
        
        return loss

    def test_step(self, batch, batch_idx):
        """
        The `test_step` function processes a batch of data for testing a model, computes metrics such as
        loss and accuracy, logs the results, and stores them for further analysis.
        
        :param batch: The `batch` parameter in the `test_step` function is expected to contain input data,
        attention masks (optional), target outputs, and task IDs. The function processes the batch based on
        its format and performs inference using a model. It calculates loss, accuracy, and differential
        pixel accuracy metrics for evaluation
        :param batch_idx: Batch index is used to keep track of the current batch being processed during
        testing. It helps in identifying and logging information specific to each batch, such as loss and
        accuracy values. The batch index is typically an integer value that increments for each batch
        processed during testing
        :return: The `test_step` method returns a dictionary named `result` containing the keys 'loss',
        'accuracy', 'task_ids', and 'test_loss'. Additionally, it logs various metrics such as test_loss,
        test_accuracy, and test_diff_accuracy. The method also appends the result to `self.test_outputs` and
        calculates task success for TSR, logging it as well.
        """
        logger.debug(f"DEBUG: Test step - Batch type: {type(batch)}, length: {len(batch)}")

        attention_mask = None
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            inputs, outputs, task_ids = batch[:3]
            if len(batch) == 4:
                attention_mask = batch[3]
        elif isinstance(batch, dict):
            inputs = batch.get('input_ids')
            outputs = batch.get('labels')
            task_ids = batch.get('task_ids')
            attention_mask = batch.get('attention_mask')
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}. Content: {batch}")

        logger.debug(f"DEBUG: Task IDs in batch: {task_ids}")

        inputs = inputs.float()
        outputs = outputs.long()

        if attention_mask is None:
            attention_mask = torch.ones(inputs.size(0), inputs.size(2) * inputs.size(3), dtype=torch.float32, device=inputs.device)

        model_outputs = self(inputs, attention_mask)
        loss = self.compute_loss(model_outputs, outputs)

        accuracies = self.compute_accuracy(model_outputs, outputs)
        diff_accuracies = self.compute_diff_accuracy(inputs, outputs, model_outputs)

        logger.debug(f"DEBUG: Computed accuracies: {accuracies}")
        logger.debug(f"DEBUG: Computed diff accuracies: {diff_accuracies}")

        result = {
            'test_loss': loss.item(),
            'test_accuracy': accuracies.tolist(),
            'test_diff_accuracy': diff_accuracies.tolist(),
            'task_ids': task_ids,
        }

        for task_id, accuracy, diff_accuracy in zip(task_ids, accuracies, diff_accuracies):
            self.log(f"{task_id}_test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{task_id}_test_diff_accuracy", diff_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        try:
            self.writer.add_scalar('test/loss', result['test_loss'], self.current_epoch)
            self.writer.add_scalar('test/avg_accuracy', sum(result['test_accuracy']) / len(result['test_accuracy']), self.current_epoch)
            self.writer.add_scalar('test/diff_accuracy', result['test_diff_accuracy'], self.current_epoch)
            logger.debug(f"DEBUG: Logged test metrics for epoch {self.current_epoch}: loss={result['test_loss']}, avg_accuracy={sum(result['test_accuracy']) / len(result['test_accuracy'])}, diff_accuracy={result['test_diff_accuracy']}")
        except Exception as e:
            logger.error(f"DEBUG: Error logging test step: {str(e)}")

        logger.debug(f"DEBUG: Test step result: {result}")

        return result

    def compute_diff_accuracy(self, inputs, targets, outputs):
        predictions = outputs.argmax(dim=-1)
        diff_accuracies, _, _ = differential_pixel_accuracy(inputs, targets, predictions)
        return diff_accuracies
        predictions = outputs.argmax(dim=-1)
        # Reshape predictions to match the target shape
        predictions = predictions.view(targets.size())
        # Calculate accuracy for each sample in the batch
        accuracies = (predictions == targets).float().mean(dim=[1, 2, 3])
        return accuracies
        
    def on_validation_epoch_end(self):
        # Compute average validation loss
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            avg_val_loss = val_loss.item()
        else:
            avg_val_loss = float('inf')  # Default to a high value if val_loss is not available

        # Update best_val_loss and best_epoch
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_epoch = self.current_epoch

        # Log validation metrics
        self.log('val_loss', avg_val_loss)
        self.log('best_val_loss', self.best_val_loss)
        self.log('best_epoch', self.best_epoch)

        # Update the results collector
        self.results_collector.update_val_metrics(self.current_epoch, {
            "avg_loss": avg_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch
        })

        # Log additional information
        self.log('epoch', self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95),
            'name': 'learning_rate',
        }
        return [optimizer], [lr_scheduler]

    def on_fit_end(self):
        self.results_collector.save_to_json(f"results/experiment_{self.results_collector.experiment_id}.json")
        try:
            self.writer.close()
            print("DEBUG: TensorBoard writer closed successfully.")
        except Exception as e:
            print(f"DEBUG: Error closing TensorBoard writer: {str(e)}")
        print("DEBUG: Results saved and TensorBoard writer closed.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
        logger.debug(f"DEBUG: Test dataloader created with {len(loader)} batches")
        return loader

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def compute_loss(self, outputs, labels):
        return nn.CrossEntropyLoss()(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)
    def log_hyperparameters(self):
        hparams = {
            'learning_rate': self.config.training.learning_rate,
            'batch_size': self.config.training.batch_size,
            'n_embd': self.config.model.n_embd,
            'n_head': self.config.model.n_head,
            'n_layer': self.config.model.n_layer,
        }
        metric_dict = {
            'train_loss': 0,
            'val_loss': 0,
            'test_accuracy': 0,
        }
        try:
            self.writer.add_hparams(hparams, metric_dict)
            print(f"DEBUG: Successfully logged hyperparameters: {hparams}")
        except Exception as e:
            print(f"DEBUG: Error logging hyperparameters: {str(e)}")
