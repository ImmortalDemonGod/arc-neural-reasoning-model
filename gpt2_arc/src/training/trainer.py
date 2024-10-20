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
from gpt2_arc.src.utils.training_helpers import get_num_workers
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy
from ..utils.results_collector import ResultsCollector
from torch.utils.data import DataLoader
from gpt2_arc.src.data.arc_dataset import ARCDataset

logger = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
from gpt2_arc.src.utils.training_helpers import get_num_workers
import os
from optuna.exceptions import TrialPruned
from pytorch_lightning.callbacks import Callback
import torch

logger = logging.getLogger(__name__)

class NanLossPruningCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Extract loss from outputs
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if loss is not None:
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected at epoch {trainer.current_epoch}, batch {batch_idx}: {loss.item()}")
                raise TrialPruned("Invalid loss encountered, pruning this trial.")


class ARCTrainer(pl.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, config: Config, args, results_collector=None, test_dataset=None):
        logger.debug("Initializing ARCTrainer")
        super().__init__()
        logger.debug(f"ARCTrainer received args.accelerator: {args.accelerator}")
        self.model = model
        # Determine the device type based on the model's parameters
        device = next(self.model.parameters()).device
        logger.debug(f"ARCTrainer initialization on device: {device}")
        if not args.fast_dev_run and device.type != "cpu":
            logger.info("Compiling the model with torch.compile for improved performance.")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        else:
            logger.info("torch.compile not applied (fast_dev_run=True or using CPU).")
        logger.debug(f"Model is on device: {device}")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.test_dataset = test_dataset  # Add this line
        self.batch_size = config.training.batch_size
        self.lr = config.training.learning_rate
        self.train_losses = []
        self.logged_metrics = {}
        self.test_outputs = []  # Initialize an empty list to store test outputs
        self.test_results = []  # Initialize test results for storing test outcomes
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.results_collector = results_collector if results_collector else ResultsCollector(config)
        self.writer = SummaryWriter(f"runs/experiment_{self.results_collector.experiment_id}")
        self.args = args  # Add this line to assign args
    
    
    def train_dataloader(self):
        logger.info("Creating training DataLoader with centralized num_workers")

        if self.config.training.balance_symbols:
            if self.config.training.balancing_method == "weighting":
                # Compute class weights (inverse of frequencies)
                class_weights = 1.0 / torch.tensor(
                    list(self.config.training.symbol_freq.values()), dtype=torch.float
                )

                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.training.batch_size,
                    num_workers=get_num_workers(self.config.training),
                    sampler=self.train_dataset.sampler,  # Use sampler instead of shuffle
                    pin_memory=True if self.args.use_gpu else False,
                    prefetch_factor=self.config.training.prefetch_factor,
                    persistent_workers=self.config.training.persistent_workers,
                    collate_fn=self.train_dataset.dataset.collate_fn if isinstance(self.train_dataset, torch.utils.data.Subset) else ARCDataset.collate_fn
                )
            elif self.config.training.balancing_method == "oversampling":
                # Placeholder for oversampling implementation
                logger.info("Oversampling method selected, but not yet implemented.")

                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.training.batch_size,
                    num_workers=get_num_workers(self.config.training),
                    shuffle=True,  # Enable shuffle if not using a sampler
                    pin_memory=True if self.args.use_gpu else False,
                    prefetch_factor=self.config.training.prefetch_factor,
                    persistent_workers=self.config.training.persistent_workers,
                    collate_fn=self.train_dataset.dataset.collate_fn if isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset.collate_fn
                )
            else:
                logger.warning(f"Unknown balancing method: {self.config.training.balancing_method}. Skipping balancing.")

                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.training.batch_size,
                    num_workers=get_num_workers(self.config.training),
                    shuffle=True,  # Enable shuffle
                    pin_memory=True if self.args.use_gpu else False,
                    prefetch_factor=self.config.training.prefetch_factor,
                    persistent_workers=self.config.training.persistent_workers,
                    collate_fn=self.train_dataset.dataset.collate_fn if isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset.collate_fn
                )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.training.batch_size,
                num_workers=get_num_workers(self.config.training),
                shuffle=True,  # Enable shuffle
                pin_memory=self.config.training.pin_memory,
                prefetch_factor=self.config.training.prefetch_factor,
                persistent_workers=self.config.training.persistent_workers,
                collate_fn=self.train_dataset.dataset.collate_fn if isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset.collate_fn
            )

        logger.debug(f"Training DataLoader created with num_workers={get_num_workers(self.config.training)}")
        return train_loader

    def val_dataloader(self):
        logger.debug("Entering ARCTrainer.val_dataloader")
        collate_fn = self.val_dataset.dataset.collate_fn if isinstance(self.val_dataset, torch.utils.data.Subset) else ARCDataset.collate_fn
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,  # Updated num_workers
            pin_memory=self.config.training.pin_memory,    # Updated pin_memory
            prefetch_factor=self.config.training.prefetch_factor,
            persistent_workers=self.config.training.persistent_workers,
            collate_fn=collate_fn
        )
        logger.debug("Exiting ARCTrainer.val_dataloader")
        return dataloader

    def test_dataloader(self):
        if self.test_dataset is None:
            logger.error("Test dataset is not provided. Please ensure that the test dataset is correctly loaded.")
            raise ValueError("Test dataset is not provided.")
        collate_fn = self.test_dataset.dataset.collate_fn if isinstance(self.test_dataset, torch.utils.data.Subset) else ARCDataset.collate_fn
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=get_num_workers(self.config.training),
            shuffle=False,
            pin_memory=self.config.training.pin_memory,
            prefetch_factor=self.config.training.prefetch_factor,
            persistent_workers=self.config.training.persistent_workers,
            collate_fn=collate_fn
        )

    
    def get_tensorboard_logger(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                return logger.experiment
        logger.debug("DEBUG: No TensorBoardLogger found in trainer.loggers")
        return None

    def training_step(self, batch, batch_idx):
        logger.debug(f"Starting training step {batch_idx}")
        inputs, targets, _ = batch
        logger.debug(f"  Inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
        logger.debug(f"  Targets shape: {targets.shape}, dtype: {targets.dtype}")
        
        outputs = self(inputs)
        logger.debug(f"  Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
        
        loss = self.compute_loss(outputs, targets)
        logger.debug(f"Loss computed: {loss.item()}")
        
        preds = torch.argmax(outputs, dim=-1)
        logger.debug(f"  Preds shape: {preds.shape}, dtype: {preds.dtype}")
        
        # Reshape preds to match targets if necessary
        try:
            preds = preds.view(targets.shape)
            logger.debug(f"  Reshaped preds to match targets shape: {preds.shape}")
        except RuntimeError as e:
            logger.error(f"  Failed to reshape preds: {e}")
            raise e
        
        accuracy = (preds == targets).float().mean()
        logger.debug(f"  Training accuracy: {accuracy.item()}")

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        logger.debug(f"Starting validation step {batch_idx}")
        inputs, targets, _ = batch
        logger.debug(f"Validation Inputs shape: {inputs.shape}, Targets shape: {targets.shape}, Targets dtype: {targets.dtype}")
        logger.debug(f"Validation batch input shape: {batch[0].shape}, Validation batch target shape: {batch[1].shape}")
        targets = targets.long()  # Ensure targets are of type Long
        logger.debug(f"Targets dtype after casting: {targets.dtype}")
        
        outputs = self(inputs)
        logger.debug(f"Validation Outputs shape: {outputs.shape}")
        
        loss = self.compute_loss(outputs, targets)
        logger.debug(f"Validation loss computed: {loss.item()}")
        
        preds = torch.argmax(outputs, dim=-1)
        logger.debug(f"  Preds shape before reshape: {preds.shape}, dtype: {preds.dtype}")

        # Reshape preds to match targets shape using view_as for safety
        try:
            preds = preds.view_as(targets)
            logger.debug(f"  Reshaped preds to match targets shape: {preds.shape}, dtype: {preds.dtype}")
        except RuntimeError as e:
            logger.error(f"  Failed to reshape preds: {e}")
            raise e

        accuracy = (preds == targets).float().mean()
        logger.debug(f"  Validation accuracy: {accuracy.item()}")

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_test_epoch_start(self):
        self.test_outputs = []  # Clear previous test outputs

    def test_step(self, batch, batch_idx):
        logger.debug(f"DEBUG: test_step input - batch: {batch}, batch_idx: {batch_idx}")
        logger.debug(f"Test batch input shape: {batch[0].shape}, Test batch target shape: {batch[1].shape}")
        logger.debug(f"DEBUG: Test step - Batch type: {type(batch)}, length: {len(batch)}")

        # Unpack batch
        if len(batch) == 3:
            inputs, outputs, task_ids = batch
        elif len(batch) == 2:
            inputs, outputs = batch
            logger.error("Batch does not contain 'task_ids'. Ensure the dataset provides 'task_ids'.")
            raise ValueError("Batch does not contain 'task_ids'. Ensure the dataset provides 'task_ids'.")
        else:
            raise ValueError(f"Unexpected batch format with length {len(batch)}")
        logger.debug(f"Received task_ids: {task_ids}")

        inputs = inputs.float()
        outputs = outputs.long()  # Ensure outputs are of type Long

        attention_mask = torch.ones(inputs.size(0), inputs.size(2) * inputs.size(3), dtype=torch.float32, device=inputs.device)

        model_outputs = self(inputs, attention_mask)
        loss = self.compute_loss(model_outputs, outputs)

        accuracies = []
        diff_accuracies = []
        
        # Compute batch-wise accuracy
        accuracy = self.compute_accuracy(model_outputs, outputs)
        diff_accuracy = self.compute_diff_accuracy(inputs, outputs, model_outputs)

        # Append batch metrics
        accuracies.append(accuracy)
        diff_accuracies.append(diff_accuracy)

        logger.debug(f"DEBUG: Batch accuracy: {accuracy}, Batch diff_accuracy: {diff_accuracy}")

        result = {
            'test_loss': loss.item(),
            'task_ids': task_ids,
            'test_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'test_diff_accuracy': sum(diff_accuracies) / len(diff_accuracies) if diff_accuracies else 0,
        }
        logger.debug(f"DEBUG: Test loss: {result['test_loss']}, Avg accuracy: {result['test_accuracy']}, Avg diff accuracy: {result['test_diff_accuracy']}")

        # Log task-specific metrics if task_ids are available
        if task_ids is not None:
            # Aggregate task-specific metrics across the batch
            for task_id in set(task_ids):  # Remove .tolist()
                if task_id == "default_task":
                    logger.error(f"'default_task' detected in task_id: {task_id}")
                    raise ValueError(f"'default_task' detected in task_id: {task_id}")

        try:
            self.writer.add_scalar('test/loss', result['test_loss'], self.current_epoch)
            self.writer.add_scalar('test/avg_accuracy', result['test_accuracy'], self.current_epoch)
            self.writer.add_scalar('test/diff_accuracy', result['test_diff_accuracy'], self.current_epoch)
            logger.debug(f"DEBUG: Logged test metrics for epoch {self.current_epoch}: loss={result['test_loss']}, avg_accuracy={result['test_accuracy']}, diff_accuracy={result['test_diff_accuracy']}")
        except Exception as e:
            logger.error(f"DEBUG: Error logging test step: {str(e)}")

        logger.debug(f"DEBUG: test_step output - result: {result}")
        logger.debug(f"DEBUG: Test step result: {result}")

        # Add per-task metrics to ResultsCollector
        for i, task_id in enumerate(task_ids):
            task_accuracy = self.compute_accuracy(model_outputs[i], outputs[i])
            task_diff_accuracy = self.compute_diff_accuracy(inputs[i], outputs[i], model_outputs[i])
            self.results_collector.add_task_specific_result(task_id, {
                "test_accuracy": task_accuracy,
                "test_diff_accuracy": task_diff_accuracy
            })

        # Append the result to self.test_outputs
        self.test_outputs.append(result)

        return result

    def on_test_epoch_end(self):
        total_loss = torch.stack([torch.tensor(x['test_loss']) for x in self.test_outputs]).mean()
        all_accuracies = []
        all_diff_accuracies = []

        per_task_accuracy = {}
        per_task_diff_accuracy = {}

        for output in self.test_outputs:
            if 'test_accuracy' in output:
                all_accuracies.append(output['test_accuracy'])
            if 'test_diff_accuracy' in output:
                all_diff_accuracies.append(output['test_diff_accuracy'])

            # Collect per-task metrics
            for key, value in output.items():
                if key.endswith('_test_accuracy'):
                    per_task_accuracy[key] = value
                elif key.endswith('_test_diff_accuracy'):
                    per_task_diff_accuracy[key] = value

        avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
        avg_diff_accuracy = sum(all_diff_accuracies) / len(all_diff_accuracies) if all_diff_accuracies else 0

        self.log('avg_test_loss', total_loss, prog_bar=True)
        self.log('avg_test_accuracy', avg_accuracy, prog_bar=True)
        self.log('avg_test_diff_accuracy', avg_diff_accuracy, prog_bar=True)

        print(f"DEBUG: Test epoch end - Avg loss: {total_loss}, Avg accuracy: {avg_accuracy}, Avg diff accuracy: {avg_diff_accuracy}")

        # Prepare final metrics including per-task metrics
        final_metrics = {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "final_test_loss": total_loss.item(),
            "final_test_accuracy": avg_accuracy,
            "final_test_diff_accuracy": avg_diff_accuracy
        }

        # Retrieve per-task metrics from ResultsCollector and include them in final_metrics
        for task_id, metrics in self.results_collector.task_specific_results.items():
            final_metrics.update({
                f"{task_id}_test_accuracy": metrics.get("test_accuracy", 0.0),
                f"{task_id}_test_diff_accuracy": metrics.get("test_diff_accuracy", 0.0)
            })

        logger.debug(f"DEBUG: Final metrics including per-task metrics: {final_metrics}")
        self.results_collector.set_final_metrics(final_metrics)

    def compute_accuracy(self, outputs, targets):
        predictions = outputs.argmax(dim=-1)
        # Reshape predictions to match the target shape
        predictions = predictions.view(targets.size())
        # Calculate accuracy over all elements
        accuracy = (predictions == targets).float().mean()
        logger.debug(f"DEBUG: compute_accuracy - Accuracy: {accuracy.item()}")
        return accuracy.item()

    def compute_diff_accuracy(self, inputs, targets, outputs):
        pad_symbol_idx = self.config.training.pad_symbol_idx  # Retrieve pad_symbol_idx from config
        predictions = outputs.argmax(dim=-1)
        diff_accuracy, _, _ = differential_pixel_accuracy(inputs, targets, predictions, pad_symbol_idx=pad_symbol_idx)
        logger.debug(f"Computed differential pixel accuracy (excluding padding tokens): {diff_accuracy}")
        return diff_accuracy
        
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
        try:
            self.writer.close()
            print("DEBUG: TensorBoard writer closed successfully.")
        except Exception as e:
            print(f"DEBUG: Error closing TensorBoard writer: {str(e)}")
        logger.debug("DEBUG: Results saved and TensorBoard writer closed.")


    def compute_loss(self, outputs, labels):
        labels = labels.long()  # Ensure labels are of type Long
        loss = self.model.loss_fn(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        logger.debug(f"Using model's loss function with ignore_index={self.model.loss_fn.ignore_index}")
        logger.debug(f"Computed loss: {loss.item()}")
        return loss

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
