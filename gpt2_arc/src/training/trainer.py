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
        self.model = model
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
                    shuffle=True,
                    pin_memory=True if self.args.use_gpu else False,
                    prefetch_factor=self.config.training.prefetch_factor,
                    persistent_workers=self.config.training.persistent_workers,
                    collate_fn=self.train_dataset.collate_fn  # Ensure collate_fn is applied
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
                    collate_fn=self.val_dataset.collate_fn  # Ensure collate_fn is applied
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
                    collate_fn=self.test_dataset.collate_fn  # Ensure collate_fn is applied
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
                collate_fn=self.train_dataset.collate_fn  # Updated line
            )

        logger.debug(f"Training DataLoader created with num_workers={get_num_workers(self.config.training)}")
        return train_loader

    def val_dataloader(self):
        logger.debug("Entering ARCTrainer.val_dataloader")
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=get_num_workers(self.config.training),
            pin_memory=self.config.training.pin_memory,
            prefetch_factor=self.config.training.prefetch_factor,
            persistent_workers=self.config.training.persistent_workers,
            collate_fn=self.val_dataset.collate_fn
        )
        logger.debug("Exiting ARCTrainer.val_dataloader")
        return dataloader

    def test_dataloader(self):
        if self.test_dataset is None:
            logger.error("Test dataset is not provided")
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=get_num_workers(self.config.training),
            shuffle=False,
            pin_memory=self.config.training.pin_memory,
            prefetch_factor=self.config.training.prefetch_factor,
            persistent_workers=self.config.training.persistent_workers,
            collate_fn=self.test_dataset.collate_fn  # Updated line
        )

    
    def get_tensorboard_logger(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                return logger.experiment
        logger.debug("DEBUG: No TensorBoardLogger found in trainer.loggers")
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
        self.train_losses.append(loss.item())
        self.results_collector.update_train_metrics(self.current_epoch, {"loss": loss.item()})
        
        tb_logger = self.get_tensorboard_logger()
        if tb_logger:
            tb_logger.add_scalar('train/loss', loss.item(), self.global_step)
            logger.debug(f"DEBUG: Logged training loss: {loss.item()} at step {self.global_step}")
        else:
            logger.debug("DEBUG: Failed to log training loss. No TensorBoard logger available.")
        
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
            logger.debug(f"DEBUG: Logged validation loss: {loss.item()} for epoch {self.current_epoch}")
        except Exception as e:
            logger.error(f"DEBUG: Error logging validation step: {str(e)}")
        
        return loss

    def on_test_epoch_start(self):
        self.test_outputs = []  # Clear previous test outputs

    def test_step(self, batch, batch_idx):
        logger.debug(f"DEBUG: test_step input - batch: {batch}, batch_idx: {batch_idx}")
        logger.debug(f"DEBUG: Test step - Batch type: {type(batch)}, length: {len(batch)}")

        # Unpack batch
        if len(batch) == 3:
            inputs, outputs, task_ids = batch
        elif len(batch) == 2:
            inputs, outputs = batch
            raise ValueError("Batch does not contain 'task_ids'. Ensure the dataset provides 'task_ids'.")
        else:
            raise ValueError(f"Unexpected batch format with length {len(batch)}")
        logger.debug(f"DEBUG: Task IDs in batch: {task_ids}")

        inputs = inputs.float()
        outputs = outputs.long()

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
                # Create a mask for the current task_id
                mask = [tid == task_id for tid in task_ids]  # Create mask as list of booleans
                mask = torch.tensor(mask, dtype=torch.bool, device=outputs.device)  # Convert to tensor
                task_accuracy = self.compute_accuracy(model_outputs[mask], outputs[mask])
                task_diff_accuracy = self.compute_diff_accuracy(inputs[mask], outputs[mask], model_outputs[mask])
                
                result[f"{task_id}_test_accuracy"] = task_accuracy
                result[f"{task_id}_test_diff_accuracy"] = task_diff_accuracy
                
                self.log(f"{task_id}_test_accuracy", task_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log(f"{task_id}_test_diff_accuracy", task_diff_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
                # Add the following lines to send per-task metrics to ResultsCollector
                self.results_collector.add_task_specific_result(
                    task_id, 
                    {
                        "test_accuracy": task_accuracy.item(),
                        "test_diff_accuracy": task_diff_accuracy
                    }
                )
                logger.debug(f"DEBUG: Added per-task metrics for {task_id} to ResultsCollector.")

        try:
            self.writer.add_scalar('test/loss', result['test_loss'], self.current_epoch)
            self.writer.add_scalar('test/avg_accuracy', result['test_accuracy'], self.current_epoch)
            self.writer.add_scalar('test/diff_accuracy', result['test_diff_accuracy'], self.current_epoch)
            logger.debug(f"DEBUG: Logged test metrics for epoch {self.current_epoch}: loss={result['test_loss']}, avg_accuracy={result['test_accuracy']}, diff_accuracy={result['test_diff_accuracy']}")
        except Exception as e:
            logger.error(f"DEBUG: Error logging test step: {str(e)}")

        logger.debug(f"DEBUG: test_step output - result: {result}")
        logger.debug(f"DEBUG: Test step result: {result}")

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
        return accuracy

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
        self.results_collector.save_to_json(f"results/experiment_{self.results_collector.experiment_id}.json")
        try:
            self.writer.close()
            print("DEBUG: TensorBoard writer closed successfully.")
        except Exception as e:
            print(f"DEBUG: Error closing TensorBoard writer: {str(e)}")
        logger.debug("DEBUG: Results saved and TensorBoard writer closed.")


    def compute_loss(self, outputs, labels):
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
