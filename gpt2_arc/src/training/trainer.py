# gpt2_arc/src/training/trainer.py
# Refactored Version
import pytorch_lightning as pl
import torch
import logging
from torch import nn, optim
from typing import Any, Dict, Optional
from ..config import Config
from gpt2_arc.src.utils.training_helpers import get_num_workers
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy
from ..utils.results_collector import ResultsCollector
from torch.utils.data import DataLoader
from gpt2_arc.src.data.arc_dataset import ARCDataset
from optuna.exceptions import TrialPruned
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)

class NanLossPruningCallback(Callback):
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        # Extract loss from outputs
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if loss is not None:
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(
                    f"Invalid loss detected at epoch {trainer.current_epoch}, batch {batch_idx}: {loss.item()}"
                )
                raise TrialPruned("Invalid loss encountered, pruning this trial.")

class ARCTrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: ARCDataset,
        val_dataset: ARCDataset,
        config: Config,
        args: Any,
        results_collector: Optional[ResultsCollector] = None,
        test_dataset: Optional[ARCDataset] = None
    ) -> None:
        logger.debug("Initializing ARCTrainer")
        super().__init__()
        logger.debug(f"ARCTrainer received args.accelerator: {args.accelerator}")
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.test_dataset = test_dataset  # Add this line
        self.batch_size = config.training.batch_size
        self.lr = config.training.learning_rate
        self.test_outputs = []  # Initialize an empty list to store test outputs
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.results_collector = results_collector if results_collector else ResultsCollector(config)
        self.args = args  # Assign args

    def train_dataloader(self):
        logger.info("Creating training DataLoader with centralized num_workers")
        logger.debug(f"Training dataset type: {type(self.train_dataset)}")
        logger.debug(f"Training dataset content: {self.train_dataset}")

        if self.config.training.balance_symbols:
            if self.config.training.balancing_method == "weighting":
                # Log symbol frequencies
                logger.debug(f"Symbol frequencies: {self.config.training.symbol_freq}")
                
                # Check if symbol frequencies are available
                if self.config.training.symbol_freq is not None:
                    # Compute class weights (inverse frequencies)
                    class_weights = 1.0 / torch.tensor(
                        list(self.config.training.symbol_freq.values()), dtype=torch.float
                    )
                    logger.debug(f"Computed class weights: {class_weights}")

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
                else:
                    logger.warning("Symbol frequencies not available. Using default DataLoader without weighting.")
                    train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=self.config.training.batch_size,
                        num_workers=get_num_workers(self.config.training),
                        shuffle=True,  # Enable shuffle since no sampler
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

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.compute_loss(outputs, targets)
            
        preds = torch.argmax(outputs, dim=-1)
        try:
            preds = preds.view(targets.shape)
            accuracy = (preds == targets).float().mean()
        except RuntimeError as e:
            logger.error("Failed to compute training accuracy")
            raise e

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
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

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        logger.debug(f"test_step input - batch: {batch}, batch_idx: {batch_idx}")
        logger.debug(f"Test batch input shape: {batch[0].shape}, Test batch target shape: {batch[1].shape}")
        logger.debug(f"Test step - Batch type: {type(batch)}, length: {len(batch)}")

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

        logger.debug(f"Batch accuracy: {accuracy}, Batch diff_accuracy: {diff_accuracy}")

        result = {
            'test_loss': loss.item(),
            'task_ids': task_ids,
            'test_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'test_diff_accuracy': sum(diff_accuracies) / len(diff_accuracies) if diff_accuracies else 0,
        }
        logger.debug(f"Test loss: {result['test_loss']}, Avg accuracy: {result['test_accuracy']}, Avg diff accuracy: {result['test_diff_accuracy']}")

        # Log task-specific metrics if task_ids are available
        if task_ids is not None:
            # Aggregate task-specific metrics across the batch
            for task_id in set(task_ids):  # Remove .tolist()
                if task_id == "default_task":
                    logger.error(f"'default_task' detected in task_id: {task_id}")
                    raise ValueError(f"'default_task' detected in task_id: {task_id}")

        self.log('test_loss', result['test_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_avg_accuracy', result['test_accuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_diff_accuracy', result['test_diff_accuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        logger.debug(f"Logged test metrics for epoch {self.current_epoch}: loss={result['test_loss']}, avg_accuracy={result['test_accuracy']}, diff_accuracy={result['test_diff_accuracy']}")

        logger.debug(f"test_step output - result: {result}")
        logger.debug(f"Test step result: {result}")

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

        logger.debug(f"Test epoch end - Avg loss: {total_loss}, Avg accuracy: {avg_accuracy}, Avg diff accuracy: {avg_diff_accuracy}")

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

        logger.debug(f"Final metrics including per-task metrics: {final_metrics}")
        self.results_collector.set_final_metrics(final_metrics)

    def compute_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        predictions = outputs.argmax(dim=-1)
        # Reshape predictions to match the target shape
        predictions = predictions.view(targets.size())
        # Calculate accuracy over all elements
        accuracy = (predictions == targets).float().mean()
        logger.debug(f"compute_accuracy - Accuracy: {accuracy.item()}")
        return accuracy.item()

    def compute_diff_accuracy(self, inputs: torch.Tensor, targets: torch.Tensor, outputs: torch.Tensor) -> float:
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


    # Modify on_fit_end for proper cleanup
    def on_fit_end(self):
        logger.debug("Training completed, cleaning up resources")
        # The actual cleanup is now handled by TrainingManager

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()  # Ensure labels are of type Long
        loss = self.model.loss_fn(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        logger.debug(f"Using model's loss function with ignore_index={self.model.loss_fn.ignore_index}")
        logger.debug(f"Computed loss: {loss.item()}")
        return loss

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            # Use PL's logger instead of direct TensorBoard writer
            if self.logger is not None:
                self.logger.log_hyperparams(hparams, metric_dict)
                logger.debug(f"Successfully logged hyperparameters: {hparams}")
        except Exception as e:
            logger.error(f"Error logging hyperparameters: {str(e)}")