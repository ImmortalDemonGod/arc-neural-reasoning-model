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
        self.results_collector = ResultsCollector(config)
        self.results_collector.results["train"] = []  # Initialize train results as a list

    def training_step(self, batch, batch_idx):
        """
        The `training_step` function processes a batch of data for training a model, ensuring correct tensor
        types and logging training metrics.
        
        :param batch: The `batch` parameter in the `training_step` function is the input data batch that the
        model will use for training. It can be in various formats such as a dictionary, tuple, or list
        depending on how the data is structured and passed to the model
        :param batch_idx: `batch_idx` is the index of the current batch being processed during training. It
        is used to keep track of which batch is currently being trained on
        :return: The `training_step` method returns the loss calculated during the training step.
        """
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
        elif isinstance(batch, list) and len(batch) == 3:
            input_ids, labels, task_ids = batch
            attention_mask = None
        elif isinstance(batch, list) and len(batch) == 2:
            input_ids, labels = batch
            attention_mask = None
            task_ids = None
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
        self.results_collector.update_train_metrics(self.current_epoch, {"loss": loss.item()})
        self.results_collector.results["train"].append({"epoch": self.current_epoch, "loss": loss.item()})
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
            input_ids, labels, task_ids = batch
            attention_mask = None
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
        self.results_collector.update_val_metrics(self.current_epoch, {"loss": loss.item()})
        self.logged_metrics["val_loss"] = loss.item()

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

        # Collect metrics in a dictionary
        metrics = {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_diff_accuracy': diff_accuracy
        }

        # Collect metrics in a dictionary
        metrics = {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_diff_accuracy': diff_accuracy
        }

        # Log metrics safely
        try:
            self.log_dict(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

        # Store result for later processing
        result = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'task_ids': task_ids,
            'test_loss': loss.item()
        }
        self.test_outputs.append(result)

        return result

    def on_test_epoch_end(self):
        all_task_ids = []
        all_losses = []
        all_accuracies = []

        print("Debug: Starting on_test_epoch_end")
        print(f"Debug: Number of test outputs: {len(self.test_outputs)}")

        for output in self.test_outputs:
            print(f"Debug: Processing output: {output}")
            batch_task_ids = output.get('task_ids', [])
            if isinstance(batch_task_ids, torch.Tensor):
                batch_task_ids = batch_task_ids.tolist()

            all_task_ids.extend(batch_task_ids)
            if 'loss' in output:
                all_losses.extend([output['loss'].item()] * len(batch_task_ids))
            else:
                logger.error("Missing 'loss' key in output")
            if 'accuracy' in output:
                all_accuracies.extend([output['accuracy'].item()] * len(batch_task_ids))
            else:
                logger.error("Missing 'accuracy' key in output")

            print(f"Debug: Batch task IDs: {batch_task_ids}")
            if 'loss' in output and 'accuracy' in output:
                print(f"Debug: Batch loss: {output['loss'].item()}, accuracy: {output['accuracy'].item()}")
            else:
                logger.error("Missing 'loss' or 'accuracy' key in output")

        total_tasks = len(all_task_ids)
        successful_tasks = sum(1 for result in self.test_outputs if result['accuracy'] == 1.0)
        task_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        avg_test_loss = sum(all_losses) / len(all_losses) if all_losses else 0
        avg_test_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0

        # Instead of logging, store these values in the results collector
        self.results_collector.set_test_results({
            "avg_loss": avg_test_loss,
            "avg_accuracy": avg_test_accuracy,
            "task_success_rate": task_success_rate
        })

        logger.info(f"Test Results - Avg Loss: {avg_test_loss:.4f}, Avg Accuracy: {avg_test_accuracy:.4f}, Task Success Rate: {task_success_rate:.4f}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95),
            'name': 'learning_rate',
        }
        return [optimizer], [lr_scheduler]

    def on_fit_end(self):
        # Save the results to a JSON file
        self.results_collector.save_to_json(f"results/experiment_{self.results_collector.experiment_id}.json")

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
