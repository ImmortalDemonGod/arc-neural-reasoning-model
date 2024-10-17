# gpt2_arc/src/utils/results_collector.py
import json
import time
import uuid
import torch
import platform
import os
from dataclasses import asdict
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ResultsCollector:
    def __init__(self, config):
        """Initialize the ResultsCollector with a given configuration."""
        self.experiment_id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.config = asdict(config)
        self.symbol_freq = self.config.get('training', {}).get('symbol_freq', {})
        logger.debug(f"Symbol frequencies set in ResultsCollector: {self.symbol_freq}")
        self.results = {
            "train": [],
            "validation": [],
            "test": {}
        }
        self.tensorboard_log_path = config.get('tensorboard_log_path', None)
        self.used_synthetic_data = config.training.use_synthetic_data
        print(f"DEBUG: Initialized self.results['train'] as {type(self.results['train'])}")
        self._log_results_type("After initialization")
        self.metrics = {}
        self.task_specific_results = {}
        self.environment = self._get_environment_info()
        self.checkpoint_path = None
        self.tensorboard_log_path = None

    def set_tensorboard_log_path(self, path):
        self.tensorboard_log_path = path
        print(f"DEBUG: Set TensorBoard log path in ResultsCollector: {path}")

    def _get_environment_info(self) -> Dict[str, str]:
        """Retrieve environment information such as Python and PyTorch versions."""
        return {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

    def _log_results_type(self, context: str):
        """Log the type of self.results['train'] for debugging."""
    
    def update_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        # print(f"DEBUG: self.results['train'] is of type {type(self.results['train'])}")
        """Update training metrics for a specific epoch."""
        self._log_results_type("Before checking 'train' in results")
        if "train" not in self.results:
            self.results["train"] = {}
        self._log_results_type("Before type check")
        if not isinstance(self.results["train"], dict):
            raise TypeError(f"Expected self.results['train'] to be a dict, but got {type(self.results['train'])}")
        self._log_results_type("Before setting default")
        # print(f"DEBUG: Before setting default, self.results['train'] is of type {type(self.results['train'])}")
        self.results["train"].setdefault(epoch, {})
        self._log_results_type("After setting default")
        # print(f"DEBUG: After setting default, self.results['train'] is of type {type(self.results['train'])}")
        self.results["train"][epoch].update(metrics)

    def update_val_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Update validation metrics for a specific epoch."""
        if "validation" not in self.results:
            self.results["validation"] = {}
        self.results["validation"][epoch] = metrics

    def set_test_results(self, metrics: Dict[str, float]):
        """Set the test results metrics."""
        self.results["test"] = metrics
        
    def add_task_specific_result(self, task_id: str, metrics: Dict[str, float]):
        """Add task-specific results for a given task ID."""
        if task_id not in self.task_specific_results:
            self.task_specific_results[task_id] = {}
        self.task_specific_results[task_id].update(metrics)

    def set_final_metrics(self, metrics: Dict[str, float]):
        """Set the final metrics after training."""
        self.metrics = metrics

    def set_checkpoint_path(self, path: str):
        """Set the path to the model checkpoint."""
        self.checkpoint_path = path

    def save_to_json(self, filepath: str):
        """Save the collected results to a JSON file."""
        try:
            self._ensure_directory_exists(os.path.dirname(filepath))
            data = {
                "experiment_id": self.experiment_id,
                "timestamp": self.timestamp,
                "config": self.config,
                "results": self.results,
                "metrics": self.metrics,
                "task_specific_results": self.task_specific_results,
                "environment": self.environment,
                "checkpoint_path": self.checkpoint_path,
                "used_synthetic_data": self.used_synthetic_data
            }
            if self.symbol_freq:
                data["symbol_freq"] = self.symbol_freq
            else:
                data["symbol_freq"] = {}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving results to {filepath}: {e}")

    def _ensure_directory_exists(self, directory: str):
        """Ensure that the directory exists; create it if it does not."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the results.
        
        Returns:
            Dict[str, Any]: Summary of key metrics.
        """
        summary = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "final_train_loss": self.results["train"][-1]["loss"] if self.results["train"] else None,
            "final_val_loss": self.results["validation"][-1]["loss"] if self.results["validation"] else None,
            "test_loss": self.results["test"].get("avg_loss"),
            "test_acc_with_pad": self.results["test"].get("avg_acc_with_pad"),
            "test_acc_without_pad": self.results["test"].get("avg_acc_without_pad"),
            "best_val_loss": self.results.get("best_val_loss"),
            "best_val_epoch": self.results.get("best_val_epoch"),
            "learning_rate": self.config.get("training", {}).get("learning_rate"),
            "batch_size": self.config.get("training", {}).get("batch_size"),
            "training_duration": self.results.get("training_duration"),
            "config": self._serialize_config(self.config),
            "tensorboard_log_path": self.tensorboard_log_path
        }
        logger.debug(f"DEBUG: Added TensorBoard log path to results: {summary['tensorboard_log_path']}")
        return {k: self._make_serializable(v) for k, v in summary.items()}

    def _make_serializable(self, obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)

    def _serialize_config(self, config):
        return {k: self._make_serializable(v) for k, v in config.items()}
    def _serialize_config(self, config):
        """Serialize the configuration dictionary."""
        return json.dumps(config, indent=2)
    
    def _make_serializable(self, value):
        """Ensure the value is serializable, handling non-serializable objects."""
        try:
            json.dumps(value)
            return value
        except (TypeError, OverflowError):
            return str(value)
