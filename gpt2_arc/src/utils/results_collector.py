import json
import time
import uuid
import torch
import platform
from dataclasses import asdict
from typing import Dict, Any

class ResultsCollector:
    def __init__(self, config):
        self.experiment_id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.config = asdict(config)
        self.results = {
            "train": {},
            "validation": {},
            "test": {}
        }
        self.metrics = {}
        self.task_specific_results = {}
        self.environment = self._get_environment_info()
        self.checkpoint_path = None

    def _get_environment_info(self) -> Dict[str, str]:
        return {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

    def update_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        if "train" not in self.results:
            self.results["train"] = {}
        self.results["train"][epoch] = metrics

    def update_val_metrics(self, epoch: int, metrics: Dict[str, float]):
        if "validation" not in self.results:
            self.results["validation"] = {}
        self.results["validation"][epoch] = metrics

    def set_test_results(self, metrics: Dict[str, float]):
        self.results["test"] = metrics

    def add_task_specific_result(self, task_id: str, metrics: Dict[str, float]):
        if task_id not in self.task_specific_results:
            self.task_specific_results[task_id] = {}
        self.task_specific_results[task_id].update(metrics)

    def set_final_metrics(self, metrics: Dict[str, float]):
        self.metrics = metrics

    def set_checkpoint_path(self, path: str):
        self.checkpoint_path = path

    def save_to_json(self, filepath: str):
        data = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "results": self.results,
            "metrics": self.metrics,
            "task_specific_results": self.task_specific_results,
            "environment": self.environment,
            "checkpoint_path": self.checkpoint_path
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "final_train_loss": self.results["train"][-1]["loss"] if self.results["train"] else None,
            "final_val_loss": self.results["validation"][-1]["loss"] if self.results["validation"] else None,
            "test_accuracy": self.results["test"].get("accuracy"),
            "config": self.config
        }
