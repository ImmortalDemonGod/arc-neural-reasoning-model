# gpt2_arc/src/utils/experiment_tracker.py
import wandb
import json
import time
import uuid
import torch
import platform
import os
from dataclasses import asdict
from typing import Dict, Any, Optional

class ExperimentTracker:
    def __init__(self, config: Dict[str, Any], project: str, entity: Optional[str] = None):
        self.experiment_id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.config = config
        self.project = project
        self.entity = entity
        self.run = None
        self.use_wandb = config.get('use_wandb', False)
        self.results = {
            "train": [],
            "validation": [],
            "test": {}
        }
        self.metrics = {}
        self.task_specific_results = {}
        self.environment = self._get_environment_info()
        self.checkpoint_path = None

        # Add debug logging
        print(f"ExperimentTracker initialized with config: {json.dumps(config, indent=2)}")
        print(f"Project: {project}, Entity: {entity}")
        print(f"use_wandb: {self.use_wandb}")

    def _get_environment_info(self) -> Dict[str, str]:
        return {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

    def start(self):
        if self.use_wandb:
            try:
                self.run = wandb.init(project=self.project, entity=self.entity, config=self.config)
                print(f"Wandb run initialized: {self.run.id}")
            except Exception as e:
                print(f"Error initializing wandb: {str(e)}")
                self.use_wandb = False
        
        if not self.use_wandb:
            print("Using local logging only")

    def finish(self):
        if self.use_wandb and self.run:
            try:
                wandb.finish()
                print("Wandb run finished")
            except Exception as e:
                print(f"Error finishing wandb run: {str(e)}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        if self.use_wandb:
            try:
                wandb.log({name: value}, step=step)
                print(f"Logged metric to wandb: {name}={value}, step={step}")
            except Exception as e:
                print(f"Error logging metric to wandb: {str(e)}")
        
        # Always log locally as a fallback
        print(f"Logged metric locally: {name}={value}, step={step}")

    def update_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        if "train" not in self.results:
            self.results["train"] = []
        while len(self.results["train"]) <= epoch:
            self.results["train"].append({})
        self.results["train"][epoch] = metrics
        if self.use_wandb:
            wandb.log({"train": metrics}, step=epoch)

    def update_val_metrics(self, epoch: int, metrics: Dict[str, float]):
        if "validation" not in self.results:
            self.results["validation"] = []
        while len(self.results["validation"]) <= epoch:
            self.results["validation"].append({})
        self.results["validation"][epoch] = metrics
        if self.use_wandb:
            wandb.log({"validation": metrics}, step=epoch)

    def set_test_results(self, metrics: Dict[str, float]):
        self.results["test"] = metrics
        if self.use_wandb:
            wandb.log({"test": metrics})

    def add_task_specific_result(self, task_id: str, metrics: Dict[str, float]):
        if task_id not in self.task_specific_results:
            self.task_specific_results[task_id] = {}
        self.task_specific_results[task_id].update(metrics)
        if self.use_wandb:
            wandb.log({f"task_{task_id}": metrics})

    def set_final_metrics(self, metrics: Dict[str, float]):
        self.metrics = metrics
        if self.use_wandb:
            wandb.log(metrics)

    def set_checkpoint_path(self, path: str):
        self.checkpoint_path = path
        if self.use_wandb:
            wandb.save(path)

    def save_to_json(self, filepath: str):
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
                "checkpoint_path": self.checkpoint_path
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Results saved to {filepath}")
        except IOError as e:
            print(f"Error saving results to {filepath}: {e}")

    def _ensure_directory_exists(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "final_train_loss": self.results["train"][-1]["loss"] if self.results["train"] else None,
            "final_val_loss": self.results["validation"][-1]["loss"] if self.results["validation"] else None,
            "test_accuracy": self.results["test"].get("accuracy"),
            "config": self._serialize_config(self.config)
        }
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

# Add a simple test
if __name__ == "__main__":
    config = {"learning_rate": 0.01, "batch_size": 32, "use_wandb": True}
    tracker = ExperimentTracker(config, project="test-project", entity="arc-abolition-org")
    tracker.start()
    tracker.log_metric("accuracy", 0.85, step=1)
    tracker.update_train_metrics(0, {"loss": 0.5, "accuracy": 0.8})
    tracker.update_val_metrics(0, {"loss": 0.6, "accuracy": 0.75})
    tracker.set_test_results({"loss": 0.55, "accuracy": 0.82})
    tracker.add_task_specific_result("task_1", {"accuracy": 0.9})
    tracker.set_final_metrics({"best_accuracy": 0.85})
    tracker.set_checkpoint_path("model_checkpoint.pth")
    tracker.save_to_json("results.json")
    tracker.finish()
