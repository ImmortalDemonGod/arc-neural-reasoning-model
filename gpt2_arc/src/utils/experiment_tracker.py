# gpt2_arc/src/utils/experiment_tracker.py
import logging
import wandb
import json
import time
import uuid
import torch
import platform
import os
from dataclasses import asdict
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ExperimentTracker:
    def __init__(self, config: Dict[str, Any], project: str, entity: Optional[str] = None, use_wandb: bool = False):
        self.experiment_id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.config = config.to_dict() if hasattr(config, 'to_dict') else self._config_to_dict(config)
        self.project = project
        self.entity = entity
        self.run = None
        self.use_wandb = use_wandb
        self.metrics = {}
        if self.use_wandb:
            try:
                self.run = wandb.init(project=self.project, entity=self.entity, config=self.config)
                print(f"Wandb run initialized: {self.run.id}")
            except Exception as e:
                print(f"Error initializing wandb: {str(e)}")
                self.use_wandb = False

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
        logger.debug(f"ExperimentTracker initialized with config: {json.dumps(self.config, indent=2)}")
        logger.debug(f"Project: {project}, Entity: {entity}")
        logger.debug(f"use_wandb: {self.use_wandb}")

    def _get_environment_info(self) -> Dict[str, str]:
        return {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        if isinstance(config, dict):
            return {k: self._config_to_dict(v) for k, v in config.items()}
        elif hasattr(config, '__dict__'):
            return {k: self._config_to_dict(v) for k, v in config.__dict__.items() if not k.startswith('_')}
        else:
            return config
        if self.use_wandb:
            try:
                self.run = wandb.init(project=self.project, entity=self.entity, config=self.config)
                print(f"Wandb run initialized: {self.run.id}")
            except Exception as e:
                print(f"Error initializing wandb: {str(e)}")
                self.use_wandb = False

        if not self.use_wandb:
            print("Using local logging only")

    def set_final_metrics(self, metrics: Dict[str, float]):
        """Set the final metrics and format them properly for saving."""
        # Add debug logging for metrics being set
        logger.debug(f"Setting final metrics: {metrics}")
        
        # Convert any tensor values to Python types
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                formatted_metrics[key] = value.item()
            else:
                formatted_metrics[key] = value
        
        self.metrics = formatted_metrics
        
        # Add test metrics to results if they exist
        if 'avg_test_loss' in formatted_metrics:
            self.results['test'].update({
                'loss': formatted_metrics['avg_test_loss'],
                'accuracy': formatted_metrics.get('avg_test_accuracy', 0.0),
                'diff_accuracy': formatted_metrics.get('avg_test_diff_accuracy', 0.0)
            })
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(formatted_metrics)
        
        logger.debug(f"Final metrics set and formatted: {self.metrics}")

    def finish(self):
        """Clean up and log final metrics."""
        # Log final summary
        summary = {
            'final_metrics': self.metrics,
            'test_results': self.results.get('test', {}),
            'task_specific_results': self.task_specific_results
        }
        logger.info(f"Experiment finished. Final metrics: {json.dumps(summary, indent=2)}")
        
        if self.use_wandb and self.run:
            try:
                # Log final summary to wandb
                wandb.log(summary)
                wandb.finish()
                logger.debug("Wandb run finished successfully")
            except Exception as e:
                logger.error(f"Error finishing wandb run: {str(e)}")
        
        # Save final results
        if self.metrics or self.results['test']:
            self.save_to_json(f"results/experiment_{self.experiment_id}_final.json")

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
        logger.debug(f"Added task-specific result for task_id {task_id}: {metrics}")
        if self.use_wandb:
            wandb.log({f"task_{task_id}": metrics})

    def set_checkpoint_path(self, path: str):
        self.checkpoint_path = path
        if self.use_wandb:
            wandb.save(path)

    def save_to_json(self, filepath: str):
        """Save experiment results with proper error handling."""
        try:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            data = {
                "experiment_id": self.experiment_id,
                "timestamp": self.timestamp,
                "config": self.config,
                "results": self.results,
                "metrics": self.metrics,
                "task_specific_results": self.task_specific_results,
                "environment": self.environment,
                "checkpoint_path": self.checkpoint_path,
                "final_summary": {
                    "train_loss": self.results["train"][-1]["loss"] if self.results["train"] else None,
                    "val_loss": self.results["validation"][-1]["loss"] if self.results["validation"] else None,
                    "test_metrics": self.results["test"]
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Results saved to {filepath}")
            
        except IOError as e:
            logger.error(f"Error saving results to {filepath}: {e}")
            raise

    def _ensure_directory_exists(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_summary(self) -> Dict[str, Any]:
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
            "tensorboard_log_path": getattr(self, 'tensorboard_log_path', None)
        }
        logger.debug(f"DEBUG: Added TensorBoard log path to results: {summary.get('tensorboard_log_path')}")
        return {k: self._make_serializable(v) for k, v in summary.items()}

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)

    def _serialize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._make_serializable(v) for k, v in config.items()}

# Add a simple test
if __name__ == "__main__":
    config = {"learning_rate": 0.01, "batch_size": 32, "use_wandb": True}
    tracker = ExperimentTracker(config, project="test-project")
    # Removed tracker.start() as it was undefined
    tracker.log_metric("accuracy", 0.85, step=1)
    tracker.update_train_metrics(0, {"loss": 0.5, "accuracy": 0.8})
    tracker.update_val_metrics(0, {"loss": 0.6, "accuracy": 0.75})
    tracker.set_test_results({"avg_test_loss": 0.55, "avg_test_accuracy": 0.82, "avg_test_diff_accuracy": 0.05})
    tracker.add_task_specific_result("task_1", {"accuracy": 0.9})
    tracker.set_final_metrics({"best_accuracy": 0.85, "avg_test_loss": 0.55, "avg_test_accuracy": 0.82, "avg_test_diff_accuracy": 0.05})
    tracker.set_checkpoint_path("model_checkpoint.pth")
    tracker.save_to_json("results/experiment_final.json")
    tracker.finish()