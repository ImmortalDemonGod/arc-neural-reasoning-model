import uuid
import time
from typing import Dict
from dataclasses import asdict

class ResultsCollector:
    def __init__(self, config):
        """Initialize the ResultsCollector with a given configuration."""
        self.experiment_id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.config = asdict(config)
        self.results = {
            "train": {},
            "validation": {},
            "test": {}
        }
        print(f"DEBUG: Initialized self.results['train'] as {type(self.results['train'])}")

    def update_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Update training metrics for a given epoch."""
        print(f"DEBUG: Before updating, self.results['train'] is of type {type(self.results['train'])}")
        if not isinstance(self.results["train"], dict):
            print(f"DEBUG: self.results['train'] is of type {type(self.results['train'])} before setdefault")
            raise TypeError(f"Expected self.results['train'] to be a dict, but got {type(self.results['train'])}")
        self.results["train"].setdefault(epoch, {})
        print(f"DEBUG: After updating, self.results['train'] is of type {type(self.results['train'])}")

    def update_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Update training metrics for a given epoch."""
        print(f"DEBUG: Before updating, self.results['train'] is of type {type(self.results['train'])}")
        if not isinstance(self.results["train"], dict):
            raise TypeError(f"Expected self.results['train'] to be a dict, but got {type(self.results['train'])}")
        self.results["train"].setdefault(epoch, {})
        print(f"DEBUG: After updating, self.results['train'] is of type {type(self.results['train'])}")

    def update_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Update training metrics for a given epoch."""
        print(f"DEBUG: Before updating, self.results['train'] is of type {type(self.results['train'])}")
        if not isinstance(self.results["train"], dict):
            raise TypeError(f"Expected self.results['train'] to be a dict, but got {type(self.results['train'])}")
        self.results["train"].setdefault(epoch, {})
        print(f"DEBUG: After updating, self.results['train'] is of type {type(self.results['train'])}")

    def update_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Update training metrics for a given epoch."""
        if not isinstance(self.results["train"], dict):
            print(f"DEBUG: self.results['train'] is of type {type(self.results['train'])} before setdefault")
            raise TypeError(f"Expected self.results['train'] to be a dict, but got {type(self.results['train'])}")
        self.results["train"].setdefault(epoch, {})
