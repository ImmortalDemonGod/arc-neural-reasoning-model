import uuid
import time
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

    def update_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Update training metrics for a given epoch."""
        if not isinstance(self.results["train"], dict):
            raise TypeError(f"Expected self.results['train'] to be a dict, but got {type(self.results['train'])}")
        self.results["train"].setdefault(epoch, {})
