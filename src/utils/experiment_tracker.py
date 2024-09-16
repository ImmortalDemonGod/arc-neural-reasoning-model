import wandb
import json
from typing import Dict, Any, Optional

class ExperimentTracker:
    def __init__(self, config: Dict[str, Any], project: str, entity: Optional[str] = None):
        self.config = config
        self.project = project
        self.entity = entity
        self.run = None
        self.use_wandb = config.get('use_wandb', False)

        # Add debug logging
        print(f"ExperimentTracker initialized with config: {json.dumps(config, indent=2)}")
        print(f"Project: {project}, Entity: {entity}")
        print(f"use_wandb: {self.use_wandb}")

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

# Add a simple test
if __name__ == "__main__":
    config = {"learning_rate": 0.01, "batch_size": 32, "use_wandb": True}
    tracker = ExperimentTracker(config, project="test-project", entity="test-entity")
    tracker.start()
    tracker.log_metric("accuracy", 0.85, step=1)
    tracker.finish()
