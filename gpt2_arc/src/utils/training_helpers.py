from typing import Optional
from gpt2_arc.src.config import TrainingConfig

def get_num_workers(config: TrainingConfig) -> int:
    """Determine the number of DataLoader workers based on the configuration."""
    return config.num_workers
