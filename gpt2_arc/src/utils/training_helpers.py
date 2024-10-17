from typing import Optional
from gpt2_arc.src.config import TrainingConfig

def get_num_workers(config: TrainingConfig, args_num_workers: Optional[int] = None) -> int:
    """Determine the number of DataLoader workers, allowing for command-line overrides."""
    if args_num_workers is not None:
        return args_num_workers
    return config.num_workers
