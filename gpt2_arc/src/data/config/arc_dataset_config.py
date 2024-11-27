# gpt2_arc/src/data/config/arc_dataset_config.py
from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Tuple

try:
    from arckit.data import TaskSet, Task
except ImportError:
    TaskSet = None

@dataclass  # Add this decorator
class ARCDatasetConfig:
    """Configuration for ARCDataset"""
    # Change assignments to type annotations (:)
    data_source: Union[str, List[Dict], 'TaskSet', Tuple[Union[List, 'TaskSet'], str]]
    is_test: bool = False
    num_symbols: int = 11
    test_split: float = 0.2
    pad_symbol_idx: int = 10
    symbol_freq: Optional[Dict[int, float]] = None
    debug: bool = False
    mamba_ratio: float = 1.0
    mamba_ratio_min: float = 0.25
    max_samples: Optional[int] = None

    def __post_init__(self):
        # Validate configuration
        if not 0 <= self.test_split <= 1:
            raise ValueError(f"test_split must be between 0 and 1, got {self.test_split}")
        if self.num_symbols <= self.pad_symbol_idx:
            raise ValueError(f"num_symbols ({self.num_symbols}) must be greater than pad_symbol_idx ({self.pad_symbol_idx})")
        if self.mamba_ratio < self.mamba_ratio_min:
            self.mamba_ratio = self.mamba_ratio_min