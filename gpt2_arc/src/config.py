from typing import Optional, Dict
from dataclasses import dataclass, asdict, field
import multiprocessing
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

@dataclass
class ModelConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    mamba_ratio: float = 1.0  # Number of Mamba layers per Transformer layer
    d_state: int = 16          # Mamba state dimension
    d_conv: int = 4            # Mamba convolution dimension
    mamba_depth: int = 1       # Depth of each Mamba layer
    mamba_expand: int = 2      # Expand factor for each Mamba layer

@dataclass
class TrainingConfig:
    # Existing fields...
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 10
    num_workers: int = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() else 4
    symbol_freq: Optional[Dict] = None
    prefetch_factor: int = 2
    persistent_workers: bool = True
    use_gpu: bool = True
    log_level: str = "INFO"
    use_synthetic_data: bool = False
    balance_symbols: bool = True
    balancing_method: str = "weighting"
    synthetic_data_path: Optional[str] = None

    # New fields for padding symbol
    pad_symbol: str = "<PAD>"
    pad_symbol_idx: int = field(init=False)

    def __post_init__(self):
        # Dynamically set num_classes based on symbol_freq
        if self.symbol_freq:
            max_existing_idx = max(int(k) for k in self.symbol_freq.keys())
            self.pad_symbol_idx = max_existing_idx + 1
            self.num_classes = max_existing_idx + 2  # +1 for padding
        else:
            # Default number of classes if symbol_freq is not provided
            self.pad_symbol_idx = 10  # Assuming 10 symbols (0-9)
            self.num_classes = 11  # 10 symbols + 1 padding
        print(f"TrainingConfig initialized with {self.num_classes} classes and PAD symbol index {self.pad_symbol_idx}")

@dataclass
class EvaluationConfig:
    perfect_accuracy_threshold: float = 99.9  # Set to 99.9 for near-perfect accuracy

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self):
        return asdict(self)
