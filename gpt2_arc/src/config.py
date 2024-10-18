# gpt2_arc/src/config.py
from typing import Optional, Dict
from dataclasses import dataclass, asdict, field
import multiprocessing
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

@dataclass
class ModelConfig:
    n_embd: int = 256          # Reduced from 768 to 256
    n_head: int = 2            # Increased from 1 to 2
    n_layer: int = 2           # Increased from 12 to 2
    num_classes: int = field(default=11, metadata={"description": "Number of output classes for the model."})
    dropout: float = 0.1
    mamba_ratio: float = 0.0
    d_state: int = 4
    d_conv: int = 1
    mamba_depth: int = 1
    mamba_expand: int = 2

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.n_embd >= self.n_head, f"n_embd ({self.n_embd}) must be greater than or equal to n_head ({self.n_head})"
        assert self.n_layer > 0, f"n_layer ({self.n_layer}) must be positive"
        assert self.d_state >= 1, f"d_state ({self.d_state}) must be at least 1"
        assert self.d_conv >= 1, f"d_conv ({self.d_conv}) must be at least 1"
        assert self.mamba_depth >= 1, f"mamba_depth ({self.mamba_depth}) must be at least 1"
        assert self.mamba_expand >= 2, f"mamba_expand ({self.mamba_expand}) must be at least 2"
        logger.debug("ModelConfig initialized successfully")

@dataclass
class TrainingConfig:
    batch_size: int = 16             # Reduced from 32 to 16
    learning_rate: float = 1e-4      # Keep as is or adjust if necessary
    max_epochs: int = 1              # Already set for fast dev run
    num_classes: int = field(default=11, metadata={"description": "Number of output classes for the model."})
    num_symbols: int = 11  # Ensure num_symbols is set to 11
    num_workers: int = 1             # Reduced from multiprocessing.cpu_count() to 1
    symbol_freq: Optional[Dict[int, float]] = None
    pin_memory: bool = False         # Disable if not using GPU
    prefetch_factor: int = 1         # Reduced from 2 to 1
    persistent_workers: bool = False # Ensure workers do not stay alive
    use_gpu: bool = True
    log_level: str = "INFO"
    use_synthetic_data: bool = False
    use_grokfast: bool = False
    grokfast_type: Optional[str] = field(default=None)  # 'ema' or 'ma'
    grokfast_alpha: float = field(default=0.98)
    grokfast_lamb: float = field(default=2.0)
    grokfast_window_size: Optional[int] = field(default=100)  # Only relevant if grokfast_type == 'ma'
    balance_symbols: bool = True
    balancing_method: str = "weighting"
    synthetic_data_path: Optional[str] = None
    include_pad_in_loss: bool = True  # Whether to include the padding class in the loss calculation
    include_pad_in_accuracy: bool = True  # Whether to include the padding class in accuracy calculations
    tensorboard_log_path: Optional[str] = None  # Default to None if not set

    # New fields for padding symbol
    pad_symbol: str = "<PAD>"
    pad_symbol_idx: int = field(default=10)  # Add this line

    def __post_init__(self):
        # Dynamically set num_classes based on symbol_freq
        self.pad_symbol_idx = 10  # Set to the default padding index
        print(f"TrainingConfig initialized with {self.num_classes} classes and PAD symbol index {self.pad_symbol_idx}")
        print(f"include_pad_in_loss: {self.include_pad_in_loss}")  # Added debug statement
        print(f"include_pad_in_accuracy: {self.include_pad_in_accuracy}")  # Added debug statement

@dataclass
class EvaluationConfig:
    perfect_accuracy_threshold: float = 99.9  # Set to 99.9 for near-perfect accuracy

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    estimated_memory: Optional[float] = None
    available_memory: Optional[float] = None

    def to_dict(self):
        return asdict(self)
