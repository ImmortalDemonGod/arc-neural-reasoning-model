# gpt2_arc/src/config.py
from typing import Optional, Dict
from dataclasses import dataclass, asdict, field
import multiprocessing
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

@dataclass
struct ModelConfig:
    var n_embd_multiplier: Int = field(default=1, metadata={"description": "Multiplier for n_head to determine n_embd"})
    var n_embd: Int = 256          # This will be set directly via command-line argument
    var n_head: Int = 2            # Increased from 1 to 2
    var n_layer: Int = 2           # Increased from 12 to 2
    var num_classes: Int = field(default=11, metadata={"description": "Number of output classes for the model."})
    var dropout: Float32 = 0.1
    var mamba_ratio: Float32 = 1.0  # Default set to 1.0 for equal Transformer and Mamba layers
    var d_state: Int = 4
    var d_conv: Int = 1
    var mamba_depth: Int = 1
    var mamba_expand: Int = 2

    fn __post_init__(inout self):
#        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
#        assert self.n_embd >= self.n_head, f"n_embd ({self.n_embd}) must be greater than or equal to n_head ({self.n_head})"
        assert self.n_layer > 0, f"n_layer ({self.n_layer}) must be positive"
        assert self.d_state >= 1, f"d_state ({self.d_state}) must be at least 1"
        assert self.d_conv >= 1, f"d_conv ({self.d_conv}) must be at least 1"
        assert self.mamba_depth >= 1, f"mamba_depth ({self.mamba_depth}) must be at least 1"
        assert self.mamba_expand >= 2, f"mamba_expand ({self.mamba_expand}) must be at least 2"

        # Modify the mamba_ratio validation
        if self.mamba_ratio < 0.0:
            raise ValueError("mamba_ratio must be >= 0.0 to ensure a valid number of MambaLayers")

        logger.debug("ModelConfig initialized successfully with mamba_ratio >= 0.0")

@dataclass
struct TrainingConfig:
    var batch_size: Int = 16             # Reduced from 32 to 16
    var learning_rate: Float32 = 1e-4      # Keep as is or adjust if necessary
    var max_epochs: Int = 1              # Already set for fast dev run
    var num_classes: Int = field(default=11, metadata={"description": "Number of output classes for the model."})
    var num_symbols: Int = 11  # Ensure num_symbols is set to 11
    var num_workers: Int = 1             # Reduced from multiprocessing.cpu_count() to 1
    var symbol_freq: Optional[Dict[]] = None
    var pin_memory: bool = False         # Disable if not using GPU
    var prefetch_factor: Int = 1         # Reduced from 2 to 1
    var persistent_workers: bool = False # Ensure workers do not stay alive
    var use_gpu: bool = True
    var log_level: str = "INFO"
    var use_synthetic_data: bool = False
    var use_grokfast: bool = False
    var grokfast_type: Optional[str] = field(default=None)  # 'ema' or 'ma'
    var grokfast_alpha: Float32 = field(default=0.98)
    var grokfast_lamb: Float32 = field(default=2.0)
    var grokfast_window_size: Optional[Int] = field(default=100)  # Only relevant if grokfast_type == 'ma'
    var balance_symbols: bool = True
    var balancing_method: str = "weighting"
    var synthetic_data_path: Optional[str] = None
    var include_pad_in_loss: bool = True  # Whether to include the padding class in the loss calculation
    var include_pad_in_accuracy: bool = True  # Whether to include the padding class in accuracy calculations
    var tensorboard_log_path: Optional[str] = None  # Default to None if not set

    # New fields for padding symbol
    var pad_symbol: str = "<PAD>"
    var pad_symbol_idx: Int = field(default=10)  # Add this line

    fn __post_init__(inout self):
        # Dynamically set num_classes based on symbol_freq
        self.pad_symbol_idx = 10  # Set to the default padding index
        print(f"TrainingConfig initialized with {self.num_classes} classes and PAD symbol index {self.pad_symbol_idx}")
        print(f"include_pad_in_loss: {self.include_pad_in_loss}")  # Added debug statement
        print(f"include_pad_in_accuracy: {self.include_pad_in_accuracy}")  # Added debug statement

@dataclass
struct EvaluationConfig:
    var perfect_accuracy_threshold: Float32 = 99.9  # Set to 99.9 for near-perfect accuracy

@dataclass
struct Config:
    var model: ModelConfig = field(default_factory=ModelConfig)
    var training: TrainingConfig = field(default_factory=TrainingConfig)
    var evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    var estimated_memory: Optional[Float32] = None
    var available_memory: Optional[Float32] = None

    fn to_dict(inout self):
        return asdict(self)
