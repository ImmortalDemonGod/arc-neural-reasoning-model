from typing import Optional
# gpt2_arc/src/config.py
from dataclasses import dataclass, asdict, field
from typing import Optional
import multiprocessing

@dataclass
class ModelConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    mamba_ratio: float = 1.0  # Number of Mamba layers per Transformer layer
    d_state: int = 16     # Mamba state dimension
    d_conv: int = 4       # Mamba convolution dimension
    mamba_depth: int = 1  # Depth of each Mamba layer
    mamba_expand: int = 2  # Expand factor for each Mamba layer

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.n_embd >= self.n_head, f"n_embd ({self.n_embd}) must be greater than or equal to n_head ({self.n_head})"
        assert self.n_layer > 0, f"n_layer ({self.n_layer}) must be positive"
        assert self.d_state >= 1, f"d_state ({self.d_state}) must be at least 1"
        assert self.d_conv >= 1, f"d_conv ({self.d_conv}) must be at least 1"
        assert self.mamba_depth >= 1, f"mamba_depth ({self.mamba_depth}) must be at least 1"
        assert self.mamba_expand >= 2, f"mamba_expand ({self.mamba_expand}) must be at least 2"

from dataclasses import dataclass, field
import multiprocessing

@dataclass
class TrainingConfig:
    # Grokfast-specific parameters
    use_grokfast: bool = False
    grokfast_type: Optional[str] = field(default=None)  # 'ema' or 'ma'
    grokfast_alpha: float = field(default=0.98)
    grokfast_lamb: float = field(default=2.0)
    grokfast_window_size: Optional[int] = field(default=100)  # Only relevant if grokfast_type == 'ma'
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 10
    num_workers: int = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() else 4
    symbol_freq: Optional[dict] = None
    prefetch_factor: int = 2
    persistent_workers: bool = True
    use_gpu: bool = True
    log_level: str = "INFO"
    use_synthetic_data: bool = False
    balance_symbols: bool = True  # Enable balancing
    balancing_method: str = "weighting"  # Options: "weighting", "oversampling"
    synthetic_data_path: Optional[str] = None

@dataclass
class EvaluationConfig:
    perfect_accuracy_threshold: float = 95  # Example: Change to 95

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    estimated_memory: Optional[float] = None
    available_memory: Optional[float] = None

    def to_dict(self):
        return asdict(self)
