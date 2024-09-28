# gpt2_arc/src/config.py
from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass
class ModelConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.n_embd >= self.n_head, f"n_embd ({self.n_embd}) must be greater than or equal to n_head ({self.n_head})"
        assert self.n_layer > 0, f"n_layer ({self.n_layer}) must be positive"

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 10
    use_gpu: bool = True
    log_level: str = "INFO"
    use_synthetic_data: bool = False
    synthetic_data_path: Optional[str] = None

@dataclass
class EvaluationConfig:
    perfect_accuracy_threshold: float = 98

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self):
        return asdict(self)
