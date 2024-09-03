# gpt2_arc/src/config.py
from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 10
    use_gpu: bool = True


from dataclasses import field


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
