# gpt2_arc/src/config.py
from dataclasses import dataclass, asdict


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
    log_level: str = "INFO"  # Add log_level attribute with default value
    optimizer_name: str = "Adam"  # Add optimizer_name attribute


from dataclasses import field


@dataclass
class Config:
    def to_dict(self):
        return asdict(self)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
