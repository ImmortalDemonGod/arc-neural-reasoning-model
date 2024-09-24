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


from dataclasses import field


@dataclass
class EvaluationConfig:
    perfect_accuracy_threshold: float = 1.0 - 1e-6  # Default to almost 1.0 to account for floating-point precision

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self):
        return asdict(self)
