# gpt2_arc/src/optimization/utils/config_utils.py
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.optimization.utils.hyperparameter_utils import validate_hyperparameters

def initialize_config():
    model_config = ModelConfig()
    training_config = TrainingConfig()
    config = Config(model=model_config, training=training_config)
    return config

def validate_and_adjust_hyperparameters(hyperparameters):
    n_embd = hyperparameters['n_embd']
    n_head = hyperparameters['n_head']
    n_layer = hyperparameters['n_layer']
    mamba_ratio = hyperparameters['mamba_ratio']
    d_state = hyperparameters['d_state']
    d_conv = hyperparameters['d_conv']
    dropout = hyperparameters['dropout']
    validate_hyperparameters(n_embd, n_head, n_layer, mamba_ratio, d_state, d_conv, dropout)