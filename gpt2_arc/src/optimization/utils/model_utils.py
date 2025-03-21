# gpt2_arc/src/optimization/utils/model_utils.py
import logging
import torch
import psutil
import optuna
from gpt2_arc.src.training.utils.training_manager import TrainingManager
from pytorch_lightning.utilities.model_summary import ModelSummary
import torch
logger = logging.getLogger(__name__)



def estimate_model_memory(hyperparameters, args, config):
    """
    Estimate memory usage for model with given hyperparameters.
    """
    total_mamba_layers = int(hyperparameters['n_layer'] * hyperparameters['mamba_ratio'])
    total_layers = hyperparameters['n_layer'] + total_mamba_layers
    
    total_params = calculate_params(
        n_layers=total_layers,
        n_heads=hyperparameters['n_head'],
        d_model=hyperparameters['n_embd'],
        mamba_ratio=hyperparameters['mamba_ratio'],
        d_state=hyperparameters['d_state'],
        d_conv=hyperparameters['d_conv'],
        mamba_depth=hyperparameters['mamba_depth'],
        mamba_expand=hyperparameters['mamba_expand']
    )
    
    safety_margin = 0.1  # 10% safety margin
    estimated_memory = estimate_memory_usage(
        total_params=total_params,
        batch_size=hyperparameters['batch_size'],
        height=30,  # adjust if necessary based on data
        width=30,   # adjust if necessary
        d_model=hyperparameters['n_embd']
    )
    
    available_memory = get_available_memory()
    estimated_memory *= (1 + safety_margin)
    
    logger.debug(f"Estimated memory usage: {estimated_memory:.2f} GB")
    logger.debug(f"Available memory: {available_memory:.2f} GB")
    
    return estimated_memory, available_memory

def check_memory_constraints(estimated_memory, available_memory, trial):
    if not can_fit_model(estimated_memory, available_memory * 0.8):
        logger.warning(f"Trial {trial.number}: Model too large for available memory. Skipping.")
        raise optuna.exceptions.TrialPruned()

# !Placeholder functions (You need to implement these)!
def calculate_params(n_layers: int, n_heads: int, d_model: int, mamba_ratio: float, d_state: int = 16, d_conv: int = 4, mamba_depth: int = 1, mamba_expand: int = 2) -> int:
    logger.debug(f"Executing calculate_params with mamba_ratio = {mamba_ratio}")
    transformer_params_per_layer = (
        12 * d_model * d_model + 13 * d_model
    )
    
    # Calculate the number of Mamba layers
    total_mamba_layers = int(n_layers * mamba_ratio)
    
    # Calculate parameters for Mamba layers
    # Assuming MambaBlock has parameters based on d_state, d_conv, depth, and expand
    mamba_params_per_layer = (
        d_state * d_conv * mamba_expand * mamba_depth  # Example calculation
    )
    total_mamba_params = total_mamba_layers * mamba_params_per_layer
    
    # Total parameters
    total_params = n_layers * transformer_params_per_layer + total_mamba_params
    logger.debug(f"Total parameters calculated: {total_params}")
    return total_params

def estimate_memory_usage(total_params: int, batch_size: int, height: int, width: int, d_model: int, dtype_size: int = 4) -> float:
    model_memory = total_params * dtype_size  # Model parameters
    optimizer_memory = model_memory * 2  # Adam optimizer uses 2x model size
    input_memory = batch_size * height * width * dtype_size  # Input tensors
    conv_output_memory = batch_size * height * width * d_model * dtype_size  # After conv layer
    activations_memory = batch_size * (height * width) * d_model * dtype_size * 2  # Forward & backward pass
    total_memory = model_memory + optimizer_memory + input_memory + conv_output_memory + activations_memory
    return total_memory / (1024**3)  # Convert to GB

def get_available_memory():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    else:
        return psutil.virtual_memory().total / (1024**3)  # Get actual system memory for CPU

def can_fit_model(estimated_memory: float, available_memory: float, threshold: float = 0.9) -> bool:
    return estimated_memory < available_memory * threshold

def get_device_info():
    if torch.cuda.is_available():
        return {
            "device": "GPU",
            "name": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
            "total_memory": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "cuda_version": torch.version.cuda
        }
    else:
        return {
            "device": "CPU",
            "name": "System CPU",
            "total_memory": psutil.virtual_memory().total / (1024**3),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A"
        }

def estimate_single_configuration(n_layers: int, n_heads: int, d_model: int, batch_size: int, height: int, width: int) -> None:
    device_info = get_device_info()
    available_memory = get_available_memory()
    
    print(f"Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print(f"Available memory: {available_memory:.2f} GB")

    total_params = calculate_params(n_layers, n_heads, d_model)
    estimated_memory = estimate_memory_usage(total_params, batch_size, height, width, d_model)
    
    print(f"\nConfiguration:")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_model: {d_model}")
    print(f"  batch_size: {batch_size}")
    print(f"  input_height: {height}")
    print(f"  input_width: {width}")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated memory usage: {estimated_memory:.2f} GB")
    
    if can_fit_model(estimated_memory, available_memory):
        print(f"Model should fit in {device_info['device']} memory.")
    else:
        print(f"Warning: Model may be too large for available {device_info['device']} memory!")
    
    print(f"Memory utilization: {(estimated_memory / available_memory) * 100:.2f}%")

def create_model(args, config, symbol_freq_dict, hyperparameters):
    """Create model using existing training manager functionality"""
    training_manager = TrainingManager(config, args)
    model = training_manager.initialize_model()
    if args.model_checkpoint:
        model = training_manager.load_checkpoint(model)
    return model

def generate_model_summary(model, trial):
    """Generate model summary using existing functionality"""
    try:
        model_summary = str(ModelSummary(model, max_depth=-1))
        trial.set_user_attr("model_summary", model_summary)
    except Exception as e:
        logger.error(f"Error generating model summary: {e}")
        trial.set_user_attr("model_summary", "Error generating model summary")