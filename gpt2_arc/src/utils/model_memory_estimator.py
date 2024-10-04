# gpt2_arc/src/utils/model_memory_estimator.py
import torch
import math
import psutil

def calculate_params(n_layers, n_heads, d_model, mamba_ratio=0, d_state=16, d_conv=4):
    # Parameters per Transformer layer
    transformer_params_per_layer = (
        12 * d_model * d_model + 13 * d_model
    )
    total_transformer_params = n_layers * transformer_params_per_layer
    
    # Parameters per Mamba layer
    mamba_params_per_layer = (
        4 * d_model * d_model + d_state * d_model + d_conv * d_model
    )
    total_mamba_params = n_layers * mamba_ratio * mamba_params_per_layer
    
    # Total parameters
    total_params = total_transformer_params + total_mamba_params
    return total_params

def estimate_memory_usage(total_params, batch_size, height, width, d_model, dtype_size=4):
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

def can_fit_model(estimated_memory, available_memory, threshold=0.9):
    return estimated_memory < available_memory * threshold

def estimate_single_configuration(n_layers, n_heads, d_model, batch_size, height, width):
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
import torch
import math
import psutil


def estimate_memory_usage(total_params, batch_size, height, width, d_model, dtype_size=4):
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

def can_fit_model(estimated_memory, available_memory, threshold=0.9):
    return estimated_memory < available_memory * threshold

def estimate_single_configuration(n_layers, n_heads, d_model, batch_size, height, width):
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
