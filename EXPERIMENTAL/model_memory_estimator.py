import torch
import math
import psutil

def calculate_params(n_layers, n_heads, d_model):
    return n_layers * (12 * d_model * d_model + 13 * d_model) + d_model * 10

def estimate_memory_usage(total_params, batch_size, seq_length, d_model, dtype_size=4):
    model_memory = total_params * dtype_size  # Model parameters
    optimizer_memory = model_memory * 2  # Adam optimizer uses 2x model size
    activations_memory = batch_size * seq_length * d_model * dtype_size * 2  # Forward & backward pass
    total_memory = model_memory + optimizer_memory + activations_memory
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

def test_model_configurations():
    device_info = get_device_info()
    available_memory = get_available_memory()
    
    print(f"Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print(f"Available memory: {available_memory:.2f} GB")

    configurations = [
        {"n_layers": 113, "n_heads": 64, "d_model": 4096, "batch_size": 32, "seq_length": 1024},
        {"n_layers": 96, "n_heads": 32, "d_model": 2048, "batch_size": 64, "seq_length": 512},
        {"n_layers": 48, "n_heads": 16, "d_model": 1024, "batch_size": 128, "seq_length": 256},
    ]

    for config in configurations:
        total_params = calculate_params(config["n_layers"], config["n_heads"], config["d_model"])
        estimated_memory = estimate_memory_usage(total_params, config["batch_size"], config["seq_length"], config["d_model"])
        
        print(f"\nConfiguration: {config}")
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated memory usage: {estimated_memory:.2f} GB")
        
        if can_fit_model(estimated_memory, available_memory):
            print(f"Model should fit in {device_info['device']} memory.")
        else:
            print(f"Warning: Model may be too large for available {device_info['device']} memory!")
        
        print(f"Memory utilization: {(estimated_memory / available_memory) * 100:.2f}%")

if __name__ == "__main__":
    test_model_configurations()
