import torch
import torch_xla.core.xla_model as xm
import math

def calculate_params(n_layers, n_heads, d_model):
    return n_layers * (12 * d_model * d_model + 13 * d_model) + d_model * 10

def estimate_memory_usage(total_params, batch_size, seq_length, dtype_size=4):
    model_memory = total_params * dtype_size  # Model parameters
    optimizer_memory = model_memory * 2  # Adam optimizer uses 2x model size
    activations_memory = batch_size * seq_length * d_model * dtype_size * 2  # Forward & backward pass
    total_memory = model_memory + optimizer_memory + activations_memory
    return total_memory / (1024**3)  # Convert to GB

def get_available_memory():
    device = xm.xla_device()
    mem_info = xm.get_memory_info(device)
    return mem_info['kb_free'] / (1024**2)  # Convert to GB

def can_fit_model(estimated_memory, available_memory, threshold=0.9):
    return estimated_memory < available_memory * threshold

def test_model_configurations():
    available_memory = get_available_memory()
    print(f"Available TPU memory: {available_memory:.2f} GB")

    configurations = [
        {"n_layers": 113, "n_heads": 64, "d_model": 4096, "batch_size": 32, "seq_length": 1024},
        {"n_layers": 96, "n_heads": 32, "d_model": 2048, "batch_size": 64, "seq_length": 512},
        {"n_layers": 48, "n_heads": 16, "d_model": 1024, "batch_size": 128, "seq_length": 256},
    ]

    for config in configurations:
        total_params = calculate_params(config["n_layers"], config["n_heads"], config["d_model"])
        estimated_memory = estimate_memory_usage(total_params, config["batch_size"], config["seq_length"])
        
        print(f"\nConfiguration: {config}")
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated GPU memory usage: {estimated_memory:.2f} GB")
        
        if can_fit_model(estimated_memory, available_memory):
            print("Model should fit in TPU memory.")
        else:
            print("Warning: Model may be too large for available TPU memory!")

if __name__ == "__main__":
    test_model_configurations()
