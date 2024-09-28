import torch
import math
import psutil
import argparse

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

def estimate_single_configuration(n_layers, n_heads, d_model, batch_size, seq_length):
    device_info = get_device_info()
    available_memory = get_available_memory()
    
    print(f"Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print(f"Available memory: {available_memory:.2f} GB")

    total_params = calculate_params(n_layers, n_heads, d_model)
    estimated_memory = estimate_memory_usage(total_params, batch_size, seq_length, d_model)
    
    print(f"\nConfiguration:")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_model: {d_model}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_length: {seq_length}")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated memory usage: {estimated_memory:.2f} GB")
    
    if can_fit_model(estimated_memory, available_memory):
        print(f"Model should fit in {device_info['device']} memory.")
    else:
        print(f"Warning: Model may be too large for available {device_info['device']} memory!")
    
    print(f"Memory utilization: {(estimated_memory / available_memory) * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Estimate memory usage for a transformer model configuration.")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of layers in the model")
    parser.add_argument("--n_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--d_model", type=int, required=True, help="Dimension of the model")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--seq_length", type=int, required=True, help="Sequence length for input")
    
    args = parser.parse_args()
    
    estimate_single_configuration(args.n_layers, args.n_heads, args.d_model, args.batch_size, args.seq_length)

if __name__ == "__main__":
    main()
