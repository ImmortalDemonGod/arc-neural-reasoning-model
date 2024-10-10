import os
import torch
from torch.utils.data import DataLoader
from src.data.arc_dataset import ARCDataset  # Adjust the import path if necessary
import psutil
import gc
import json

def get_system_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    mem_mb = mem_bytes / (1024 ** 2)  # Convert to MB
    return mem_mb

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    return 0.0

def initialize_dataset():
    data_source = "path/to/your/data"  # Replace with your actual data source
    dataset = ARCDataset(
        data_source=data_source,
        is_test=False,
        num_symbols=10,
        test_split=0.2,
        debug=True  # Enable debug mode if needed
    )
    return dataset

def test_memory_usage(dataset, batch_size, num_batches=10):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Adjust based on your CPU cores
        pin_memory=True if torch.cuda.is_available() else False
    )

    memory_records = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        # Garbage collect to get accurate measurements
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Measure memory before loading the batch
        mem_before = get_system_memory_usage()
        gpu_before = get_gpu_memory_usage()

        inputs, outputs, task_ids = batch  # Unpack the batch

        # Force GPU synchronization if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure memory after loading the batch
        mem_after = get_system_memory_usage()
        gpu_after = get_gpu_memory_usage()

        # Calculate memory used by the batch
        mem_used = mem_after - mem_before
        gpu_used = gpu_after - gpu_before

        memory_records.append({
            "batch_number": i + 1,
            "batch_size": batch_size,
            "system_memory_mb": mem_used,
            "gpu_memory_mb": gpu_used
        })

        print(f"Batch {i+1}/{num_batches} | Batch Size: {batch_size} | "
              f"System Memory Used: {mem_used:.2f} MB | GPU Memory Used: {gpu_used:.2f} MB")

    return memory_records

def main():
    dataset = initialize_dataset()

    # Define the range of batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32]  # Extend as needed

    all_memory_data = []

    for batch_size in batch_sizes:
        print(f"\nTesting with Batch Size: {batch_size}")
        memory_data = test_memory_usage(dataset, batch_size)
        all_memory_data.extend(memory_data)

    # Save the memory usage data to a JSON file
    output_path = os.path.join("gpt2_arc", "memory_usage_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_memory_data, f, indent=4)
    
    print(f"\nMemory usage data saved to {output_path}")

if __name__ == "__main__":
    main()
