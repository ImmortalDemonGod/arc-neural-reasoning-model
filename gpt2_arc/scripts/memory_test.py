import sys
import os

# Determine the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to the Python path
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
from gpt2_arc.src.data.arc_dataset import ARCDataset
import psutil
from arckit import load_data
import gc
from memory_profiler import profile
import json

def get_system_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    mem_mb = mem_bytes / (1024 ** 2)  # Convert to MB
    return mem_mb

@profile
def initialize_dataset():
    # Load data using arckit.load_data() as in training.py
    train_set, eval_set = load_data()
    
    # Initialize ARCDataset with the training set
    dataset = ARCDataset(
        data_source=train_set,
        is_test=False,
        num_symbols=10,
        test_split=0.2,
        debug=True  # Enable debug mode if needed
    )
    return dataset

@profile
def main():
    # Perform garbage collection to ensure accurate measurements
    gc.collect()

    # Measure memory before loading the dataset
    mem_before = get_system_memory_usage()
    print(f"Memory before loading dataset: {mem_before:.2f} MB")

    # Initialize the dataset
    dataset = initialize_dataset()

    # Measure memory after loading the dataset
    mem_after = get_system_memory_usage()
    print(f"Memory after loading dataset: {mem_after:.2f} MB")

    # Calculate memory used by the dataset
    mem_used = mem_after - mem_before
    print(f"Total Memory Used by Dataset: {mem_used:.2f} MB")

    # Calculate per-example memory usage
    total_samples = len(dataset)
    if total_samples == 0:
        print("Dataset is empty.")
        return

    per_example_mem = mem_used / total_samples
    print(f"Estimated Memory Usage per Example: {per_example_mem:.2f} KB")

    # Define the range of batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32]  # Extend as needed

    all_memory_data = []

    for batch_size in batch_sizes:
        print(f"\nTesting with Batch Size: {batch_size}")
        # Perform garbage collection before each batch test
        gc.collect()

        # Measure memory before processing the batch
        mem_before_batch = get_system_memory_usage()

        # Initialize DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Ensure main process data loading
            pin_memory=False  # Disable pin_memory for CPU
        )

        # Iterate through a subset of batches to measure memory
        num_batches = 10
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs, outputs, task_ids = batch  # Unpack the batch

            # Optionally, validate tensor shapes
            print(f"Batch {i+1}: Inputs shape: {inputs.shape}, Outputs shape: {outputs.shape}")

        # Measure memory after processing the batch
        mem_after_batch = get_system_memory_usage()
        mem_used_batch = mem_after_batch - mem_before_batch

        memory_records = {
            "batch_size": batch_size,
            "memory_used_mb": mem_used_batch
        }

        all_memory_data.append(memory_records)

        print(f"Batch Size: {batch_size} | Memory Used: {mem_used_batch:.2f} MB")

    # Save the memory usage data to a JSON file
    output_path = os.path.join("gpt2_arc", "memory_usage_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_memory_data, f, indent=4)
    
    print(f"\nMemory usage data saved to {output_path}")

if __name__ == "__main__":
    main()
