import sys
import os

# Determine the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to the Python path
sys.path.insert(0, project_root)

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
import json

def get_system_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    mem_mb = mem_bytes / (1024 ** 2)  # Convert to MB
    return mem_mb

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

    # Save the memory usage data to a JSON file
    memory_data = {
        "total_memory_mb": mem_used,
        "total_samples": total_samples,
        "memory_per_example_kb": per_example_mem
    }
    output_path = os.path.join("gpt2_arc", "memory_usage_results.json")
    with open(output_path, 'w') as f:
        json.dump(memory_data, f, indent=4)

    print(f"\nMemory usage data saved to {output_path}")

if __name__ == "__main__":
    main()
