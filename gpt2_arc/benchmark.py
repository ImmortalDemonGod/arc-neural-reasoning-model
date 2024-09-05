import torch
from torch.utils.data import DataLoader
import arckit
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import ModelConfig
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Baseline values for comparison
BASELINE_TOTAL_TIME = 1.9902
BASELINE_GRIDS_PER_SECOND = 160.79

def benchmark_model(model, dataset, batch_size=32, num_batches=10):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=ARCDataset.collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_time = 0.0
    total_grids = 0

    for i, (inputs, outputs) in enumerate(dataloader):
        if i >= num_batches:
            break

        # Create a dummy attention mask (all ones)
        attention_mask = torch.ones(inputs.size(0), inputs.size(2) * inputs.size(3), dtype=torch.float32)
        inputs, attention_mask = inputs.to(device), attention_mask.to(device)

        # Measure the time taken to process the batch
        start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            model(inputs, attention_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        batch_time = end_time - start_time
        total_time += batch_time
        total_grids += len(inputs)

        logger.info(f"Time taken for batch {i+1}: {batch_time:.4f} seconds")

    average_time = total_time / num_batches
    grids_per_second = total_grids / total_time

    logger.info(f"Average time per batch: {average_time:.4f} seconds")
    logger.info(f"Average grids per second: {grids_per_second:.2f}")
    logger.info(f"Total time for {num_batches} batches: {total_time:.4f} seconds")
    logger.info(f"Total grids processed: {total_grids}")

    # Compare with baseline
    time_improvement = total_time - BASELINE_TOTAL_TIME
    grids_per_second_improvement = grids_per_second - BASELINE_GRIDS_PER_SECOND

    time_improvement_percent = (time_improvement / BASELINE_TOTAL_TIME) * 100
    grids_per_second_improvement_percent = (grids_per_second_improvement / BASELINE_GRIDS_PER_SECOND) * 100

    if abs(time_improvement_percent) < 5:
        logger.info("No improvement in total time")
    else:
        logger.info(f"Improvement in total time: {time_improvement:.4f} seconds ({time_improvement_percent:.2f}%)")

    if abs(grids_per_second_improvement_percent) < 5:
        logger.info("No improvement in grids per second")
    else:
        logger.info(f"Improvement in grids per second: {grids_per_second_improvement:.2f} ({grids_per_second_improvement_percent:.2f}%)")

    return total_time, grids_per_second

if __name__ == "__main__":
    # Load your dataset and model
    # Load data using arckit
    train_set, _ = arckit.load_data()
    
    # Create the ARCDataset
    full_dataset = ARCDataset(train_set, is_test=False)

    # Use the full dataset for benchmarking
    train_dataset = full_dataset
    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)
    model = GPT2ARC(model_config)

    # Run the benchmark with multiple batches
    # Run the benchmark with multiple batches
    total_time, grids_per_second = benchmark_model(model, train_dataset, num_batches=10)

    # Log the results
    logger.info(f"Total time for 10 batches: {total_time:.4f} seconds")
    logger.info(f"Average grids per second: {grids_per_second:.2f}")
