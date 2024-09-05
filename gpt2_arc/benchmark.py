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

if __name__ == "__main__":
    # Load your dataset and model
    # Run the benchmark with multiple batches
    benchmark_model(model, train_dataset, num_batches=10)
    # Load data using arckit
    train_set, _ = arckit.load_data()
    
    # Create the ARCDataset
    full_dataset = ARCDataset(train_set, is_test=False)

    # Create a smaller subset of the dataset for benchmarking
    subset_size = int(0.1 * len(full_dataset))  # Use 10% of the dataset
    train_dataset, _ = torch.utils.data.random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)
    model = GPT2ARC(model_config)

    # Run the benchmark
    benchmark_model(model, train_dataset)
