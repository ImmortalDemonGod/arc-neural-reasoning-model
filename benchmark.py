import torch
from torch.utils.data import DataLoader
from src.data.arc_dataset import ARCDataset
from src.models.gpt2 import GPT2ARC
from src.config import ModelConfig
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_model(model, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=ARCDataset.collate_fn)
    inputs, attention_mask, _ = next(iter(dataloader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, attention_mask = inputs.to(device), attention_mask.to(device)
    model.to(device)

    # Measure the time taken to process the batch
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        model(inputs, attention_mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    time_taken = end_time - start_time
    grids_per_second = len(inputs) / time_taken

    logger.info(f"Time taken for batch: {time_taken:.4f} seconds")
    logger.info(f"Grids per second: {grids_per_second:.2f}")

if __name__ == "__main__":
    # Load your dataset and model
    dataset = ARCDataset(...)  # Provide appropriate arguments
    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)
    model = GPT2ARC(model_config)

    # Run the benchmark
    benchmark_model(model, dataset)
