
import logging
from gpt2_arc.src.data.arc_dataset import ARCDataset

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Specify the path to your arckit data
    data_path = '/path/to/arckit/data'  # ➡️ **Replace this with your actual data path**
    
    # Initialize the ARCDataset
    try:
        dataset = ARCDataset(data_source=data_path, is_test=False, debug=True)
        logger.info("ARCDataset initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ARCDataset: {e}")
        return
    
    # Basic Verification: Total Number of Samples
    total_samples = len(dataset)
    logger.info(f"Total number of samples in dataset: {total_samples}")
    
    # Basic Verification: Inspect First 5 Samples
    num_samples_to_inspect = min(5, total_samples)
    logger.info(f"Inspecting the first {num_samples_to_inspect} samples:")
    for i in range(num_samples_to_inspect):
        try:
            input_tensor, output_tensor, task_id = dataset[i]
            logger.debug(f"Sample {i + 1}:")
            logger.debug(f"  Task ID: {task_id}")
            logger.debug(f"  Input Tensor Shape: {input_tensor.shape}")
            logger.debug(f"  Output Tensor Shape: {output_tensor.shape}")
        except Exception as e:
            logger.error(f"Error accessing sample {i}: {e}")
    
    # Basic Verification: Symbol Frequencies
    try:
        symbol_freq = dataset.get_symbol_frequencies()
        logger.info("Symbol Frequencies:")
        for symbol, freq in symbol_freq.items():
            logger.info(f"  Symbol {symbol}: Frequency = {freq:.4f}")
    except Exception as e:
        logger.error(f"Failed to compute symbol frequencies: {e}")
    
    # Basic Verification: Grid Size Statistics
    try:
        grid_stats = dataset.get_grid_size_stats()
        logger.info(f"Grid Size Statistics: {grid_stats}")
    except Exception as e:
        logger.error(f"Failed to compute grid size statistics: {e}")

if __name__ == "__main__":
    main()
