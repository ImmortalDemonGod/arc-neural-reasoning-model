
import logging
import arckit
from gpt2_arc.src.data.arc_dataset import ARCDataset

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load data using arckit
    try:
        train_set, eval_set = arckit.load_data()
        logger.info("Data loaded successfully using arckit.load_data()")
    except Exception as e:
        logger.error(f"Failed to load data using arckit.load_data(): {e}")
        return
    
    # Initialize training dataset
    try:
        train_dataset = ARCDataset(data_source=train_set, is_test=False, debug=True)
        logger.info("Train ARCDataset initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Train ARCDataset: {e}")
        return
    
    # Initialize evaluation dataset
    try:
        eval_dataset = ARCDataset(data_source=eval_set, is_test=True, debug=True)
        logger.info("Eval ARCDataset initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Eval ARCDataset: {e}")
        return
    
    # Basic Verification: Total Number of Samples
    # Basic Verification: Total Number of Samples
    train_total_samples = len(train_dataset)
    logger.info(f"Total number of training samples in dataset: {train_total_samples}")
    
    eval_total_samples = len(eval_dataset)
    logger.info(f"Total number of evaluation samples in dataset: {eval_total_samples}")
    
    # Basic Verification: Inspect First 5 Training Samples
    num_train_samples_to_inspect = min(train_total_samples, 5)
    logger.info(f"Inspecting the first {num_train_samples_to_inspect} training samples:")
    for i in range(num_train_samples_to_inspect):
        try:
            input_tensor, output_tensor, task_id = train_dataset[i]
            logger.debug(f"Training Sample {i + 1}:")
            logger.debug(f"  Task ID: {task_id}")
            logger.debug(f"  Input Tensor Shape: {input_tensor.shape}")
            logger.debug(f"  Output Tensor Shape: {output_tensor.shape}")
        except Exception as e:
            logger.error(f"Error accessing sample {i}: {e}")
    
    # Basic Verification: Inspect First 5 Evaluation Samples
    num_eval_samples_to_inspect = min(5, eval_total_samples)
    logger.info(f"Inspecting the first {num_eval_samples_to_inspect} evaluation samples:")
    for i in range(num_eval_samples_to_inspect):
        try:
            input_tensor, output_tensor, task_id = eval_dataset[i]
            logger.debug(f"Evaluation Sample {i + 1}:")
            logger.debug(f"  Task ID: {task_id}")
            logger.debug(f"  Input Tensor Shape: {input_tensor.shape}")
            logger.debug(f"  Output Tensor Shape: {output_tensor.shape}")
        except Exception as e:
            logger.error(f"Error accessing evaluation sample {i}: {e}")
    
    # Basic Verification: Symbol Frequencies for Training Dataset
    try:
        train_symbol_freq = train_dataset.get_symbol_frequencies()
        logger.info("Training Symbol Frequencies:")
        for symbol, freq in train_symbol_freq.items():
            logger.info(f"  Symbol {symbol}: Frequency = {freq:.4f}")
    except Exception as e:
        logger.error(f"Failed to compute symbol frequencies: {e}")
    
    # Basic Verification: Symbol Frequencies for Evaluation Dataset
    try:
        eval_symbol_freq = eval_dataset.get_symbol_frequencies()
        logger.info("Evaluation Symbol Frequencies:")
        for symbol, freq in eval_symbol_freq.items():
            logger.info(f"  Symbol {symbol}: Frequency = {freq:.4f}")
    except Exception as e:
        logger.error(f"Failed to compute evaluation symbol frequencies: {e}")
    
    # Basic Verification: Grid Size Statistics for Training Dataset
    try:
        train_grid_stats = train_dataset.get_grid_size_stats()
        logger.info(f"Training Grid Size Statistics: {train_grid_stats}")
    except Exception as e:
        logger.error(f"Failed to compute training grid size statistics: {e}")
    
    # Basic Verification: Grid Size Statistics for Evaluation Dataset
    try:
        eval_grid_stats = eval_dataset.get_grid_size_stats()
        logger.info(f"Evaluation Grid Size Statistics: {eval_grid_stats}")
    except Exception as e:
        logger.error(f"Failed to compute evaluation grid size statistics: {e}")

if __name__ == "__main__":
    main()
