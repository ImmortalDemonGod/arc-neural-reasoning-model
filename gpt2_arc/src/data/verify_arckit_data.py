# gpt2_arc/src/data/verify_arckit_data.py
import argparse
import logging
import arckit
from gpt2_arc.src.data.arc_dataset import ARCDataset

SYNTHETIC_DATA_PATH = "/workspaces/arc-neural-reasoning-model/gpt2_arc/src/data/SyntheticARC/task_small"

def main(args):
    # Set up logging
    # Determine logging level based on --debug flag
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load synthetic dataset
    try:
        synthetic_dataset = ARCDataset(
            data_source=args.synthetic_data_path,
            is_test=False,  # Assuming synthetic data is for training; set to True if for testing
            use_cache=not args.no_cache  # Pass the use_cache flag
        )
        logger.info("Synthetic ARCDataset initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Synthetic ARCDataset: {e}")
        return
    try:
        train_set, eval_set = arckit.load_data()
        logger.info("Data loaded successfully using arckit.load_data()")
    except Exception as e:
        logger.error(f"Failed to load data using arckit.load_data(): {e}")
        return
    
    # Initialize training dataset
    try:
        train_dataset = ARCDataset(
            data_source=train_set,
            is_test=False,
            use_cache=not args.no_cache  # Pass the use_cache flag
        )
        logger.info("Train ARCDataset initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Train ARCDataset: {e}")
        return
    
    # Initialize evaluation dataset
    try:
        eval_dataset = ARCDataset(
            data_source=eval_set,
            is_test=True,
            use_cache=not args.no_cache  # Pass the use_cache flag
        )
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
            sample = train_dataset[i]
            input_tensor, output_tensor, task_id = sample
            logger.debug(f"Training Sample {i + 1}:")
            logger.debug(f"  Type of sample: {type(sample)}")
            logger.debug(f"  Sample content: {sample}")
            logger.debug(f"  Task ID: {task_id}")
            logger.debug(f"  Input Tensor Type: {type(input_tensor)}, Shape: {input_tensor.shape}")
            logger.debug(f"  Output Tensor Type: {type(output_tensor)}, Shape: {output_tensor.shape}")
        except Exception as e:
            logger.error(f"Error accessing sample {i}: {e}")
    
    # Basic Verification: Inspect First 5 Evaluation Samples
    num_eval_samples_to_inspect = min(5, eval_total_samples)
    logger.info(f"Inspecting the first {num_eval_samples_to_inspect} evaluation samples:")
    for i in range(num_eval_samples_to_inspect):
        try:
            sample = eval_dataset[i]
            input_tensor, output_tensor, task_id = sample
            logger.debug(f"Evaluation Sample {i + 1}:")
            logger.debug(f"  Type of sample: {type(sample)}")
            logger.debug(f"  Sample content: {sample}")
            logger.debug(f"  Task ID: {task_id}")
            logger.debug(f"  Input Tensor Type: {type(input_tensor)}, Shape: {input_tensor.shape}")
            logger.debug(f"  Output Tensor Type: {type(output_tensor)}, Shape: {output_tensor.shape}")
        except Exception as e:
            logger.error(f"Error accessing evaluation sample {i}: {e}")
    
    # Basic Verification: Total Number of Synthetic Samples
    synthetic_total_samples = len(synthetic_dataset)
    logger.info(f"Total number of synthetic training samples in dataset: {synthetic_total_samples}")
    
    # Basic Verification: Inspect First 5 Synthetic Training Samples
    num_synthetic_samples_to_inspect = min(synthetic_total_samples, 5)
    logger.info(f"Inspecting the first {num_synthetic_samples_to_inspect} synthetic training samples:")
    for i in range(num_synthetic_samples_to_inspect):
        try:
            sample = synthetic_dataset[i]
            if isinstance(sample, tuple) and len(sample) == 3:
                input_tensor, output_tensor, task_id = sample
                logger.debug(f"Synthetic Training Sample {i + 1}:")
                logger.debug(f"  Type of sample: {type(sample)}")
                logger.debug(f"  Sample content: {sample}")
                logger.debug(f"  Task ID: {task_id}")
                logger.debug(f"  Input Tensor Type: {type(input_tensor)}, Shape: {input_tensor.shape}")
                logger.debug(f"  Output Tensor Type: {type(output_tensor)}, Shape: {output_tensor.shape}")
            else:
                logger.warning(f"Synthetic Sample {i + 1} has unexpected format: {type(sample)} with content: {sample}")
        except Exception as e:
            logger.error(f"Error accessing synthetic training sample {i}: {e}")
    
    # Basic Verification: Symbol Frequencies for Synthetic Training Dataset
    try:
        synthetic_symbol_freq = synthetic_dataset.get_symbol_frequencies()
        logger.info("Synthetic Training Symbol Frequencies:")
        for symbol, freq in synthetic_symbol_freq.items():
            logger.info(f"  Symbol {symbol}: Frequency = {freq:.4f}")
    except Exception as e:
        logger.error(f"Failed to compute synthetic training symbol frequencies: {e}")
    
    # Basic Verification: Grid Size Statistics for Synthetic Training Dataset
    try:
        synthetic_grid_stats = synthetic_dataset.get_grid_size_stats()
        logger.info(f"Synthetic Training Grid Size Statistics: {synthetic_grid_stats}")
    except Exception as e:
        logger.error(f"Failed to compute synthetic training grid size statistics: {e}")
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
    parser = argparse.ArgumentParser(description="Verify ARC and Synthetic Datasets")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    parser.add_argument(
        "--synthetic_data_path",
        type=str,
        default=SYNTHETIC_DATA_PATH,
        help="Path to the synthetic data directory"
    )
    
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching of the dataset"
    )
    
    args = parser.parse_args()
    
    main(args)
