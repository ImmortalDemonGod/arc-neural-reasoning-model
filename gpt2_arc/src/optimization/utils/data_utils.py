# gpt2_arc/src/optimization/utils/data_utils.py
import logging
import numpy as np
import torch
from gpt2_arc.src.training.utils.data_manager import DataManager

logger = logging.getLogger(__name__)

def load_datasets(trial, args, config, all_synthetic_data):
    """
    Load training, validation and test datasets using DataManager.
    
    Args:
        trial: Optuna trial object
        args: Command line arguments
        config: Model configuration
        all_synthetic_data: Pre-loaded synthetic data if using synthetic data
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # Initialize data manager
    data_manager = DataManager(config=config, args=args)
    
    if args.use_synthetic_data:
        train_data = all_synthetic_data['train_dataset']
    else:
        # Load full datasets using data manager's methods
        train_data, val_data, test_data = data_manager.load_all_datasets()

        # Log dataset sizes
        logger.debug(f"trial {trial.number}: loaded {len(train_data)} training samples.")
        logger.debug(f"trial {trial.number}: loaded {len(val_data)} validation samples.") 
        logger.debug(f"trial {trial.number}: loaded {len(test_data)} test samples.")

        return train_data, val_data, test_data

def calculate_symbol_frequencies(args, config, train_data):
    """
    Calculate symbol frequencies for the training data.
    
    Args:
        args: Command line arguments
        config: Model configuration  
        train_data: Training dataset
        
    Returns:
        tuple: (symbol_freq_dict, balance_symbols, balancing_method)
    """
    symbol_freq_dict = {}

    if args.fast_dev_run:
        balance_symbols = False
        balancing_method = "none"
        logger.debug("fast_dev_run enabled. disabling symbol frequency balancing.")
    else:
        if args.enable_symbol_freq:
            logger.debug("calculating symbol frequencies.")
            symbol_freq = train_data.get_symbol_frequencies()
            symbol_freq_dict = process_symbol_freq(symbol_freq, config)
            balance_symbols = True
            balancing_method = "weighting"
        else:
            logger.debug("symbol frequency calculation disabled.")
            balance_symbols = False
            balancing_method = "none"

    return symbol_freq_dict, balance_symbols, balancing_method

def process_symbol_freq(symbol_freq, config):
    """
    Process raw symbol frequencies into a standardized dictionary format.
    
    Args:
        symbol_freq: Raw symbol frequencies (numpy array or dict)
        config: Model configuration
        
    Returns:
        dict: Processed symbol frequency dictionary
    """
    # Ensure symbol_freq_dict is a dictionary
    if isinstance(symbol_freq, np.ndarray):
        symbol_freq_dict = {i: float(freq) for i, freq in enumerate(symbol_freq)}
        logger.debug("converted symbol_freq numpy array to dictionary.")
    elif isinstance(symbol_freq, dict):
        symbol_freq_dict = symbol_freq.copy()
        logger.debug("copied symbol_freq dictionary.")
    else:
        raise TypeError(
            f"unexpected type for symbol_freq: {type(symbol_freq)}. Expected dict or np.ndarray."
        )

    # Remove padding symbol
    pad_symbol_idx = config.training.pad_symbol_idx
    symbol_freq_dict.pop(pad_symbol_idx, None)
    logger.debug(f"removed pad_symbol_idx ({pad_symbol_idx}) from symbol_freq_dict.")

    # Validate symbol_freq_dict length
    num_classes = config.training.num_classes
    if len(symbol_freq_dict) != num_classes - 1:
        raise ValueError(
            f"length of symbol_freq_dict ({len(symbol_freq_dict)}) must match num_classes minus padding ({num_classes - 1})."
        )

    return symbol_freq_dict