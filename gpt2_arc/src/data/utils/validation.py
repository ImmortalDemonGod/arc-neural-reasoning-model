# gpt2_arc/src/data/utils/validation.py
from typing import Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)

def validate_sample(sample: Dict[str, Any], num_symbols: int) -> bool:
    """Validates a single data sample"""
    try:
        # Check required keys
        required_keys = {"input", "output", "task_id"}
        if not required_keys.issubset(sample.keys()):
            missing = required_keys - sample.keys()
            logger.error(f"Sample missing required keys: {missing}")
            return False

        # Validate tensors
        for key in ["input", "output"]:
            if not isinstance(sample[key], torch.Tensor):
                logger.error(f"'{key}' is not a tensor")
                return False
            
            if sample[key].ndimension() != 3 or sample[key].shape[0] != 1:
                logger.error(f"'{key}' has invalid shape: {sample[key].shape}")
                return False
            
            max_symbol = sample[key].max()
            if max_symbol >= num_symbols:
                logger.error(f"Symbol {max_symbol.item()} exceeds allowed maximum ({num_symbols-1})")
                return False

        # Validate task_id
        if not isinstance(sample["task_id"], str) or not sample["task_id"]:
            logger.error("Invalid task_id")
            return False

        return True
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False