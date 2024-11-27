# gp2_arc/src/data/arc_dataset.py
# PHASE 1 Refactor ASSEMENT
import os
import random
from typing import Union, List, Dict, Tuple, Any, Optional
from typing import List, Dict, Union, Optional
import json
from cysimdjson import JSONParser, JSONArray
import numpy as np
import pickle
import hashlib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
import logging
import ijson  # Import ijson for streaming JSON parsing
import json  # Temporarily switch to json for better error messages
from tqdm import tqdm  # Import tqdm for progress bars
import sys  # Import sys for handling tqdm output
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
import multiprocessing  # To determine CPU count
from threading import Lock
from jsonschema import validate, ValidationError
from torch.utils.data import get_worker_info
from .config.arc_dataset_config import ARCDatasetConfig  # Fixed import
from .utils.validation import validate_sample
from .utils.grid_ops import GridOperations
from .utils.custom_exceptions import ARCDatasetError, DataLoadingError, ValidationError, ResourceError
from arckit.data import TaskSet, Task
import logging
logger = logging.getLogger(__name__)
from .loaders import create_loader
from .utils.statistics import DatasetStatistics
from .cache import ARCCache

# Create a handler that writes to stderr
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# Determine debug mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "False") == "True"

# Set logging levels based on DEBUG_MODE
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)

# Define JSON Schema for task validation
TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "input": {
            "type": "array",
            "items": {"type": "number"}
        },
        "output": {
            "type": "array",
            "items": {"type": "number"}
        }
    },
    "required": ["input", "output"],
    "additionalProperties": False
}

def set_debug_mode(debug: bool = False) -> None:
    if debug:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
        handler.setLevel(logging.ERROR)

class ARCDataset(Dataset):
    def __init__(self, config: ARCDatasetConfig):
        # Define acceptable key names for input and output
        self.INPUT_KEYS = ['input', 'inputs']
        self.OUTPUT_KEYS = ['output', 'outputs']

        # Transfer config values to instance variables
        self.config = config  # Store full config for cache
        self.is_test = config.is_test
        self.num_symbols = config.num_symbols
        self.test_split = config.test_split
        self.pad_symbol_idx = config.pad_symbol_idx
        self.mamba_ratio = config.mamba_ratio
        self.symbol_freq = config.symbol_freq if config.symbol_freq is not None else {}
        self.data_source = config.data_source
        self.max_samples = config.max_samples

        logger.debug(f"Initialized ARCDataset with pad_symbol_idx: {self.pad_symbol_idx}")
        if self.symbol_freq:
            logger.debug("Symbol frequencies provided; initializing WeightedRandomSampler.")
        else:
            logger.debug("No symbol frequencies provided.")
        self.data_files = []
        self.num_samples = 0

        self.json_parser = JSONParser()  # Initialize the JSON parser here

        # Initialize cache only if enabled
        if config.enable_caching:
            self.cache = ARCCache(config.cache_dir)
            cache_path = self.cache.generate_path(config)
            cache_result = self.cache.load(cache_path)

            if cache_result is not None:
                self.data, self.statistics = cache_result
                self.num_samples = len(self.data)
                logger.debug(f"Loaded {self.num_samples} samples from cache")
            else:
                self._load_and_process_data(config)
        else:
            logger.debug("Caching disabled, loading data from source")
            self._load_and_process_data(config)

        if self.symbol_freq:
            # Calculate weights for each sample based on symbol frequencies
            weights = []
            for sample in self.data:
                input_freq = torch.tensor([self.symbol_freq.get(symbol.item(), 0.0) for symbol in sample["input"].flatten()])
                output_freq = torch.tensor([self.symbol_freq.get(symbol.item(), 0.0) for symbol in sample["output"].flatten()])
                sample_freq = torch.cat((input_freq, output_freq)).mean()
                weights.append(1.0 / (sample_freq + 1e-8))  # Add epsilon to avoid division by zero

            self.sample_weights = torch.tensor(weights, dtype=torch.float)
            self.sampler = WeightedRandomSampler(self.sample_weights, num_samples=len(self.sample_weights), replacement=True)
            logger.debug("WeightedRandomSampler initialized based on symbol frequencies.")
        else:
            self.sampler = None
            logger.debug("No symbol frequencies provided; sampler not initialized.")

        if DEBUG_MODE:
            logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
            logger.debug("Debug mode is enabled for ARCDataset.")
        else:
            logger.setLevel(logging.INFO)
            handler.setLevel(logging.INFO)

        logger.debug("ARCDataset initialization completed.")
    
    def _load_and_process_data(self, config: ARCDatasetConfig):
        """Load and process data from source, computing statistics."""
        try:
            logger.debug("Loading data from data source")
            self.data = self._load_data(self.data_source)

            if not self.data:
                logger.error("No valid samples loaded. Ensure that all samples have 'input' and 'output' keys.")
                raise ValueError("No valid samples loaded. Ensure that all samples have 'input' and 'output' keys.")

            logger.debug(f"Data loaded successfully from data source with {len(self.data)} samples.")

            # Apply sample limit if specified
            if self.max_samples is not None:
                self.data = self.data[:self.max_samples]
                logger.debug(f"Limited dataset to {self.max_samples} samples.")

            self.num_samples = len(self.data)

            # Compute statistics
            logger.info("Starting computation of dataset statistics.")
            stats_computer = DatasetStatistics(self.num_symbols)
            self.statistics = stats_computer.compute_all_statistics(self.data)

            if 'grid_size_stats' in self.statistics:
                self.max_grid_size = (
                    self.statistics['grid_size_stats']['max_height'],
                    self.statistics['grid_size_stats']['max_width']
                )

            # Save to cache if enabled
            if config.enable_caching:
                self.cache.save(config, self.data, self.statistics)
                logger.info("Dataset statistics have been cached successfully.")

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise

    def _load_data(self, data_source: Union[str, List[Dict], 'TaskSet']) -> List[Dict]:
        """Load dataset using appropriate loader based on data source type."""
        try:
            loader = create_loader(data_source, self.num_symbols, self.pad_symbol_idx)
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _validate_data(self):
        """Validates the dataset samples"""
        for idx, sample in enumerate(self.data):
            if not validate_sample(sample, self.num_symbols):
                raise ValidationError(f"Invalid sample at index {idx}")
        
    def __len__(self):
        return len(self.data)

    def get_num_samples(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.data[idx]
        task_id = sample["task_id"]
        assert task_id != "default_task", f"Sample at index {idx} has 'default_task' as task_id."
        input_tensor = sample["input"]  # Already padded
        output_tensor = sample["output"]  # Already padded
        if idx < 5:  # Log only the first 5 samples to avoid clutter
            logger.debug(f"Sample {idx} - Task ID: {task_id}")
        return input_tensor, output_tensor, task_id

    def get_grid_size_stats(self) -> Dict[str, Any]:
        """
        Returns the precomputed grid size statistics.
        
        Returns:
            Dict[str, Any]: A dictionary containing grid size statistics.
        """
        if hasattr(self, 'statistics') and 'grid_size_stats' in self.statistics:
            return self.statistics['grid_size_stats']
        else:
            logger.warning("Grid size statistics not available.")
            return {}
    
    def get_symbol_frequencies(self) -> Dict[int, float]:
        """
        Returns the precomputed symbol frequencies.
        
        Returns:
            Dict[int, float]: A dictionary mapping symbols to their frequencies.
        """
        if hasattr(self, 'statistics') and 'symbol_frequencies' in self.statistics:
            return self.statistics['symbol_frequencies']
        else:
            logger.warning("Symbol frequencies not available.")
            return {}

    def _preprocess_grid(self, grid: Union[Dict, List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        return GridOperations.preprocess_grid(grid, self.pad_symbol_idx)
        
    def _pad_grid_torch(self, grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return GridOperations.pad_grid(grid, height, width, self.pad_symbol_idx)

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        # Debugging: Check batch size
        logger.debug(f"Collating batch of size: {len(batch)}")
        
        if not batch:
            logger.warning("Empty batch received")
            return torch.tensor([]), torch.tensor([]), []

        inputs, outputs, task_ids = zip(*batch)
    
        # Since all samples are already padded to 30x30, no additional padding is required here.
        # However, to ensure consistency, you can verify the shapes.
    
        padded_inputs = torch.stack(inputs)
        padded_outputs = torch.stack(outputs)
    
        # Debugging: Verify shapes after stacking
        #print(f"Padded inputs shape: {padded_inputs.shape}")
        #print(f"Padded outputs shape: {padded_outputs.shape}")
    
        return padded_inputs, padded_outputs, list(task_ids)