# gp2_arc/src/data/arc_dataset.py
import os
import json
import random
from typing import Union, List, Dict, Tuple, Any, Optional
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)
import numpy as np
import pickle
import hashlib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
import logging
import ijson  # Import ijson for streaming JSON parsing
import cysimdjson  # New import added
from tqdm import tqdm  # Import tqdm for progress bars
import sys  # Import sys for handling tqdm output
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
import multiprocessing  # To determine CPU count
from threading import Lock
from jsonschema import validate, ValidationError
from torch.utils.data import get_worker_info

try:
    from arckit.data import TaskSet, Task
except ImportError:
    TaskSet = None

logger = logging.getLogger(__name__)

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
def set_debug_mode(debug=False):
    if debug:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
        handler.setLevel(logging.ERROR)

class ARCDataset(Dataset):
    def __init__(
        self,
        data_source: Union[str, List[Dict], 'TaskSet', Tuple[Union[List, 'TaskSet'], str]],
        is_test: bool = False,
        num_symbols: int = 11,
        test_split: float = 0.2,
        pad_symbol_idx: int = 10,
        symbol_freq: Optional[Dict[int, float]] = None,
        debug: bool = False,
    ):
        self.is_test = is_test
        self.num_symbols = num_symbols
        self.test_split = test_split
        self.pad_symbol_idx = pad_symbol_idx
        self.symbol_freq = symbol_freq if symbol_freq is not None else {}

        logger.debug(f"Initialized ARCDataset with pad_symbol_idx: {self.pad_symbol_idx}")
        if self.symbol_freq:
            logger.debug("Symbol frequencies provided; initializing WeightedRandomSampler.")
        else:
            logger.debug("No symbol frequencies provided.")
        self.data_files = []
        self.data_source = data_source
        self.num_samples = 0

        # Initialize cysimdjson JSONParser
        self.json_parser = cysimdjson.JSONParser()
        logger.debug("Initialized cysimdjson.JSONParser instance")
        
        self.cache_path = self._generate_cache_path(
            data_source=self.data_source,
            num_symbols=self.num_symbols,
            is_test=self.is_test,
            test_split=self.test_split
        )
        
        if not self._load_cache(self.cache_path):
            # Load data only if not loaded from cache
            try:
                logger.debug("Loading data from data source as cache was not found or failed.")
                self.data = self._load_data(data_source)

                if not self.data:
                    logger.error("No valid samples loaded. Ensure that all samples have 'input' and 'output' keys.")
                    raise ValueError("No valid samples loaded. Ensure that all samples have 'input' and 'output' keys.")
                
                logger.debug(f"Data loaded successfully from data source with {len(self.data)} samples.")
            except Exception as e:
                logger.error(f"Failed to load data: {e}", exc_info=True)
                raise

            self.num_samples = len(self.data)
            self._compute_and_cache_statistics()
            self._save_cache(self.cache_path)

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

        if debug:
            logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
            logger.debug("Debug mode is enabled for ARCDataset.")
        else:
            logger.setLevel(logging.INFO)
            handler.setLevel(logging.INFO)

        logger.debug("ARCDataset initialization completed.")

    def _process_single_task(self, task: Union[Dict, List], task_id: str) -> List[Dict]:
        """
        Processes a single task dictionary or list and returns a list of samples.
        
        Args:
            task (Union[Dict, List]): The task data containing 'input' and 'output', or a list of such dictionaries.
            task_id (str): Identifier for the task.
        
        Returns:
            List[Dict]: List of processed sample dictionaries.
        """
        samples = []
        sample_count = 0  # Initialize sample counter
        missing_id_logged = False  # Flag to track if warning has been logged for this file
        try:
            if isinstance(task, dict):
                # Existing processing for dictionary tasks
                for ex in task.get('train', []):
                    logger.debug(f"Processing training example keys: {ex.keys()}")
                    logger.debug(f"Processing example keys: {ex.keys()}")
                    input_tensor = self._preprocess_grid(ex['input'])
                    output_tensor = self._preprocess_grid(ex['output'])
                    samples.append({
                        "input": input_tensor,
                        "output": output_tensor,
                        "task_id": task_id
                    })
                
                for ex in task.get('test', []):
                    input_tensor = self._preprocess_grid(ex['input'])
                    output_tensor = self._preprocess_grid(ex['output'])
                    samples.append({
                        "input": input_tensor,
                        "output": output_tensor,
                        "task_id": task_id
                    })
            elif isinstance(task, list):
                # New processing for list-type tasks
                for ex in task:
                    input_tensor = self._preprocess_grid(ex['input'])
                    output_tensor = self._preprocess_grid(ex['output'])
                    samples.append({
                        "input": input_tensor,
                        "output": output_tensor,
                        "task_id": task_id
                    })
            else:
                raise ValueError(f"Unsupported task type: {type(task)}")
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
        return samples
    
    
    def _process_single_file_streaming(self, file_path: str) -> List[Dict]:
        """
        Processes a single JSON file using streaming parsing and returns the list of samples.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Returns:
            List[Dict]: List of processed samples from the file.
        """
        samples = []
        samples = []
        sample_count = 0  # Initialize sample counter
        missing_id_logged = False  # Flag to track if warning has been logged for this file

        # Skip empty files early
        if os.path.getsize(file_path) == 0:
            logger.warning(f"Empty JSON file detected: {file_path}. Skipping.")
            return samples

        try:
            with open(file_path, 'rb') as f:  # Open in binary mode
                # Use cysimdjson for efficient parsing
                parsed_json = self.json_parser.parse(f.read())
                # Convert parsed_json to native Python structures
                parsed_py = self._cysimdjson_to_native(parsed_json)
                
                if isinstance(parsed_py, list):
                    for ex in parsed_py:
                        if 'input' in ex and 'output' in ex:
                            try:
                                input_tensor = self._preprocess_grid(ex['input'])
                                output_tensor = self._preprocess_grid(ex['output'])
                                task_id = ex.get('id', f"default_task_{sample_count}")
                                if not isinstance(task_id, str) or not task_id:
                                    if not missing_id_logged:
                                        task_id = f"default_task_{sample_count}"
                                        logger.warning(f"Sample missing valid 'id'. Assigned task_id: {task_id}")
                                        missing_id_logged = True
                                    else:
                                        task_id = f"default_task_{sample_count}"
                                samples.append({
                                    "input": input_tensor,
                                    "output": output_tensor,
                                    "task_id": task_id
                                })
                                sample_count += 1
                            except Exception as e:
                                logger.error(f"Error preprocessing sample in file {file_path}: {e}", exc_info=True)
                        else:
                            logger.warning(f"Sample missing 'input' or 'output' keys in file {file_path}. Skipping.")
                elif isinstance(parsed_py, dict):
                    # Existing processing for dictionary tasks
                    for ex in parsed_py.get('train', []):
                        try:
                            logger.debug(f"Processing training example keys: {ex.keys()}")
                            input_tensor = self._preprocess_grid(ex['input'])
                            output_tensor = self._preprocess_grid(ex['output'])
                            task_id = parsed_py.get('id', f"default_task_{sample_count}")
                            if not isinstance(task_id, str) or not task_id:
                                if not missing_id_logged:
                                    task_id = f"default_task_{sample_count}"
                                    logger.warning(f"Task missing valid 'id'. Assigned task_id: {task_id}")
                                    missing_id_logged = True
                                else:
                                    task_id = f"default_task_{sample_count}"
                            samples.append({
                                "input": input_tensor,
                                "output": output_tensor,
                                "task_id": task_id
                            })
                            sample_count += 1
                        except Exception as e:
                            logger.error(f"Error preprocessing training sample in file {file_path}: {e}", exc_info=True)
                    
                    for ex in parsed_py.get('test', []):
                        try:
                            input_tensor = self._preprocess_grid(ex['input'])
                            output_tensor = self._preprocess_grid(ex['output'])
                            task_id = parsed_py.get('id', f"default_task_{sample_count}")
                            if not isinstance(task_id, str) or not task_id:
                                if not missing_id_logged:
                                    task_id = f"default_task_{sample_count}"
                                    logger.warning(f"Task missing valid 'id'. Assigned task_id: {task_id}")
                                    missing_id_logged = True
                                else:
                                    task_id = f"default_task_{sample_count}"
                            samples.append({
                                "input": input_tensor,
                                "output": output_tensor,
                                "task_id": task_id
                            })
                            sample_count += 1
                        except Exception as e:
                            logger.error(f"Error preprocessing testing sample in file {file_path}: {e}", exc_info=True)
                else:
                    logger.warning(f"Unexpected JSON structure in file {file_path}. Skipping.")
        except Exception as e:  # Catch all exceptions related to parsing
            logger.error(f"cysimdjson failed to parse file {file_path}: {e}. Skipping.")

        return samples

    def _cysimdjson_to_native(self, parsed_json):
        """
        Recursively converts cysimdjson parsed objects to native Python lists and dicts.
        
        Args:
            parsed_json (cysimdjson.cysimdjson.JSONValue): Parsed JSON object.
        
        Returns:
            Union[dict, list, primitive]: Native Python data structure.
        """
        if isinstance(parsed_json, cysimdjson.JSONObject):
            return {k: self._cysimdjson_to_native(v) for k, v in parsed_json.items()}
        elif isinstance(parsed_json, cysimdjson.JSONArray):
            return [self._cysimdjson_to_native(item) for item in parsed_json]
        elif isinstance(parsed_json, (int, float, str, bool)):
            return parsed_json
        elif parsed_json is None:
            return None
        else:
            logger.warning(f"Unknown JSON type encountered: {type(parsed_json)}")
            return None
        """
        Wrapper method to process a single file in parallel.
        
        Args:
            file_path (str): Path to the JSON file.
            
        Returns:
            List[Dict]: List of processed samples from the file.
        """
        """
        Wrapper method to process a single file in parallel.
        
        Args:
            file_path (str): Path to the JSON file.
            
        Returns:
            List[Dict]: List of processed samples from the file.
        """
        return self._process_single_file_streaming
    
    
    def _save_cache(self, cache_path: str, data_only=False):
        """
        Saves the dataset and its statistics to the specified cache path using pickle.
    
        Args:
            cache_path (str): The file path where the cache will be saved.
            data_only (bool): If True, only save the data without statistics.
        """
        try:
            if data_only:
                cache_data = {
                    "data": self.data
                }
            else:
                cache_data = {
                    "data": self.data,
                    "statistics": self.statistics
                }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Saved {'data only ' if data_only else ''}cache to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")

        # Add data validation
        self._validate_data()

    def _validate_data(self):
        """
        Validates the dataset to ensure each sample contains the required keys and correct data types.
        Raises:
            ValueError: If any sample is missing required keys or has incorrect types.
        """
        required_keys = {"input", "output", "task_id"}
        for idx, sample in enumerate(self.data):
            # Check for required keys
            if not required_keys.issubset(sample.keys()):
                missing = required_keys - sample.keys()
                raise KeyError(f"Sample at index {idx} is missing keys: {missing}")
            
            # Validate 'input' and 'output' types
            for key in ["input", "output"]:
                if not isinstance(sample[key], torch.Tensor):
                    raise TypeError(f"Sample at index {idx} has '{key}' of type {type(sample[key])}, expected torch.Tensor.")
                
                if sample[key].ndimension() != 3 or sample[key].shape[0] != 1:
                    raise ValueError(f"Sample at index {idx} has '{key}' with shape {sample[key].shape}, expected shape (1, H, W).")
                
                # Validate that symbols are within the allowed range based on num_symbols
                max_symbol_allowed = self.num_symbols - 1
                max_symbol = sample[key].max()
                if max_symbol > max_symbol_allowed:
                    logger.error(
                        f"Sample at index {idx} has symbol {max_symbol.item()} exceeding the allowed maximum ({max_symbol_allowed})."
                    )
                    raise ValueError(
                        f"Sample at index {idx} has symbol {max_symbol.item()} exceeding the allowed maximum ({max_symbol_allowed})."
                    )
            
            # Validate 'task_id' type
            if not isinstance(sample["task_id"], str):
                raise TypeError(f"Sample at index {idx} has 'task_id' of type {type(sample['task_id'])}, expected str.")
        
        logger.debug("All samples passed validation.")
    
    def __len__(self):
        return len(self.data)

    def get_num_samples(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        task_id = sample["task_id"]
        assert task_id != "default_task", f"Sample at index {idx} has 'default_task' as task_id."
        input_tensor = sample["input"]  # Already padded
        output_tensor = sample["output"]  # Already padded
        if idx < 5:  # Log only the first 5 samples to avoid clutter
            logger.debug(f"Sample {idx} - Task ID: {task_id}")
        return input_tensor, output_tensor, task_id


    @staticmethod
    def _generate_cache_path(data_source: Union[str, List[Dict], 'TaskSet', Tuple[Union[List, 'TaskSet'], str]], num_symbols: int, is_test: bool, test_split: float) -> str:
        dataset_version = "v1"
        
        # Create a stable representation of data_source based on its type
        if isinstance(data_source, str):
            data_source_str = os.path.abspath(data_source)  # Use absolute path for consistency
        elif isinstance(data_source, TaskSet):
            data_source_str = f"TaskSet:{len(data_source.tasks)}"  # Use number of tasks as identifier
        elif isinstance(data_source, list):
            data_source_str = f"List:{len(data_source)}"  # Use length of the list
        else:
            data_source_str = str(data_source)  # Fallback to string representation
        
        # Create a JSON string with stable identifiers
        hash_input = json.dumps({
            'version': dataset_version,
            'data_source': data_source_str,
            'num_symbols': num_symbols,
            'is_test': is_test,
            'test_split': test_split
        }, sort_keys=True).encode('utf-8')
        
        # Generate MD5 hash for the cache filename
        hash_digest = hashlib.md5(hash_input).hexdigest()
        cache_filename = f"arc_dataset_cache_{hash_digest}.pkl"
        
        # Define the cache directory relative to the current file
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, cache_filename)

    def _load_cache(self, cache_path: str) -> bool:
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                self.data = cache_data.get("data", [])
                self.statistics = cache_data.get("statistics", {})
                self.num_samples = len(self.data)
                logger.debug(f"Loaded cached data from {cache_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load cache from {cache_path}: {e}")
        return False

    def _compute_and_cache_statistics(self):
        """
        Computes dataset statistics and caches them alongside the dataset cache.
        """
        logger.debug("Computing dataset statistics")
        grid_size_stats = self._compute_grid_size_stats()
        symbol_frequencies = self._compute_symbol_frequencies()
        
        statistics = {
            "grid_size_stats": grid_size_stats,
            "symbol_frequencies": symbol_frequencies
        }
        
        # Update the cache dictionary with statistics
        self.statistics = statistics
        self._save_cache(self.cache_path)  # Ensure statistics are saved in the cache
        logger.debug("Dataset statistics computed and cached successfully")

        if self.symbol_freq:
            logger.debug(f"Sampling weights - min: {self.sample_weights.min().item()}, max: {self.sample_weights.max().item()}, mean: {self.sample_weights.mean().item()}")


    def _process_list_data(self, data_list: List[Dict], task_id: Optional[str] = None) -> List[Dict]:
        processed_data = []
        logger.debug(f"Processing list data with {len(data_list)} items")
        for idx, example in enumerate(data_list):
            if 'input' in example and 'output' in example:
                input_grid = self._preprocess_grid(example['input'])
                output_grid = self._preprocess_grid(example['output'])
                
                task_id_sample = task_id if task_id else example.get('task_id')
                if not task_id_sample or task_id_sample == "default_task":
                    task_id_sample = f"default_task_{idx}"
                    logger.warning(f"Sample at index {idx} has invalid 'task_id'. Assigning new task_id: {task_id_sample}")
                
                processed_data.append({
                    "input": input_grid,
                    "output": output_grid,
                    "task_id": task_id_sample
                })
            else:
                logger.warning(f"Example at index {idx} missing 'input' or 'output' keys. Skipping.")
        logger.debug(f"Processed {len(processed_data)} samples")
        return processed_data


    def _combine_data(self, official_data, synthetic_data_path):
        official_processed = self._process_arckit_data(official_data) if TaskSet is not None and isinstance(official_data, TaskSet) else official_data
        synthetic_processed = self._process_synthetic_data(synthetic_data_path)
        return official_processed + synthetic_processed

    def _process_synthetic_data(self, directory: str):
        self.data_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                self.data_files.append(file_path)
                logger.debug(f"Processing file: {file_path}")
                with open(file_path, 'r') as f:
                    try:
                        task_data = json.load(f)
                        # Assign task_id from filename
                        task_id = os.path.splitext(filename)[0]
                        processed_samples = self._process_single_task(task_data, task_id=task_id)
                        self.data.extend(processed_samples)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from file {file_path}: {e}")

    def _process_arckit_data(self, taskset: 'TaskSet') -> List[Dict]:
        """
        Processes data from an arckit TaskSet and returns a list of samples.
        
        Args:
            taskset (TaskSet): The TaskSet object containing tasks.
            
        Returns:
            List[Dict]: List of processed sample dictionaries.
        """
        processed_data = []
        logger.debug(f"Processing TaskSet with {len(taskset.tasks)} tasks")
        for task in taskset.tasks:
            logger.debug(f"Processing task: {task.id}")
            logger.debug(f"Train samples: {len(task.train)}, Test samples: {len(task.test)}")
            # Process training samples
            for ex in task.train:
                try:
                    input_tensor = self._preprocess_grid(ex[0])
                    output_tensor = self._preprocess_grid(ex[1])
                    processed_data.append({
                        "input": input_tensor,
                        "output": output_tensor,
                        "task_id": task.id
                    })
                except Exception as e:
                    logger.error(f"Error processing training example in task {task.id}: {e}", exc_info=True)
            
            # Process testing samples
            for ex in task.test:
                try:
                    input_tensor = self._preprocess_grid(ex[0])
                    output_tensor = self._preprocess_grid(ex[1])
                    processed_data.append({
                        "input": input_tensor,
                        "output": output_tensor,
                        "task_id": task.id
                    })
                except Exception as e:
                    logger.error(f"Error processing testing example in task {task.id}: {e}", exc_info=True)
            
            logger.debug(f"Processed task {task.id}: Total samples added: {len(task.train) + len(task.test)}")
        
        logger.debug(f"Total samples processed from TaskSet: {len(processed_data)}")
        return processed_data


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

    def _compute_grid_size_stats(self):
        max_height, max_width = 0, 0
        for sample in self.data:
            # Assuming sample["input"] and sample["output"] have shape [C, H, W]
            max_height = max(max_height, sample["input"].shape[1], sample["output"].shape[1])
            max_width = max(max_width, sample["input"].shape[2], sample["output"].shape[2])
        grid_size_stats = {"max_height": max_height, "max_width": max_width}
        self.max_grid_size = (max_height, max_width)
        return grid_size_stats

    def _compute_symbol_frequencies(self):
        symbol_counts = np.zeros(self.num_symbols, dtype=int)
        max_symbol_in_data = 0
        for sample in self.data:
            input_symbols = sample["input"].flatten().numpy().astype(int)
            output_symbols = sample["output"].flatten().numpy().astype(int)
            if input_symbols.size > 0:
                max_symbol_in_data = max(max_symbol_in_data, input_symbols.max())
            if output_symbols.size > 0:
                max_symbol_in_data = max(max_symbol_in_data, output_symbols.max())
            symbol_counts += np.bincount(input_symbols, minlength=self.num_symbols)
            symbol_counts += np.bincount(output_symbols, minlength=self.num_symbols)
        
        logger.debug(f"Maximum symbol index in data: {max_symbol_in_data}")
        logger.debug(f"Symbol counts length: {len(symbol_counts)}")
        
        if max_symbol_in_data >= self.num_symbols:
            logger.error(f"Found symbol index {max_symbol_in_data} exceeding num_symbols - 1 ({self.num_symbols - 1}).")
            raise ValueError(f"Symbol index {max_symbol_in_data} exceeds the allowed range.")
        
        total_symbols = symbol_counts.sum()
        if total_symbols == 0:
            logger.warning("Total de símbolos es 0. Evitando la división por cero.")
            symbol_freq = np.zeros_like(symbol_counts, dtype=float)
        else:
            symbol_freq = symbol_counts / total_symbols
        
        return symbol_freq
    
    def _preprocess_grid(self, grid: Union[Dict, List, np.ndarray, torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        logger.debug(f"Preprocessing grid with initial type: {type(grid)}")
        
        # Convert grid to torch.Tensor if it's a list or numpy array
        if isinstance(grid, list):
            grid_tensor = torch.as_tensor(grid, dtype=torch.float32)
            logger.debug(f"Converted list to tensor with shape: {grid_tensor.shape}")
        elif isinstance(grid, np.ndarray):
            grid_tensor = torch.as_tensor(grid, dtype=torch.float32)
            logger.debug(f"Converted numpy array to tensor with shape: {grid_tensor.shape}")
        elif isinstance(grid, torch.Tensor):
            grid_tensor = grid.float()
            logger.debug(f"Using existing tensor with shape: {grid_tensor.shape}")
        elif isinstance(grid, int):
            logger.debug("Grid is of type int. Converting to 1x1 grid.")
            grid = [[grid]]
            grid_tensor = torch.as_tensor(grid, dtype=torch.float32)
            logger.debug(f"Converted int to tensor with shape: {grid_tensor.shape}")
        else:
            raise ValueError(f"Unexpected grid type: {type(grid)}")
    
        # Ensure grid_tensor has three dimensions [C, H, W]
        if grid_tensor.ndim == 2:
            logger.debug("Grid tensor is 2D. Adding channel dimension.")
            grid_tensor = grid_tensor.unsqueeze(0)  # Add channel dimension
            logger.debug(f"Grid tensor shape after unsqueeze: {grid_tensor.shape}")
        elif grid_tensor.ndim != 3:
            raise ValueError(f"Unexpected grid tensor dimensions: {grid_tensor.ndim}. Expected 2D or 3D tensor.")

        logger.debug(f"Grid shape before padding: {grid_tensor.shape}")

        # Apply padding using PyTorch's built-in functions with correct pad_value
        padded_grid = self._pad_grid_torch(grid_tensor, height=30, width=30, pad_value=self.pad_symbol_idx)

        logger.debug(f"Grid shape after padding: {padded_grid.shape}")
        logger.debug(f"Padded grid with pad_symbol_idx: {self.pad_symbol_idx}, resulting shape: {padded_grid.shape}")
        return padded_grid
    
    
    def kronecker_scale(self, X, target_height=30, target_width=30):
        logger.debug(f"Kronecker scaling input shape: {X.shape}")
        h, w = X.shape
        scale_h = target_height / h
        scale_w = target_width / w
        d = int(np.floor(min(scale_h, scale_w)))
        
        X_scaled = np.kron(X, np.ones((d, d)))
        logger.debug(f"Kronecker scaled output shape: {X_scaled.shape}")
        return X_scaled


    def reverse_scaling(self, X_orig, X_pred):
        logger.debug(f"Reverse scaling - Original shape: {X_orig.shape}, Prediction shape: {X_pred.shape}")
        h, w = X_orig.shape
        # Reshape X_pred to 2D if it's 1D
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape((int(np.sqrt(X_pred.size)), -1))
        
        X_pred_cropped = X_pred[:h, :w]  # Crop to original size
        
        if h == X_pred.shape[0] and w == X_pred.shape[1]:
            logger.debug("No rescaling needed")
            return X_pred_cropped
        
        # Calculate the downscale factor
        d_h = X_pred_cropped.shape[0] // h
        d_w = X_pred_cropped.shape[1] // w
        
        # Ensure the dimensions are compatible for reshaping
        if d_h > 0 and d_w > 0:
            try:
                X_rev = X_pred_cropped.reshape(h, d_h, w, d_w).mean(axis=(1, 3))
            except ValueError as e:
                logger.error(f"Error during reshaping: {e}")
                logger.debug(f"X_pred_cropped shape: {X_pred_cropped.shape}, h: {h}, w: {w}, d_h: {d_h}, d_w: {d_w}")
                raise
        else:
            logger.warning(f"Invalid downscale factors: d_h={d_h}, d_w={d_w}")
            raise ValueError("Invalid dimensions for reverse scaling")
        # Resize the result to match the original target shape
        result = np.resize(X_rev.round().astype(int), X_orig.shape)
        logger.debug(f"Reverse scaled output shape: {result.shape}")
        return result

    def _scale_grid(self, grid: np.ndarray, height: int, width: int) -> np.ndarray:
        return grid  # No scaling, preserve original size

    def _pad_grid_torch(self, grid: torch.Tensor, height: int, width: int, pad_value: int = 0) -> torch.Tensor:
        """
        Pads the input grid tensor to the specified height and width using PyTorch's functional padding.
        
        Args:
            grid (torch.Tensor): The input grid tensor with shape [C, H, W].
            height (int): The target height after padding.
            width (int): The target width after padding.
        
        Returns:
            torch.Tensor: The padded grid tensor.
        """
        _, h, w = grid.shape
        pad_h = max((height - h) // 2, 0)
        pad_w = max((width - w) // 2, 0)

        # Calculate padding for top, bottom, left, and right
        padding = (pad_w, width - w - pad_w, pad_h, height - h - pad_h)  # (left, right, top, bottom)
        logger.debug(f"Padding applied: left={pad_w}, right={width - w - pad_w}, top={pad_h}, bottom={height - h - pad_h}")
    
        # Apply padding using PyTorch's functional pad
        padded_grid = F.pad(grid, padding, mode='constant', value=pad_value)
        return padded_grid


    @staticmethod
    def collate_fn(batch):
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
    
    def _load_data(self, data_source):
        logger.debug(f"Loading data from source type: {type(data_source)}")
        if isinstance(data_source, list):
            return self._process_list_data(data_source)
        elif isinstance(data_source, TaskSet):
            return self._process_arckit_data(data_source)
        elif isinstance(data_source, str):
            if os.path.isdir(data_source):
                logger.debug(f"Loading data from directory: {data_source}")
                return self._load_directory(data_source)
            elif os.path.isfile(data_source):
                logger.debug(f"Loading data from file: {data_source}")
                # Since synthetic data is preloaded and passed via 'all_synthetic_data', avoid reloading here
                # Instead, assume 'all_synthetic_data' contains the necessary datasets
                samples = self._process_single_file_parallel(data_source)
                return samples
            else:
                raise ValueError(f"Invalid data source path: {data_source}")
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")
    def _load_directory(self, directory_path: str) -> List[Dict]:
        """
        Loads all JSON files from the specified directory and processes them.

        Args:
            directory_path (str): Path to the directory containing JSON files.

        Returns:
            List[Dict]: List of processed samples from all JSON files in the directory.
        """
        all_samples = []
        file_paths = []

        # Collect all JSON file paths
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                    logger.debug(f"Queued file for processing: {file_path}")

        # Define the number of threads (adjust based on your system's resources)
        max_workers = min(32, os.cpu_count() + 4)  # Example adjustment

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks to the executor
            future_to_file = {executor.submit(self._process_single_file_streaming, fp): fp for fp in file_paths}

            with tqdm(total=len(file_paths), desc="Loading synthetic data", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    fp = future_to_file[future]
                    try:
                        samples = future.result()
                        all_samples.extend(samples)
                        logger.debug(f"Completed processing file: {fp} with {len(samples)} samples")
                    except Exception as e:
                        logger.error(f"Error processing file {fp}: {e}", exc_info=True)
                    pbar.update(1)
        logger.debug(f"Loaded {len(all_samples)} samples from directory {directory_path}")
        return all_samples

    def _load_single_file(self, file_path: str) -> List[Dict]:
        """
        Loads and processes a single JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            List[Dict]: List of processed samples from the JSON file.
        """
        try:
            return self._process_single_file_streaming(file_path)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            return []
