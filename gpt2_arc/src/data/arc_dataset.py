# gp2_arc/src/data/arc_dataset.py
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
from config.arc_dataset_config import ARCDatasetConfig
from utils.validation import validate_sample
from utils.grid_ops import GridOperations
from utils.custom_exceptions import ARCDatasetError, DataLoadingError, ValidationError, ResourceError
from arckit.data import TaskSet, Task
import logging
logger = logging.getLogger(__name__)

from utils.statistics import DatasetStatistics

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

            # Apply sample limit if specified
            if max_samples is not None:
                self.data = self.data[:max_samples]
                logger.debug(f"Limited dataset to {max_samples} samples.")
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

    
    def _process_single_file_streaming(self, file_path: str) -> List[Dict]:
        """
        Processes a single JSON file using streaming parsing and returns the list of samples.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Returns:
            List[Dict]: List of processed samples from the file.
        """
        samples = []
        sample_count = 0  # Initialize sample counter
        missing_id_logged = False  # Flag to track if warning has been logged for this file

        # Skip empty files early
        if os.path.getsize(file_path) == 0:
            logger.warning(f"Empty JSON file detected: {file_path}. Skipping.")
            return samples

        try:
            with open(file_path, 'rb') as f:
                try:
                    # Attempt to parse with cysimdjson
                    parsed_json = self.json_parser.parse(f.read())
                    # Direct Conversion Using as_list() and as_dict()
                    if isinstance(parsed_json, JSONArray):
                        parsed_py = parsed_json.tolist(recursive=True)
                        logger.debug(f"Parsed JSON is a list with {len(parsed_py)} items.")
                    elif parsed_json.is_object():
                        parsed_py = parsed_json.as_dict(recursive=True)
                        logger.debug(f"Parsed JSON is a dict with keys: {list(parsed_py.keys())}")
                    else:
                        logger.warning(f"Parsed JSON is neither a list nor a dict for file {file_path}: {type(parsed_json)}. Skipping file.")
                        return samples

                    # Add Type Assertions
                    assert isinstance(parsed_py, (list, dict)), f"Parsed JSON is of type {type(parsed_py)}, expected list or dict."

                except Exception as e:
                    logger.exception(f"cysimdjson failed to parse file {file_path}. Attempting standard json parser.")
                    logger.error(f"Exception type: {type(e).__name__}")

                    # Read a larger snippet of the file for debugging
                    f.seek(0)
                    file_snippet = f.read(2048).decode('utf-8', errors='replace')  # Read first 2KB
                    logger.debug(f"Snippet from {file_path} for debugging:\n{file_snippet}\n")
                    
                    f.seek(0)  # Reset file pointer to the beginning
                    try:
                        parsed_py = json.load(f)
                        logger.info(f"Successfully parsed {file_path} with standard json parser.")
                        
                        # Log the type and structure of parsed_py
                        logger.debug(f"Type of parsed_py: {type(parsed_py)}")
                        if isinstance(parsed_py, list):
                            logger.debug(f"Parsed JSON is a list with {len(parsed_py)} items.")
                            if parsed_py:
                                logger.debug(f"First item in list: {parsed_py[0]}")
                        elif isinstance(parsed_py, dict):
                            logger.debug(f"Parsed JSON is a dict with keys: {list(parsed_py.keys())}")
                            for key in ['samples', 'data', 'entries', 'train', 'test']:
                                if key in parsed_py and isinstance(parsed_py[key], list):
                                    logger.debug(f"Key '{key}' contains {len(parsed_py[key])} items.")
                                    if parsed_py[key]:
                                        logger.debug(f"First item under '{key}': {parsed_py[key][0]}")
        
                    except json.JSONDecodeError as je:
                        logger.error(f"Standard json parser failed to parse file {file_path}: {je}. Skipping file.")
                        return samples  # Skip this file as both parsers failed
                    except Exception as je:
                        logger.exception(f"Unexpected error during standard json parsing of file {file_path}: {je}. Skipping file.")
                        return samples  # Skip this file due to unexpected parsing error

            # Proceed with data extraction as per the existing logic
            # Determine how to extract samples based on JSON structure
            if isinstance(parsed_py, list):
                data_iterable = parsed_py
            elif isinstance(parsed_py, dict):
                # Attempt to extract samples from common keys like 'data', 'samples', 'train', 'test'
                if 'samples' in parsed_py:
                    data_iterable = parsed_py['samples']
                elif 'entries' in parsed_py:
                    data_iterable = parsed_py['entries']
                elif 'train' in parsed_py:
                    data_iterable = parsed_py['train']
                elif 'test' in parsed_py:
                    data_iterable = parsed_py['test']
                else:
                    logger.warning(f"No recognizable keys found in {file_path}. Skipping file.")
                    return samples
            else:
                logger.warning(f"Parsed JSON is neither a list nor a dict for file {file_path}: {type(parsed_py)}. Skipping file.")
                return samples

            logger.debug(f"Total samples to process from {file_path}: {len(data_iterable)}")

            for ex in data_iterable:
                # Log the type and keys of each example
                if isinstance(ex, dict):
                    logger.debug(f"Processing example of type dict with keys: {list(ex.keys())}")
                else:
                    logger.warning(f"Expected example to be a dict, but got {type(ex)}. Skipping example.")
                    continue  # Skip non-dict examples

                # Handle cases where 'input' and 'output' might be nested differently with variations
                input_key = next((k for k in ex.keys() if k.lower() in self.INPUT_KEYS), None)
                output_key = next((k for k in ex.keys() if k.lower() in self.OUTPUT_KEYS), None)

                if input_key and output_key:
                    try:
                        input_tensor = self._preprocess_grid(ex[input_key])
                        output_tensor = self._preprocess_grid(ex[output_key])
                        task_id = ex.get('id', f"default_task_{sample_count}")
                        if not isinstance(task_id, str) or not task_id:
                            if not missing_id_logged:
                                task_id = f"default_task_{sample_count}"
                                logger.warning(f"Sample missing valid 'id'. Assigned task_id: {task_id}")
                                missing_id_logged = True
                            else:
                                task_id = f"default_task_{sample_count}"
                        
                        # Add prefix if not already present
                        if isinstance(task_id, str) and not (task_id.startswith('synthetic_task_') or task_id.startswith('default_task')):
                            task_id = f"synthetic_task_{task_id}"
                            logger.debug(f"Prefixed task_id to: {task_id}")
                        samples.append({
                            "input": input_tensor,
                            "output": output_tensor,
                            "task_id": task_id
                        })
                        sample_count += 1
                    except Exception as e:
                        logger.exception(
                            f"Error preprocessing sample {sample_count} (Task ID: {task_id}) in file {file_path}: {e}"
                        )
                else:
                    logger.warning(f"Sample missing 'input' or 'output' keys in file {file_path}. Skipping.")
                    logger.debug(f"Sample keys: {list(ex.keys())}")

        except Exception as e:  # Catch all exceptions related to parsing
            logger.exception(f"Failed to process file {file_path}: {e}", exc_info=True)

        logger.info(f"Finished processing synthetic data file: {file_path}. Extracted {len(samples)} samples.")
        return samples
    
    
    def _process_single_file_parallel(self, file_path: str) -> List[Dict]:

        """
        Wrapper method to process a single file in parallel.
        
        Args:
            file_path (str): Path to the JSON file.
            
        Returns:
            List[Dict]: List of processed samples from the file.
        """
        return self._process_single_file_streaming(file_path)
    
    
    def _save_cache(self, cache_path: str, data_only: bool = False) -> None:
        """
        Saves the dataset and its statistics to the specified cache path using pickle.
    
        Args:
            cache_path (str): The file path where the cache will be saved.
            data_only (bool): If True, only save the data without statistics.
        """
        logger.debug(f"Attempting to save {'data only ' if data_only else ''}cache to {cache_path}")
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
            logger.info(f"Successfully saved {'data only ' if data_only else ''}cache to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}", exc_info=True)

        # Add data validation
        self._validate_data()

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
        logger.debug(f"Attempting to load cache from: {cache_path}")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                self.data = cache_data.get("data", [])
                self.statistics = cache_data.get("statistics", {})
                self.num_samples = len(self.data)
                logger.info(f"Successfully loaded cache from {cache_path} with {self.num_samples} samples.")
                return True
            except Exception as e:
                logger.error(f"Failed to load cache from {cache_path}: {e}", exc_info=True)
        else:
            logger.warning(f"Cache file does not exist at: {cache_path}. Proceeding without cache.")
        return False

    def _compute_and_cache_statistics(self):
        """Computes dataset statistics and caches them alongside the dataset cache."""
        logger.info("Starting computation of dataset statistics.")
        try:
            stats_computer = DatasetStatistics(self.num_symbols)
            self.statistics = stats_computer.compute_all_statistics(self.data)
            
            if 'grid_size_stats' in self.statistics:
                self.max_grid_size = (
                    self.statistics['grid_size_stats']['max_height'],
                    self.statistics['grid_size_stats']['max_width']
                )
            
            self._save_cache(self.cache_path)
            logger.info("Dataset statistics have been cached successfully.")
            
            if self.symbol_freq:
                logger.debug(f"Sampling weights - min: {self.sample_weights.min().item()}, "
                            f"max: {self.sample_weights.max().item()}, "
                            f"mean: {self.sample_weights.mean().item()}")
        except (ValidationError, DataLoadingError) as e:
            logger.error(f"Failed to compute statistics: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in statistics computation: {e}")
            raise DataLoadingError(f"Statistics computation failed: {str(e)}")
    
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
    
    def _load_data(self, data_source: Union[str, List[Dict], 'TaskSet']) -> List[Dict]:
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
