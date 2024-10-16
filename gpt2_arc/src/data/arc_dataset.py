# gp2_arc/src/data/arc_dataset.py
import os
import json
import random
from typing import Union, List, Dict, Tuple, Any
import numpy as np
import pickle
import hashlib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging
import ijson  # Import ijson for streaming JSON parsing
from tqdm import tqdm  # Import tqdm for progress bars
import sys  # Import sys for handling tqdm output
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
logger.setLevel(logging.ERROR)  # Set to ERROR by default

# Create a handler that writes to stderr
handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)

# Create a formatting for the logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

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
        num_symbols: int = 10,
        test_split: float = 0.2,
        pad_symbol_idx: int = 10,  # Add this parameter with a default value
        debug=False,
    ):
        self.test_split = test_split
        self.is_test = is_test
        self.num_symbols = num_symbols
        self.pad_symbol_idx = pad_symbol_idx  # Store it as an instance variable
        self.data_files = []  # Initialize data_files as an empty list
        self.data_source = data_source
        self.num_samples = 0
        self.data = []
        self.cache_path = self._generate_cache_path(
            data_source=self.data_source,
            num_symbols=self.num_symbols,
            is_test=self.is_test,
            test_split=self.test_split
        )

        if self._load_cache(self.cache_path):
            logger.debug("Data loaded from cache successfully.")
            return
        if debug:
            logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.ERROR)
            handler.setLevel(logging.ERROR)
        logger.debug("Starting ARCDataset initialization")
        logger.debug(f"data_source type: {type(data_source)}")
        logger.debug(f"data_source content: {data_source}")
        logger.debug(f"self.test_split is set to: {self.test_split}")

        if isinstance(data_source, str):
            if os.path.isdir(data_source):
                logger.debug("Initializing dataset with data from directory")
                self.data_dir = data_source
                self.data_files = [
                    os.path.join(data_source, f)
                    for f in os.listdir(data_source)
                    if f.endswith('.json')
                ]
                random.shuffle(self.data_files)
                # Determine the number of workers based on CPU count
                cpu_count = multiprocessing.cpu_count() or 1
                max_workers = min(16, cpu_count + 2)  # Adjusted heuristic for ThreadPoolExecutor
                
                logger.debug(f"Using ThreadPoolExecutor with {max_workers} workers for parallel processing.")
                
                logger.debug(f"Using ProcessPoolExecutor with {max_workers} workers for parallel processing.")

                logger.debug(f"Using ProcessPoolExecutor with {max_workers} workers for parallel processing.")

                # Initialize tqdm progress bar and process files in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all file processing tasks
                    future_to_file = {executor.submit(self._process_single_file_parallel, fp): fp for fp in self.data_files}
                    
                    # Wrap the as_completed iterator with tqdm for the progress bar
                    for future in tqdm(
                        as_completed(future_to_file),
                        total=len(future_to_file),
                        desc="Loading JSON Files",
                        unit="file",
                        file=sys.stdout,
                        ncols=100,              # Optional: Set width of the progress bar
                        mininterval=0.5,        # Optional: Minimum interval between updates
                        colour='green'          # Optional: Set progress bar color (requires tqdm >=4.46.0)
                    ):
                        file_path = future_to_file[future]
                        try:
                            samples = future.result()
                            self.data.extend(samples)  # No lock needed
                            logger.debug(f"Added {len(samples)} samples from file {file_path}")
                        except Exception as exc:
                            logger.error(f"{file_path} generated an exception: {exc}", exc_info=True)
            elif os.path.isfile(data_source):
                with open(data_source, 'r') as f:
                    task_data = json.load(f)
                if isinstance(task_data, dict):
                    task_id = task_data.get('id', "default_task")
                    samples = self._process_single_task(task_data, task_id=task_id)
                    self.data.extend(samples)
                elif isinstance(task_data, list):
                    samples = self._process_list_data(task_data,task_id=task_id)
                    self.data.extend(samples)
                else:
                    logger.error(f"Unexpected data format in file {data_source}: {type(task_data)}")
            else:
                raise FileNotFoundError(f"Data source file or directory not found: {data_source}")
        elif TaskSet is not None and isinstance(data_source, TaskSet):
            logger.debug(f"TaskSet attributes before access: {dir(data_source)}")
            logger.debug(f"Does TaskSet have 'dataset' attribute? {hasattr(data_source, 'dataset')}")
            samples = self._process_arckit_data(data_source)
            self.data.extend(samples)
        elif isinstance(data_source, list):
            samples = self._process_list_data(data_source, task_id="default_task")
            self.data.extend(samples)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        self.num_samples = len(self.data)
        self._compute_and_cache_statistics()
        self._save_cache(self.cache_path)

    def _process_single_task(self, task: Dict, task_id: str) -> List[Dict]:
        """
        Processes a single task dictionary and returns a list of samples.
        
        Args:
            task (Dict): The task data containing 'input' and 'output'.
            task_id (str): Identifier for the task.
        
        Returns:
            List[Dict]: List of processed sample dictionaries.
        """
        samples = []
        try:
            # Process training samples
            for ex in task.get('train', []):
                input_tensor = self._preprocess_grid(ex['input'])
                output_tensor = self._preprocess_grid(ex['output'])
                samples.append({
                    "input": input_tensor,
                    "output": output_tensor,
                    "task_id": task_id
                })
            
            # Process testing samples
            for ex in task.get('test', []):
                input_tensor = self._preprocess_grid(ex['input'])
                output_tensor = self._preprocess_grid(ex['output'])
                samples.append({
                    "input": input_tensor,
                    "output": output_tensor,
                    "task_id": task_id
                })
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
        try:
            # Skip empty files early
            if os.path.getsize(file_path) == 0:
                logger.warning(f"Empty JSON file detected: {file_path}. Skipping.")
                return samples
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use ijson for efficient streaming if the file is large
                parser = ijson.parse(f)
                current_object = {}
                current_key = None
                for prefix, event, value in parser:
                    if (prefix, event) == ('', 'start_map'):
                        current_object = {}
                    elif event == 'map_key':
                        current_key = value
                    elif event in ('string', 'number', 'boolean', 'null'):
                        current_object[current_key] = value
                    elif (prefix, event) == ('', 'end_map'):
                        # Determine the structure based on keys
                        if 'train' in current_object and 'test' in current_object:
                            # Task-Based Structure
                            try:
                                validate(instance=current_object, schema=TASK_SCHEMA)
                                task_id = current_object.get('id', os.path.splitext(os.path.basename(file_path))[0])
                                task_samples = self._process_single_task(current_object, task_id=task_id)
                                samples.extend(task_samples)
        except ijson.JSONError as e:
            logger.warning(f"Malformed JSON in file {file_path}: {e}. Skipping.")
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error in file {file_path}: {e}. Skipping.")
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {e}", exc_info=True)
        
        return samples

    def _process_single_file_parallel(self, file_path: str) -> List[Dict]:
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
        return self._process_single_file_streaming(file_path)
    
    
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
        # Since all samples are already padded to 30x30 during preprocessing, no additional padding is required here.
        input_tensor = sample["input"]  # Already padded
        output_tensor = sample["output"]  # Already padded
        return input_tensor, output_tensor, sample["task_id"]



    @staticmethod
    def _generate_cache_path(self, data_source: Union[str, List[Dict], 'TaskSet', Tuple[Union[List, 'TaskSet'], str]], num_symbols: int, is_test: bool, test_split: float) -> str:
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


    def _process_list_data(self, data_list: List[Dict], task_id: str) -> List[Dict]:
        processed_data = []
        for idx, example in enumerate(data_list):
            if 'input' in example and 'output' in example and isinstance(example['input'], (list, np.ndarray)) and isinstance(example['output'], (list, np.ndarray)):
                # Preprocess the grids
                input_grid = self._preprocess_grid(example['input'])
                output_grid = self._preprocess_grid(example['output'])

                # Assign task_id if not present
                # Assign task_id from parameter, overriding any existing task_id in the data

                processed_data.append({
                    "input": input_grid,
                    "output": output_grid,
                    "task_id": task_id
                })
            else:
                logger.warning(f"Example at index {idx} missing 'input' or 'output' keys or has incorrect types.")
                # Optionally, skip or raise an error
                # raise ValueError("Unexpected item format in data_source.")
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
    
    def get_symbol_frequencies(self) -> Dict[str, float]:
        """
        Returns the precomputed symbol frequencies.
        
        Returns:
            Dict[str, float]: A dictionary mapping symbols to their frequencies.
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
        for sample in self.data:
            symbol_counts += np.bincount(sample["input"].flatten(), minlength=self.num_symbols)
            symbol_counts += np.bincount(sample["output"].flatten(), minlength=self.num_symbols)
        return symbol_counts / symbol_counts.sum()
    
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

        # Apply padding using PyTorch's built-in functions
        padded_grid = self._pad_grid_torch(grid_tensor, height=30, width=30)

        logger.debug(f"Grid shape after padding: {padded_grid.shape}")
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


    def collate_fn(self, batch):
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
        print(f"Padded inputs shape: {padded_inputs.shape}")
        print(f"Padded outputs shape: {padded_outputs.shape}")
    

        return padded_inputs, padded_outputs, list(task_ids)
