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
from torch.utils.data import get_worker_info
import math  # Import math module for ceiling division

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

# Function to set debug mode
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
        debug=False,
    ):
        self.test_split = test_split
        self.is_test = is_test
        self.num_symbols = num_symbols
        self.data_files = []
        self.data_source = data_source
        self.index_mapping = []
        self.file_samples_count = {}

        self.cache_path = self._generate_cache_path(
            data_source=self.data_source,
            num_symbols=self.num_symbols,
            is_test=self.is_test,
            test_split=self.test_split
        )
        
        if self._load_cache(self.cache_path):
            logger.debug("Data index loaded from cache successfully.")
            self.num_samples = len(self.index_mapping)
            return

        set_debug_mode(debug)
        logger.debug("Starting ARCDataset initialization with lazy loading")

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
                self._build_index_from_files(self.data_files)
            elif os.path.isfile(data_source):
                logger.debug("Initializing dataset with single file")
                self.data_files = [data_source]
                self._build_index_from_files(self.data_files)
                self.num_samples = len(self.index_mapping)
            else:
                raise FileNotFoundError(f"Data source file or directory not found: {data_source}")
        elif TaskSet is not None and isinstance(data_source, TaskSet):
            logger.debug("Initializing dataset with TaskSet data")
            self.data = self._process_arckit_data(data_source)
            self.num_samples = len(self.data)
        elif isinstance(data_source, list):
            logger.debug("Initializing dataset with list data")
            self.data_files = []
            self._process_list_data_indices(data_source)
            self.num_samples = len(self.index_mapping)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")
        self._save_cache(self.cache_path)

    def _build_index_from_files(self, data_files: List[str]):
        """
        Builds the index mapping and file samples count from the provided list of data files.
        
        Args:
            data_files (List[str]): List of file paths to process.
        """
        self.index_mapping = []
        self.file_samples_count = {}

        logger.debug("Building index from files")
        
        for file_path in data_files:
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                
                if isinstance(task_data, dict):
                    samples = task_data.get('test', []) if self.is_test else task_data.get('train', [])
                elif isinstance(task_data, list):
                    samples = task_data
                else:
                    raise ValueError(f"Unexpected data format in file {file_path}: {type(task_data)}")
                
                num_samples = len(samples)
                self.file_samples_count[file_path] = num_samples
                
                for sample_idx in range(num_samples):
                    self.index_mapping.append((file_path, sample_idx))
                
                logger.debug(f"Processed {num_samples} samples from {file_path}")
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                continue  # Skip problematic files and continue with others

        logger.debug(f"Total indexed samples: {len(self.index_mapping)}")

    def _save_cache(self, cache_path: str):
        """
        Saves the dataset index and statistics to the specified cache path using pickle.
        """
        try:
            cache_data = {
                "index_mapping": self.index_mapping,
                "file_samples_count": self.file_samples_count,
                "statistics": self.statistics if hasattr(self, 'statistics') else {}
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Saved cache to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
    
    def __len__(self):
        return len(self.data)

    def get_num_samples(self):
        if hasattr(self, 'data') and self.data:
            return len(self.data)
        else:
            return self.num_samples
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")

        file_path, sample_idx = self.index_mapping[idx]

        if file_path is None:
            # In-memory list data
            sample = self.data_source[sample_idx]
            input_tensor = self._preprocess_grid(sample['input'])
            output_tensor = self._preprocess_grid(sample['output'])
            task_id = sample.get('task_id', "default_task")
        else:
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                if isinstance(task_data, dict):
                    samples = task_data.get('test', []) if self.is_test else task_data.get('train', [])
                    sample = samples[sample_idx]
                    task_id = task_data.get('id', os.path.splitext(os.path.basename(file_path))[0])
                elif isinstance(task_data, list):
                    sample = task_data[sample_idx]
                    task_id = os.path.splitext(os.path.basename(file_path))[0]
                else:
                    raise ValueError(f"Unexpected data format in file {file_path}: {type(task_data)}")
                
                input_tensor = self._preprocess_grid(sample['input'])
                output_tensor = self._preprocess_grid(sample['output'])
            except Exception as e:
                logger.error(f"Error loading sample {sample_idx} from file {file_path}: {e}", exc_info=True)
                # Optionally, you can skip this sample or return a default value
                raise e

        return input_tensor, output_tensor, task_id

    def _count_samples_in_directory(self, directory: str):
        num_samples = 0
        file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
        logger.debug(f"Counting samples in {len(file_list)} files")
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                if isinstance(task_data, dict):
                    sample_count = len(task_data.get('test', [])) if self.is_test else len(task_data.get('train', []))
                    num_samples += sample_count
                    logger.debug(f"File {file_path}: {sample_count} samples")
                elif isinstance(task_data, list):
                    num_samples += len(task_data)
                    logger.debug(f"File {file_path}: {len(task_data)} samples")
                else:
                    logger.error(f"Unexpected data format in file {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                continue  # Skip this file and proceed to the next
        logger.debug(f"Total samples counted: {num_samples}")
        return num_samples


    @staticmethod
    def _generate_cache_path(data_source: Union[str, List[Dict], 'TaskSet', Tuple[Union[List, 'TaskSet'], str]], num_symbols: int, is_test: bool, test_split: float) -> str:
        dataset_version = "v1"
        hash_input = json.dumps({
            'version': dataset_version,
            'data_source': data_source if isinstance(data_source, list) else data_source.__str__(),
            'num_symbols': num_symbols,
            'is_test': is_test,
            'test_split': test_split
        }, sort_keys=True).encode('utf-8')
        hash_digest = hashlib.md5(hash_input).hexdigest()
        cache_filename = f"arc_dataset_cache_{hash_digest}.pkl"
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, cache_filename)

    def _load_cache(self, cache_path: str) -> bool:
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                self.index_mapping = cache_data.get("index_mapping", [])
                self.file_samples_count = cache_data.get("file_samples_count", {})
                self.statistics = cache_data.get("statistics", {})
                logger.debug(f"Loaded cached index from {cache_path}")
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
        Computes grid size statistics by iterating over samples on-the-fly.
        Returns:
            Dict[str, Any]: A dictionary containing grid size statistics.
        """
        max_height, max_width = 0, 0
        for file_path, sample_idx in self.index_mapping:
            if file_path is None:
                sample = self.data_source[sample_idx]
                input_shape = self._get_grid_shape(sample['input'])
                output_shape = self._get_grid_shape(sample['output'])
            else:
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)
                    if isinstance(task_data, dict):
                        samples = task_data.get('test', []) if self.is_test else task_data.get('train', [])
                        sample = samples[sample_idx]
                        input_shape = self._get_grid_shape(sample['input'])
                        output_shape = self._get_grid_shape(sample['output'])
                    elif isinstance(task_data, list):
                        sample = task_data[sample_idx]
                        input_shape = self._get_grid_shape(sample['input'])
                        output_shape = self._get_grid_shape(sample['output'])
                    else:
                        raise ValueError(f"Unexpected data format in file {file_path}: {type(task_data)}")
                except Exception as e:
                    logger.error(f"Error processing sample {sample_idx} from file {file_path}: {e}", exc_info=True)
                    continue  # Skip this sample

            max_height = max(max_height, input_shape[1], output_shape[1])
            max_width = max(max_width, input_shape[2], output_shape[2])

        grid_size_stats = {"max_height": max_height, "max_width": max_width}
        logger.debug(f"Computed grid size stats: {grid_size_stats}")
        return grid_size_stats

    def get_symbol_frequencies(self) -> Dict[str, float]:
        """
        Computes symbol frequencies by iterating over samples on-the-fly.
        Returns:
            Dict[str, float]: A dictionary mapping symbols to their frequencies.
        """
        symbol_counts = np.zeros(self.num_symbols, dtype=int)
        total_symbols = 0

        for file_path, sample_idx in self.index_mapping:
            if file_path is None:
                sample = self.data_source[sample_idx]
                input_symbols = self._get_symbols(sample['input'])
                output_symbols = self._get_symbols(sample['output'])
            else:
                try:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)
                    if isinstance(task_data, dict):
                        samples = task_data.get('test', []) if self.is_test else task_data.get('train', [])
                        sample = samples[sample_idx]
                        input_symbols = self._get_symbols(sample['input'])
                        output_symbols = self._get_symbols(sample['output'])
                    elif isinstance(task_data, list):
                        sample = task_data[sample_idx]
                        input_symbols = self._get_symbols(sample['input'])
                        output_symbols = self._get_symbols(sample['output'])
                    else:
                        raise ValueError(f"Unexpected data format in file {file_path}: {type(task_data)}")
                except Exception as e:
                    logger.error(f"Error processing sample {sample_idx} from file {file_path}: {e}", exc_info=True)
                    continue  # Skip this sample

            symbol_counts += np.bincount(input_symbols.flatten(), minlength=self.num_symbols)
            symbol_counts += np.bincount(output_symbols.flatten(), minlength=self.num_symbols)
            total_symbols += len(input_symbols.flatten()) + len(output_symbols.flatten())

        if total_symbols == 0:
            logger.warning("No symbols found in the dataset.")
            return {}

        symbol_freq = {str(symbol): float(count) / total_symbols for symbol, count in enumerate(symbol_counts)}
        logger.debug(f"Computed symbol frequencies: {symbol_freq}")
        return symbol_freq

    def _get_grid_shape(self, grid: Union[Dict, List, np.ndarray, torch.Tensor]) -> Tuple[int, int, int]:
        """
        Retrieves the shape of the grid without preprocessing.
        Returns:
            Tuple[int, int, int]: Shape in [C, H, W]
        """
        if isinstance(grid, list):
            grid_tensor = torch.tensor(grid, dtype=torch.float32)
        elif isinstance(grid, np.ndarray):
            grid_tensor = torch.from_numpy(grid).float()
        elif isinstance(grid, torch.Tensor):
            grid_tensor = grid.float()
        else:
            raise ValueError(f"Unexpected grid type: {type(grid)}")

        if grid_tensor.ndim == 2:
            grid_tensor = grid_tensor.unsqueeze(0)  # Add channel dimension

        return grid_tensor.shape  # [C, H, W]

    def _get_symbols(self, grid: Union[Dict, List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Retrieves symbol tensor without preprocessing.
        """
        if isinstance(grid, list):
            grid_tensor = torch.tensor(grid, dtype=torch.int64)
        elif isinstance(grid, np.ndarray):
            grid_tensor = torch.from_numpy(grid).long()
        elif isinstance(grid, torch.Tensor):
            grid_tensor = grid.long()
        else:
            raise ValueError(f"Unexpected grid type: {type(grid)}")
        
        if grid_tensor.ndim == 2:
            grid_tensor = grid_tensor.unsqueeze(0)  # Add channel dimension

        return grid_tensor

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
    
    def _preprocess_grid(self, grid: Union[Dict, List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        logger.debug(f"Preprocessing grid with initial type: {type(grid)}")
        
        # Convert grid to torch.Tensor if it's a list or numpy array
        if isinstance(grid, list):
            grid_tensor = torch.tensor(grid, dtype=torch.float32)
            logger.debug(f"Converted list to tensor with shape: {grid_tensor.shape}")
        elif isinstance(grid, np.ndarray):
            grid_tensor = torch.from_numpy(grid).float()
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
        print(f"Kronecker scaling input shape: {X.shape}")
        h, w = X.shape
        scale_h = target_height / h
        scale_w = target_width / w
        d = int(np.floor(min(scale_h, scale_w)))
        
        X_scaled = np.kron(X, np.ones((d, d)))
        print(f"Kronecker scaled output shape: {X_scaled.shape}")
        return X_scaled


    def reverse_scaling(self, X_orig, X_pred):
        print(f"Reverse scaling - Original shape: {X_orig.shape}, Prediction shape: {X_pred.shape}")
        h, w = X_orig.shape
        # Reshape X_pred to 2D if it's 1D
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape((int(np.sqrt(X_pred.size)), -1))
        
        X_pred_cropped = X_pred[:h, :w]  # Crop to original size
        
        if h == X_pred.shape[0] and w == X_pred.shape[1]:
            print("No rescaling needed")
            return X_pred_cropped
        
        # Calculate the downscale factor
        d_h = X_pred_cropped.shape[0] // h
        d_w = X_pred_cropped.shape[1] // w
        
        # Ensure the dimensions are compatible for reshaping
        if d_h > 0 and d_w > 0:
            try:
                X_rev = X_pred_cropped.reshape(h, d_h, w, d_w).mean(axis=(1, 3))
            except ValueError as e:
                print(f"Error during reshaping: {e}")
                print(f"X_pred_cropped shape: {X_pred_cropped.shape}, h: {h}, w: {w}, d_h: {d_h}, d_w: {d_w}")
                raise
        else:
            print(f"Invalid downscale factors: d_h={d_h}, d_w={d_w}")
            raise ValueError("Invalid dimensions for reverse scaling")
        # Resize the result to match the original target shape
        result = np.resize(X_rev.round().astype(int), X_orig.shape)
        print(f"Reverse scaled output shape: {result.shape}")
        return result

    def _scale_grid(self, grid: np.ndarray, height: int, width: int) -> np.ndarray:
        return grid  # No scaling, preserve original size

    def _pad_grid_torch(self, grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
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
        padded_grid = F.pad(grid, padding, mode='constant', value=0)
        return padded_grid


    @staticmethod
    def collate_fn(batch):
        # Debugging: Check batch size
        logger.debug(f"Collating batch of size: {len(batch)}")
        
        if not batch:
            logger.warning("Empty batch received")
            return torch.tensor([]), torch.tensor([]), []

        inputs, outputs, task_ids = zip(*batch)

        # Find maximum dimensions in the batch
        max_h = max(input_tensor.size(1) for input_tensor in inputs)
        max_w = max(input_tensor.size(2) for input_tensor in outputs)

        # Debugging: Print maximum dimensions
        logger.debug(f"Maximum height in batch: {max_h}")
        logger.debug(f"Maximum width in batch: {max_w}")

        # Pad inputs and outputs to the maximum size
        padded_inputs = torch.stack([
            F.pad(input_tensor, (0, max_w - input_tensor.size(2), 0, max_h - input_tensor.size(1)))
            for input_tensor in inputs
        ])

        padded_outputs = torch.stack([
            F.pad(output_tensor, (0, max_w - output_tensor.size(2), 0, max_h - output_tensor.size(1)))
            for output_tensor in outputs
        ])

        # Debugging: Verify shapes after padding
        print(f"Padded inputs shape: {padded_inputs.shape}")
        print(f"Padded outputs shape: {padded_outputs.shape}")

        return padded_inputs, padded_outputs, list(task_ids)
