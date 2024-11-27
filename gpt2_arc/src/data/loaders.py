# gpt2_arc/src/data/loaders.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import os
import json
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from cysimdjson import JSONParser
import tempfile  # Added tempfile to imports

from .utils.validation import validate_sample
from .utils.custom_exceptions import DataLoadingError, ValidationError
from .utils.grid_ops import GridOperations
try:
    from arckit.data import TaskSet, Task
except ImportError:
    TaskSet = None
    Task = None

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy arrays and tensors."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)


class BaseLoader(ABC):
    """Abstract base class for loading ARC dataset samples from different sources."""
    
    def __init__(self, num_symbols: int, pad_symbol_idx: int = 10):
        """Initialize loader with configuration parameters.
        
        Args:
            num_symbols: Number of unique symbols in the dataset
            pad_symbol_idx: Index used for padding
        """
        self.json_parser = JSONParser()
        self.num_symbols = num_symbols
        self.pad_symbol_idx = pad_symbol_idx
        # Device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load data from source and return list of samples.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing 'input', 'output', and 'task_id'
        
        Raises:
            DataLoadingError: If data cannot be loaded or is invalid
        """
        pass

    def _prepare_tensor(self, data: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Prepare data as proper tensor with correct shape and device placement."""
        try:
            # Convert to tensor if needed
            if not isinstance(data, torch.Tensor):
                tensor = torch.tensor(data, dtype=torch.float32)
            else:
                tensor = data.float()
            
            # Add batch dimension if needed
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            
            # Move to correct device
            tensor = tensor.to(self.device)
            
            return GridOperations.preprocess_grid(tensor, self.pad_symbol_idx)
        except Exception as e:
            raise ValidationError(f"Failed to prepare tensor: {str(e)}")

    def _validate_tensors(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> bool:
        """Validate tensor shapes and values.
        
        Args:
            input_tensor: Input grid tensor
            output_tensor: Output grid tensor
            
        Returns:
            bool: True if tensors are valid
            
        Raises:
            ValidationError: If tensors are invalid
        """
        try:
            # Check dimensionality
            if input_tensor.ndim != 3 or output_tensor.ndim != 3:
                raise ValidationError(f"Invalid tensor dimensions: input {input_tensor.ndim}D, output {output_tensor.ndim}D")

            # Check channel dimension
            if input_tensor.size(0) != 1 or output_tensor.size(0) != 1:
                raise ValidationError(f"Invalid channel dimension: input {input_tensor.size(0)}, output {output_tensor.size(0)}")

            # Check symbol range
            if input_tensor.max() >= self.num_symbols or output_tensor.max() >= self.num_symbols:
                raise ValidationError(f"Symbol values exceed allowed range (0-{self.num_symbols-1})")

            return True
        except ValidationError as e:
            logger.error(f"Tensor validation error: {str(e)}")
            raise


def create_loader(data_source: Union[str, List[Dict], 'TaskSet'], num_symbols: int, pad_symbol_idx: int = 10) -> BaseLoader:
    """Create appropriate loader based on data source type.
    
    Args:
        data_source: Source of the data (file path, directory path, list, or TaskSet)
        num_symbols: Number of unique symbols in the dataset
        pad_symbol_idx: Index used for padding
        
    Returns:
        BaseLoader: Appropriate loader instance for the data source
        
    Raises:
        ValueError: If data source type is not supported
        DataLoadingError: If temporary file creation fails
    """
    if isinstance(data_source, str):
        if os.path.isdir(data_source):
            return DirectoryLoader(data_source, num_symbols, pad_symbol_idx)
        else:
            return JSONFileLoader(data_source, num_symbols, pad_symbol_idx)
    elif isinstance(data_source, TaskSet):
        return TaskSetLoader(data_source, num_symbols, pad_symbol_idx)
    elif isinstance(data_source, list):
        # Create temporary file in a directory we know we have write access to
        try:
            # Create a temporary file with a .json extension
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_path = temp_file.name
                logger.debug(f"Created temporary file at: {temp_path}")
                
                # Write the data using the custom encoder
                json.dump(data_source, temp_file, cls=NumpyEncoder)
                temp_file.flush()  # Ensure all data is written
                
            # Create and return the loader
            loader = JSONFileLoader(temp_path, num_symbols, pad_symbol_idx)
            
            # Wrap the loader's load method to ensure cleanup
            original_load = loader.load
            def wrapped_load():
                try:
                    return original_load()
                finally:
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            logger.debug(f"Cleaned up temporary file: {temp_path}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temporary file {temp_path}: {e}")
            
            loader.load = wrapped_load
            return loader
            
        except Exception as e:
            # Clean up if something goes wrong
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            raise DataLoadingError(f"Failed to create temporary data file: {str(e)}")
    else:
        raise ValueError(f"Unsupported data source type: {type(data_source)}")


class JSONFileLoader(BaseLoader):
    """Loads ARC dataset samples from a single JSON file."""
    
    def __init__(self, file_path: str, num_symbols: int, pad_symbol_idx: int = 10):
        """Initialize loader with JSON file path.
        
        Args:
            file_path: Path to JSON file containing samples
            num_symbols: Number of unique symbols in the dataset
            pad_symbol_idx: Index used for padding
        """
        super().__init__(num_symbols, pad_symbol_idx)
        self.file_path = file_path
    
    def load(self) -> List[Dict[str, Any]]:
        """Load and parse samples from JSON file.
        
        Returns:
            List[Dict[str, Any]]: List of valid samples
            
        Raises:
            DataLoadingError: If file cannot be read or parsed
        """
        try:
            if not os.path.exists(self.file_path):
                raise DataLoadingError(f"File not found: {self.file_path}")
                
            if os.path.getsize(self.file_path) == 0:
                raise DataLoadingError(f"Empty file: {self.file_path}")
            
            samples = []
            raw_samples = self._load_json_content()
            
            # Process and validate samples
            for idx, sample in enumerate(raw_samples):
                try:
                    # Prepare tensors
                    input_tensor = self._prepare_tensor(sample['input'])
                    output_tensor = self._prepare_tensor(sample['output'])
                    
                    # Validate tensors (will raise ValidationError if invalid)
                    self._validate_tensors(input_tensor, output_tensor)
                    
                    # Handle task_id
                    task_id = sample.get('task_id', sample.get('id', f"task_{idx}"))
                    if not isinstance(task_id, str) or not task_id:
                        task_id = f"task_{idx}"
                        logger.warning(f"Invalid task_id at index {idx}, assigned: {task_id}")

                    # Create validated sample
                    processed_sample = {
                        "input": input_tensor,
                        "output": output_tensor,
                        "task_id": task_id
                    }
                    
                    if validate_sample(processed_sample, self.num_symbols):
                        samples.append(processed_sample)
                    else:
                        logger.warning(f"Sample validation failed at index {idx}")
                    
                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {str(e)}")
                    continue
            
            if not samples:
                raise DataLoadingError(f"No valid samples found in {self.file_path}")
                
            logger.info(f"Successfully loaded {len(samples)} samples from {self.file_path}")
            return samples
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load {self.file_path}: {str(e)}")

    def _load_json_content(self) -> List[Dict]:
        """Load and parse JSON content from file."""
        with open(self.file_path, 'rb') as f:
            try:
                # Try fast parser first
                parsed_json = self.json_parser.parse(f.read())
                
                # Convert to Python objects
                if parsed_json.is_array():
                    data = parsed_json.as_list()
                elif parsed_json.is_object():
                    data = parsed_json.as_dict()
                else:
                    raise DataLoadingError(f"Invalid JSON format in {self.file_path}")
                    
            except Exception as e:
                logger.warning(f"Fast JSON parsing failed, falling back to standard parser: {e}")
                f.seek(0)
                data = json.load(f)

        # Extract samples based on file structure
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Look for common container keys
            for key in ['samples', 'train', 'test', 'data']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise DataLoadingError(f"No valid sample array found in {self.file_path}")
        else:
            raise DataLoadingError(f"Unexpected data format in {self.file_path}")


class DirectoryLoader(BaseLoader):
    """Loads ARC dataset samples from a directory of JSON files."""
    
    def __init__(self, directory_path: str, num_symbols: int, pad_symbol_idx: int = 10):
        super().__init__(num_symbols, pad_symbol_idx)
        self.directory_path = directory_path
        
    def load(self) -> List[Dict[str, Any]]:
        """Load samples from all JSON files in directory."""
        try:
            if not os.path.exists(self.directory_path):
                raise DataLoadingError(f"Directory not found: {self.directory_path}")
                
            file_paths = []
            for root, _, files in os.walk(self.directory_path):
                for file in files:
                    if file.endswith('.json'):
                        file_paths.append(os.path.join(root, file))

            if not file_paths:
                raise DataLoadingError(f"No JSON files found in {self.directory_path}")

            all_samples = []
            max_workers = min(32, os.cpu_count() + 4)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._load_single_file, fp): fp 
                    for fp in file_paths
                }
                
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        samples = future.result()
                        all_samples.extend(samples)
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {str(e)}")

            if not all_samples:
                raise DataLoadingError("No valid samples loaded from any file")

            return all_samples

        except Exception as e:
            raise DataLoadingError(f"Failed to load directory {self.directory_path}: {str(e)}")

    def _load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load samples from a single file using JSONFileLoader.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List[Dict[str, Any]]: List of processed samples
        """
        return JSONFileLoader(file_path, self.num_symbols, self.pad_symbol_idx).load()


class TaskSetLoader(BaseLoader):
    """Loads ARC dataset samples from an ARCkit TaskSet."""
    
    def __init__(self, taskset: 'TaskSet', num_symbols: int, pad_symbol_idx: int = 10):
        """Initialize loader with TaskSet.
        
        Args:
            taskset: ARCkit TaskSet containing tasks
            num_symbols: Number of unique symbols in the dataset
            pad_symbol_idx: Index used for padding
        
        Raises:
            DataLoadingError: If TaskSet support is not available
        """
        if TaskSet is None:
            raise DataLoadingError("TaskSet loading requires ARCkit to be installed")
        
        super().__init__(num_symbols, pad_symbol_idx)
        self.taskset = taskset
        
    def load(self) -> List[Dict[str, Any]]:
        """Load samples from TaskSet.
        
        Returns:
            List[Dict[str, Any]]: List of processed samples
            
        Raises:
            DataLoadingError: If data cannot be loaded or processed
        """
        try:
            processed_data = []
            logger.debug(f"Processing TaskSet with {len(self.taskset.tasks)} tasks")
            
            for task in self.taskset.tasks:
                logger.debug(f"Processing task: {task.id}")
                
                # Process training samples
                for input_grid, output_grid in task.train:
                    try:
                        input_tensor = self._prepare_tensor(input_grid)
                        output_tensor = self._prepare_tensor(output_grid)
                        
                        if self._validate_tensors(input_tensor, output_tensor):
                            sample = {
                                "input": input_tensor,
                                "output": output_tensor,
                                "task_id": task.id
                            }
                            if validate_sample(sample, self.num_symbols):
                                processed_data.append(sample)
                            
                    except Exception as e:
                        logger.error(f"Error processing training sample in task {task.id}: {str(e)}")
                
                # Process test samples
                for input_grid, output_grid in task.test:
                    try:
                        input_tensor = self._prepare_tensor(input_grid)
                        output_tensor = self._prepare_tensor(output_grid)
                        
                        if self._validate_tensors(input_tensor, output_tensor):
                            sample = {
                                "input": input_tensor,
                                "output": output_tensor,
                                "task_id": task.id
                            }
                            if validate_sample(sample, self.num_symbols):
                                processed_data.append(sample)
                            
                    except Exception as e:
                        logger.error(f"Error processing test sample in task {task.id}: {str(e)}")
                
                logger.debug(f"Processed task {task.id}: {len(task.train)} train + {len(task.test)} test samples")
            
            if not processed_data:
                raise DataLoadingError("No valid samples extracted from TaskSet")
                
            logger.info(f"Successfully loaded {len(processed_data)} total samples from TaskSet")
            return processed_data
            
        except Exception as e:
            raise DataLoadingError(f"Failed to process TaskSet: {str(e)}")