# gp2_arc/src/data/arc_dataset.py
import os
import json
import random
from typing import Union, List, Dict, Tuple
import numpy as np
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
        self.data_files = []  # Initialize data_files as an empty list
        self.data_source = data_source
        self.num_samples = 0
        self.data = []
        set_debug_mode(debug)
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
                for file_path in self.data_files:
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)
                    if isinstance(task_data, dict):
                        samples = self._process_single_task(task_data)
                        self.data.extend(samples)
                    elif isinstance(task_data, list):
                        samples = self._process_list_data(task_data)
                        self.data.extend(samples)
                    else:
                        logger.error(f"Unexpected data format in file {file_path}: {type(task_data)}")
            elif os.path.isfile(data_source):
                with open(data_source, 'r') as f:
                    task_data = json.load(f)
                if isinstance(task_data, dict):
                    samples = self._process_single_task(task_data)
                    self.data.extend(samples)
                elif isinstance(task_data, list):
                    samples = self._process_list_data(task_data)
                    self.data.extend(samples)
                else:
                    logger.error(f"Unexpected data format in file {data_source}: {type(task_data)}")
            else:
                raise FileNotFoundError(f"Data source file or directory not found: {data_source}")
        elif TaskSet is not None and isinstance(data_source, TaskSet):
            samples = self._process_arckit_data(data_source)
            self.data.extend(samples)
        elif isinstance(data_source, list):
            samples = self._process_list_data(data_source)
            self.data.extend(samples)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        self.num_samples = len(self.data)

        # Add data validation
        self._validate_data()


    def __len__(self):
        return len(self.data)

    def get_num_samples(self):
        return self.num_samples
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["input"], sample["output"], sample["task_id"]

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


    def _process_single_task(self, task_data: Dict) -> List[Dict]:
        split_key = 'test' if self.is_test else 'train'
        samples = []
        task_id = task_data.get('id', f"task_{len(self.data) + 1}")  # Assign a default ID if not present
        for example in task_data.get(split_key, []):
            input_grid = self._preprocess_grid(example['input'])
            output_grid = self._preprocess_grid(example['output'])
            samples.append({
                "input": input_grid,
                "output": output_grid,
                "task_id": task_id
            })
        return samples

    def _process_arckit_data(self, taskset: 'TaskSet') -> List[Dict]:
        samples = []
        for task in taskset.tasks:
            examples = task.test if self.is_test else task.train
            for example in examples:
                input_tensor = self._preprocess_grid(input_grid)
                output_tensor = self._preprocess_grid(output_grid)
                samples.append({
                    "input": input_tensor,
                    "output": output_tensor,
                    "task_id": task.id
                })
        return samples

    def _process_list_data(self, data_list: List[Dict]) -> List[Dict]:
        samples = []
        for example in data_list:
            if 'input' in example and 'output' in example:
                input_grid = self._preprocess_grid(example['input'])
                output_grid = self._preprocess_grid(example['output'])
                task_id = example.get('task_id', f"task_{len(samples) + 1}")  # Assign a default ID if not present
                samples.append({
                    "input": input_grid,
                    "output": output_grid,
                    "task_id": task_id
                })
            else:
                logger.warning("Example missing 'input' or 'output' keys.")
        return samples


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
                        task_id = os.path.splitext(filename)[0]  # Use the filename (without extension) as the task ID
                        processed_task = self._process_single_task(task_data)
                        processed_task["id"] = task_id
                        self.data.append(processed_task)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from file {file_path}: {e}")



    def _preprocess_grid(self, example: Dict) -> Dict:
        if isinstance(example, dict) and 'input' in example and 'output' in example:
            input_grid = torch.tensor(example["input"], dtype=torch.float32).unsqueeze(0)
            output_grid = torch.tensor(example["output"], dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected example format: {example}")
        return {"input": input_grid, "output": output_grid}

    def _process_single_task(self, task_data: Dict) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        split_key = 'test' if self.is_test else 'train'
        samples = []
        for example in task_data.get(split_key, []):
            input_grid = self._preprocess_grid(example['input'])
            output_grid = self._preprocess_grid(example['output'])
            samples.append((input_grid, output_grid))
        return samples

    def _process_arckit_data(self, taskset: 'TaskSet') -> List[Dict]:
        processed_data = []
        logger.debug(f"Processing TaskSet with {len(taskset.tasks)} tasks")
        for task in taskset.tasks:
            logger.debug(f"Task ID: {task.id}")
        for task in taskset.tasks:
            logger.debug(f"Processing task: {task.id}")
            logger.debug(f"Task ID: {task.id}")
            logger.debug(f"Train samples: {len(task.train)}, Test samples: {len(task.test)}")
            processed_task = {
                "id": task.id,
                "train": [
                    {"input": np.array(ex[0]), "output": np.array(ex[1])}
                    for ex in task.train
                ],
                "test": [
                    {"input": np.array(ex[0]), "output": np.array(ex[1])}
                    for ex in task.test
                ]
            }
            processed_data.append(processed_task)
            logger.debug(f"Processed task {task.id}: Train samples: {len(processed_task['train'])}, Test samples: {len(processed_task['test'])}")
        logger.debug(f"Processed {len(processed_data)} tasks")
        return processed_data


    def _validate_data(self):
        for task in self.data:
            for split in ["train", "test"]:
                if split not in task:
                    continue
                for idx, sample in enumerate(task[split]):
                    if "input" not in sample or "output" not in sample:
                        raise KeyError(f"Sample {idx} in task {split} set is missing 'input' or 'output' key")
                    input_data = sample["input"]
                    output_data = sample["output"]
                    if not (isinstance(input_data, (list, np.ndarray)) and isinstance(output_data, (list, np.ndarray))):
                        logger.warning(f"Sample {idx} in task {split} set 'input' or 'output' must be a list or numpy array")
                        continue
                    if isinstance(input_data, list):
                        input_data = np.array(input_data)
                    if isinstance(output_data, list):
                        output_data = np.array(output_data)
                    if input_data.ndim != 2 or output_data.ndim != 2:
                        raise ValueError(f"Sample {idx} in task {split} set 'input' and 'output' must be 2D lists")
                    if np.any(input_data >= self.num_symbols) or np.any(output_data >= self.num_symbols):
                        logger.warning(f"Sample {idx} in task {split} set contains invalid symbols (>= {self.num_symbols})")

    def _compute_grid_size_stats(self):
        max_height, max_width = 0, 0
        for task in self.data:
            for split in ["train", "test"]:
                for sample in task[split]:
                    max_height = max(max_height, sample["input"].shape[0], sample["output"].shape[0])
                    max_width = max(max_width, sample["input"].shape[1], sample["output"].shape[1])
        self.max_grid_size = (max_height, max_width)

    def _compute_symbol_frequencies(self):
        symbol_counts = np.zeros(self.num_symbols, dtype=int)
        for task in self.data:
            for split in ["train", "test"]:
                for sample in task[split]:
                    symbol_counts += np.bincount(sample["input"].flatten(), minlength=self.num_symbols)
                    symbol_counts += np.bincount(sample["output"].flatten(), minlength=self.num_symbols)
        return symbol_counts / symbol_counts.sum()
    
    def _preprocess_grid(self, grid: Union[Dict, List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        logger.debug(f"Preprocessing grid with initial type: {type(grid)}")
        if isinstance(grid, list):
            grid_array = np.array(grid)
        elif isinstance(grid, np.ndarray):
            grid_array = grid
        else:
            raise ValueError(f"Unexpected grid type: {type(grid)}")

        logger.debug(f"Grid shape before padding: {grid_array.shape}")
        padded_grid = self._pad_grid(grid_array, height=30, width=30)

        logger.debug(f"Grid shape after padding: {padded_grid.shape}")
        grid_tensor = torch.tensor(padded_grid, dtype=torch.float32).unsqueeze(0)
        return grid_tensor
    def kronecker_scale(self, X, target_height=30, target_width=30):
        print(f"Kronecker scaling input shape: {X.shape}")
        h, w = X.shape
        scale_h = target_height / h
        scale_w = target_width / w
        d = int(np.floor(min(scale_h, scale_w)))
        
        X_scaled = np.kron(X, np.ones((d, d)))
        print(f"Kronecker scaled output shape: {X_scaled.shape}")
        return X_scaled

    def pad_grid(self, X, target_height=30, target_width=30):
        print(f"Padding input shape: {X.shape}")
        h, w = X.shape
        pad_h = (target_height - h) // 2
        pad_w = (target_width - w) // 2
        padded = np.pad(X, ((pad_h, target_height - h - pad_h), 
                            (pad_w, target_width - w - pad_w)), 
                        mode='constant')
        print(f"Padded output shape: {padded.shape}")
        return padded

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

    def _pad_grid(self, grid: np.ndarray, height: int, width: int) -> np.ndarray:
        h, w = grid.shape
        logger.debug(f"Grid shape before padding/cropping: (h={h}, w={w}), target: (height={height}, width={width})")

        if h > height or w > width:
            logger.debug("Grid is larger than target size. Cropping the grid.")
            grid = grid[:height, :width]
            h, w = grid.shape
            logger.debug(f"Grid shape after cropping: (h={h}, w={w})")

        pad_h = (height - h) // 2
        pad_w = (width - w) // 2
        pad_top = pad_h
        pad_bottom = height - h - pad_h
        pad_left = pad_w
        pad_right = width - w - pad_w

        logger.debug(f"Calculated padding - pad_top: {pad_top}, pad_bottom: {pad_bottom}, pad_left: {pad_left}, pad_right: {pad_right}")

        pad_top = max(0, pad_top)
        pad_bottom = max(0, pad_bottom)
        pad_left = max(0, pad_left)
        pad_right = max(0, pad_right)

        padded_grid = np.pad(
            grid,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant'
        )

        logger.debug(f"Padded grid shape: {padded_grid.shape}")

        return padded_grid
    def _process_list_data(self, data_source):
        # print(f"DEBUG: Processing {len(data_source)} items")
        processed_data = []
        for idx, item in enumerate(data_source):
            # print(f"DEBUG: Processing item {idx}")
            # print(f"DEBUG: Item type: {type(item)}")
            # print(f"DEBUG: Item content: {item}")

            if 'input' in item and 'output' in item and isinstance(item['input'], (list, np.ndarray)) and isinstance(item['output'], (list, np.ndarray)):
                processed_data.append(item)
            else:
                raise ValueError("Unexpected item format in data_source.")
    
        return processed_data


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
