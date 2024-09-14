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

try:
    from arckit.data import TaskSet
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
        logger.debug(f"Initializing ARCDataset with data_source: {data_source}")
        # Add logging to verify task_id presence
        if isinstance(data_source, list):
            for item in data_source:
                if 'task_id' not in item:
                    logger.warning(f"Missing task_id in data item: {item}")
        self.debug_attr = "test"  # Simple attribute for testing
        set_debug_mode(debug)  # Set debug mode based on parameter
        logger.debug(f"Initializing ARCDataset with data_source type: {type(data_source)}")
        if isinstance(data_source, str):
            logger.debug(f"Data source path: {data_source}")
            if os.path.isdir(data_source):
                logger.debug("Processing synthetic data from directory")
                self.data = self._process_synthetic_data(data_source)
            elif os.path.isfile(data_source):
                logger.debug("Processing JSON data from file")
                with open(data_source, 'r') as f:
                    raw_data = json.load(f)
                self.data = self._process_json_data(raw_data)
            else:
                raise FileNotFoundError(f"Data source file or directory not found: {data_source}")
        elif isinstance(data_source, list):
            logger.debug("Processing list data")
            self.data = self._process_list_data(data_source)
        elif isinstance(data_source, tuple):
            logger.debug("Processing combined data")
            self.data = self._combine_data(*data_source)
        elif TaskSet is not None and isinstance(data_source, TaskSet):
            logger.debug("Processing ARCkit data")
            self.data = self._process_arckit_data(data_source)
        else:
            logger.error(f"Invalid data_source type: {type(data_source)}")
            raise ValueError("Data source must be either a file path, a list of tasks, or a TaskSet")

        logger.debug(f"Number of tasks: {len(self.data)}")
        logger.debug(f"First task structure: {self.data[0].keys()}")
        logger.debug(f"First train sample structure: {self.data[0]['train'][0].keys()}")
        logger.debug(f"First train input shape: {np.array(self.data[0]['train'][0]['input']).shape}")
        self.is_test = is_test
        self.num_symbols = num_symbols
        self.test_split = test_split
        logger.debug(f"test_split set to: {self.test_split}")
        self.test_split = test_split
        self.samples = []
        if TaskSet is not None and isinstance(data_source, TaskSet):
            for task in data_source.tasks:
                self.samples.extend(task.train)
                self.samples.extend(task.test)
        
        if isinstance(data_source, str):
            if os.path.isdir(data_source):
                self.data = self._process_synthetic_data(data_source)
            elif os.path.isfile(data_source):
                with open(data_source, 'r') as f:
                    raw_data = json.load(f)
                self.data = self._process_json_data(raw_data)
            else:
                raise FileNotFoundError(f"Data source file or directory not found: {data_source}")
        elif isinstance(data_source, list):
            self.data = self._process_list_data(data_source)
        elif isinstance(data_source, tuple):
            self.data = self._combine_data(*data_source)
        elif TaskSet is not None and isinstance(data_source, TaskSet):
            self.data = self._process_arckit_data(data_source)
        else:
            logger.error(f"Invalid data_source type: {type(data_source)}")
            raise ValueError("Data source must be either a file path, a list of tasks, or a TaskSet")
        
        print(f"Number of train samples: {sum(len(task['train']) for task in self.data)}")
        print(f"Number of test samples: {sum(len(task['test']) for task in self.data)}")
        self.max_grid_size = self._compute_max_grid_size()
        self._validate_data()

    def _process_json_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed_data = []
        for task in raw_data:
            logger.debug(f"Processing task: {task}")
            processed_task = {
                "train": [
                    {"input": np.array(example["input"]), "output": np.array(example["output"])}
                    for example in task["train"]
                ],
                "test": [
                    {"input": np.array(example["input"]), "output": np.array(example["output"])}
                    for example in task["test"]
                ]
            }
            processed_data.append(processed_task)
        # Flatten the data structure
        flattened_data = []
        for task in processed_data:
            flattened_data.extend(task['train'])
            flattened_data.extend(task['test'])
        
        return flattened_data

    def _validate_data(self):
        for task in self.data:
            for split in ["train", "test"]:
                if split in task:
                    for sample in task[split]:
                        if not ("input" in sample and "output" in sample):
                            raise ValueError(f"Each sample must contain 'input' and 'output'. Task: {task.get('id', 'unknown')}")
        print("Data validation passed.")

    def _compute_max_grid_size(self):
        max_h, max_w = 0, 0
        for task in self.data:
            for split in ['train', 'test']:
                for sample in task[split]:
                    if isinstance(sample['input'], torch.Tensor):
                        if sample['input'].dim() == 3:
                            h, w = sample['input'].shape[1], sample['input'].shape[2]
                        elif sample['input'].dim() == 2:
                            h, w = sample['input'].shape
                        else:
                            raise ValueError(f"Unexpected tensor dimensions: {sample['input'].dim()}")
                    elif isinstance(sample['input'], np.ndarray):
                        h, w = sample['input'].shape
                    elif isinstance(sample['input'], list):
                        h, w = len(sample['input']), len(sample['input'][0])
                    else:
                        raise TypeError(f"Unexpected input type: {type(sample['input'])}")
                    
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)
        
        logger.debug(f"Computed max grid size: ({max_h}, {max_w})")
        return (max_h, max_w)

    def _combine_data(self, official_data, synthetic_data_path):
        official_processed = self._process_arckit_data(official_data) if TaskSet is not None and isinstance(official_data, TaskSet) else official_data
        synthetic_processed = self._process_synthetic_data(synthetic_data_path)
        return official_processed + synthetic_processed

    def _process_synthetic_data(self, directory: str) -> List[Dict]:
        processed_data = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                with open(os.path.join(directory, filename), 'r') as f:
                    task_data = json.load(f)
                    processed_data.append(self._process_single_task(task_data))
        return processed_data

    def _process_single_task(self, task_data: Union[Dict, List]) -> Dict:
        logger.debug(f"Inside _process_single_task, test_split is: {self.test_split}")
        if isinstance(task_data, dict):
            train_examples = task_data.get("train", [])
            test_examples = task_data.get("test", [])
        elif isinstance(task_data, list):
            split_idx = int(len(task_data) * (1 - self.test_split))
            train_examples = task_data[:split_idx]
            test_examples = task_data[split_idx:]
        else:
            raise ValueError("Task data must be either a dictionary or a list")

        return {
            "train": [self._preprocess_grid(example) for example in train_examples],
            "test": [self._preprocess_grid(example) for example in test_examples]
        }

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

    def __len__(self) -> int:
        if self.is_test:
            total_samples = sum(len(task['test']) for task in self.data)
        else:
            total_samples = sum(len(task['train']) for task in self.data)
        logger.debug(f"Total samples in dataset: {total_samples}")
        return total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"ARCDataset __getitem__ called with idx {idx}")
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range (total samples: {len(self)})")

        current_idx = 0
        for task in self.data:
            split = 'test' if self.is_test else 'train'
            if idx < current_idx + len(task[split]):
                sample = task[split][idx - current_idx]
                input_grid = self._preprocess_grid(sample["input"])
                output_grid = self._preprocess_grid(sample["output"])
                logger.debug(f"Returning input shape: {input_grid.shape}, output shape: {output_grid.shape}")
                logger.debug(f"__getitem__ input dtype: {input_grid.dtype}, output dtype: {output_grid.dtype}")
                task_id = task.get("id", -1)  # Default to -1 if no id is found
                return input_grid, output_grid, task_id
            current_idx += len(task[split])

        raise RuntimeError("Unexpected error in __getitem__")

    def _validate_data(self):
        for task in self.data:
            for split in ["train", "test"]:
                if split not in task:
                    continue
                for idx, sample in enumerate(task[split]):
                    if "input" not in sample or "output" not in sample:
                        raise ValueError(f"Sample {idx} in task {split} set is missing 'input' or 'output' key")
                    input_data = sample["input"]
                    output_data = sample["output"]
                    if not (isinstance(input_data, (list, np.ndarray)) and isinstance(output_data, (list, np.ndarray))):
                        raise ValueError(f"Sample {idx} in task {split} set 'input' or 'output' must be a list or numpy array")
                    if isinstance(input_data, list):
                        input_data = np.array(input_data)
                    if isinstance(output_data, list):
                        output_data = np.array(output_data)
                    if input_data.ndim != 2 or output_data.ndim != 2:
                        raise ValueError(f"Sample {idx} in task {split} set 'input' and 'output' must be 2D lists")
                    if np.any(input_data >= self.num_symbols) or np.any(output_data >= self.num_symbols):
                        raise ValueError(f"Sample {idx} in task {split} set contains invalid symbols (>= {self.num_symbols})")

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
        X_pred_cropped = X_pred[:h, :w]  # Crop to original size
        
        if h == X_pred.shape[0] and w == X_pred.shape[1]:
            print("No rescaling needed")
            return X_pred_cropped
        
        d = X_pred.shape[0] // h
        X_rev = X_pred_cropped.reshape(h, d, w, d).mean(axis=(1, 3))
        result = X_rev.round().astype(int)
        print(f"Reverse scaled output shape: {result.shape}")
        return result
        if isinstance(grid, dict):
            input_grid = np.array(grid['input'])
            logger.debug(f"Original grid shape: {input_grid.shape}")
            logger.debug(f"Original grid content:\n{input_grid}")
        elif isinstance(grid, np.ndarray):
            input_grid = grid
            logger.debug(f"Original grid shape: {input_grid.shape}")
            logger.debug(f"Original grid content:\n{input_grid}")
        else:
            raise ValueError(f"Unexpected grid type: {type(grid)}")

        print(f"Preprocessing grid, input type: {type(grid)}")
        if isinstance(grid, dict):
            input_grid = np.array(grid['input'])
        elif isinstance(grid, np.ndarray):
            input_grid = grid
        else:
            raise ValueError(f"Unexpected grid type: {type(grid)}")

        print(f"Original grid shape: {input_grid.shape}")
        scaled_grid = self.kronecker_scale(input_grid)
        padded_grid = self.pad_grid(scaled_grid)
        
        grid_tensor = torch.tensor(padded_grid, dtype=torch.float32).unsqueeze(0)
        print(f"Final preprocessed grid shape: {grid_tensor.shape}")
        return grid_tensor

    def _scale_grid(self, grid: np.ndarray, height: int, width: int) -> np.ndarray:
        return grid  # No scaling, preserve original size

    def _pad_grid(self, grid: np.ndarray, height: int, width: int) -> np.ndarray:
        h, w = grid.shape
        pad_h = (height - h) // 2
        pad_w = (width - w) // 2
        return np.pad(grid, ((pad_h, height - h - pad_h), (pad_w, width - w - pad_w)), mode='constant')
    def _process_list_data(self, data_source: List[Dict]) -> List[Dict]:
        processed_data = []
        for item in data_source:
            processed_item = {
                "train": [{"input": np.array(item["input"]), "output": np.array(item["output"])}],
                "test": []
            }
            processed_data.append(processed_item)
        return processed_data

    @staticmethod
    def collate_fn(batch):
    @staticmethod
    def collate_fn(batch):
        print(f"Collating batch of size: {len(batch)}")
        inputs, outputs, task_ids = zip(*batch)
        
        print(f"Input shapes: {[i.shape for i in inputs]}")
        print(f"Output shapes: {[o.shape for o in outputs]}")

        # Find max dimensions in the batch
        max_h = max(i.size(1) for i in inputs)
        max_w = max(i.size(2) for i in inputs)

        print(f"Max dimensions: height={max_h}, width={max_w}")

        # Pad inputs and outputs to max size in the batch
        padded_inputs = torch.stack([F.pad(i, (0, max_w - i.size(2), 0, max_h - i.size(1))) for i in inputs])
        padded_outputs = torch.stack([F.pad(o, (0, max_w - o.size(2), 0, max_h - o.size(1))) for o in outputs])

        print(f"Padded input shape: {padded_inputs.shape}")
        print(f"Padded output shape: {padded_outputs.shape}")

        return padded_inputs, padded_outputs, list(task_ids)
