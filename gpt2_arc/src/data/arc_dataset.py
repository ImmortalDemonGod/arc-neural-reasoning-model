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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARCDataset(Dataset):
    def __init__(
        self,
        data_source: Union[str, List[Dict], 'TaskSet', Tuple[Union[List, 'TaskSet'], str]],
        is_test: bool = False,
        num_symbols: int = 10,
        test_split: float = 0.2,
    ):
        self.is_test = is_test
        self.num_symbols = num_symbols
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
        return processed_data

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
                    h, w = sample['input'].shape
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)
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
            logger.debug(f"Processing task: {task.id}")
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
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range (total samples: {len(self)})")

        current_idx = 0
        for task in self.data:
            split = 'test' if self.is_test else 'train'
            if idx < current_idx + len(task[split]):
                sample = task[split][idx - current_idx]
                input_grid = self._preprocess_grid(sample["input"])
                output_grid = self._preprocess_grid(sample["output"])
                logger.debug(f"__getitem__ input dtype: {input_grid.dtype}, output dtype: {output_grid.dtype}")
                return input_grid, output_grid
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

    def _preprocess_grid(self, grid: np.ndarray) -> torch.Tensor:
        logger.debug(f"Original grid shape: {grid.shape}")
        logger.debug(f"Original grid content:\n{grid}")

        # Pad the grid to 30x30
        padded_grid = self._pad_grid(grid, height=30, width=30)

        # Convert to tensor and add channel dimension
        grid_tensor = torch.tensor(padded_grid, dtype=torch.float32).unsqueeze(0)

        logger.debug(f"Preprocessed grid shape: {grid_tensor.shape}")
        logger.debug(f"Preprocessed grid content:\n{grid_tensor}")

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
        # This method will be used by DataLoader to prepare batches
        inputs, outputs = zip(*batch)
        
        # Find max dimensions in the batch
        max_h = max(i.size(1) for i in inputs)
        max_w = max(i.size(2) for i in inputs)

        # Pad inputs and outputs to max size in the batch
        padded_inputs = torch.stack([F.pad(i, (0, max_w - i.size(2), 0, max_h - i.size(1))) for i in inputs])
        padded_outputs = torch.stack([F.pad(o, (0, max_w - o.size(2), 0, max_h - o.size(1))) for o in outputs])

        return padded_inputs, padded_outputs
