# gp2_arc/src/data/arc_dataset.py
import os
import json
import random
from typing import Union, List, Dict, Tuple
import numpy as np
import torch
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
        max_grid_size: Tuple[int, int] = (30, 30),
        num_symbols: int = 10,
        test_split: float = 0.2,
    ):
        self.is_test = is_test
        self.max_grid_size = max_grid_size
        self.num_symbols = num_symbols
        self.test_split = test_split

        if isinstance(data_source, str):
            if os.path.isdir(data_source):
                self.data = self._process_synthetic_data(data_source)
            elif os.path.isfile(data_source):
                with open(data_source, 'r') as f:
                    self.data = json.load(f)
            else:
                raise FileNotFoundError(f"Data source file or directory not found: {data_source}")
        elif isinstance(data_source, tuple):
            official_data, synthetic_data_path = data_source
            self.data = self._combine_data(official_data, synthetic_data_path)
        elif isinstance(data_source, list):
            self.data = self._process_list_data(data_source)
        elif isinstance(data_source, dict):
            self.data = self._process_synthetic_data(data_source)
        elif TaskSet is not None and isinstance(data_source, TaskSet):
            self.data = self._process_arckit_data(data_source)
        else:
            raise ValueError(
                "Data must be either a file path, a list of tasks, a dictionary of synthetic data, a TaskSet from arckit, or a tuple of (official_data, synthetic_data_path)"
            )

        self._validate_data()
        self._compute_grid_size_stats()
        self.symbol_frequencies = self._compute_symbol_frequencies()

    def _combine_data(self, official_data, synthetic_data_path):
        official_processed = self._process_arckit_data(official_data) if TaskSet is not None and isinstance(official_data, TaskSet) else official_data
        synthetic_processed = self._process_synthetic_data(synthetic_data_path)
        return official_processed + synthetic_processed

    def _process_synthetic_data(self, data_source: Union[str, Dict]) -> List[Dict]:
        if isinstance(data_source, str):
            return self._load_synthetic_data_from_directory(data_source)
        elif isinstance(data_source, dict):
            return [self._process_single_task(data_source)]
        else:
            raise ValueError("Synthetic data must be either a directory path or a dictionary")

    def _load_synthetic_data_from_directory(self, directory: str) -> List[Dict]:
        processed_data = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                with open(os.path.join(directory, filename), 'r') as f:
                    task_data = json.load(f)
                    processed_data.append(self._process_single_task(task_data))
        return processed_data

    def _process_single_task(self, task_data: Union[Dict, List]) -> Dict:
        if isinstance(task_data, dict):
            all_examples = task_data.get("train", []) + task_data.get("test", [])
        elif isinstance(task_data, list):
            all_examples = task_data
        else:
            raise ValueError("Task data must be either a dictionary or a list")

        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * (1 - self.test_split))

        return {
            "id": task_data.get("id", "unknown") if isinstance(task_data, dict) else "unknown",
            "train": [
                {"input": np.array(ex["input"]), "output": np.array(ex["output"])}
                for ex in all_examples[:split_idx]
            ],
            "test": [
                {"input": np.array(ex["input"]), "output": np.array(ex["output"])}
                for ex in all_examples[split_idx:]
            ]
        }

    def _process_arckit_data(self, taskset: 'TaskSet') -> List[Dict]:
        processed_data = []
        logger.debug(f"Processing TaskSet with {len(taskset.tasks)} tasks")
        for task in taskset.tasks:
            logger.debug(f"Processing task: {task.id}")
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
        logger.debug(f"Processed {len(processed_data)} tasks")
        return processed_data

    def __len__(self) -> int:
        return sum(len(task['train']) + len(task['test']) for task in self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        task_idx = 0
        split = "test" if self.is_test else "train"
        remaining_idx = idx

        while task_idx < len(self.data):
            if split in self.data[task_idx] and remaining_idx < len(self.data[task_idx][split]):
                sample = self.data[task_idx][split][remaining_idx]
                input_grid = self._preprocess_grid(sample["input"])
                output_grid = self._preprocess_grid(sample["output"])
                return input_grid, output_grid
            
            if split in self.data[task_idx]:
                remaining_idx -= len(self.data[task_idx][split])
            task_idx += 1

        raise IndexError("Index out of range")

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
        if np.any(grid >= self.num_symbols):
            raise ValueError(f"Grid contains invalid symbols (>= {self.num_symbols})")

        # Pad the grid to max_grid_size
        padded_grid = np.zeros(self.max_grid_size, dtype=int)
        padded_grid[:grid.shape[0], :grid.shape[1]] = grid

        # One-hot encode the padded grid
        one_hot_grid = np.eye(self.num_symbols)[padded_grid]

        # Ensure the output shape is correct (num_symbols, height, width)
        one_hot_grid = np.transpose(one_hot_grid, (2, 0, 1))

        # Pad or crop to match the expected dimensions
        expected_shape = (self.num_symbols, self.max_grid_size[0], self.max_grid_size[1])
        if one_hot_grid.shape != expected_shape:
            padded_one_hot = np.zeros(expected_shape, dtype=one_hot_grid.dtype)
            padded_one_hot[:, :one_hot_grid.shape[1], :one_hot_grid.shape[2]] = one_hot_grid
            one_hot_grid = padded_one_hot

        return torch.tensor(one_hot_grid, dtype=torch.float32)
    def _process_list_data(self, data_source: List[Dict]) -> List[Dict]:
        processed_data = []
        for item in data_source:
            processed_item = {
                "train": [{"input": np.array(item["input"]), "output": np.array(item["output"])}],
                "test": []
            }
            processed_data.append(processed_item)
        return processed_data
