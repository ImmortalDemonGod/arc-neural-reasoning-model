# gp2_arc/src/data/arc_dataset.py

import json
import logging
import os
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from arckit.data import TaskSet
except ImportError:
    TaskSet = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_source: Union[str, List[Dict], 'TaskSet'],
        max_grid_size: Tuple[int, int] = (30, 30),
        num_symbols: int = 10,
    ):
        logger.debug(
            f"Initializing ArcDataset with data_source type: {type(data_source)}"
        )

        if isinstance(data_source, str):
            logger.debug(f"Loading data from file: {data_source}")
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"File not found: {data_source}")
            with open(data_source, "r") as f:
                self.data = json.load(f)
        elif isinstance(data_source, list):
            logger.debug("Using provided list data directly")
            self.data = data_source
        elif TaskSet is not None and (isinstance(data_source, TaskSet) or (hasattr(data_source, 'tasks') and isinstance(data_source.tasks, list))):
            logger.debug("Processing arckit TaskSet")
            self.data = self._process_taskset(data_source)
        else:
            raise ValueError(
                "data_source must be either a file path (str), a list of dictionaries, or an arckit TaskSet"
            )
        
        self.max_grid_size = max_grid_size
        self.num_symbols = num_symbols
        self._validate_data()
        logger.info(f"Initialized ArcDataset with {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def _process_taskset(self, taskset: 'TaskSet') -> List[Dict]:
        processed_data = []
        for task in taskset.tasks:
            for example in task.train + task.test:
                input_data, output_data = example
                processed_data.append({
                    "input": input_data.tolist() if isinstance(input_data, np.ndarray) else input_data,
                    "output": output_data.tolist() if isinstance(output_data, np.ndarray) else output_data
                })
        return processed_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        if "input" in sample and "output" in sample:
            input_grid = self._preprocess_grid(sample["input"])
            output_grid = self._preprocess_grid(sample["output"])
        else:
            raise IndexError(f"Sample {idx} is missing 'input' or 'output' key")
        logger.debug(
            f"Retrieved sample {idx}: input shape {input_grid.shape}, output shape {output_grid.shape}"
        )
        return input_grid, output_grid

    def _validate_data(self):
        for idx, sample in enumerate(self.data):
            if "input" not in sample or "output" not in sample:
                raise ValueError(f"Sample {idx} is missing 'input' or 'output' key")
            if not isinstance(sample["input"], list) or not isinstance(
                sample["output"], list
            ):
                raise ValueError(f"Sample {idx} 'input' or 'output' is not a list")
            if not all(isinstance(row, list) for row in sample["input"]) or not all(
                isinstance(row, list) for row in sample["output"]
            ):
                raise ValueError(f"Sample {idx} 'input' or 'output' is not a 2D list")
            if any(max(row) >= self.num_symbols for row in sample["input"]) or any(
                max(row) >= self.num_symbols for row in sample["output"]
            ):
                raise ValueError(
                    f"Sample {idx} contains invalid symbols (>= {self.num_symbols})"
                )

    def _preprocess_grid(self, grid: List[List[int]]) -> torch.Tensor:
        grid_array = np.array(grid)

        if np.any(grid_array >= self.num_symbols):
            raise ValueError(f"Grid contains invalid symbols (>= {self.num_symbols})")

        padded_grid = np.zeros(self.max_grid_size, dtype=int)
        padded_grid[: grid_array.shape[0], : grid_array.shape[1]] = grid_array
        logger.debug(f"Padded grid: \n{padded_grid}")

        one_hot_grid = np.eye(self.num_symbols)[padded_grid]

        # Ensure padding remains zero after one-hot encoding
        one_hot_grid[grid_array.shape[0] :, :, :] = 0
        one_hot_grid[:, grid_array.shape[1] :, :] = 0

        return torch.tensor(one_hot_grid, dtype=torch.float32)
