# gp2_arc/src/data/arc_dataset.py

import logging
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArcDataset(Dataset):
    def __init__(self, data: List[Dict], max_grid_size: Tuple[int, int] = (30, 30), num_symbols: int = 10):
        self.data = data
        self.max_grid_size = max_grid_size
        self.num_symbols = num_symbols
        self._validate_data()
        logger.info(f"Initialized ArcDataset with {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        input_grid = self._preprocess_grid(sample['input'])
        output_grid = self._preprocess_grid(sample['output'])
        logger.debug(f"Retrieved sample {idx}: input shape {input_grid.shape}, output shape {output_grid.shape}")
        return input_grid, output_grid

    def _validate_data(self):
        for idx, sample in enumerate(self.data):
            if 'input' not in sample or 'output' not in sample:
                raise ValueError(f"Sample {idx} is missing 'input' or 'output' key")
            if not isinstance(sample['input'], list) or not isinstance(sample['output'], list):
                raise ValueError(f"Sample {idx} 'input' or 'output' is not a list")
            if not all(isinstance(row, list) for row in sample['input']) or not all(isinstance(row, list) for row in sample['output']):
                raise ValueError(f"Sample {idx} 'input' or 'output' is not a 2D list")
            if any(max(row) >= self.num_symbols for row in sample['input']) or any(max(row) >= self.num_symbols for row in sample['output']):
                raise ValueError(f"Sample {idx} contains invalid symbols (>= {self.num_symbols})")

    def _preprocess_grid(self, grid: List[List[int]]) -> torch.Tensor:
        grid_array = np.array(grid)
        
        if np.any(grid_array >= self.num_symbols):
            raise ValueError(f"Grid contains invalid symbols (>= {self.num_symbols})")
        
        padded_grid = np.zeros(self.max_grid_size, dtype=int)
        padded_grid[:grid_array.shape[0], :grid_array.shape[1]] = grid_array
        logger.debug(f"Padded grid: \n{padded_grid}")
        
        one_hot_grid = np.eye(self.num_symbols)[padded_grid]
        
        return torch.tensor(one_hot_grid, dtype=torch.float32)
