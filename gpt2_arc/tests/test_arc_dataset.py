import pytest
import torch
import numpy as np
from src.data.arc_dataset import ArcDataset

@pytest.fixture
def sample_data():
    return [
        {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]},
        {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
    ]

def test_arc_dataset_initialization(sample_data):
    dataset = ArcDataset(sample_data)
    assert len(dataset) == 2, "Dataset should have 2 samples"

def test_arc_dataset_getitem(sample_data):
    dataset = ArcDataset(sample_data)
    input_grid, output_grid = dataset[0]
    
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (30, 30, 10), "Input grid should have shape (30, 30, 10)"
    assert output_grid.shape == (30, 30, 10), "Output grid should have shape (30, 30, 10)"
    
    assert torch.all(input_grid[0, 0] == torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])), "Input grid values are incorrect"
    assert torch.all(output_grid[0, 0] == torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])), "Output grid values are incorrect"

def test_arc_dataset_len(sample_data):
    dataset = ArcDataset(sample_data)
    assert len(dataset) == len(sample_data), "Dataset length should match input data length"

def test_arc_dataset_invalid_data():
    invalid_data = [{'input': [1, 0], 'output': [[0, 1], [1, 0]]}]
    with pytest.raises(ValueError):
        ArcDataset(invalid_data)

def test_arc_dataset_invalid_symbols():
    invalid_data = [{'input': [[10, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}]
    with pytest.raises(ValueError):
        ArcDataset(invalid_data)

def test_arc_dataset_preprocessing():
    data = [{'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}]
    dataset = ArcDataset(data, max_grid_size=(5, 5), num_symbols=3)
    input_grid, _ = dataset[0]
    
    assert input_grid.shape == (5, 5, 3), "Preprocessed grid should have shape (5, 5, 3)"
    assert torch.all(input_grid[2:, :, :] == 0), "Padding should be all zeros"
    assert torch.all(input_grid[:, 2:, :] == 0), "Padding should be all zeros"
