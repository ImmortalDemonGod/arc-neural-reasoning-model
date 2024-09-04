# gpt2_arc/tests/test_arc_dataset.py

import numpy as np
import pytest
import torch
from src.data.arc_dataset import ArcDataset
from unittest.mock import Mock


@pytest.fixture
def sample_data():
    return [
        {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
    ]


@pytest.fixture
def mock_taskset():
    mock_task = Mock()
    mock_task.train = [
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]))
    ]
    mock_task.test = [
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [1, 1]]))
    ]
    
    mock_taskset = Mock()
    mock_taskset.tasks = [mock_task]
    return mock_taskset

def test_arc_dataset_initialization(sample_data):
    dataset = ArcDataset(sample_data)
    assert len(dataset) == 2, "Dataset should have 2 samples"
    input_grid, output_grid = dataset[0]
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (30, 30, 10), "Input grid should have shape (30, 30, 10)"
    assert output_grid.shape == (30, 30, 10), "Output grid should have shape (30, 30, 10)"
    dataset = ArcDataset(mock_taskset)
    assert len(dataset) == 3, "Dataset should have 3 samples (2 train + 1 test)"
    
    input_grid, output_grid = dataset[0]
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (30, 30, 10), "Input grid should have shape (30, 30, 10)"
    assert output_grid.shape == (30, 30, 10), "Output grid should have shape (30, 30, 10)"
    dataset = ArcDataset(sample_data)
    assert len(dataset) == 2, "Dataset should have 2 samples"


def test_arc_dataset_getitem(sample_data):
    dataset = ArcDataset(sample_data)
    input_grid, output_grid = dataset[0]

    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (30, 30, 10), "Input grid should have shape (30, 30, 10)"
    assert output_grid.shape == (
        30,
        30,
        10,
    ), "Output grid should have shape (30, 30, 10)"

    assert torch.all(
        input_grid[0, 0] == torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    ), "Input grid values are incorrect"
    assert torch.all(
        output_grid[0, 0] == torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ), "Output grid values are incorrect"


def test_arc_dataset_len(sample_data):
    dataset = ArcDataset(sample_data)
    assert len(dataset) == len(
        sample_data
    ), "Dataset length should match input data length"


def test_arc_dataset_invalid_data(sample_data):
    invalid_data = [{"input": [1, 0], "output": [[0, 1], [1, 0]]}]
    with pytest.raises(ValueError, match="is not a 2D list"):
        ArcDataset(invalid_data)

    invalid_data = [{"input": [[1, 0], [0, 1]], "output": "not a list"}]
    with pytest.raises(ValueError, match="is not a list"):
        ArcDataset(invalid_data)

    invalid_data = [{"input": [[1, 0], [0, 1]], "output": [[10, 1], [1, 0]]}]
    with pytest.raises(ValueError, match="contains invalid symbols"):
        ArcDataset(invalid_data)

    invalid_data = [{"input": [[1, 0], [0, 1]]}]  # Missing 'output'
    with pytest.raises(ValueError, match="missing 'input' or 'output' key"):
        ArcDataset(invalid_data)

    invalid_data = [{"input": "not a list", "output": [[0, 1], [1, 0]]}]
    with pytest.raises(ValueError, match="'input' or 'output' is not a list"):
        ArcDataset(invalid_data)


def test_arc_dataset_preprocess_grid(sample_data):
    dataset = ArcDataset(sample_data, max_grid_size=(5, 5), num_symbols=3)
    input_grid, output_grid = dataset[0]

    assert input_grid.shape == (
        5,
        5,
        3,
    ), "Preprocessed grid should have shape (5, 5, 3)"
    assert torch.all(input_grid[2:, :, :] == 0), "Padding should be all zeros in rows"
    assert torch.all(
        input_grid[:, 2:, :] == 0
    ), "Padding should be all zeros in columns"
    assert torch.all(output_grid[2:, :, :] == 0), "Padding should be all zeros in rows"
    assert torch.all(
        output_grid[:, 2:, :] == 0
    ), "Padding should be all zeros in columns"
