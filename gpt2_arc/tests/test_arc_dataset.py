# gpt2_arc/tests/test_arc_dataset.py

import numpy as np
import pytest
import torch
import random
from src.data.arc_dataset import ARCDataset
from unittest.mock import Mock
from arckit.data import TaskSet


@pytest.fixture
def sample_data():
    return [
        {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
    ]


@pytest.fixture
def mock_taskset():
    mock_task = Mock()
    mock_task.id = "mock_task_1"
    mock_task.train = [
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]))
    ]
    mock_task.test = [
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [1, 1]]))
    ]
    
    mock_taskset = Mock(spec=TaskSet)
    mock_taskset.tasks = [mock_task]
    return mock_taskset

def test_arc_dataset_initialization(sample_data):
    dataset = ARCDataset(sample_data)
    assert len(dataset) == 2, "Dataset should have 2 samples"
    input_grid, output_grid = dataset[0]
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (30, 30, 10), "Input grid should have shape (30, 30, 10)"
    assert output_grid.shape == (30, 30, 10), "Output grid should have shape (30, 30, 10)"

def test_arc_dataset_synthetic_data():
    synthetic_data_path = "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/syntheticARC/tasks"
    dataset = ARCDataset(synthetic_data_path, is_test=False)

    assert len(dataset) > 0, "Synthetic dataset should not be empty"
    print(f"Loaded {len(dataset)} synthetic tasks")

    # Test a few random samples
    for i in range(3):
        idx = random.randint(0, len(dataset) - 1)
        input_grid, output_grid = dataset[idx]
        print(f"\nSample {i + 1}:")
        print(f"Input grid shape: {input_grid.shape}")
        print(f"Output grid shape: {output_grid.shape}")

    # Verify grid sizes and symbol range
    assert dataset.max_grid_size[0] <= 30 and dataset.max_grid_size[1] <= 30, "Grid size exceeds maximum allowed"
    assert max(dataset.symbol_frequencies) < dataset.num_symbols, "Symbol values exceed allowed range"

    print(f"Maximum grid size: {dataset.max_grid_size}")
    print(f"Symbol frequencies: {dataset.symbol_frequencies}")


def test_arc_dataset_taskset_initialization(mock_taskset):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.debug(f"Mock TaskSet: {mock_taskset}")
    logger.debug(f"Mock TaskSet attributes: {dir(mock_taskset)}")
    
    dataset = ARCDataset(mock_taskset)
    
    logger.debug(f"Dataset length: {len(dataset)}")
    
    assert len(dataset) == 3, "Dataset should have 3 samples (2 train + 1 test)"
    input_grid, output_grid = dataset[0]
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (30, 30, 10), "Input grid should have shape (30, 30, 10)"
    assert output_grid.shape == (30, 30, 10), "Output grid should have shape (30, 30, 10)"


def test_arc_dataset_getitem(sample_data):
    dataset = ARCDataset(sample_data)
    input_grid, output_grid = dataset[0]

    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (30, 30, 10), "Input grid should have shape (30, 30, 10)"
    assert output_grid.shape == (30, 30, 10), "Output grid should have shape (30, 30, 10)"

    assert torch.all(
        input_grid[0, 0] == torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    ), "Input grid values are incorrect"
    assert torch.all(
        output_grid[0, 0] == torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ), "Output grid values are incorrect"


def test_arc_dataset_len(sample_data):
    print("Debugging: Entering test_arc_dataset_len")
    print(f"Debugging: sample_data = {sample_data}")
    dataset = ARCDataset(sample_data)
    print(f"Debugging: len(dataset) = {len(dataset)}, len(sample_data) = {len(sample_data)}")
    assert len(dataset) == len(sample_data), "Dataset length should match input data length"
    print("Debugging: Exiting test_arc_dataset_len")


def test_arc_dataset_invalid_data(sample_data):
    invalid_data = [{"input": [1, 0], "output": [[0, 1], [1, 0]]}]
    with pytest.raises(ValueError, match="must be 2D lists"):
        ARCDataset(invalid_data)

    invalid_data = [{"input": [[1, 0], [0, 1]], "output": "not a list"}]
    with pytest.raises(ValueError, match="must be a list or numpy array"):
        ARCDataset(invalid_data)

    invalid_data = [{"input": [[1, 0], [0, 1]], "output": [[10, 1], [1, 0]]}]
    with pytest.raises(ValueError, match="contains invalid symbols"):
        ARCDataset(invalid_data)

    invalid_data = [{"input": [[1, 0], [0, 1]]}]  # Missing 'output'
    with pytest.raises(ValueError, match="missing 'input' or 'output' key"):
        ARCDataset(invalid_data)

    invalid_data = [{"input": "not a list", "output": [[0, 1], [1, 0]]}]
    with pytest.raises(ValueError, match="'input' is not a list"):
        ARCDataset(invalid_data)

    invalid_data = [{"input": [[0, 1], [1, 0]], "output": 5}]
    with pytest.raises(ValueError, match="'output' is not a list"):
        ARCDataset(invalid_data)


def test_arc_dataset_preprocess_grid(sample_data):
    dataset = ARCDataset(sample_data, max_grid_size=(5, 5), num_symbols=3)
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

@pytest.fixture
def mock_taskset():
    mock_task = Mock()
    mock_task.id = "mock_task_1"
    mock_task.train = [
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]))
    ]
    mock_task.test = [
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [1, 1]]))
    ]
    
    mock_taskset = Mock(spec=TaskSet)
    mock_taskset.tasks = [mock_task]
    return mock_taskset

def test_arc_dataset_taskset_initialization(mock_taskset):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.debug(f"Mock TaskSet: {mock_taskset}")
    logger.debug(f"Mock TaskSet attributes: {dir(mock_taskset)}")
    
    dataset = ARCDataset(mock_taskset)
    
    logger.debug(f"Dataset length: {len(dataset)}")
    
    assert len(dataset) == 3, "Dataset should have 3 samples (2 train + 1 test)"
    input_grid, output_grid = dataset[0]
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (30, 30, 10), "Input grid should have shape (30, 30, 10)"
    assert output_grid.shape == (30, 30, 10), "Output grid should have shape (30, 30, 10)"
