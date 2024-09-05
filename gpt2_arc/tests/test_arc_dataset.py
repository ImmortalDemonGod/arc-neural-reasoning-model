# gpt2_arc/tests/test_arc_dataset.py

import numpy as np
import pytest
import torch
import random
import logging
from src.data.arc_dataset import ARCDataset
from unittest.mock import Mock
from arckit.data import TaskSet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    assert input_grid.shape == (10, 2, 2), "Input grid should have shape (10, 2, 2)"
    assert output_grid.shape == (10, 2, 2), "Output grid should have shape (10, 2, 2)"

def test_arc_dataset_synthetic_data():
    synthetic_data_path = "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/syntheticARC/tasks"
    dataset = ARCDataset(synthetic_data_path, is_test=False)

    assert len(dataset) > 0, "Synthetic dataset should not be empty"
    print(f"Loaded {len(dataset.data)} synthetic tasks")
    print(f"Total dataset length: {len(dataset)}")
    
    total_train = sum(len(task['train']) for task in dataset.data)
    total_test = sum(len(task['test']) for task in dataset.data)
    print(f"Total train samples: {total_train}")
    print(f"Total test samples: {total_test}")

    for i, task in enumerate(dataset.data):
        print(f"Task {i} - Train samples: {len(task['train'])}, Test samples: {len(task['test'])}")

    if len(dataset) == 0:
        pytest.skip("Dataset is empty; skipping random sample tests.")

    # Test a few random samples
    for i in range(3):
        idx = random.randint(0, len(dataset) - 1)
        try:
            input_grid, output_grid = dataset[idx]
            print(f"\nSample {i + 1}:")
            print(f"Index: {idx}")
            print(f"Input grid shape: {input_grid.shape}")
            print(f"Output grid shape: {output_grid.shape}")
        except IndexError as e:
            pytest.fail(f"Generated index {idx} out of range for dataset size {len(dataset)}: {str(e)}")

    # Verify grid sizes
    max_h, max_w = dataset.max_grid_size
    assert max_h > 0 and max_w > 0, "Grid size should be positive"
    print(f"Maximum grid size: {dataset.max_grid_size}")

    # Verify that we can access train and test splits
    assert len(dataset.data) > 0, "Dataset should contain at least one task"
    assert 'train' in dataset.data[0], "Each task should have a 'train' split"
    assert 'test' in dataset.data[0], "Each task should have a 'test' split"


def test_arc_dataset_taskset_initialization(mock_taskset):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.debug(f"Mock TaskSet: {mock_taskset}")
    logger.debug(f"Mock TaskSet attributes: {dir(mock_taskset)}")
    logger.debug(f"Mock TaskSet tasks: {mock_taskset.tasks}")
    logger.debug(f"Length of Mock TaskSet tasks: {len(mock_taskset.tasks)}")

    dataset = ARCDataset(mock_taskset)
    
    logger.debug(f"Dataset length after initialization: {len(dataset)}")
    logger.debug(f"Dataset details: {[len(task['train']) + len(task['test']) for task in dataset.data]}")

    assert len(dataset) == 3, "Dataset should have 3 samples (2 train + 1 test)"
    input_grid, output_grid = dataset[0]
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (10, 2, 2), "Input grid should have shape (10, 2, 2)"
    assert output_grid.shape == (10, 2, 2), "Output grid should have shape (10, 2, 2)"


def test_arc_dataset_getitem(sample_data):
    dataset = ARCDataset(sample_data)
    input_grid, output_grid = dataset[0]

    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (10, 2, 2), "Input grid should have shape (10, 2, 2)"
    assert output_grid.shape == (10, 2, 2), "Output grid should have shape (10, 2, 2)"

    assert torch.all(
        input_grid[:, 0, 0] == torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    ), "Input grid values are incorrect"
    assert torch.all(
        output_grid[:, 0, 0] == torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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
    with pytest.raises(ValueError):
        ARCDataset(invalid_data)

    invalid_data = [{"input": [[1, 0], [0, 1]], "output": "not a list"}]
    with pytest.raises(ValueError):
        ARCDataset(invalid_data)

def test_arc_dataset_preprocess_grid(sample_data):
    dataset = ARCDataset(sample_data, num_symbols=3)
    input_grid, output_grid = dataset[0]

    print(f"Input grid shape: {input_grid.shape}")
    print(f"Output grid shape: {output_grid.shape}")
    print(f"Input grid content:\n{input_grid}")
    print(f"Output grid content:\n{output_grid}")

    # Check that the grids are indeed 3D
    assert input_grid.ndim == 3, f"Expected 3D input grid, got {input_grid.ndim}D"
    assert output_grid.ndim == 3, f"Expected 3D output grid, got {output_grid.ndim}D"

    assert input_grid.shape == (3, 2, 2), f"Preprocessed grid should have shape (3, 2, 2), but got {input_grid.shape}"
    assert output_grid.shape == (3, 2, 2), f"Preprocessed grid should have shape (3, 2, 2), but got {output_grid.shape}"

    # Check if the original data is preserved
    print(f"Shape of eye tensor: {torch.eye(input_grid.size(0)).shape}")
    print(f"Input grid size(0): {input_grid.size(0)}")
    
    # Instead of using torch.eye, let's create the expected input manually
    expected_input = torch.zeros((3, 2, 2))
    expected_input[0, 0, 0] = 1  # Representing 1 in the original input
    expected_input[1, 0, 1] = 1  # Representing 0 in the original input
    expected_input[1, 1, 0] = 1  # Representing 0 in the original input
    expected_input[0, 1, 1] = 1  # Representing 1 in the original input

    expected_output = torch.zeros((3, 2, 2))
    expected_output[1, 0, 0] = 1  # Representing 0 in the original output
    expected_output[0, 0, 1] = 1  # Representing 1 in the original output
    expected_output[0, 1, 0] = 1  # Representing 1 in the original output
    expected_output[1, 1, 1] = 1  # Representing 0 in the original output

    print(f"Expected input:\n{expected_input}")
    print(f"Expected output:\n{expected_output}")

    assert torch.all(input_grid == expected_input), "Input grid data mismatch"
    assert torch.all(output_grid == expected_output), "Output grid data mismatch"

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
    assert input_grid.shape == (10, 2, 2), "Input grid should have shape (10, 2, 2)"
    assert output_grid.shape == (10, 2, 2), "Output grid should have shape (10, 2, 2)"

from torch.utils.data import DataLoader

def test_arc_dataset_collate_fn(sample_data):
    logger.debug("Starting test_arc_dataset_collate_fn")
    dataset = ARCDataset(sample_data)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=ARCDataset.collate_fn)
    batch = next(iter(dataloader))
    input_batch, output_batch = batch
    logger.debug(f"Collated batch shapes - inputs: {input_batch.shape}, outputs: {output_batch.shape}")
    assert input_batch.shape == (2, 10, 2, 2), "Batched input should maintain original size"
    assert output_batch.shape == (2, 10, 2, 2), "Batched output should maintain original size"
    logger.debug("Completed test_arc_dataset_collate_fn")

def test_arc_dataset_variable_size_grids(sample_data):
    logger.debug("Starting test_arc_dataset_variable_size_grids")
    variable_data = sample_data + [{"input": [[1, 0, 2], [0, 2, 1], [2, 1, 0]], "output": [[2, 1, 0], [1, 0, 2], [0, 2, 1]]}]
    dataset = ARCDataset(variable_data)
    
    # Check first sample (2x2)
    input_grid_1, output_grid_1 = dataset[0]
    assert input_grid_1.shape == (10, 2, 2), "First sample should maintain 2x2 shape"
    assert output_grid_1.shape == (10, 2, 2), "First sample should maintain 2x2 shape"
    
    # Check third sample (3x3)
    input_grid_2, output_grid_2 = dataset[2]
    assert input_grid_2.shape == (10, 3, 3), "Third sample should maintain 3x3 shape"
    assert output_grid_2.shape == (10, 3, 3), "Third sample should maintain 3x3 shape"
    
    logger.debug("Completed test_arc_dataset_variable_size_grids")
