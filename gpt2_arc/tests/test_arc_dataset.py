# gpt2_arc/tests/test_arc_dataset.py

import os
import numpy as np
import pytest
import torch
import random
import logging
import arckit
from torch.utils.data import DataLoader
from src.data.arc_dataset import ARCDataset, set_debug_mode
from src.utils.experiment_tracker import ExperimentTracker

# Set up logging for tests
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

@pytest.fixture(scope="module")
def debug_mode():
    set_debug_mode(True)
    yield
    set_debug_mode(False)
from unittest.mock import Mock
from arckit.data import TaskSet


@pytest.fixture
def sample_data():
    return [
        {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]},
        {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}
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
def test_dataset_statistics_computation(sample_data):
    dataset = ARCDataset(sample_data, debug=True)
    
    # Retrieve statistics
    grid_size_stats = dataset.get_grid_size_stats()
    symbol_frequencies = dataset.get_symbol_frequencies()
    
    # Assertions for grid size statistics
    assert "max_height" in grid_size_stats, "Grid size stats should include 'max_height'"
    assert "max_width" in grid_size_stats, "Grid size stats should include 'max_width'"
    assert grid_size_stats["max_height"] == 2, "Max height should be 2 for sample data"
    assert grid_size_stats["max_width"] == 2, "Max width should be 2 for sample data"
    
    # Assertions for symbol frequencies
    expected_frequencies = {
        0: 4 / 8,  # 4 occurrences of symbol '0'
        1: 4 / 8,  # 4 occurrences of symbol '1'
        # Assuming num_symbols=10, remaining symbols have frequency 0
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        9: 0.0
    }
    for symbol, freq in expected_frequencies.items():
        assert symbol_frequencies.get(symbol, 0.0) == freq, f"Frequency for symbol {symbol} should be {freq}"

def test_dataset_statistics_caching(sample_data):
    dataset = ARCDataset(sample_data, debug=True)
    
    # Ensure that statistics are cached
    assert hasattr(dataset, 'statistics'), "Dataset should have a 'statistics' attribute after caching"
    assert "grid_size_stats" in dataset.statistics, "Statistics should include 'grid_size_stats'"
    assert "symbol_frequencies" in dataset.statistics, "Statistics should include 'symbol_frequencies'"

    # Reload the dataset to ensure statistics are loaded from cache
    dataset_reloaded = ARCDataset(sample_data, debug=True)
    grid_size_stats = dataset_reloaded.get_grid_size_stats()
    symbol_frequencies = dataset_reloaded.get_symbol_frequencies()
    
    # Assertions to verify consistency
    assert grid_size_stats == dataset.statistics["grid_size_stats"], "Grid size stats should match after reloading from cache"
    assert symbol_frequencies == dataset.statistics["symbol_frequencies"], "Symbol frequencies should match after reloading from cache"
    dataset = ARCDataset(sample_data, debug=True)
    logger.debug(f"Dataset length: {len(dataset)}, expected: {len(sample_data)}")
    assert len(dataset) == len(sample_data), "Dataset length mismatch"
    
    input_grid, output_grid, *_ = dataset[0]
    logger.debug(f"Input grid shape: {input_grid.shape}, expected: (1, 30, 30)")
    logger.debug(f"Output grid shape: {output_grid.shape}, expected: (1, 30, 30)")
    
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    
    # Update the shape check to match the new preprocessing logic
    assert input_grid.shape == (1, 30, 30), "Input grid should have shape (1, 30, 30)"
    assert output_grid.shape == (1, 30, 30), "Output grid should have shape (1, 30, 30)"
    
    # Verify that the original data is preserved in the center of the padded grid
    center_input = input_grid[0, 14:16, 14:16]
    center_output = output_grid[0, 14:16, 14:16]
    
    logger.debug(f"Center input:\n{center_input}")
    logger.debug(f"Center output:\n{center_output}")
    
    assert torch.allclose(center_input, torch.tensor([[1., 0.], [0., 1.]])), "Input data not preserved correctly"
    assert torch.allclose(center_output, torch.tensor([[0., 1.], [1., 0.]])), "Output data not preserved correctly"
    dataset = ARCDataset(sample_data)
    assert len(dataset) == 2, "Dataset should have 2 samples"
    
    input_grid, output_grid, *_ = dataset[0]
    
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    
    # Update the shape check to match the new preprocessing logic
    assert input_grid.shape == (1, 30, 30), "Input grid should have shape (1, 30, 30)"
    assert output_grid.shape == (1, 30, 30), "Output grid should have shape (1, 30, 30)"
    
    # Verify that the original data is preserved in the center of the padded grid
    center_input = input_grid[0, 14:16, 14:16]
    center_output = output_grid[0, 14:16, 14:16]
    
    assert torch.allclose(center_input, torch.tensor([[1., 0.], [0., 1.]])), "Input data not preserved correctly"
    assert torch.allclose(center_output, torch.tensor([[0., 1.], [1., 0.]])), "Output data not preserved correctly"

#Skip
@pytest.mark.skip(reason="Skipping test for synthetic data because test is problematic")
def test_arc_dataset_synthetic_data(debug_mode):
    synthetic_data_path = "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/syntheticARC/tasks"
    assert os.path.isdir(synthetic_data_path), f"Directory does not exist: {synthetic_data_path}"
    train_dataset = ARCDataset(synthetic_data_path, is_test=False, debug=True)
    test_dataset = ARCDataset(synthetic_data_path, is_test=True, debug=True)

    assert len(train_dataset) > 0, "Synthetic train dataset should not be empty"
    assert len(test_dataset) > 0, "Synthetic test dataset should not be empty"
    logger.debug(f"Loaded {len(train_dataset.data)} synthetic tasks")
    logger.debug(f"Total train dataset length: {len(train_dataset)}")
    logger.debug(f"Total test dataset length: {len(test_dataset)}")

    total_train = sum(len(task['train']) for task in train_dataset.data)
    total_test = sum(len(task['test']) for task in test_dataset.data)
    logger.debug(f"Total train samples: {total_train}")
    logger.debug(f"Total test samples: {total_test}")

    for i, task in enumerate(train_dataset.data):
        print(f"Task {i} - Train samples: {len(task['train'])}, Test samples: {len(task['test'])}")

    assert len(train_dataset) == total_train, f"Train dataset length ({len(train_dataset)}) should match total train samples ({total_train})"
    assert len(test_dataset) == total_test, f"Test dataset length ({len(test_dataset)}) should match total test samples ({total_test})"

    if len(train_dataset) == 0:
        pytest.skip("Train dataset is empty; skipping random sample tests.")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    if len(train_dataset) < 3:
        pytest.skip("Not enough data in the train dataset for random sampling tests.")
    
    # Test a few random samples from the train dataset
    for i in range(3):
        idx = random.choice(range(len(train_dataset)))
        try:
            print(f"\nTrain Sample {i + 1}:")
            print(f"Generated index: {idx}")
            input_grid, output_grid = train_dataset[idx]
            print(f"Input grid shape: {input_grid.shape}")
            print(f"Output grid shape: {output_grid.shape}")
        except IndexError as e:
            print(f"Error: Attempted to access index {idx} which is out of range. Train dataset size is {len(train_dataset)}.")
            pytest.fail(f"Generated index {idx} out of range for train dataset size {len(train_dataset)}: {str(e)}")

    # Verify grid sizes
    max_h, max_w = train_dataset.max_grid_size
    assert max_h > 0 and max_w > 0, "Grid size should be positive"
    print(f"Maximum grid size: {train_dataset.max_grid_size}")

    # Verify access to train and test splits
    assert len(train_dataset.data) > 0, "Dataset should contain at least one task"
    assert 'train' in train_dataset.data[0], "Each task should have a 'train' split"
    assert 'test' in train_dataset.data[0], "Each task should have a 'test' split"

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")





def test_data_loader_parameters(sample_data):                         
    dataset = ARCDataset(sample_data)
    num_workers = 4  # Example value
    prefetch_factor = 2
    persistent_workers = True

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    assert train_loader.num_workers == num_workers, "num_workers should be set correctly"
    assert train_loader.prefetch_factor == prefetch_factor, "prefetch_factor should be set correctly"
    assert train_loader.persistent_workers == persistent_workers, "persistent_workers should be set correctly"

def test_dataloader_loading(sample_data):
    dataset = ARCDataset(sample_data)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        prefetch_factor=1,
        persistent_workers=True
    )

    for batch in data_loader:
        inputs, outputs, task_ids = batch
        assert inputs.shape == (1, 1, 30, 30), "Input shape mismatch"
        assert outputs.shape == (1, 1, 30, 30), "Output shape mismatch"
    dataset = ARCDataset(sample_data)
    input_grid, output_grid, *_ = dataset[0]

    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (1, 30, 30), "Input grid should have shape (1, 30, 30)"
    assert output_grid.shape == (1, 30, 30), "Output grid should have shape (1, 30, 30)"

    # Check if the original data is preserved in the center
    center_input = input_grid[0, 14:16, 14:16]
    center_output = output_grid[0, 14:16, 14:16]
    assert torch.allclose(center_input, torch.tensor([[1., 0.], [0., 1.]])), "Input data not preserved correctly"
    assert torch.allclose(center_output, torch.tensor([[0., 1.], [1., 0.]])), "Output data not preserved correctly"


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
    dataset = ARCDataset(sample_data, num_symbols=10)
    input_grid, output_grid, *_ = dataset[0]

    print(f"Input grid shape: {input_grid.shape}")
    print(f"Output grid shape: {output_grid.shape}")
    print(f"Input grid content:\n{input_grid}")
    print(f"Output grid content:\n{output_grid}")

    # Check that the grids are indeed 3D
    assert input_grid.ndim == 3, f"Expected 3D input grid, got {input_grid.ndim}D"
    assert output_grid.ndim == 3, f"Expected 3D output grid, got {output_grid.ndim}D"

    # Check the shape (1, 30, 30)
    assert input_grid.shape == (1, 30, 30), f"Preprocessed grid should have shape (1, 30, 30), but got {input_grid.shape}"
    assert output_grid.shape == (1, 30, 30), f"Preprocessed grid should have shape (1, 30, 30), but got {output_grid.shape}"

    # Check if the original data is preserved in the center
    expected_input = torch.zeros((1, 30, 30))
    expected_input[0, 14:16, 14:16] = torch.tensor([[1., 0.], [0., 1.]])

    expected_output = torch.zeros((1, 30, 30))
    expected_output[0, 14:16, 14:16] = torch.tensor([[0., 1.], [1., 0.]])

    print(f"Expected input:\n{expected_input}")
    print(f"Expected output:\n{expected_output}")

    assert torch.allclose(input_grid, expected_input), "Input grid data mismatch"
    assert torch.allclose(output_grid, expected_output), "Output grid data mismatch"

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
def test_experiment_tracker_logs_dataset_statistics(sample_data):
    # Initialize ARCDataset
    dataset = ARCDataset(sample_data, debug=True)
    
    # Initialize ExperimentTracker with mock config
    mock_config = {"dummy_key": "dummy_value"}
    tracker = ExperimentTracker(config=mock_config, project="test_project")
    
    # Retrieve dataset statistics
    train_grid_stats = dataset.get_grid_size_stats()
    train_symbol_freq = dataset.get_symbol_frequencies()
    val_grid_stats = dataset.get_grid_size_stats()  # Assuming same dataset for simplicity
    val_symbol_freq = dataset.get_symbol_frequencies()
    
    # Log metrics
    tracker.log_metric("train_max_grid_height", train_grid_stats.get("max_height", 0))
    tracker.log_metric("train_max_grid_width", train_grid_stats.get("max_width", 0))
    tracker.log_metric("train_symbol_frequencies", train_symbol_freq)

    tracker.log_metric("val_max_grid_height", val_grid_stats.get("max_height", 0))
    tracker.log_metric("val_max_grid_width", val_grid_stats.get("max_width", 0))
    tracker.log_metric("val_symbol_frequencies", val_symbol_freq)
    
    # Assertions to verify metrics are logged correctly
    assert tracker.metrics["train_max_grid_height"] == train_grid_stats.get("max_height", 0)
    assert tracker.metrics["train_max_grid_width"] == train_grid_stats.get("max_width", 0)
    assert tracker.metrics["train_symbol_frequencies"] == train_symbol_freq

    assert tracker.metrics["val_max_grid_height"] == val_grid_stats.get("max_height", 0)
    assert tracker.metrics["val_max_grid_width"] == val_grid_stats.get("max_width", 0)
    assert tracker.metrics["val_symbol_frequencies"] == val_symbol_freq
def test_collate_fn_output(): 
    sample_data = [
        {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
    ]
    dataset = ARCDataset(sample_data)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=ARCDataset.collate_fn)
    batch = next(iter(dataloader))

    assert isinstance(batch, list), "Collate function should return a list"
    assert len(batch) == 3, "Collate function should return a list with 3 elements"
    assert isinstance(batch[0], torch.Tensor), "First element should be a tensor (inputs)"
    assert isinstance(batch[1], torch.Tensor), "Second element should be a tensor (outputs)"
    assert batch[0].shape == (2, 1, 30, 30), "Input tensor should have shape (batch_size, 1, 30, 30)"
    assert batch[1].shape == (2, 1, 30, 30), "Output tensor should have shape (batch_size, 1, 30, 30)"
    assert batch[0].dtype == torch.float32, "Input tensor should be of type float32"
    assert batch[1].dtype == torch.float32, "Output tensor should be of type float32"

def test_getitem_output():
    sample_data = [
        {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
    ]
    dataset = ARCDataset(sample_data)
    input_grid, output_grid, *_ = dataset[0]

    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (1, 30, 30), "Input grid should have shape (1, 30, 30)"
    assert output_grid.shape == (1, 30, 30), "Output grid should have shape (1, 30, 30)"
    assert input_grid.dtype == torch.float32, "Input grid should be float32"
    assert output_grid.dtype == torch.float32, "Output grid should be float32"

#Skip
@pytest.mark.skip(reason="Skipping because test is problematic")
def test_arc_dataset_taskset_initialization(mock_taskset):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.debug(f"Mock TaskSet: {mock_taskset}")
    logger.debug(f"Mock TaskSet attributes: {dir(mock_taskset)}")
    
    print(f"Mock task train data: {mock_taskset.tasks[0].train}")
    print(f"Mock task test data: {mock_taskset.tasks[0].test}")
    dataset = ARCDataset(mock_taskset)
    
    logger.debug(f"Dataset length: {len(dataset)}")
    print(f"Dataset length: {len(dataset)}, Expected: 3")
    
    assert len(dataset) == 3, "Dataset should have 3 samples (2 train + 1 test)"
    input_grid, output_grid, *_ = dataset[0]
    print(f"Input grid shape: {input_grid.shape}, Expected: (1, 30, 30)")
    print(f"Output grid shape: {output_grid.shape}, Expected: (1, 30, 30)")
    
    assert isinstance(input_grid, torch.Tensor), "Input should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output should be a torch.Tensor"
    assert input_grid.shape == (1, 30, 30), "Input grid should have shape (1, 30, 30)"
    assert output_grid.shape == (1, 30, 30), "Output grid should have shape (1, 30, 30)"
    
    # Check if the original data is preserved in the center
    center_input = input_grid[0, 14:16, 14:16]
    center_output = output_grid[0, 14:16, 14:16]
    print(f"Center input: {center_input}")
    print(f"Center output: {center_output}")
    
    assert torch.allclose(center_input, torch.tensor([[1., 0.], [0., 1.]])), "Input data not preserved correctly"
    assert torch.allclose(center_output, torch.tensor([[0., 1.], [1., 0.]])), "Output data not preserved correctly"
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
    assert input_grid.shape == (1, 30, 30), "Input grid should have shape (1, 30, 30)"
    assert output_grid.shape == (1, 30, 30), "Output grid should have shape (1, 30, 30)"
    
    # Check if the original data is preserved in the center
    center_input = input_grid[0, 14:16, 14:16]
    center_output = output_grid[0, 14:16, 14:16]
    assert torch.allclose(center_input, torch.tensor([[1., 0.], [0., 1.]])), "Input data not preserved correctly"
    assert torch.allclose(center_output, torch.tensor([[0., 1.], [1., 0.]])), "Output data not preserved correctly"

from torch.utils.data import DataLoader

def test_arc_dataset_collate_fn(sample_data):
    logger.debug("Starting test_arc_dataset_collate_fn")
    dataset = ARCDataset(sample_data)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=ARCDataset.collate_fn)
    batch = next(iter(dataloader))
    input_batch, output_batch, *_ = batch
    logger.debug(f"Collated batch shapes - inputs: {input_batch.shape}, outputs: {output_batch.shape}")
    assert input_batch.shape == (2, 1, 30, 30), "Batched input should have shape (2, 1, 30, 30)"
    assert output_batch.shape == (2, 1, 30, 30), "Batched output should have shape (2, 1, 30, 30)"
    logger.debug("Completed test_arc_dataset_collate_fn")

def test_arc_dataset_variable_size_grids(sample_data):
    logger.debug("Starting test_arc_dataset_variable_size_grids")
    variable_data = sample_data + [{"input": [[1, 0, 2], [0, 2, 1], [2, 1, 0]], "output": [[2, 1, 0], [1, 0, 2], [0, 2, 1]]}]
    dataset = ARCDataset(variable_data)
    
    # Check first sample (2x2)
    input_grid_1, output_grid_1, *_ = dataset[0]
    assert input_grid_1.shape == (1, 30, 30), "First sample should have shape (1, 30, 30)"
    assert output_grid_1.shape == (1, 30, 30), "First sample should have shape (1, 30, 30)"
    
    # Check center of first sample (2x2)
    center_input_1 = input_grid_1[0, 14:16, 14:16]
    center_output_1 = output_grid_1[0, 14:16, 14:16]
    assert torch.allclose(center_input_1, torch.tensor([[1., 0.], [0., 1.]])), "First sample input data not preserved correctly"
    assert torch.allclose(center_output_1, torch.tensor([[0., 1.], [1., 0.]])), "First sample output data not preserved correctly"
    
    # Check third sample (3x3)
    input_grid_2, output_grid_2, *_ = dataset[2]
    assert input_grid_2.shape == (1, 30, 30), "Third sample should have shape (1, 30, 30)"
    assert output_grid_2.shape == (1, 30, 30), "Third sample should have shape (1, 30, 30)"
    
    # Check center of third sample (3x3)
    center_input_2 = input_grid_2[0, 13:16, 13:16]
    center_output_2 = output_grid_2[0, 13:16, 13:16]
    assert torch.allclose(center_input_2, torch.tensor([[1., 0., 2.], [0., 2., 1.], [2., 1., 0.]])), f"Third sample input data not preserved correctly. Got:\n{center_input_2}"
    assert torch.allclose(center_output_2, torch.tensor([[2., 1., 0.], [1., 0., 2.], [0., 2., 1.]])), f"Third sample output data not preserved correctly. Got:\n{center_output_2}"
    
    logger.debug("Completed test_arc_dataset_variable_size_grids")

def test_arc_dataset_with_arckit_data_get_task_id():
    # Load data using arckit
    train_set, _ = arckit.load_data()

    # Initialize the dataset
    dataset = ARCDataset(train_set, is_test=False)

    # Check that __getitem__ returns the correct structure
    input_grid, output_grid, task_id = dataset[0]
    assert isinstance(input_grid, torch.Tensor), "Input grid should be a torch.Tensor"
    assert isinstance(output_grid, torch.Tensor), "Output grid should be a torch.Tensor"
    assert isinstance(task_id, str), "Task ID should be a string"

    # Test the collate_fn
    batch = [dataset[i] for i in range(2)]  # Create a batch of two samples
    collated_inputs, collated_outputs, collated_task_ids = ARCDataset.collate_fn(batch)

    assert len(collated_task_ids) == 2, "Batch size should be 2"
    assert collated_inputs.shape[0] == 2, "Batch size should be 2"
    assert collated_outputs.shape[0] == 2, "Batch size should be 2"
