import pytest
import torch
import numpy as np
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch

@pytest.fixture
def valid_grid_data():
    """Provides sample grid data in multiple formats for testing."""
    base_data = [[1, 2], [3, 4]]
    return {
        'list': base_data,
        'numpy': np.array(base_data),
        'tensor': torch.tensor(base_data),
        'expected': torch.tensor(base_data).float()
    }

@pytest.fixture
def valid_sample():
    """Provides a complete valid sample with all required fields."""
    return {
        'input': [[1, 2], [3, 4]],
        'output': [[4, 3], [2, 1]],
        'task_id': 'test_task_001'
    }

@pytest.fixture
def temp_json_file(valid_sample):
    """Creates a temporary JSON file with valid sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump([valid_sample], tf)
        path = tf.name
    yield path
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def temp_json_dir(valid_sample):
    """Creates a temporary directory with multiple JSON files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(3):
            file_path = Path(temp_dir) / f'sample_{i}.json'
            sample = valid_sample.copy()
            sample['task_id'] = f'task_{i}'
            with file_path.open('w') as f:
                json.dump([sample], f)
        yield temp_dir

@pytest.fixture
def mock_grid_ops():
    """Provides a mock for GridOperations."""
    with patch('gpt2_arc.src.data.loaders.GridOperations') as mock:
        mock.preprocess_grid.return_value = torch.ones(1, 3, 3)
        yield mock

@pytest.fixture
def mock_validator():
    """Provides a mock for sample validation."""
    with patch('gpt2_arc.src.data.loaders.validate_sample') as mock:
        mock.return_value = True
        yield mock

@pytest.fixture
def mock_taskset():
    """Provides a mock TaskSet with predefined train/test samples."""
    class MockTask:
        def __init__(self, task_id, train_samples, test_samples):
            self.id = task_id
            self.train = train_samples
            self.test = test_samples

    class MockTaskSet:
        def __init__(self):
            self.tasks = [
                MockTask(
                    'task_001',
                    [(np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]]))],
                    [(np.array([[2, 3], [4, 5]]), np.array([[5, 4], [3, 2]]))]
                )
            ]

    return MockTaskSet()
