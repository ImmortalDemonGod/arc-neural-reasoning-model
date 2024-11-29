# test_arc_dataset/
# └── conftest.py
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

@pytest.fixture
def real_arc_tasks() -> List[Dict]:
    """Provide realistic ARC task data for testing."""
    return [
        {
            "task_id": "test_task_1",
            "input": [[[1, 0], [0, 1]]],
            "output": [[[0, 1], [1, 0]]]
        },
        {
            "task_id": "test_task_2",
            "input": [[[0, 1, 2], [1, 2, 0], [2, 0, 1]]],
            "output": [[[2, 1, 0], [1, 0, 2], [0, 2, 1]]]
        }
    ]

@pytest.fixture
def mixed_size_tasks() -> List[Dict]:
    """Provide tasks with varying grid sizes."""
    return [
        {
            "task_id": "small_grid",
            "input": [[[1]]],
            "output": [[[0]]]
        },
        {
            "task_id": "large_grid",
            "input": [[[i % 3 for i in range(5)] for _ in range(5)]],
            "output": [[[i % 3 for i in range(5)] for _ in range(5)]]
        }
    ]

@pytest.fixture
def mock_statistics(mocker):
    """Mock statistics computation with realistic returns."""
    stats = mocker.MagicMock()
    stats.compute_all_statistics.return_value = {
        'grid_size_stats': {'max_height': 30, 'max_width': 30},
        'symbol_frequencies': np.array([0.1] * 11)
    }
    return mocker.patch('src.data.arc_dataset.DatasetStatistics', return_value=stats)

@pytest.fixture
def mock_cache(tmp_path):
    """Provide a real temporary cache directory with mocked cache operations."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with patch('src.data.arc_dataset.ARCCache') as mock:
        instance = mock.return_value
        instance.load.return_value = None
        instance.cache_dir = cache_dir
        yield instance
