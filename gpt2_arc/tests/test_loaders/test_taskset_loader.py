from gpt2_arc.src.data.loaders import TaskSetLoader, DataLoadingError, ValidationError
import pytest
import torch
from unittest.mock import Mock

class TestTaskSetLoader:
    """Test suite for TaskSetLoader functionality."""

    def test_load_valid_taskset(self, mock_taskset, mock_grid_ops, mock_validator):
        loader = TaskSetLoader(mock_taskset, num_symbols=10)
        samples = loader.load()
        assert len(samples) == 2  # One train + one test sample
        assert all(s['task_id'] == 'task_001' for s in samples)
        assert all(isinstance(s['input'], torch.Tensor) for s in samples)
        assert all(isinstance(s['output'], torch.Tensor) for s in samples)

    def test_empty_taskset(self):
        empty_taskset = Mock(tasks=[])
        loader = TaskSetLoader(empty_taskset, num_symbols=10)
        with pytest.raises(DataLoadingError, match="No valid samples"):
            loader.load()

    @pytest.mark.parametrize("invalid_grid", [
        None,
        "invalid",
        [[1, 2], [3, "4"]],  # Mixed types
        [[1, 2, 3], [4, 5]]  # Inconsistent dimensions
    ])
    def test_invalid_grids(self, mock_taskset, invalid_grid):
        mock_taskset.train = [(invalid_grid, invalid_grid)]
        loader = TaskSetLoader(mock_taskset, num_symbols=10)
        with pytest.raises((DataLoadingError, ValidationError)):
            loader.load()
