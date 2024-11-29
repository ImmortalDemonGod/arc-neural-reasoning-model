# test_arc_dataset/test_initialization.py
import pytest
import torch
from src.data.arc_dataset import ARCDataset
from src.data.utils.custom_exceptions import DataLoadingError
import logging


class TestARCDatasetInitialization:
    """Test suite for ARCDataset initialization and configuration."""
    
    def test_basic_initialization(self, real_arc_tasks):
        """Test basic initialization with minimal parameters."""
        dataset = ARCDataset(data_source=real_arc_tasks)
        assert len(dataset) == len(real_arc_tasks)
        assert dataset.num_symbols == 11
        assert dataset.pad_symbol_idx == 10
        
        input_tensor, output_tensor, task_id = dataset[0]
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(output_tensor, torch.Tensor)
        assert isinstance(task_id, str)

    def test_initialization_with_config(self, real_arc_tasks, mock_cache, tmp_path):
        """Test full configuration initialization."""
        symbol_freq = {i: 1/11 for i in range(11)}
        dataset = ARCDataset(
            data_source=real_arc_tasks,
            is_test=True,
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=symbol_freq,
            debug=True,
            max_samples=2,
            enable_caching=True,
            cache_dir=str(tmp_path)
        )
        assert dataset.is_test
        assert dataset.max_samples == 2
        assert dataset.symbol_freq == symbol_freq

    def test_empty_source_handling(self):
        """Test handling of empty data source."""
        with pytest.raises(DataLoadingError):
            ARCDataset(data_source=[])

    def test_missing_field_handling(self):
        """Test handling of data with missing required fields."""
        incomplete_data = [
            {"input": [[[1, 2], [3, 4]]]},  # Missing output
            {"output": [[[5, 6], [7, 8]]]}   # Missing input
        ]
        with pytest.raises(DataLoadingError):
            dataset = ARCDataset(data_source=incomplete_data)
            _ = dataset[0]  # Force data loading

    def test_invalid_grid_format(self):
        """Test handling of invalid grid formats."""
        invalid_grids = [
            {
                "task_id": "invalid_1",
                "input": 123,  # Not a grid at all
                "output": [[[1, 2], [3, 4]]]
            }
        ]
        with pytest.raises(DataLoadingError):
            dataset = ARCDataset(data_source=invalid_grids)
            _ = dataset[0]

    def test_default_task_id_generation(self):
        """Test automatic generation of task IDs when missing."""
        data_without_ids = [
            {
                "input": [[[1, 2], [3, 4]]],
                "output": [[[5, 6], [7, 8]]]
            }
        ]
        dataset = ARCDataset(data_source=data_without_ids)
        _, _, task_id = dataset[0]
        assert task_id.startswith("task_")
        assert task_id != "default_task"