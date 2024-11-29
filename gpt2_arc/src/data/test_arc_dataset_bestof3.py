import pytest
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch, MagicMock

from src.data.arc_dataset import ARCDataset
from src.data.utils.custom_exceptions import DataLoadingError, ValidationError
from torch.utils.data import DataLoader

# Configure logging for tests
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============= Test Data Fixtures =============

@pytest.fixture
def real_arc_tasks() -> List[Dict]:
    """Provide realistic ARC task data for testing."""
    return [
        {
            "task_id": "test_task_1",
            "input": [[[1, 0], [0, 1]]],  # Note: Proper grid nesting
            "output": [[[0, 1], [1, 0]]]
        },
        {
            "task_id": "test_task_2",
            "input": [[[0, 1, 2], [1, 2, 0], [2, 0, 1]]],  # Different size grid
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

# ============= Mock Fixtures =============

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

# ============= Test Classes =============

class TestARCDatasetInitialization:
    """Test suite for ARCDataset initialization and configuration."""
    
    def test_basic_initialization(self, real_arc_tasks):
        """Test basic initialization with minimal parameters."""
        dataset = ARCDataset(data_source=real_arc_tasks)
        assert len(dataset) == len(real_arc_tasks)
        assert dataset.num_symbols == 11
        assert dataset.pad_symbol_idx == 10

    def test_initialization_with_config(self, real_arc_tasks, mock_cache, tmp_path):
        """Test full configuration initialization."""
        dataset = ARCDataset(
            data_source=real_arc_tasks,
            is_test=True,
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq={i: 1/11 for i in range(11)},
            debug=True,
            max_samples=2,
            enable_caching=True,
            cache_dir=str(tmp_path)
        )
        assert dataset.is_test
        assert dataset.max_samples == 2
        assert dataset.symbol_freq

    @pytest.mark.parametrize("invalid_source", [
        [],  # Empty list
        [{}],  # Empty dict
        [{"input": [[[1]]], "output": [[[1]]]}]  # Missing task_id
    ])
    def test_invalid_initialization(self, invalid_source):
        """Test initialization with invalid data sources."""
        with pytest.raises((DataLoadingError, ValidationError)):
            ARCDataset(data_source=invalid_source)

class TestARCDatasetProcessing:
    """Test suite for data processing and transformation."""

    def test_grid_preprocessing(self, real_arc_tasks):
        """Test grid preprocessing and padding."""
        dataset = ARCDataset(data_source=real_arc_tasks)
        input_tensor, output_tensor, _ = dataset[0]
        
        # Check tensor properties
        assert isinstance(input_tensor, torch.Tensor)
        assert input_tensor.shape == (1, 30, 30)  # Standard padded size
        assert torch.any(input_tensor == dataset.pad_symbol_idx)  # Padding exists

    def test_variable_size_handling(self, mixed_size_tasks):
        """Test handling of different grid sizes."""
        dataset = ARCDataset(data_source=mixed_size_tasks)
        
        # Test small grid
        small_input, small_output, _ = dataset[0]
        assert small_input.shape == (1, 30, 30)
        assert small_input[0, 14, 14] == 1  # Center value check
        
        # Test large grid
        large_input, large_output, _ = dataset[1]
        assert large_input.shape == (1, 30, 30)
        center_slice = large_input[0, 13:18, 13:18]  # 5x5 center
        assert torch.any(center_slice != dataset.pad_symbol_idx)  # Contains data

class TestARCDatasetStatistics:
    """Test suite for dataset statistics computation and access."""

    def test_symbol_frequencies(self, real_arc_tasks):
        """Test symbol frequency computation and access."""
        dataset = ARCDataset(data_source=real_arc_tasks)
        freqs = dataset.get_symbol_frequencies()
        
        assert isinstance(freqs, np.ndarray)
        assert len(freqs) == dataset.num_symbols
        assert np.all(freqs >= 0) and np.all(freqs <= 1)  # Valid probabilities

    def test_grid_size_statistics(self, mixed_size_tasks):
        """Test grid size statistics computation."""
        dataset = ARCDataset(data_source=mixed_size_tasks)
        stats = dataset.get_grid_size_stats()
        
        assert 'max_height' in stats
        assert 'max_width' in stats
        assert stats['max_height'] >= 5  # From mixed_size_tasks fixture
        assert stats['max_width'] >= 5

class TestARCDatasetSampling:
    """Test suite for dataset sampling and iteration."""

    def test_weighted_sampling(self, real_arc_tasks):
        """Test weighted sampling with symbol frequencies."""
        symbol_freq = {i: 1/(i+1) for i in range(11)}  # Non-uniform frequencies
        dataset = ARCDataset(
            data_source=real_arc_tasks,
            symbol_freq=symbol_freq
        )
        assert dataset.sampler is not None
        assert isinstance(dataset.sampler, torch.utils.data.WeightedRandomSampler)

    def test_dataloader_integration(self, real_arc_tasks):
        """Test integration with PyTorch DataLoader."""
        dataset = ARCDataset(data_source=real_arc_tasks)
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=dataset.collate_fn,
            num_workers=0  # For testing
        )
        
        batch = next(iter(loader))
        inputs, outputs, task_ids = batch
        
        assert inputs.shape == (2, 1, 30, 30)
        assert outputs.shape == (2, 1, 30, 30)
        assert len(task_ids) == 2

    def test_empty_batch_handling(self):
        """Test handling of empty batches in collate_fn."""
        empty_batch = []
        inputs, outputs, task_ids = ARCDataset.collate_fn(empty_batch)
        
        assert inputs.nelement() == 0
        assert outputs.nelement() == 0
        assert task_ids == []

class TestARCDatasetCaching:
    """Test suite for dataset caching functionality."""

    def test_cache_creation_and_loading(self, real_arc_tasks, tmp_path):
        """Test cache creation and subsequent loading."""
        cache_dir = tmp_path / "cache"
        
        # First creation
        dataset1 = ARCDataset(
            data_source=real_arc_tasks,
            enable_caching=True,
            cache_dir=str(cache_dir)
        )
        
        # Second creation should load from cache
        dataset2 = ARCDataset(
            data_source=real_arc_tasks,
            enable_caching=True,
            cache_dir=str(cache_dir)
        )
        
        assert len(dataset1) == len(dataset2)
        assert dataset1.get_grid_size_stats() == dataset2.get_grid_size_stats()

    def test_cache_invalidation(self, real_arc_tasks, tmp_path):
        """Test cache invalidation with changed parameters."""
        cache_dir = tmp_path / "cache"
        
        # Create with initial parameters
        dataset1 = ARCDataset(
            data_source=real_arc_tasks,
            num_symbols=11,
            cache_dir=str(cache_dir)
        )
        
        # Create with different parameters
        dataset2 = ARCDataset(
            data_source=real_arc_tasks,
            num_symbols=12,  # Different parameter
            cache_dir=str(cache_dir)
        )
        
        # Should have different cache paths
        assert dataset1.cache.generate_path(dataset1.config) != \
               dataset2.cache.generate_path(dataset2.config)