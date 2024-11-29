# test_arc_dataset/test_statistics.py
import numpy as np
from src.data.arc_dataset import ARCDataset

class TestARCDatasetStatistics:
    """Test suite for dataset statistics computation and access."""

    def test_symbol_frequencies(self, real_arc_tasks):
        """Test symbol frequency computation and access."""
        dataset = ARCDataset(data_source=real_arc_tasks)
        freqs = dataset.get_symbol_frequencies()
        
        assert isinstance(freqs, np.ndarray)
        assert len(freqs) == dataset.num_symbols
        assert np.all(freqs >= 0) and np.all(freqs <= 1)

    def test_grid_size_statistics(self, mixed_size_tasks):
        """Test grid size statistics computation."""
        dataset = ARCDataset(data_source=mixed_size_tasks)
        stats = dataset.get_grid_size_stats()
        
        assert 'max_height' in stats
        assert 'max_width' in stats
        assert stats['max_height'] >= 5
        assert stats['max_width'] >= 5
