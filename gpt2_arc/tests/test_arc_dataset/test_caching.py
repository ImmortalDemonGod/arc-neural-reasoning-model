# test_arc_dataset/test_caching.py
from src.data.arc_dataset import ARCDataset

class TestARCDatasetCaching:
    """Test suite for dataset caching functionality."""

    def test_cache_creation_and_loading(self, real_arc_tasks, tmp_path):
        """Test cache creation and subsequent loading."""
        cache_dir = tmp_path / "cache"
        
        dataset1 = ARCDataset(
            data_source=real_arc_tasks,
            enable_caching=True,
            cache_dir=str(cache_dir)
        )
        
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
        
        dataset1 = ARCDataset(
            data_source=real_arc_tasks,
            num_symbols=11,
            cache_dir=str(cache_dir)
        )
        
        dataset2 = ARCDataset(
            data_source=real_arc_tasks,
            num_symbols=12,  # Different parameter
            cache_dir=str(cache_dir)
        )
        
        assert dataset1.cache.generate_path(dataset1.config) != \
               dataset2.cache.generate_path(dataset2.config)