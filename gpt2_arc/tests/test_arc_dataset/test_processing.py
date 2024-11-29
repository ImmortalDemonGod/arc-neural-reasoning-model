# test_arc_dataset/test_processing.py
import torch
from src.data.arc_dataset import ARCDataset

class TestARCDatasetProcessing:
    """Test suite for data processing and transformation."""

    def test_grid_preprocessing(self, real_arc_tasks):
        """Test grid preprocessing and padding."""
        dataset = ARCDataset(data_source=real_arc_tasks)
        input_tensor, output_tensor, _ = dataset[0]
        
        assert isinstance(input_tensor, torch.Tensor)
        assert input_tensor.shape == (1, 30, 30)
        assert torch.any(input_tensor == dataset.pad_symbol_idx)

    def test_variable_size_handling(self, mixed_size_tasks):
        """Test handling of different grid sizes."""
        dataset = ARCDataset(data_source=mixed_size_tasks)
        
        small_input, small_output, _ = dataset[0]
        assert small_input.shape == (1, 30, 30)
        assert small_input[0, 14, 14] == 1
        
        large_input, large_output, _ = dataset[1]
        assert large_input.shape == (1, 30, 30)
        center_slice = large_input[0, 13:18, 13:18]
        assert torch.any(center_slice != dataset.pad_symbol_idx)

