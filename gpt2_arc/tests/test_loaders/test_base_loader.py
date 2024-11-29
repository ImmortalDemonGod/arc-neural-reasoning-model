from gpt2_arc.src.data.loaders import BaseLoader, ValidationError
import pytest
import torch
import numpy as np

class TestBaseLoader:
    """Test suite for BaseLoader functionality."""
    
    class ConcreteLoader(BaseLoader):
        def load(self):
            return []

    def test_initialization(self):
        """Test proper initialization of BaseLoader."""
        loader = self.ConcreteLoader(num_symbols=10)
        assert loader.num_symbols == 10
        assert loader.pad_symbol_idx == 10
        assert isinstance(loader.device, torch.device)
        
    @pytest.mark.parametrize("data_type", ['list', 'numpy', 'tensor'])
    def test_prepare_tensor_valid_inputs(self, valid_grid_data, data_type, mock_grid_ops):
        loader = self.ConcreteLoader(num_symbols=10)
        result = loader._prepare_tensor(valid_grid_data[data_type])
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.device == loader.device

    @pytest.mark.parametrize("invalid_input", [
        "invalid string", 123, None,
        [[1, "2"], [3, 4]],  # Mixed types
        [[1, 2], [3, 4, 5]]  # Inconsistent dimensions
    ])
    def test_prepare_tensor_invalid_inputs(self, invalid_input):
        loader = self.ConcreteLoader(num_symbols=10)
        with pytest.raises(ValidationError):
            loader._prepare_tensor(invalid_input)

    @pytest.mark.parametrize("input_shape,output_shape,should_pass", [
        ((1, 3, 3), (1, 3, 3), True),
        ((1, 5, 5), (1, 5, 5), True),
        ((2, 3, 3), (1, 3, 3), False),  # Wrong batch size
        ((1, 3, 3), (3, 3), False),     # Missing dimension
        ((3, 3), (1, 3, 3), False),     # Missing dimension
    ])
    def test_validate_tensors(self, input_shape, output_shape, should_pass):
        loader = self.ConcreteLoader(num_symbols=10)
        input_tensor = torch.ones(*input_shape)
        output_tensor = torch.ones(*output_shape)
        
        if should_pass:
            assert loader._validate_tensors(input_tensor, output_tensor)
        else:
            with pytest.raises(ValidationError):
                loader._validate_tensors(input_tensor, output_tensor)
