from gpt2_arc.src.data.loaders import JSONFileLoader, DataLoadingError
import pytest
import torch
import json
from unittest.mock import patch

class TestJSONFileLoader:
    """Test suite for JSONFileLoader functionality."""

    def test_load_valid_file(self, temp_json_file, mock_grid_ops, mock_validator):
        loader = JSONFileLoader(temp_json_file, num_symbols=10)
        samples = loader.load()
        assert len(samples) == 1
        assert all(key in samples[0] for key in ['input', 'output', 'task_id'])
        assert isinstance(samples[0]['input'], torch.Tensor)
        assert isinstance(samples[0]['output'], torch.Tensor)

    def test_load_nonexistent_file(self):
        loader = JSONFileLoader('nonexistent.json', num_symbols=10)
        with pytest.raises(DataLoadingError, match="File not found"):
            loader.load()

    def test_load_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")
        
        loader = JSONFileLoader(str(empty_file), num_symbols=10)
        with pytest.raises(DataLoadingError, match="Empty file"):
            loader.load()

    def test_load_invalid_json(self, tmp_path):
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid json content")
        
        loader = JSONFileLoader(str(invalid_file), num_symbols=10)
        with pytest.raises(DataLoadingError):
            loader.load()

def test_fallback_to_standard_parser(self, temp_json_file, mock_grid_ops):
    with patch('gpt2_arc.src.data.loaders.fast_parser', side_effect=Exception):
        loader = JSONFileLoader(temp_json_file, num_symbols=10)
        samples = loader.load()
        assert len(samples) > 0
