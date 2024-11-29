from gpt2_arc.src.data.loaders import DirectoryLoader, DataLoadingError
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

class TestDirectoryLoader:
    """Test suite for DirectoryLoader functionality."""

    def test_load_valid_directory(self, temp_json_dir, mock_grid_ops, mock_validator):
        loader = DirectoryLoader(temp_json_dir, num_symbols=10)
        samples = loader.load()
        assert len(samples) == 3
        assert all(isinstance(s['task_id'], str) for s in samples)
        assert all(isinstance(s['input'], torch.Tensor) for s in samples)
        assert all(isinstance(s['output'], torch.Tensor) for s in samples)

    def test_load_empty_directory(self, tmp_path):
        loader = DirectoryLoader(str(tmp_path), num_symbols=10)
        with pytest.raises(DataLoadingError, match="No JSON files found"):
            loader.load()

    def test_load_directory_with_some_invalid_files(self, temp_json_dir):
        invalid_file = Path(temp_json_dir) / "invalid.json"
        invalid_file.write_text("invalid json")
        
        loader = DirectoryLoader(temp_json_dir, num_symbols=10)
        samples = loader.load()
        assert len(samples) == 3  # Only valid files should be processed

def test_concurrent_loading(self, temp_json_dir, mock_grid_ops):
    loader = DirectoryLoader(temp_json_dir, num_symbols=10)
    
    def mock_load_file(file_path):
        return [{'input': torch.ones(1,3,3), 
                'output': torch.ones(1,3,3), 
                'task_id': 'test'}]

    with patch('concurrent.futures.ThreadPoolExecutor', autospec=True) as mock_executor:
        mock_pool = MagicMock()
        mock_pool.submit.return_value.result.return_value = mock_load_file('')
        mock_executor.return_value.__enter__.return_value = mock_pool
        
        samples = loader.load()
        assert mock_executor.called
        assert len(samples) > 0