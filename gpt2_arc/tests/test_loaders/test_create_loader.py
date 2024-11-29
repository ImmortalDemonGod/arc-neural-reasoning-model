from gpt2_arc.src.data.loaders import create_loader, JSONFileLoader, DirectoryLoader, TaskSetLoader
import pytest
import torch

class TestCreateLoader:
    """Test suite for loader creation functionality."""

    def test_create_json_file_loader(self, temp_json_file):
        loader = create_loader(temp_json_file, num_symbols=10)
        assert isinstance(loader, JSONFileLoader)

    def test_create_directory_loader(self, temp_json_dir):
        loader = create_loader(temp_json_dir, num_symbols=10)
        assert isinstance(loader, DirectoryLoader)

    def test_create_taskset_loader(self, mock_taskset):
        loader = create_loader(mock_taskset, num_symbols=10)
        assert isinstance(loader, TaskSetLoader)

    def test_create_loader_from_list(self, valid_sample):
        loader = create_loader([valid_sample], num_symbols=10)
        assert isinstance(loader, JSONFileLoader)

    def test_invalid_source_type(self):
        with pytest.raises(ValueError, match="Unsupported data source type"):
            create_loader(None, num_symbols=10)
