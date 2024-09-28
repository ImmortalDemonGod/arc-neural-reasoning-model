# gpt2_arc/tests/test_synthetic_arc_dataset.py
import os
import pytest
import logging
from gpt2_arc.src.data.arc_dataset import ARCDataset

SYNTHETIC_DATA_PATH = "/workspaces/arc-neural-reasoning-model/gpt2_arc/src/data/SyntheticARC/tasks"

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def synthetic_dataset():
    logger.debug("Creating synthetic dataset")
    return ARCDataset(SYNTHETIC_DATA_PATH)

def test_synthetic_data_loading(synthetic_dataset):
    logger.debug("Testing synthetic data loading")
    assert len(synthetic_dataset) > 0, "Synthetic dataset is empty"

def test_synthetic_data_structure(synthetic_dataset):
    logger.debug("Testing synthetic data structure")
    sample = synthetic_dataset[0]
    assert isinstance(sample, tuple), "Sample should be a tuple"
    assert len(sample) == 3, "Sample should contain input, output, and task_id"
    input_grid, output_grid, task_id = sample
    
    assert input_grid.dim() == 3, "Input grid should be 3-dimensional (channel, height, width)"
    assert output_grid.dim() == 3, "Output grid should be 3-dimensional (channel, height, width)"
    assert isinstance(task_id, (int, str)), "Task ID should be an integer or string"

def test_all_synthetic_files_loaded():
    logger.debug("Testing all synthetic files loaded")
    file_count = len([f for f in os.listdir(SYNTHETIC_DATA_PATH) if f.endswith('.json')])
    dataset = ARCDataset(SYNTHETIC_DATA_PATH)
    assert len(dataset.data) == file_count, f"Number of loaded tasks ({len(dataset.data)}) doesn't match the number of JSON files ({file_count})"

def test_synthetic_data_content(synthetic_dataset):
    logger.debug("Testing synthetic data content")
    for i in range(len(synthetic_dataset)):
        input_grid, output_grid, _ = synthetic_dataset[i]
        assert input_grid.min() >= 0 and input_grid.max() <= 9, f"Input grid values should be between 0 and 9 (sample {i})"
        assert output_grid.min() >= 0 and output_grid.max() <= 9, f"Output grid values should be between 0 and 9 (sample {i})"

if __name__ == "__main__":
    pytest.main([__file__])
