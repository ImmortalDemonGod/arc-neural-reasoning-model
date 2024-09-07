import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gpt2_arc.benchmark import benchmark_model, main, BASELINES

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.config.n_embd = 64
    model.config.n_head = 2
    model.config.n_layer = 1
    return model

@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    # Return a tuple of two tensors for each item
    dataset.__getitem__.return_value = (torch.randn(1, 30, 30), torch.randn(1, 30, 30))
    print(f"Created mock dataset with length {dataset.__len__.return_value}")
    return dataset

def test_benchmark_model_normal_operation(mock_model, mock_dataset):
    print("Starting test_benchmark_model_normal_operation")
    with patch('torch.device', return_value=torch.device('cpu')) as mock_device, \
         patch('time.time', side_effect=[i for i in range(100)]) as mock_time, \
         patch('psutil.cpu_percent', return_value=50) as mock_cpu, \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=60)) as mock_memory:

        print(f"Mocked device: {mock_device.return_value}")
        print(f"Mocked time side effect: {mock_time.side_effect[:5]}...")
        print(f"Mocked CPU percent: {mock_cpu.return_value}")
        print(f"Mocked memory percent: {mock_memory.return_value.percent}")

        avg_time, avg_grids = benchmark_model(mock_model, mock_dataset, num_runs=1)

        print(f"Benchmark results - avg_time: {avg_time}, avg_grids: {avg_grids}")

        assert isinstance(avg_time, float), f"avg_time is not a float: {type(avg_time)}"
        assert isinstance(avg_grids, float), f"avg_grids is not a float: {type(avg_grids)}"
        assert avg_time > 0, f"avg_time is not positive: {avg_time}"
        assert avg_grids > 0, f"avg_grids is not positive: {avg_grids}"

    print("test_benchmark_model_normal_operation completed successfully")
    with patch('torch.device', return_value=torch.device('cpu')), \
         patch('time.time', side_effect=[0, 1] * 30), \
         patch('psutil.cpu_percent', return_value=50), \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=60)):
        
        avg_time, avg_grids = benchmark_model(mock_model, mock_dataset)
        
        assert isinstance(avg_time, float)
        assert isinstance(avg_grids, float)
        assert avg_time > 0
        assert avg_grids > 0

@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_benchmark_model_device_types(mock_model, mock_dataset, device_type):
    with patch('torch.device', return_value=torch.device(device_type)), \
         patch('time.time', side_effect=[0, 1] * 30), \
         patch('psutil.cpu_percent', return_value=50), \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=60)), \
         patch('torch.cuda.is_available', return_value=device_type == "cuda"), \
         patch('torch.backends.mps.is_available', return_value=device_type == "mps"):
        
        avg_time, avg_grids = benchmark_model(mock_model, mock_dataset, device_type=device_type)
        
        assert isinstance(avg_time, float)
        assert isinstance(avg_grids, float)
        assert avg_time > 0
        assert avg_grids > 0

@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_benchmark_model_precision(mock_model, mock_dataset, precision):
    with patch('torch.device', return_value=torch.device('cpu')), \
         patch('time.time', side_effect=[0, 1] * 30), \
         patch('psutil.cpu_percent', return_value=50), \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=60)):
        
        avg_time, avg_grids = benchmark_model(mock_model, mock_dataset, precision=precision)
        
        assert isinstance(avg_time, float)
        assert isinstance(avg_grids, float)
        assert avg_time > 0
        assert avg_grids > 0

def test_benchmark_model_empty_dataset(mock_model):
    empty_dataset = MagicMock()
    empty_dataset.__len__.return_value = 0
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        benchmark_model(mock_model, empty_dataset)

@pytest.mark.parametrize("batch_size", [1, 1000])
def test_benchmark_model_extreme_batch_sizes(mock_model, mock_dataset, batch_size):
    with patch('torch.device', return_value=torch.device('cpu')), \
         patch('time.time', side_effect=[0, 1] * 30), \
         patch('psutil.cpu_percent', return_value=50), \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=60)):
        
        avg_time, avg_grids = benchmark_model(mock_model, mock_dataset, batch_size=batch_size)
        
        assert isinstance(avg_time, float)
        assert isinstance(avg_grids, float)
        assert avg_time > 0
        assert avg_grids > 0

# Add more tests here for error handling, performance testing, etc.
"""Next steps:

Implement error handling tests
Add performance tests with simple thresholds
Implement mutation tests for critical parts (statistical calculations and error handling)
Create basic tests for the main function"""
