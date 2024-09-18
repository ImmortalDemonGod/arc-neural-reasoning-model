# gpt2_arc/tests/test_benchmark.py

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from benchmark import benchmark_model, main, BASELINES
from src.config import ModelConfig
from src.models.gpt2 import GPT2ARC

# Mock classes and fixtures

@pytest.fixture
def mock_model():
    return MagicMock(spec=GPT2ARC)

@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.__getitem__.return_value = (
        torch.randn(1, 30, 30),  # inputs
        torch.randint(0, 10, (1, 30, 30)),  # outputs
        "task_1"  # task_id
    )
    dataset.__len__.return_value = 100
    return dataset

@pytest.fixture
def mock_dataloader():
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([
        (
            torch.randn(32, 1, 30, 30),  # inputs
            torch.randn(32, 1, 30, 30),  # outputs
            f"task_{i}"                  # task_ids
        )
        for i in range(10)
    ])
    return dataloader


# Tests for benchmark_model function

def test_benchmark_model_basic(mock_model, mock_dataset, mock_dataloader):
    with patch('gpt2_arc.benchmark.DataLoader', return_value=mock_dataloader), \
         patch('gpt2_arc.benchmark.torch.cuda.is_available', return_value=False), \
         patch('gpt2_arc.benchmark.torch.backends.mps.is_available', return_value=False):
        avg_time, avg_grids = benchmark_model(mock_model, mock_dataset)
    
    assert isinstance(avg_time, (float, int))
    assert isinstance(avg_grids, (float, int))
    assert avg_time > 0
    assert avg_grids > 0

@pytest.mark.parametrize("batch_size,num_batches,num_runs", [
    (16, 5, 10),
    (64, 20, 5),
    (128, 2, 3)
])
def test_benchmark_model_parameters(mock_model, mock_dataset, mock_dataloader, batch_size, num_batches, num_runs):
    with patch('gpt2_arc.benchmark.DataLoader', return_value=mock_dataloader), \
         patch('gpt2_arc.benchmark.torch.cuda.is_available', return_value=False), \
         patch('gpt2_arc.benchmark.torch.backends.mps.is_available', return_value=False):
        avg_time, avg_grids = benchmark_model(
            mock_model, mock_dataset, batch_size=batch_size, num_batches=num_batches, num_runs=num_runs
        )
    
    assert isinstance(avg_time, (float, int))
    assert isinstance(avg_grids, (float, int))

def test_benchmark_model_cuda(mock_model, mock_dataset, mock_dataloader):
    with patch('benchmark.torch.cuda.is_available', return_value=True), \
         patch('benchmark.torch.cuda.synchronize'), \
         patch('benchmark.DataLoader', return_value=mock_dataloader), \
         patch('benchmark.torch.compile', return_value=mock_model):
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this system")
        
        try:
            avg_time, avg_grids = benchmark_model(mock_model, mock_dataset, device_type='cuda')
        except AssertionError as e:
            if "Torch not compiled with CUDA enabled" in str(e):
                pytest.skip("PyTorch not compiled with CUDA support")
            else:
                raise
        
        assert isinstance(avg_time, float)
        assert isinstance(avg_grids, float)
        assert avg_time >= 0
        assert avg_grids >= 0

def test_benchmark_model_mps(mock_model, mock_dataset, mock_dataloader):
    with patch('benchmark.torch.backends.mps.is_available', return_value=True), \
         patch('benchmark.DataLoader', return_value=mock_dataloader):
        avg_time, avg_grids = benchmark_model(mock_model, mock_dataset, device_type='mps')
    
    assert isinstance(avg_time, (float, int))
    assert isinstance(avg_grids, (float, int))

def test_benchmark_model_error_handling(mock_model, mock_dataset):
    with pytest.raises(ValueError, match="Invalid device type"):
        benchmark_model(mock_model, mock_dataset, device_type='invalid_device')

# Tests for main function

@pytest.fixture
def mock_argparse():
    with patch('benchmark.argparse.ArgumentParser') as mock_argparse:
        mock_args = MagicMock()
        mock_args.num_runs = 5
        mock_args.num_full_runs = 1
        mock_args.batch_size = 32
        mock_args.num_batches = 10
        mock_args.n_embd = 64
        mock_args.n_head = 2
        mock_args.n_layer = 1
        mock_args.device = 'cpu'
        mock_args.precision = 'highest'
        mock_argparse.return_value.parse_args.return_value = mock_args
        yield mock_argparse

def test_main_function(mock_argparse, mock_dataset, mock_model):
    with patch('benchmark.arckit.load_data', return_value=(mock_dataset, None)), \
         patch('benchmark.ARCDataset', return_value=mock_dataset), \
         patch('benchmark.GPT2ARC', return_value=mock_model), \
         patch('benchmark.benchmark_model', return_value=(1.0, 100.0)):
        main(mock_argparse.return_value.parse_args())

# Performance tests

@pytest.mark.benchmark(group="benchmark_model")
def test_benchmark_model_performance(benchmark, mock_model):
    # Create a mock dataset with one item
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 1
    mock_dataset.__getitem__.return_value = (
        torch.randn(1, 30, 30),  # input
        torch.randint(0, 10, (1, 30, 30)),  # output
        "task_1"  # task_id
    )

    # Create a mock dataloader that returns the mock dataset item
    mock_dataloader = MagicMock()
    mock_dataloader.__iter__.return_value = iter([mock_dataset.__getitem__()])

    with patch('gpt2_arc.benchmark.DataLoader', return_value=mock_dataloader), \
         patch('gpt2_arc.benchmark.torch.cuda.is_available', return_value=False), \
         patch('gpt2_arc.benchmark.torch.backends.mps.is_available', return_value=False):
        result = benchmark(
            benchmark_model,
            mock_model,
            mock_dataset,
            batch_size=1,
            num_batches=1,
            device_type='cpu',
            precision='medium',
            model_checkpoint=None
        )

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected tuple of length 2, got length {len(result)}"
    
    avg_time, grids_per_second = result
    print(f"Benchmark result - Average Time: {avg_time}, Grids per Second: {grids_per_second}")
    
    assert isinstance(avg_time, float), f"Expected float for avg_time, got {type(avg_time)}"
    assert isinstance(grids_per_second, float), f"Expected float for grids_per_second, got {type(grids_per_second)}"
    assert avg_time >= 0, f"Average time should be non-negative, got {avg_time}"
    assert grids_per_second >= 0, f"Grids per second should be non-negative, got {grids_per_second}"

    if avg_time > 0:
        assert grids_per_second > 0, f"Grids per second should be positive when avg_time > 0, got {grids_per_second}"

# Edge case tests

def test_benchmark_model_empty_dataset(mock_model):
    empty_dataset = MagicMock()
    empty_dataset.__len__.return_value = 0

    with pytest.raises(ValueError, match="Dataset is empty"):
        benchmark_model(mock_model, empty_dataset)

def test_benchmark_model_single_item_dataset(mock_model):
    single_item_dataset = MagicMock()
    single_item_dataset.__len__.return_value = 1
    mock_dataloader = MagicMock()
    mock_dataloader.__iter__.return_value = iter([
        (torch.randn(1, 1, 30, 30), torch.randn(1, 1, 30, 30), "task_1")
    ])
    
    with patch('benchmark.DataLoader', return_value=mock_dataloader):
        avg_time, avg_grids = benchmark_model(mock_model, single_item_dataset, batch_size=1, num_batches=1)
    
    assert isinstance(avg_time, (float, int))
    assert isinstance(avg_grids, (float, int))

# Error handling tests

def test_benchmark_model_with_correct_data(mock_model, mock_dataset, mock_dataloader):
    with patch('benchmark.DataLoader', return_value=mock_dataloader):
        avg_time, avg_grids = benchmark_model(mock_model, mock_dataset)
        
        assert isinstance(avg_time, float), "avg_time should be a float"
        assert isinstance(avg_grids, float), "avg_grids should be a float"
        assert avg_time > 0, "avg_time should be positive"
        assert avg_grids > 0, "avg_grids should be positive"

def test_benchmark_model_model_error(mock_model, mock_dataset, mock_dataloader):
    # Mock the model to raise a RuntimeError during execution
    mock_model.forward.side_effect = RuntimeError("Model execution failed")
    
    with patch('gpt2_arc.benchmark.DataLoader', return_value=mock_dataloader):
        with pytest.raises(RuntimeError, match="Model execution failed"):
            benchmark_model(mock_model, mock_dataset, device_type='cpu')

#skip
@pytest.mark.skip(reason="I dont want to crash my computer")
def test_benchmark_model_out_of_memory(mock_model, mock_dataset, mock_dataloader):
    mock_model.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
    
    with patch('benchmark.DataLoader', return_value=mock_dataloader), \
         patch('benchmark.torch.cuda.is_available', return_value=True), \
         pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA out of memory"):
        benchmark_model(mock_model, mock_dataset, device_type='cuda')

# Precision tests

@pytest.fixture
def mock_torch():
    return MagicMock()

@pytest.mark.parametrize("precision", ['highest', 'high', 'medium'])
def test_benchmark_model_precision(mock_model, mock_dataset, mock_torch, precision):
    with patch('gpt2_arc.benchmark.DataLoader') as mock_dataloader_class:
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = iter([
            (
                torch.randn(1, 1, 30, 30),  # inputs
                torch.randint(0, 10, (1, 30, 30)),  # outputs
                "task_1"  # task_id
            )
        ])
        mock_dataloader_class.return_value = mock_dataloader

        with patch('gpt2_arc.benchmark.torch.set_float32_matmul_precision') as mock_set_precision:
            benchmark_model(mock_model, mock_dataset, precision=precision)
    
    mock_set_precision.assert_called_once_with(precision)

# CSV output tests

def test_csv_output(mock_model, mock_dataset, mock_dataloader, tmp_path):
    csv_file = tmp_path / "benchmark_results.csv"
    stats_csv_file = tmp_path / "benchmark_statistics.csv"
    
    with patch('benchmark.DataLoader', return_value=mock_dataloader), \
         patch('benchmark.csv.writer') as mock_csv_writer:
        benchmark_model(mock_model, mock_dataset)
    
    assert mock_csv_writer.call_count == 2  # One for results, one for statistics

# Test suite execution

if __name__ == '__main__':
    pytest.main(['-v', '--cov=benchmark', '--cov-report=term-missing'])
