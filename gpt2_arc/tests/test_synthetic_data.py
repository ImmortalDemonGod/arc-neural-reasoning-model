import json
import tempfile
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.training.train import main

@pytest.fixture
def synthetic_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "synthetic_data"
        data_path.mkdir()
        with open(data_path / "task1.json", "w") as f:
            json.dump({
                "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
                "test": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}]
            }, f)
        yield str(data_path)

def test_synthetic_data_loading(synthetic_data):
    dataset = ARCDataset(synthetic_data)
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 3  # input, output, task_id
    assert sample[0].shape == (1, 2, 2)  # Assuming 2x2 grid
    assert sample[1].shape == (1, 2, 2)

@pytest.mark.parametrize("use_synthetic", [True, False])
def test_main_with_synthetic_data(synthetic_data, use_synthetic):
    args = MagicMock()
    args.use_synthetic_data = use_synthetic
    args.synthetic_data_path = synthetic_data if use_synthetic else None
    args.max_epochs = 1
    args.fast_dev_run = True
    args.use_gpu = False
    args.no_logging = True
    args.no_checkpointing = True
    args.no_progress_bar = True
    args.project = "test_project"
    args.log_level = "DEBUG"
    args.batch_size = 1  # Set batch_size to a positive integer
    args.learning_rate = 1e-4  # Set a default learning rate
    print(f"DEBUG: args.batch_size = {args.batch_size}")
    print(f"DEBUG: args.learning_rate = {args.learning_rate}")

    with patch("gpt2_arc.src.training.train.pl.Trainer") as mock_pl_trainer, \
         patch("gpt2_arc.src.training.train.ARCDataset") as mock_dataset, \
         patch("gpt2_arc.src.training.train.GPT2ARC") as mock_model, \
         patch("gpt2_arc.src.training.train.ARCTrainer") as mock_arc_trainer:
        print("DEBUG: Inside test_main_with_synthetic_data")
        print(f"DEBUG: mock_pl_trainer = {mock_pl_trainer}")
        print(f"DEBUG: mock_arc_trainer = {mock_arc_trainer}")
        main(args)
        mock_pl_trainer.assert_called_once()

def test_synthetic_data_argument_parsing():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-synthetic-data", action="store_true")
    parser.add_argument("--synthetic-data-path", type=str)

    # Test with synthetic data
    args = parser.parse_args(["--use-synthetic-data", "--synthetic-data-path", "/path/to/data"])
    assert args.use_synthetic_data
    assert args.synthetic_data_path == "/path/to/data"

    # Test without synthetic data
    args = parser.parse_args([])
    assert not args.use_synthetic_data
    assert args.synthetic_data_path is None
