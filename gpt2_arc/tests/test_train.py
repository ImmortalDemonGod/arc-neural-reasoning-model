# gpt2_arc/tests/test_train.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pytest
import sys
import os

# Add the project root to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from unittest.mock import patch, MagicMock, ANY
from gpt2_arc.src.data.arc_dataset import ArcDataset
import argparse
import pytorch_lightning as pl
from gpt2_arc.src.training.train import main
from gpt2_arc.src.data.arc_dataset import ArcDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.trainer import ARCTrainer

@pytest.fixture
def mock_args():
    args = argparse.Namespace()
    args.train_data = "mock_train_data.json"
    args.val_data = "mock_val_data.json"
    args.batch_size = 32
    args.learning_rate = 1e-4
    args.max_epochs = 10
    args.use_gpu = False
    return args

@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=ArcDataset)
    dataset.data = [{"input": "mock input", "output": "mock output"}]
    dataset.__len__.return_value = 100
    return dataset

@pytest.fixture
def model():
    return GPT2ARC()

@pytest.fixture
def mock_trainer():
    return MagicMock(spec=ARCTrainer)

@pytest.fixture
def mock_pl_trainer():
    return MagicMock(spec=pl.Trainer)

# Existing GPT2ARC model tests

def test_gpt2arc_initialization(model):
    assert isinstance(model, GPT2ARC)
    assert hasattr(model, 'gpt2')
    assert hasattr(model, 'config')

def test_gpt2arc_forward_pass(model):
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    output = model(input_ids, attention_mask)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_length, model.config.n_embd)

def test_gpt2arc_output_values(model):
    batch_size = 1
    seq_length = 5
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    attention_mask = torch.ones((batch_size, seq_length))
    output = model(input_ids, attention_mask)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinity values"

def test_gpt2arc_attention_mask(model):
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.zeros((batch_size, seq_length))
    attention_mask[:, :5] = 1  # Only attend to first 5 tokens
    output_with_mask = model(input_ids, attention_mask)
    output_without_mask = model(input_ids)
    assert not torch.allclose(output_with_mask, output_without_mask), "Attention mask should affect the output"

# New tests for train.py

def test_main_execution(mock_args, mock_dataset, model, mock_trainer, mock_pl_trainer):
    with patch('gpt2_arc.src.training.train.ArcDataset', return_value=mock_dataset), \
         patch('gpt2_arc.src.training.train.GPT2ARC', return_value=model), \
         patch('gpt2_arc.src.training.train.ARCTrainer', return_value=mock_trainer), \
         patch('gpt2_arc.src.training.train.pl.Trainer', return_value=mock_pl_trainer), \
         patch('gpt2_arc.src.training.train.TensorBoardLogger'), \
         patch('gpt2_arc.src.training.train.ModelCheckpoint'):
        
        main(mock_args)
        
        mock_pl_trainer.fit.assert_called_once_with(mock_trainer)

def test_data_loading(mock_args):
    with patch('gpt2_arc.src.data.arc_dataset.ArcDataset.__init__', return_value=None) as mock_init:
        ArcDataset(mock_args.train_data)
        mock_init.assert_called_once_with(mock_args.train_data)

def test_trainer_initialization(model, mock_dataset):
    trainer = ARCTrainer(
        model=model,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        batch_size=32,
        lr=1e-4
    )
    assert isinstance(trainer, ARCTrainer)
    assert trainer.model == model
    assert trainer.train_dataset == mock_dataset
    assert trainer.val_dataset == mock_dataset
    assert trainer.batch_size == 32
    assert trainer.lr == 1e-4

@pytest.mark.parametrize("batch_size", [1, 1000000])
def test_batch_size_extremes(mock_args, batch_size):
    mock_args.batch_size = batch_size
    with patch('gpt2_arc.src.training.train.ArcDataset'), \
         patch('gpt2_arc.src.training.train.GPT2ARC'), \
         patch('gpt2_arc.src.training.train.ARCTrainer'), \
         patch('gpt2_arc.src.training.train.pl.Trainer'):
        
        main(mock_args)  # Should not raise an exception

@pytest.mark.parametrize("learning_rate", [1e-10, 1000])
def test_learning_rate_extremes(mock_args, learning_rate):
    mock_args.learning_rate = learning_rate
    with patch('gpt2_arc.src.training.train.ArcDataset'), \
         patch('gpt2_arc.src.training.train.GPT2ARC'), \
         patch('gpt2_arc.src.training.train.ARCTrainer'), \
         patch('gpt2_arc.src.training.train.pl.Trainer'):
        
        main(mock_args)  # Should not raise an exception

def test_non_existent_train_data(mock_args):
    mock_args.train_data = "non_existent_path.json"
    with pytest.raises(FileNotFoundError):
        main(mock_args)

def test_gpu_not_available(mock_args):
    mock_args.use_gpu = True
    with patch('torch.cuda.is_available', return_value=False), \
         patch('gpt2_arc.src.training.train.ArcDataset'), \
         patch('gpt2_arc.src.training.train.GPT2ARC'), \
         patch('gpt2_arc.src.training.train.ARCTrainer'), \
         patch('gpt2_arc.src.training.train.pl.Trainer') as mock_trainer:
        
        main(mock_args)
        mock_trainer.assert_called_with(max_epochs=mock_args.max_epochs, 
                                        logger=ANY,
                                        callbacks=[ANY],
                                        gpus=0)

from hypothesis import given, strategies as st, settings, HealthCheck

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(batch_size=st.integers(min_value=1, max_value=1024))
def test_valid_batch_sizes(mock_args, batch_size):
    mock_args.batch_size = batch_size
    with patch('gpt2_arc.src.training.train.ArcDataset'), \
         patch('gpt2_arc.src.training.train.GPT2ARC'), \
         patch('gpt2_arc.src.training.train.ARCTrainer'), \
         patch('gpt2_arc.src.training.train.pl.Trainer'):
        
        main(mock_args)  # Should not raise an exception

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(learning_rate=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_valid_learning_rates(mock_args, learning_rate):
    mock_args.learning_rate = learning_rate
    with patch('gpt2_arc.src.training.train.ArcDataset'), \
         patch('gpt2_arc.src.training.train.GPT2ARC'), \
         patch('gpt2_arc.src.training.train.ARCTrainer'), \
         patch('gpt2_arc.src.training.train.pl.Trainer'):
        
        main(mock_args)  # Should not raise an exception

def test_end_to_end_training(mock_args, tmp_path):
    mock_args.max_epochs = 2  # Short training run
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    mock_args.checkpoint_dir = str(checkpoint_dir)
    
    with patch('gpt2_arc.src.training.train.ArcDataset'), \
         patch('gpt2_arc.src.training.train.GPT2ARC'), \
         patch('gpt2_arc.src.training.train.ARCTrainer'), \
         patch('gpt2_arc.src.training.train.pl.Trainer') as mock_trainer, \
         patch('gpt2_arc.src.training.train.ModelCheckpoint') as mock_checkpoint:
        
        main(mock_args)
        
        mock_trainer.return_value.fit.assert_called_once()
        mock_checkpoint.assert_called_once()

def test_tensorboard_logging(mock_args, tmp_path):
    log_dir = tmp_path / "tb_logs"
    log_dir.mkdir()
    
    with patch('gpt2_arc.src.training.train.ArcDataset'), \
         patch('gpt2_arc.src.training.train.GPT2ARC'), \
         patch('gpt2_arc.src.training.train.ARCTrainer'), \
         patch('gpt2_arc.src.training.train.pl.Trainer'), \
         patch('gpt2_arc.src.training.train.TensorBoardLogger') as mock_logger:
        
        main(mock_args)
        
        mock_logger.assert_called_once_with("tb_logs", name="arc_model")

# Additional test for GPT2ARC model in training context

def test_gpt2arc_in_training_loop(model, mock_dataset):
    trainer = ARCTrainer(
        model=model,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        batch_size=2,
        lr=1e-4
    )
    
    # Simulate a single training step
    vocab_size = 10  # Use a small vocab size for testing
    seq_length = 10
    batch = {
        'input_ids': torch.randint(0, vocab_size, (2, seq_length)).long(),
        'attention_mask': torch.ones((2, seq_length)).float(),
        'labels': torch.randint(0, vocab_size, (2, seq_length)).long()
    }
    
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert not torch.isnan(loss).any(), "Training loss is NaN"
    assert not torch.isinf(loss).any(), "Training loss is infinite"
