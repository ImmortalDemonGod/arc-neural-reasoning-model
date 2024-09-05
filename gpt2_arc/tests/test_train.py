# gpt2_arc/tests/test_train.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import os
import sys

import pytest

# Add the project root to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse
from unittest.mock import ANY, MagicMock, patch

import pytorch_lightning as pl
import torch

from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.train import main
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
    dataset = MagicMock(spec=ARCDataset)
    dataset.data = [{"input": "mock input", "output": "mock output"}]
    dataset.__len__.return_value = 100
    return dataset


from src.config import Config, ModelConfig, TrainingConfig


@pytest.fixture
def model():
    config = Config(model=ModelConfig(), training=TrainingConfig())
    return GPT2ARC(config.model)


@pytest.fixture
def mock_trainer():
    return MagicMock(spec=ARCTrainer)


@pytest.fixture
def mock_pl_trainer():
    return MagicMock(spec=pl.Trainer)


# Existing GPT2ARC model tests


def test_gpt2arc_initialization(model):
    assert isinstance(model, GPT2ARC)
    assert hasattr(model, "conv1")  # Check for conv1 instead of token_embedding
    assert hasattr(model, "blocks")
    assert hasattr(model, "ln_f")
    assert hasattr(model, "config")


def test_gpt2arc_forward_pass(model):
    batch_size = 2
    channels = 1
    height = 30
    width = 30
    input_ids = torch.randint(0, 2, (batch_size, channels, height, width))
    attention_mask = torch.ones((batch_size, height * width))
    output = model(input_ids, attention_mask)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, height * width, model.config.n_embd)


def test_gpt2arc_output_values(model):
    batch_size = 1
    channels = 1
    height = 30
    width = 30
    input_ids = torch.randint(0, 2, (batch_size, channels, height, width))
    attention_mask = torch.ones((batch_size, height * width))
    output = model(input_ids, attention_mask)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinity values"


def test_gpt2arc_attention_mask(model):
    batch_size = 2
    channels = 1
    height = 30
    width = 30
    input_ids = torch.randint(0, 2, (batch_size, channels, height, width))
    attention_mask = torch.zeros((batch_size, height * width))
    attention_mask[:, :450] = 1  # Only attend to first half of the pixels
    output_with_mask = model(input_ids, attention_mask)
    output_without_mask = model(input_ids)
    assert not torch.allclose(output_with_mask, output_without_mask), "Attention mask should affect the output"


# New tests for train.py


def test_logging(mock_args, mock_dataset, model, mock_trainer, mock_pl_trainer):
    with patch(
        "gpt2_arc.src.training.train.ARCDataset", return_value=mock_dataset
    ), patch("gpt2_arc.src.training.train.GPT2ARC", return_value=model), patch(
        "gpt2_arc.src.training.train.ARCTrainer", return_value=mock_trainer
    ), patch(
        "gpt2_arc.src.training.train.pl.Trainer", return_value=mock_pl_trainer
    ), patch("gpt2_arc.src.training.train.TensorBoardLogger") as mock_logger, patch(
        "gpt2_arc.src.training.train.ModelCheckpoint"
    ) as mock_checkpoint:
        main(mock_args)

        mock_logger.assert_called_once_with("tb_logs", name="arc_model")
        mock_checkpoint.assert_called_once_with(
            dirpath="checkpoints",
            filename="arc_model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )


def test_fit_call(mock_args, mock_dataset, model, mock_trainer, mock_pl_trainer):
    with patch(
        "gpt2_arc.src.training.train.ARCDataset", return_value=mock_dataset
    ), patch("gpt2_arc.src.training.train.GPT2ARC", return_value=model), patch(
        "gpt2_arc.src.training.train.ARCTrainer", return_value=mock_trainer
    ), patch(
        "gpt2_arc.src.training.train.pl.Trainer", return_value=mock_pl_trainer
    ), patch("gpt2_arc.src.training.train.TensorBoardLogger"), patch(
        "gpt2_arc.src.training.train.ModelCheckpoint"
    ):
        main(mock_args)

        mock_pl_trainer.fit.assert_called_once_with(mock_trainer)


def test_data_loading(mock_args):
    with patch(
        "gpt2_arc.src.data.arc_dataset.ARCDataset.__init__", return_value=None
    ) as mock_init:
        ARCDataset(mock_args.train_data)
        mock_init.assert_called_once_with(mock_args.train_data)


def test_trainer_initialization(model, mock_dataset):
    config = Config(model=ModelConfig(), training=TrainingConfig())
    trainer = ARCTrainer(
        model=model, train_dataset=mock_dataset, val_dataset=mock_dataset, config=config
    )
    assert isinstance(trainer, ARCTrainer)
    assert trainer.model == model
    assert trainer.train_dataset == mock_dataset
    assert trainer.val_dataset == mock_dataset
    assert trainer.batch_size == 32
    assert trainer.lr == 1e-4


@pytest.mark.parametrize("batch_size", [1, 1000000])
def test_batch_size_extremes(mock_args, batch_size):
    model_config = ModelConfig(n_embd=96, n_head=3, n_layer=1)
    config = Config(model=model_config, training=TrainingConfig(batch_size=32, learning_rate=5e-4, max_epochs=2))
    mock_args.batch_size = batch_size
    with patch("gpt2_arc.src.training.train.ARCDataset"), patch(
        "gpt2_arc.src.training.train.GPT2ARC"
    ), patch("gpt2_arc.src.training.train.ARCTrainer"), patch(
        "gpt2_arc.src.training.train.pl.Trainer"
    ) as mock_trainer:
        main(mock_args)

        mock_trainer.assert_called_with(
            max_epochs=config.training.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            gradient_clip_val=1.0
        )


@pytest.mark.parametrize("learning_rate", [1e-10, 1000])
def test_learning_rate_extremes(mock_args, learning_rate):
    mock_args.learning_rate = learning_rate
    with patch("gpt2_arc.src.training.train.ARCDataset"), patch(
        "gpt2_arc.src.training.train.GPT2ARC"
    ), patch("gpt2_arc.src.training.train.ARCTrainer"), patch(
        "gpt2_arc.src.training.train.pl.Trainer"
    ):
        main(mock_args)  # Should not raise an exception


def test_non_existent_train_data(mock_args):
    mock_args.train_data = "non_existent_path.json"
    with pytest.raises(FileNotFoundError):
        main(mock_args)


def test_gpu_not_available(mock_args):
    mock_args.use_gpu = True
    with patch("torch.cuda.is_available", return_value=False), patch(
        "gpt2_arc.src.training.train.ARCDataset"
    ), patch("gpt2_arc.src.training.train.GPT2ARC"), patch(
        "gpt2_arc.src.training.train.ARCTrainer"
    ), patch("gpt2_arc.src.training.train.pl.Trainer") as mock_trainer:
        main(mock_args)
        mock_trainer.assert_called_with(
            max_epochs=mock_args.max_epochs, logger=ANY, callbacks=[ANY], accelerator='cpu'
        )


from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(batch_size=st.integers(min_value=1, max_value=1024))
def test_valid_batch_sizes(mock_args, batch_size):
    mock_args.batch_size = batch_size
    with patch("gpt2_arc.src.training.train.ARCDataset"), patch(
        "gpt2_arc.src.training.train.GPT2ARC"
    ), patch("gpt2_arc.src.training.train.ARCTrainer"), patch(
        "gpt2_arc.src.training.train.pl.Trainer"
    ):
        main(mock_args)  # Should not raise an exception


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    learning_rate=st.floats(
        min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
    )
)
def test_valid_learning_rates(mock_args, learning_rate):
    mock_args.learning_rate = learning_rate
    with patch("gpt2_arc.src.training.train.ARCDataset"), patch(
        "gpt2_arc.src.training.train.GPT2ARC"
    ), patch("gpt2_arc.src.training.train.ARCTrainer"), patch(
        "gpt2_arc.src.training.train.pl.Trainer"
    ):
        main(mock_args)  # Should not raise an exception


def test_end_to_end_training(mock_args, tmp_path):
    model_config = ModelConfig(n_embd=96, n_head=3, n_layer=1)
    config = Config(model=model_config, training=TrainingConfig(batch_size=32, learning_rate=5e-4, max_epochs=2))
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    mock_args.checkpoint_dir = str(checkpoint_dir)

    with patch("gpt2_arc.src.training.train.ARCDataset"), patch(
        "gpt2_arc.src.training.train.GPT2ARC"
    ), patch("gpt2_arc.src.training.train.ARCTrainer"), patch(
        "gpt2_arc.src.training.train.pl.Trainer"
    ) as mock_trainer, patch(
        "gpt2_arc.src.training.train.ModelCheckpoint"
    ) as mock_checkpoint:
        main(mock_args)

        mock_trainer.return_value.fit.assert_called_once()
        mock_checkpoint.assert_called_once()


def test_tensorboard_logging(mock_args, tmp_path):
    log_dir = tmp_path / "tb_logs"
    log_dir.mkdir()

    with patch("gpt2_arc.src.training.train.ARCDataset"), patch(
        "gpt2_arc.src.training.train.GPT2ARC"
    ), patch("gpt2_arc.src.training.train.ARCTrainer"), patch(
        "gpt2_arc.src.training.train.pl.Trainer"
    ), patch("gpt2_arc.src.training.train.TensorBoardLogger") as mock_logger:
        main(mock_args)

        mock_logger.assert_called_once_with("tb_logs", name="arc_model")


# Additional test for GPT2ARC model in training context


def test_gpt2arc_in_training_loop(model, mock_dataset):
    config = Config(model=ModelConfig(), training=TrainingConfig())
    trainer = ARCTrainer(
        model=model, train_dataset=mock_dataset, val_dataset=mock_dataset, config=config
    )

    # Simulate a single training step
    batch_size = 2
    channels = 1
    height = 30
    width = 30
    batch = {
        "input_ids": torch.randint(0, 2, (batch_size, channels, height, width)).float(),
        "attention_mask": torch.ones((batch_size, height * width)).float(),
        "labels": torch.randint(0, 2, (batch_size, channels, height, width)).long(),
    }

    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar tensor
    assert not torch.isnan(loss).any(), "Training loss is NaN"
    assert not torch.isinf(loss).any(), "Training loss is infinite"
