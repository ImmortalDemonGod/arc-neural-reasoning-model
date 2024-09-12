# gpt2_arc/tests/test_trainer.py
import pytest
import torch
from src.config import Config, ModelConfig, TrainingConfig
from src.data.arc_dataset import ARCDataset
from src.models.gpt2 import GPT2ARC
from src.training.trainer import ARCTrainer


@pytest.fixture
def sample_data():
    return [
        {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
    ]


@pytest.fixture
def model():
    config = ModelConfig()
    return GPT2ARC(config)


@pytest.fixture
def trainer(model, sample_data):
    config = Config(model=ModelConfig(), training=TrainingConfig())
    train_dataset = ARCDataset(sample_data)
    val_dataset = ARCDataset(sample_data)
    return ARCTrainer(model, train_dataset, val_dataset, config)


def test_arctrainer_initialization(trainer):
    assert isinstance(trainer, ARCTrainer)
    assert hasattr(trainer, "model")
    assert hasattr(trainer, "train_dataset")
    assert hasattr(trainer, "val_dataset")


def test_arctrainer_forward_pass(trainer):
    batch_size = 2
    seq_length = 900  # 30x30 grid
    input_ids = torch.randint(0, 2, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    output = trainer(input_ids, attention_mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_length, trainer.model.config.n_embd)


@pytest.mark.parametrize("batch_format", ["tuple", "dict"])
def test_arctrainer_training_step(trainer, batch_format):
    batch_size = 2
    seq_length = 900  # 30x30 grid
    vocab_size = 10  # Use a small vocab size for testing
    if batch_format == "tuple":
        batch = (
            torch.randint(0, vocab_size, (batch_size, seq_length)).long(),
            torch.ones((batch_size, seq_length)).float(),
            torch.randint(0, vocab_size, (batch_size, seq_length)).long(),
        )
    else:
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)).long(),
            "attention_mask": torch.ones((batch_size, seq_length)).float(),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_length)).long(),
        }
    loss = trainer.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert not torch.isnan(loss).any(), "Loss contains NaN values"
    assert not torch.isinf(loss).any(), "Loss contains infinity values"


def test_training_step_with_list_input():
    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)
    config = Config(model=model_config, training=TrainingConfig(batch_size=2, learning_rate=1e-4, max_epochs=2))
    model = GPT2ARC(config.model)
    trainer = ARCTrainer(model, None, None, config)

    batch_size = 2
    vocab_size = 10

    # Create a batch as a list (simulating the output from collate_fn)
    batch = [
        torch.randint(0, vocab_size, (batch_size, 1, 30, 30)).float(),  # inputs
        torch.randint(0, vocab_size, (batch_size, 1, 30, 30)).long(),   # outputs
    ]

    loss = trainer.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.shape == torch.Size([]), "Loss should be a scalar"
    assert not torch.isnan(loss).any(), "Loss should not be NaN"
    assert not torch.isinf(loss).any(), "Loss should not be infinity"

def test_validation_step_with_list_input():
    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)
    config = Config(model=model_config, training=TrainingConfig(batch_size=2, learning_rate=1e-4, max_epochs=2))
    model = GPT2ARC(config.model)
    trainer = ARCTrainer(model, None, None, config)

    batch_size = 2
    vocab_size = 10

    # Create a batch as a list (simulating the output from collate_fn)
    batch = [
        torch.randint(0, vocab_size, (batch_size, 1, 30, 30)).float(),  # inputs
        torch.randint(0, vocab_size, (batch_size, 1, 30, 30)).long(),   # outputs
    ]

    trainer.validation_step(batch, 0)

    assert "val_loss" in trainer.logged_metrics, "Validation loss should be logged"
    assert isinstance(trainer.logged_metrics["val_loss"], float), "Logged validation loss should be a float"

@pytest.mark.parametrize("batch_format", ["tuple", "dict"])
def test_arctrainer_validation_step(trainer, batch_format):
    batch_size = 2
    seq_length = 900  # 30x30 grid
    vocab_size = 10  # Use a small vocab size for testing
    if batch_format == "tuple":
        batch = (
            torch.randint(0, vocab_size, (batch_size, seq_length)).long(),
            torch.ones((batch_size, seq_length)).float(),
            torch.randint(0, vocab_size, (batch_size, seq_length)).long(),
        )
    else:
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)).long(),
            "attention_mask": torch.ones((batch_size, seq_length)).float(),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_length)).long(),
        }
    trainer.validation_step(batch, 0)

    # Check if val_loss is logged
    assert "val_loss" in trainer.logged_metrics


def test_arctrainer_configure_optimizers(trainer):
    optimizer = trainer.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)  # Use torch.optim.AdamW


def test_arctrainer_train_dataloader(trainer):
    dataloader = trainer.train_dataloader()
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert len(dataloader.dataset) == len(trainer.train_dataset)


def test_arctrainer_val_dataloader(trainer):
    dataloader = trainer.val_dataloader()
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert len(dataloader.dataset) == len(trainer.val_dataset)
