import pytest
import torch
from src.models.gpt2 import GPT2ARC
from src.training.trainer import ARCTrainer
from src.data.arc_dataset import ArcDataset

@pytest.fixture
def sample_data():
    return [
        {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]},
        {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
    ]

@pytest.fixture
def model():
    return GPT2ARC()

@pytest.fixture
def trainer(model, sample_data):
    train_dataset = ArcDataset(sample_data)
    val_dataset = ArcDataset(sample_data)
    return ARCTrainer(model, train_dataset, val_dataset)

def test_arctrainer_initialization(trainer):
    assert isinstance(trainer, ARCTrainer)
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'train_dataset')
    assert hasattr(trainer, 'val_dataset')

def test_arctrainer_forward_pass(trainer):
    batch_size = 2
    seq_length = 900  # 30x30 grid
    input_ids = torch.randint(0, 2, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    output = trainer(input_ids, attention_mask)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_length, trainer.model.config.n_embd)

def test_arctrainer_training_step(trainer):
    batch_size = 2
    seq_length = 900  # 30x30 grid
    input_ids = torch.randint(0, 2, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    labels = torch.randint(0, 2, (batch_size, seq_length))
    
    batch = (input_ids, attention_mask, labels)
    loss = trainer.training_step(batch, 0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert not torch.isnan(loss).any(), "Loss contains NaN values"
    assert not torch.isinf(loss).any(), "Loss contains infinity values"

def test_arctrainer_validation_step(trainer):
    batch_size = 2
    seq_length = 900  # 30x30 grid
    input_ids = torch.randint(0, 2, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    labels = torch.randint(0, 2, (batch_size, seq_length))
    
    batch = (input_ids, attention_mask, labels)
    trainer.validation_step(batch, 0)
    
    # Check if val_loss is logged
    assert 'val_loss' in trainer.logged_metrics

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
