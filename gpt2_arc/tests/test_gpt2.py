import pytest
import torch
from src.models.gpt2 import GPT2ARC

@pytest.fixture
def model():
    return GPT2ARC()

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
