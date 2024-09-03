# gpt2_arc/tests/test_gpt2.py
import pytest
import torch
from src.models.gpt2 import GPT2ARC, Attention, FeedForward, TransformerBlock
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def model():
    return GPT2ARC()

def test_gpt2arc_initialization(model):
    assert isinstance(model, GPT2ARC)
    assert hasattr(model, 'token_embedding')
    assert hasattr(model, 'position_embedding')
    assert hasattr(model, 'blocks')
    assert hasattr(model, 'ln_f')
    assert hasattr(model, 'config')

def test_gpt2arc_forward_pass(model):
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    output = model(input_ids, attention_mask)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_length, model.config['n_embd'])

    logger.debug(f"Output shape: {output.shape}")

def test_gpt2arc_output_values(model):
    logger.debug("Testing GPT2ARC output values")
    batch_size = 1
    seq_length = 5
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    attention_mask = torch.ones((batch_size, seq_length))
    
    output = model(input_ids, attention_mask)
    
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinity values"

    logger.debug(f"Output min: {output.min()}, max: {output.max()}")

def test_gpt2arc_attention_mask(model):
    logger.debug("Testing GPT2ARC attention mask")
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.zeros((batch_size, seq_length))
    attention_mask[:, :5] = 1  # Only attend to first 5 tokens
    
    output_with_mask = model(input_ids, attention_mask)
    output_without_mask = model(input_ids)
    
    assert not torch.allclose(output_with_mask, output_without_mask), "Attention mask should affect the output"
