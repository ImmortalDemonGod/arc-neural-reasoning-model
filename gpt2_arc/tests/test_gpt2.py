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
def test_gpt2arc_forward_pass(model):

    logger.debug(f"Difference between outputs: {(output_with_mask - output_without_mask).abs().mean()}")

def test_attention_module():
    logger.debug("Testing Attention module")
    attention = Attention(n_embd=768, n_head=12)
    x = torch.randn(2, 10, 768)
    output = attention(x)
    assert output.shape == x.shape
    logger.debug(f"Attention input shape: {x.shape}, output shape: {output.shape}")

def test_feedforward_module():
    logger.debug("Testing FeedForward module")
    ff = FeedForward(n_embd=768)
    x = torch.randn(2, 10, 768)
    output = ff(x)
    assert output.shape == x.shape
    logger.debug(f"FeedForward input shape: {x.shape}, output shape: {output.shape}")

def test_transformer_block():
    logger.debug("Testing TransformerBlock")
    block = TransformerBlock(n_embd=768, n_head=12)
    x = torch.randn(2, 10, 768)
    output = block(x)
    assert output.shape == x.shape
    logger.debug(f"TransformerBlock input shape: {x.shape}, output shape: {output.shape}")
