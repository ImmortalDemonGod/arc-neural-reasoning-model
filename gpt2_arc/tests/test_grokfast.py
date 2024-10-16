import pytest
from collections import deque
from gpt2_arc.src.utils.grokfast import gradfilter_ma, gradfilter_ema
from gpt2_arc.src.utils.grokfast_callback import GrokfastCallback
import torch.nn as nn
import torch
from unittest.mock import MagicMock

def test_gradfilter_ma():
    # Initialize a simple model
    model = nn.Linear(10, 10)
    
    # Create dummy gradients and initialize the grads dictionary
    initial_grad = torch.ones(10, 10)
    model.weight.grad = initial_grad.clone()
    grads = {name: deque(maxlen=5) for name, param in model.named_parameters() if param.requires_grad}
    
    # Apply gradfilter_ma
    updated_grads = gradfilter_ma(
        m=model,
        grads=grads,
        window_size=5,
        lamb=2.0,
        filter_type='mean',
        warmup=False,
        trigger=False
    )
    
    # Assertions to verify gradients are updated correctly
    assert "weight" in updated_grads
    assert len(updated_grads["weight"]) == 1
    expected_grad = initial_grad + (initial_grad * 2.0)
    assert torch.allclose(model.weight.grad, expected_grad)

def test_gradfilter_ema():
    # Initialize a simple model
    model = nn.Linear(10, 10)
    
    # Create dummy gradients and initialize the grads dictionary
    initial_grad = torch.ones(10, 10)
    model.weight.grad = initial_grad.clone()
    grads = None  # For EMA, grads start as None
    
    # Apply gradfilter_ema for the first time
    updated_grads = gradfilter_ema(
        m=model,
        grads=grads,
        alpha=0.9,
        lamb=2.0
    )
    
    # Expected gradient after first update
    expected_grad = initial_grad * 2.0
    assert "weight" in updated_grads
    assert torch.allclose(model.weight.grad, expected_grad)
    
    # Apply gradfilter_ema again with new gradients
    new_grad = torch.ones(10, 10) * 2.0
    model.weight.grad = new_grad.clone()
    updated_grads = gradfilter_ema(
        m=model,
        grads=updated_grads,
        alpha=0.9,
        lamb=2.0
    )
    
    # Expected gradient after second update
    expected_grad = new_grad + (updated_grads["weight"] * 2.0)
    assert torch.allclose(model.weight.grad, expected_grad)

class TestGrokfastCallback:
    def test_grokfast_callback_ema(self):
        # Initialize a simple model
        model = nn.Linear(10, 10)
        
        # Initialize the callback with EMA settings
        callback = GrokfastCallback(
            filter_type='ema',
            alpha=0.9,
            lamb=2.0,
            window_size=100,
            warmup=True,
            trigger=False
        )
        
        # Mock trainer and pl_module
        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.model = model
        
        # Set initial gradients
        initial_grad = torch.ones(10, 10)
        model.weight.grad = initial_grad.clone()
        
        # Call the callback after backward
        callback.on_after_backward(trainer, pl_module)
        
        # Expected gradient update
        expected_grad = initial_grad + (initial_grad * 2.0)
        assert torch.allclose(model.weight.grad, expected_grad)
    
    def test_grokfast_callback_ma(self):
        # Initialize a simple model
        model = nn.Linear(10, 10)
        
        # Initialize the callback with MA settings
        callback = GrokfastCallback(
            filter_type='ma',
            alpha=0.9,  # Not used for MA but required parameter
            lamb=2.0,
            window_size=5,
            warmup=False,
            trigger=False
        )
        
        # Mock trainer and pl_module
        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.model = model
        
        # Set initial gradients
        initial_grad = torch.ones(10, 10)
        model.weight.grad = initial_grad.clone()
        
        # Call the callback after backward
        callback.on_after_backward(trainer, pl_module)
        
        # Expected gradient update
        expected_grad = initial_grad + (initial_grad * 2.0)
        assert torch.allclose(model.weight.grad, expected_grad)
        
        # Apply again to test window functionality
        new_grad = torch.ones(10, 10) * 2.0
        model.weight.grad = new_grad.clone()
        callback.on_after_backward(trainer, pl_module)
        
        # Expected gradient after second update
        expected_grad = new_grad + (initial_grad * 2.0)
        assert torch.allclose(model.weight.grad, expected_grad)
