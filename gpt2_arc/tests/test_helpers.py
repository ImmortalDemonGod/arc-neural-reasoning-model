import pytest
import torch
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy

@pytest.fixture
def padding_indices():
    return 10  # Assuming 10 is the pad_symbol_idx

def test_differential_pixel_accuracy_ignore_padding(padding_indices):
    # Create sample input, target, and prediction tensors with padding tokens
    # Format: [batch_size, 1, H, W]
    input_tensor = torch.tensor([[[[1, 2, padding_indices],
                                   [3, padding_indices, 5],
                                   [padding_indices, 7, 8]]]], dtype=torch.float)
    
    target_tensor = torch.tensor([[[[1, 2, padding_indices],
                                    [3, padding_indices, 0],
                                    [padding_indices, 7, padding_indices]]]], dtype=torch.long)
    
    prediction_tensor = torch.tensor([[[[1, 2, padding_indices],
                                        [3, padding_indices, 5],
                                        [padding_indices, 7, 8]]]], dtype=torch.long)
    
    # Expected behavior:
    # - Only positions where target != pad_symbol_idx are considered
    # - Calculate accuracy where target != pad_symbol_idx
    # Here, let's calculate manually:
    # Mask: positions where target != 10
    # input vs target: (5 != 0), (8 != 10) => two different pixels
    # prediction = [1,2,10], [3,10,5], [10,7,8]
    # correct_diff_predictions: (prediction=5 vs target=0: False), (prediction=8 vs target=10: False)
    # So, correct_diff_pixels = 0
    # total_diff_pixels = 2
    # accuracy = 0 / 2 = 0.0
    
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor, pad_symbol_idx=padding_indices)
    expected_accuracy = 0.0
    assert accuracy == expected_accuracy, f"Expected accuracy {expected_accuracy}, got {accuracy}"
