import torch
import pytest

def test_actual_checkpoint_loading():
    # Path to the actual checkpoint file
    checkpoint_path = 'final_model_4fe9801e-c839-454f-a46c-6e94e3c04e81.pth'
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Print the keys for debugging purposes
    print("Checkpoint keys:", checkpoint.keys())
    
    # Check if the checkpoint contains expected keys
    expected_keys = ['conv1.weight', 'conv1.bias', 'blocks.0.attention.key.weight']
    for key in expected_keys:
        assert key in checkpoint, f"Checkpoint does not contain expected key: {key}"

if __name__ == "__main__":
    pytest.main([__file__])
