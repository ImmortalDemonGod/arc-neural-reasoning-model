import torch
import pytest

def test_actual_checkpoint_loading():
    # Path to the actual checkpoint file
    checkpoint_path = 'final_model_4fe9801e-c839-454f-a46c-6e94e3c04e81.pth'
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Print the keys for debugging purposes
    print("Checkpoint keys:", checkpoint.keys())
    
    # Check if 'state_dict' is in the checkpoint
    assert 'state_dict' in checkpoint, "Checkpoint does not contain 'state_dict' key"

if __name__ == "__main__":
    pytest.main([__file__])
