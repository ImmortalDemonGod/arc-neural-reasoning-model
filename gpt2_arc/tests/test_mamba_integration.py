import unittest
import torch
from gpt2_arc.src.models.gpt2 import MambaLayer, GPT2ARC
from gpt2_arc.src.config import ModelConfig

class TestMambaLayer(unittest.TestCase):
    def test_mamba_layer_forward(self):
        # Initialize MambaLayer with test parameters
        n_embd = 64
        d_state = 16
        d_conv = 4
        dropout = 0.1
        mamba_layer = MambaLayer(n_embd, d_state, d_conv, dropout)

        # Create a sample input tensor
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, n_embd)

        # Forward pass
        output = mamba_layer(x)

        # Assert output shape is correct
        self.assertEqual(output.shape, x.shape)

        # Optional: Check for NaNs or infinite values
        self.assertTrue(torch.all(torch.isfinite(output)))

class TestGPT2ARCWithMamba(unittest.TestCase):
    def test_gpt2arc_with_mamba_forward(self):
        # Define model configuration with Mamba parameters
        model_config = ModelConfig(
            n_embd=64,
            n_head=4,
            n_layer=2,
            mamba_ratio=1,
            d_state=16,
            d_conv=4,
            dropout=0.1
        )
        num_classes = 10  # Adjust based on your dataset
        model = GPT2ARC(config=model_config, num_classes=num_classes)

        # Create a sample input tensor (e.g., a batch of grids)
        batch_size = 2
        height = width = 6
        x = torch.randint(0, num_classes, (batch_size, 1, height, width), dtype=torch.long)

        # Forward pass
        output = model(x)

        # Assert output shape is correct
        expected_output_shape = (batch_size, height * width, num_classes)
        self.assertEqual(output.shape, expected_output_shape)

        # Optional: Check for NaNs or infinite values
        self.assertTrue(torch.all(torch.isfinite(output)))

if __name__ == '__main__':
    unittest.main()
