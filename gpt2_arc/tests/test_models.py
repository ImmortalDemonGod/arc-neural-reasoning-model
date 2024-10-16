import unittest
import torch
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig

class TestGPT2ARC(unittest.TestCase):
    def setUp(self):
        # Define model and training configurations
        model_config = ModelConfig(
            n_embd=16,
            n_head=2,
            n_layer=2,
            mamba_ratio=1,
            d_state=4,
            d_conv=1,
            dropout=0.05
        )
        training_config = TrainingConfig(
            batch_size=2,
            learning_rate=0.001,
            max_epochs=10,
            use_gpu=False,
            log_level="DEBUG",
            use_synthetic_data=False,
            balance_symbols=True,
            balancing_method="weighting",
            synthetic_data_path=None,
            symbol_freq={"0": 0.5, "1": 0.2, "2": 0.1, "3": 0.1, "4": 0.05, "5": 0.05}
        )
        self.config = Config(model=model_config, training=training_config)
        self.model = GPT2ARC(config=self.config, num_classes=6, symbol_freq=self.config.training.symbol_freq)

    def test_model_initialization_with_class_weights(self):
        expected_weights = torch.tensor([2.0, 5.0, 10.0, 10.0, 20.0, 20.0])
        self.assertTrue(torch.allclose(self.model.loss_fn.weight, expected_weights),
                        "Class weights in loss function do not match expected values.")

    def test_model_forward_pass(self):
        dummy_input = torch.zeros(1, 1, 6, 6)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 1, 6), "Model output shape mismatch.")

if __name__ == '__main__':
    unittest.main()
