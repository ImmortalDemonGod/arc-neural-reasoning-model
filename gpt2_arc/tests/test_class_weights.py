
import unittest
import torch
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.models.gpt2 import GPT2ARC

class TestClassWeights(unittest.TestCase):
    def setUp(self):
        # Define a sample symbol frequency dictionary
        self.symbol_freq = {
            "0": 0.5,
            "1": 0.2,
            "2": 0.1,
            "3": 0.1,
            "4": 0.05,
            "5": 0.05
        }
        
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
            symbol_freq=self.symbol_freq
        )
        
        # Create a Config object
        self.config = Config(model=model_config, training=training_config)
        
        # Initialize the GPT2ARC model with symbol_freq
        self.model = GPT2ARC(config=self.config, num_classes=len(self.symbol_freq), symbol_freq=self.symbol_freq)
    
    def test_class_weights_correctness(self):
        # Calculate expected class weights
        expected_weights = {k: 1.0 / v for k, v in self.symbol_freq.items()}
        expected_weights_tensor = torch.tensor([expected_weights[str(i)] for i in range(len(self.symbol_freq))])
        
        # Retrieve class weights from the model's loss function
        if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'weight'):
            actual_weights = self.model.loss_fn.weight
            # Assert that actual_weights matches expected_weights_tensor
            self.assertTrue(torch.allclose(actual_weights, expected_weights_tensor), 
                            f"Actual class weights {actual_weights} do not match expected {expected_weights_tensor}")
        else:
            self.fail("The model's loss function does not have 'weight' attribute.")

    def test_class_weights_application(self):
        # Create dummy outputs and labels
        outputs = torch.tensor([[0.1, 0.6, 0.3, 0.0, 0.0, 0.0],
                                [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]], requires_grad=True)
        labels = torch.tensor([1, 2])
        
        # Compute loss
        loss = self.model.loss_fn(outputs, labels)
        
        # Manually calculate expected loss
        expected_loss = (-torch.log(torch.tensor(0.6)) / self.symbol_freq["1"]) + (-torch.log(torch.tensor(0.2)) / self.symbol_freq["2"])
        expected_loss = expected_loss / 2  # Average over batch
        
        # Check if calculated loss matches expected loss
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4, 
                               msg=f"Computed loss {loss.item()} does not match expected {expected_loss.item()}")

if __name__ == '__main__':
    unittest.main()
