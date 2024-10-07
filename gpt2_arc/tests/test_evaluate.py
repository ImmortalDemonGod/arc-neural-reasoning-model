import unittest
import torch
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.data.arc_dataset import ARCDataset

class TestEvaluationMetrics(unittest.TestCase):
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
        self.train_dataset = ARCDataset(data_source="path/to/mock_data")
        self.val_dataset = ARCDataset(data_source="path/to/mock_data")

    def test_evaluation_metrics_computation(self):
        trainer = ARCTrainer(model=self.model, train_dataset=self.train_dataset, val_dataset=self.val_dataset, config=self.config)
        # Create dummy outputs and labels
        outputs = torch.tensor([[0.1, 0.6, 0.3, 0.0, 0.0, 0.0],
                                [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]], requires_grad=True)
        labels = torch.tensor([1, 2])
        loss = trainer.compute_loss(outputs, labels)
        self.assertGreater(loss.item(), 0, "Loss should be positive.")
        accuracy = trainer.compute_accuracy(outputs, labels)
        self.assertGreaterEqual(accuracy.item(), 0, "Accuracy should be non-negative.")
        self.assertLessEqual(accuracy.item(), 1, "Accuracy should not exceed 1.")

if __name__ == '__main__':
    unittest.main()
