import unittest
import json
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig

class TestExperimentTracker(unittest.TestCase):
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

    def test_experiment_tracker_logging(self):
        tracker = ExperimentTracker(config=self.config, project="test_project")
        tracker.log_metric("test_metric", 0.95)
        self.assertIn("test_metric", tracker.metrics, "Metric should be logged in tracker.metrics.")
        self.assertEqual(tracker.metrics["test_metric"], 0.95, "Logged metric value mismatch.")

    def test_experiment_tracker_save_to_json(self):
        tracker = ExperimentTracker(config=self.config, project="test_project")
        tracker.log_metric("test_metric", 0.95)
        tracker.save_to_json("test_results.json")
        with open("test_results.json", 'r') as f:
            data = json.load(f)
        self.assertIn("test_metric", data, "Metric should be present in saved JSON.")
        self.assertEqual(data["test_metric"], 0.95, "Saved metric value mismatch.")

if __name__ == '__main__':
    unittest.main()
