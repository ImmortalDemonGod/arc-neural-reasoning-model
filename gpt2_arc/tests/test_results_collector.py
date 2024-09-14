import unittest
from gpt2_arc.src.utils.results_collector import ResultsCollector
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig

class TestResultsCollector(unittest.TestCase):
    def setUp(self):
        model_config = ModelConfig(n_embd=96, n_head=3, n_layer=1)
        training_config = TrainingConfig(batch_size=32, learning_rate=1e-4, max_epochs=10)
        config = Config(model=model_config, training=training_config)
        self.results_collector = ResultsCollector(config)

    def test_initialization(self):
        self.assertIsNotNone(self.results_collector.experiment_id)
        self.assertIsNotNone(self.results_collector.timestamp)
        self.assertEqual(self.results_collector.config['model']['n_embd'], 96)

    def test_update_train_metrics(self):
        self.results_collector.update_train_metrics(1, {"loss": 0.5})
        self.assertIn(1, self.results_collector.results["train"])
        self.assertEqual(self.results_collector.results["train"][1]["loss"], 0.5)

    def test_update_val_metrics(self):
        self.results_collector.update_val_metrics(1, {"loss": 0.3})
        self.assertIn(1, self.results_collector.results["validation"])
        self.assertEqual(self.results_collector.results["validation"][1]["loss"], 0.3)

    def test_set_test_results(self):
        self.results_collector.set_test_results({"accuracy": 0.8})
        self.assertEqual(self.results_collector.results["test"]["accuracy"], 0.8)

    def test_add_task_specific_result(self):
        self.results_collector.add_task_specific_result("task_1", {"accuracy": 0.9})
        self.assertIn("task_1", self.results_collector.task_specific_results)
        self.assertEqual(self.results_collector.task_specific_results["task_1"]["accuracy"], 0.9)

    def test_get_summary(self):
        summary = self.results_collector.get_summary()
        self.assertEqual(summary["experiment_id"], self.results_collector.experiment_id)

if __name__ == '__main__':
    unittest.main()
