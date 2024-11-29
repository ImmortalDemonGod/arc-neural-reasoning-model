import unittest
from unittest.mock import patch
import sys
import os

# Adjust the import path as necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from gpt2_arc.src.optimization.optimizer import run_optimization

class TestHyperparameterOptimization(unittest.TestCase):
    @patch('optuna.create_study')
    def test_run_optimization(self, mock_create_study):
        # Mock the study object and its methods
        mock_study = mock_create_study.return_value
        mock_study.optimize.return_value = None
        mock_study.best_trial = None  # Simulate no successful trials

        # Call the function with minimal parameters
        try:
            run_optimization(n_trials=1, n_jobs=1)
            success = True
        except Exception as e:
            success = False
            print(f"Optimization failed with exception: {e}")

        self.assertTrue(success, "Hyperparameter optimization should run without errors")

if __name__ == '__main__':
    unittest.main()
