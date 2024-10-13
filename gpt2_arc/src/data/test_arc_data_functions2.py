# gpt2_arc/src/data/test_arc_data_functions2.py

import unittest
import os
import json
import torch
import numpy as np

from gpt2_arc.src.data.arc_dataset import ARCDataset

# ===========================
# Global Configuration
# ===========================

# Define global paths for the data directory and data file
DATA_DIR = '/workspaces/arc-neural-reasoning-model/gpt2_arc/src/data/SyntheticARC/task_small'
DATA_FILE = '/workspaces/arc-neural-reasoning-model/gpt2_arc/src/data/SyntheticARC/tasks/ecdecbb3.json'
INVALID_PATH = 'invalid_path_for_testing'

# ===========================
# Test Classes
# ===========================

class TestARCDatasetInitialization(unittest.TestCase):
    def test_init_with_directory(self):
        """
        Test Initialization with a Directory Data Source

        Purpose:
        Verify that the dataset initializes correctly when provided with a directory containing JSON files.
        """
        dataset = ARCDataset(data_source=DATA_DIR, is_test=False, use_cache=False)
        self.assertGreater(len(dataset), 0, "Dataset should not be empty when initialized with a valid directory.")

    def test_init_with_file(self):
        """
        Test Initialization with a File Data Source

        Purpose:
        Verify that the dataset initializes correctly when provided with a single JSON file.
        """
        dataset = ARCDataset(data_source=DATA_FILE, is_test=False, use_cache=False)
        self.assertGreater(len(dataset), 0, "Dataset should not be empty when initialized with a valid file.")

    def test_init_with_invalid_source(self):
        """
        Test Initialization with Invalid Data Source

        Purpose:
        Ensure that the dataset raises an appropriate error when given an invalid data source.
        """
        with self.assertRaises(FileNotFoundError):
            dataset = ARCDataset(data_source=INVALID_PATH, is_test=False, use_cache=False)


class TestARCDatasetIndexing(unittest.TestCase):
    def test_build_index_from_files(self):
        """
        Test the _build_index_from_files Method

        Purpose:
        Verify that the index mapping is correctly built from data files.
        """
        dataset = ARCDataset(data_source=DATA_DIR, is_test=False, use_cache=False)
        expected_num_samples = 0

        # Manually count the number of samples in the data directory
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(DATA_DIR, filename)
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    if isinstance(task_data, dict):
                        num_samples = len(task_data.get('train', []))
                    elif isinstance(task_data, list):
                        num_samples = len(task_data)
                    else:
                        num_samples = 0
                    expected_num_samples += num_samples

        actual_num_samples = len(dataset)
        self.assertEqual(actual_num_samples, expected_num_samples, "Index mapping should contain all samples from data files.")


class TestARCDatasetGetItem(unittest.TestCase):
    def test_get_item(self):
        """
        Test Retrieving an Item

        Purpose:
        Verify that items can be retrieved correctly using __getitem__.
        """
        dataset = ARCDataset(data_source=DATA_FILE, is_test=False, use_cache=False)
        sample = dataset[0]  # Get the first sample
        self.assertIsInstance(sample, tuple, "Sample should be a tuple.")
        self.assertEqual(len(sample), 3, "Sample tuple should contain three elements.")
        input_tensor, output_tensor, task_id = sample
        self.assertIsInstance(input_tensor, torch.Tensor, "Input should be a torch.Tensor.")
        self.assertIsInstance(output_tensor, torch.Tensor, "Output should be a torch.Tensor.")
        self.assertIsInstance(task_id, str, "Task ID should be a string.")

    def test_get_item_out_of_bounds(self):
        """
        Test Index Out of Bounds

        Purpose:
        Ensure that accessing an out-of-bounds index raises an IndexError.
        """
        dataset = ARCDataset(data_source=DATA_FILE, is_test=False, use_cache=False)
        with self.assertRaises(IndexError):
            sample = dataset[len(dataset)]  # Accessing beyond the dataset size


class TestARCDatasetPreprocessGrid(unittest.TestCase):
    def test_preprocess_grid_with_list(self):
        """
        Test _preprocess_grid Method with List Input

        Purpose:
        Verify that the grid preprocessing works correctly for list input types.
        """
        dataset = ARCDataset(data_source=[], use_cache=False)  # Empty data source for testing
        grid = [[1, 2], [3, 4]]
        processed_grid = dataset._preprocess_grid(grid)
        self.assertIsInstance(processed_grid, torch.Tensor)
        self.assertEqual(processed_grid.shape, (1, 30, 30), "Processed grid should have shape (1, 30, 30).")

    def test_preprocess_grid_with_numpy_array(self):
        """
        Test _preprocess_grid Method with NumPy Array Input

        Purpose:
        Verify that the grid preprocessing works correctly for NumPy array input types.
        """
        dataset = ARCDataset(data_source=[], use_cache=False)
        grid = np.array([[1, 2], [3, 4]])
        processed_grid = dataset._preprocess_grid(grid)
        self.assertIsInstance(processed_grid, torch.Tensor)
        self.assertEqual(processed_grid.shape, (1, 30, 30))

    def test_preprocess_grid_with_tensor(self):
        """
        Test _preprocess_grid Method with Torch Tensor Input

        Purpose:
        Verify that the grid preprocessing works correctly for torch tensor input types.
        """
        dataset = ARCDataset(data_source=[], use_cache=False)
        grid = torch.tensor([[1, 2], [3, 4]])
        processed_grid = dataset._preprocess_grid(grid)
        self.assertIsInstance(processed_grid, torch.Tensor)
        self.assertEqual(processed_grid.shape, (1, 30, 30))

    def test_preprocess_grid_with_invalid_input(self):
        """
        Test _preprocess_grid Method with Invalid Input

        Purpose:
        Ensure that the method raises a ValueError when given invalid input.
        """
        dataset = ARCDataset(data_source=[], use_cache=False)
        with self.assertRaises(ValueError):
            processed_grid = dataset._preprocess_grid(None)


class TestARCDatasetCaching(unittest.TestCase):
    def test_cache_save_and_load(self):
        """
        Test Cache Saving and Loading

        Purpose:
        Verify that the dataset correctly saves and loads the cache.
        """
        dataset = ARCDataset(data_source=DATA_FILE, is_test=False, use_cache=True)
        cache_path = dataset.cache_path
        self.assertTrue(os.path.exists(cache_path), "Cache file should be created.")

        # Load a new instance to see if it uses the cache
        dataset_loaded = ARCDataset(data_source=DATA_FILE, is_test=False, use_cache=True)
        self.assertEqual(len(dataset), len(dataset_loaded), "Dataset loaded from cache should have the same number of samples.")

    def test_dataset_without_cache(self):
        """
        Test Dataset Functionality Without Cache

        Purpose:
        Ensure that the dataset functions correctly when caching is disabled.
        """
        dataset = ARCDataset(data_source=DATA_FILE, is_test=False, use_cache=False)
        self.assertGreater(len(dataset), 0, "Dataset should load correctly without cache.")


class TestARCDatasetProcessSingleTask(unittest.TestCase):
    def test_process_single_task_with_dict(self):
        """
        Test _process_single_task Method with Valid Dictionary Input

        Purpose:
        Verify that _process_single_task processes task data correctly.
        """
        dataset = ARCDataset(data_source=[], use_cache=False)
        task_data = {
            "train": [([[0]], [[1]])],
            "test": [([[1]], [[0]])]
        }
        task_id = "test_task"
        processed_samples = dataset._process_single_task(task_data, task_id)
        self.assertEqual(len(processed_samples), 2, "Should process both train and test samples.")
        for sample in processed_samples:
            self.assertIn('input', sample)
            self.assertIn('output', sample)
            self.assertEqual(sample['task_id'], task_id)

    def test_process_single_task_with_invalid_data(self):
        """
        Test _process_single_task Method with Invalid Data Input

        Purpose:
        Ensure that invalid task data raises a ValueError.
        """
        dataset = ARCDataset(data_source=[], use_cache=False)
        task_data = {"invalid_key": []}
        task_id = "test_task"
        with self.assertRaises(ValueError):
            processed_samples = dataset._process_single_task(task_data, task_id)


# ===========================
# Test Runner
# ===========================

if __name__ == '__main__':
    unittest.main()
