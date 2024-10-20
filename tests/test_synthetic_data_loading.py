import unittest
import os
import json
import torch
from gpt2_arc.src.data.arc_dataset import ARCDataset
import logging

logger = logging.getLogger(__name__)

class TestSyntheticDataLoading(unittest.TestCase):
    def setUp(self):
        """
        Initialize necessary attributes to use the specific large synthetic data file.
        """
        # Path to the specific large synthetic data file
        self.large_file_path = "/workspaces/arc-neural-reasoning-model/gpt2_arc/src/data/SyntheticARC/small_tasks/1c786137.json"
        
        # Ensure the large synthetic data file exists
        self.assertTrue(os.path.isfile(self.large_file_path), f"Large synthetic data file does not exist: {self.large_file_path}")
        

        # Clear existing cache related to ARCDataset
        cache_dir = os.path.join(os.path.dirname(__file__), '../src/data/cache')
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.startswith('arc_dataset_cache_') and f.endswith('.pkl')]
            for cache_file in cache_files:
                cache_path = os.path.join(cache_dir, cache_file)
                try:
                    os.remove(cache_path)
                    self.addCleanup(os.remove, cache_path)  # Ensure cache is removed even if tests fail
                    logger.debug(f"Removed cache file: {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file {cache_path}: {e}")

    def tearDown(self):
        """
        No teardown needed since we're using an existing data file.
        """
        pass
    def _count_json_samples(self, file_path: str) -> int:
        """
        Counts the number of valid samples in the specified JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            int: Number of valid samples.
        """
        count = 0
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    # Common keys that contain sample lists
                    for key in ['train', 'test', 'samples', 'entries']:
                        if key in data and isinstance(data[key], list):
                            count += len(data[key])
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in file: {file_path}")
            except Exception as e:
                logger.error(f"Error counting samples in file {file_path}: {e}")
        return count

    def test_dataset_loading(self):
        """
        Test if the dataset loads the expected number of samples from a specific JSON file.
        """
        # Use the provided large synthetic data file
        data_source = self.large_file_path
        
        # Use the provided large synthetic data file
        data_source = self.large_file_path
        
        dataset = ARCDataset(
            data_source=data_source,
            data_source=data_source,
            is_test=False,
            max_samples=None,  # Load all samples
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # Determine the expected number of samples in the large file
        expected_samples = self._count_json_samples(data_source)
        self.assertEqual(len(dataset), expected_samples)
    
    def test_sample_structure(self):
        """
        Test if each loaded synthetic sample has 'input', 'output', and 'task_id' keys with correct types.
        """
        dataset = ARCDataset(
            is_test=False,
            max_samples=3,
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        for i in range(len(dataset)):
            input_tensor, output_tensor, task_id = dataset[i]
            
            # Check types
            self.assertIsInstance(input_tensor, torch.Tensor)
            self.assertIsInstance(output_tensor, torch.Tensor)
            self.assertIsInstance(task_id, str)
            
            # Check tensor shapes
            self.assertEqual(input_tensor.ndimension(), 3)  # [C, H, W]
            self.assertEqual(output_tensor.ndimension(), 3)  # [C, H, W]
            self.assertEqual(input_tensor.shape[0], 1)  # Assuming 1 channel
            self.assertEqual(output_tensor.shape[0], 1)
            self.assertTrue(input_tensor.shape[1] <= 30)  # Based on padding logic
            self.assertTrue(input_tensor.shape[2] <= 30)
            self.assertTrue(output_tensor.shape[1] <= 30)
            self.assertTrue(output_tensor.shape[2] <= 30)
            
            # Check task_id format
            self.assertTrue(task_id.startswith('synthetic_task_') or task_id.startswith('default_task'))
    
    def test_handling_invalid_json(self):
        """
        Test that invalid synthetic JSON files are skipped and do not affect the dataset loading.
        """
        # Path to the specific large synthetic data file
        data_source = self.large_file_path

        # Create an invalid JSON file in the same directory
        invalid_file = os.path.join(os.path.dirname(self.large_file_path), 'synthetic_invalid.json')
        with open(invalid_file, 'w') as f:
            f.write("{invalid_json: True,}")  # Malformed JSON

        dataset = ARCDataset(
            data_source=os.path.dirname(self.large_file_path),  # Point to the directory containing both files
            is_test=False,
            max_samples=None,
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # The invalid file should be skipped; only count samples from the large valid file
        expected_samples = self._count_json_samples(data_source)
        self.assertEqual(len(dataset), expected_samples)
        
        # Clean up the invalid file after test
        os.remove(invalid_file)
    def test_empty_file_handling(self):
        """
        Test that empty synthetic JSON files are skipped without affecting the dataset loading.
        """
        # Path to the specific large synthetic data file
        data_source_dir = os.path.dirname(self.large_file_path)
        
        # Create an empty synthetic JSON file within the data directory
        empty_file = os.path.join(data_source_dir, 'synthetic_empty.json')
        with open(empty_file, 'w') as f:
            pass  # Create an empty file
        
        dataset = ARCDataset(
            data_source=data_source_dir,
            is_test=False,
            max_samples=None,
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # The presence of the empty file should not affect the loading of valid samples
        expected_samples = self._count_json_samples(self.large_file_path)
        self.assertEqual(len(dataset), expected_samples)
        
        # Clean up the empty file after test
        os.remove(empty_file)
    def test_synthetic_data_directory_exists(self):
        """
        Test that the synthetic data directory exists.
        """
        self.assertTrue(os.path.isdir(os.path.dirname(self.large_file_path)),
                        f"Synthetic data directory does not exist: {os.path.dirname(self.large_file_path)}")

    def test_symbol_freq_handling_with_existing_data(self):
        """
        Test if symbol frequencies are correctly handled when provided for synthetic data.
        """
        """
        Test if symbol frequencies are correctly handled when provided for synthetic data.
        """
        symbol_freq = {i: 1.0 for i in range(10)}  # Uniform frequencies
        
        dataset = ARCDataset(
            data_source=self.large_file_path,
            is_test=False,
            max_samples=3,
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=symbol_freq,
            debug=True
        )
        
        # Check that sampler is initialized
        self.assertIsNotNone(dataset.sampler)
        self.assertEqual(len(dataset.sampler), len(dataset))
    
