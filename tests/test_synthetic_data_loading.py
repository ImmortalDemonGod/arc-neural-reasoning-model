import unittest
import os
import json
import torch
from gpt2_arc.src.data.arc_dataset import ARCDataset
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

class TestSyntheticDataLoading(unittest.TestCase):
    def setUp(self):
        """
        Set up a temporary directory for synthetic data and initialize necessary attributes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.synthetic_data_dir = self.temp_dir

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
        Remove the temporary directory after tests.
        """
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_loading(self):
        """
        Test if the dataset loads the expected number of synthetic samples.
        """
        dataset = ARCDataset(
            data_source=self.synthetic_data_dir,
            is_test=False,
            max_samples=None,  # Load all samples
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # Assert that the expected number of synthetic samples are loaded
        expected_samples = self._count_json_samples_directory(self.synthetic_data_dir)
        self.assertEqual(len(dataset), expected_samples)
    
    def test_sample_structure(self):
        """
        Test if each loaded synthetic sample has 'input', 'output', and 'task_id' keys with correct types.
        """
        dataset = ARCDataset(
            data_source=self.synthetic_data_dir,
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
            self.assertEqual(input_tensor.ndim, 3)  # [C, H, W]
            self.assertEqual(output_tensor.ndim, 3)  # [C, H, W]
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
        dataset = ARCDataset(
            data_source=self.temp_dir,
            is_test=False,
            max_samples=10,  # More than the valid synthetic samples to see if invalid ones are skipped
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # Determine the expected number of samples in the large file
        expected_samples = self._count_json_samples(self.temp_dir)
        self.assertEqual(len(dataset), expected_samples)
    
    def test_empty_file_handling(self):
        """
        Test that empty synthetic JSON files are skipped without affecting the dataset loading.
        """
        # Create an empty synthetic JSON file
        empty_file = os.path.join(self.temp_dir, 'synthetic_empty.json')
        with open(empty_file, 'w') as f:
            pass  # Create an empty file
        
        dataset = ARCDataset(
            data_source=self.temp_dir,
            is_test=False,
            max_samples=4,  # Total valid synthetic samples are 3
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # Expect still 3 valid synthetic samples; empty file is skipped
        self.assertEqual(len(dataset), 3)
    
    def test_synthetic_data_directory_exists(self):
        """
        Test that the synthetic data directory exists.
        """
        self.assertTrue(os.path.isdir(self.synthetic_data_dir), f"Synthetic data directory does not exist: {self.synthetic_data_dir}")

    def test_symbol_freq_handling_with_existing_data(self):
        """
        Test if symbol frequencies are correctly handled when provided for synthetic data.
        """
        # Create a symbol frequency dictionary
        symbol_freq = {i: 1.0 for i in range(10)}  # Uniform frequencies
        
        dataset = ARCDataset(
            data_source=self.temp_dir,
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
    
    def _count_json_samples_directory(self, directory):
        logger = logging.getLogger(__name__)
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            count += len(data)
                        elif isinstance(data, dict):
                            for key in ['train', 'test', 'samples', 'entries']:
                                if key in data and isinstance(data[key], list):
                                    count += len(data[key])
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON files
        return count

    def _count_json_samples_file(self, file_path: str) -> int:
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
        """
        Test if the dataset can handle loading a large number of synthetic samples.
        """
        # Use the provided large synthetic data file
        large_file_path = self.large_file_path

        dataset = ARCDataset(
            data_source=large_file_path,
            is_test=False,
            max_samples=None,  # Load all samples from the large file
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # Expecting 104 valid synthetic samples
        self.assertEqual(len(dataset), 104)

if __name__ == '__main__':
    unittest.main()
