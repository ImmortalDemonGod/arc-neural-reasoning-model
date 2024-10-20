import unittest
import os
import json
import torch
from gpt2_arc.src.data.arc_dataset import ARCDataset

class TestSyntheticDataLoading(unittest.TestCase):
    def setUp(self):
        """
        Set up the test to use existing synthetic data from the specified directory.
        """
        self.synthetic_data_dir = "/workspaces/arc-neural-reasoning-model/gpt2_arc/src/data/SyntheticARC/small_tasks"
        self.large_file_path = "/workspaces/arc-neural-reasoning-model/gpt2_arc/src/data/SyntheticARC/small_tasks/1c786137.json"
        self.assertTrue(os.path.isfile(self.large_file_path), f"Large synthetic data file does not exist: {self.large_file_path}")
    
    def tearDown(self):
        """
        Remove the temporary directory after tests.
        """
        # No need to remove directory as it's existing synthetic data
    
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
        expected_samples = self._count_json_samples(self.synthetic_data_dir)
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
        
        # Expecting 3 valid synthetic samples from two valid files
        self.assertEqual(len(dataset), 3)
    
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
    
    def _count_json_samples(self, directory):
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
        """
        Test if the dataset can handle loading a large number of synthetic samples.
        """
        # Create a large number of mock synthetic samples
        large_samples = []
        for i in range(100):
            large_samples.append({
                "input": [[i % 10, (i+1) % 10, (i+2) % 10], [(i+3) % 10, (i+4) % 10, (i+5) % 10]],
                "output": [[(i+6) % 10, (i+7) % 10, (i+8) % 10], [(i+9) % 10, (i+10) % 10, (i+11) % 10]],
                "id": f"synthetic_task_{i+5}"
            })
        
        # Add a new synthetic mock file with large samples
        self.large_mock_file = os.path.join(self.temp_dir, 'synthetic_mock_large.json')
        with open(self.large_mock_file, 'w') as f:
            json.dump(large_samples, f)
        
        dataset = ARCDataset(
            data_source=self.temp_dir,
            is_test=False,
            max_samples=103,  # Total valid synthetic samples: 2 + 2 + 100 = 104
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # Expecting 104 valid synthetic samples
        self.assertEqual(len(dataset), 104)

if __name__ == '__main__':
    unittest.main()
