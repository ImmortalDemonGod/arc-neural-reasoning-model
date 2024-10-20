import unittest
import tempfile
import os
import json
import torch
from gpt2_arc.src.data.arc_dataset import ARCDataset
import shutil

class TestSyntheticDataLoading(unittest.TestCase):
    def setUp(self):
        """
        Set up a temporary directory with mock JSON files for testing synthetic data loading.
        """
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Create mock JSON files within the temporary directory

        # File 1: Well-formed JSON array of samples
        self.mock_file1 = os.path.join(self.temp_dir, 'synthetic_mock1.json')
        samples1 = [
            {
                "input": [[0, 1, 2], [3, 4, 5]],
                "output": [[6, 7, 8], [9, 10, 0]],
                "id": "synthetic_task_1"
            },
            {
                "input": [[1, 2, 3], [4, 5, 6]],
                "output": [[7, 8, 9], [0, 1, 2]],
                "id": "synthetic_task_2"
            }
        ]
        with open(self.mock_file1, 'w') as f:
            json.dump(samples1, f)
        
        # File 2: Well-formed JSON dictionary with 'train' and 'test' keys
        self.mock_file2 = os.path.join(self.temp_dir, 'synthetic_mock2.json')
        samples2 = {
            "train": [
                {
                    "input": [[2, 3, 4], [5, 6, 7]],
                    "output": [[8, 9, 0], [1, 2, 3]],
                    "id": "synthetic_task_3"
                }
            ],
            "test": [
                {
                    "input": [[3, 4, 5], [6, 7, 8]],
                    "output": [[9, 0, 1], [2, 3, 4]],
                    "id": "synthetic_task_4"
                }
            ]
        }
        with open(self.mock_file2, 'w') as f:
            json.dump(samples2, f)
        
        # File 3: Invalid JSON to test error handling
        self.mock_file3 = os.path.join(self.temp_dir, 'synthetic_mock3.json')
        with open(self.mock_file3, 'w') as f:
            f.write('{"input": [[4, 5, 6], [7, 8, 9]], "output": [[0, 1, 2], [3, 4, 5]},')  # Missing closing brace
    
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
            data_source=self.temp_dir,
            is_test=False,
            max_samples=1,  # Expecting 3 valid synthetic samples
            num_symbols=11,
            pad_symbol_idx=10,
            symbol_freq=None,
            debug=True
        )
        
        # Assert that 3 synthetic samples are loaded
        self.assertEqual(len(dataset), 3)
    
    def test_sample_structure(self):
        """
        Test if each loaded synthetic sample has 'input', 'output', and 'task_id' keys with correct types.
        """
        dataset = ARCDataset(
            data_source=self.temp_dir,
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
    
    def test_symbol_freq_handling(self):
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
    
    def test_large_dataset_loading(self):
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
