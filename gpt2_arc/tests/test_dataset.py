import unittest
import torch
from gpt2_arc.src.data.arc_dataset import ARCDataset

class TestARCDataset(unittest.TestCase):
    def setUp(self):
        # Initialize dataset with a mock data source
        self.dataset = ARCDataset(data_source="path/to/mock_data")

    def test_task_ids_loaded_from_filenames(self):
        # Mock the os.listdir to return predefined filenames
        synthetic_filenames = ['task_alpha.json', 'task_beta.json']
        with patch('os.listdir', return_value=synthetic_filenames):
            # Mock the open function to return empty JSON data
            mock_data = json.dumps({'train': [], 'test': []})
            with patch('builtins.open', mock_open(read_data=mock_data)):
                dataset = ARCDataset(data_source='path/to/synthetic_data', debug=True)
                expected_task_ids = ['task_alpha', 'task_beta']
                actual_task_ids = [sample['task_id'] for sample in dataset.data]
                self.assertEqual(actual_task_ids, expected_task_ids, "Task IDs do not match filenames.")
        # Initialize dataset with a mock directory
        dataset = ARCDataset(data_source="path/to/mock_directory")
        self.assertGreater(len(dataset), 0, "Dataset should contain samples loaded from the directory.")

    def test_dataset_preprocessing(self):
        # Initialize dataset with a mock data source
        dataset = ARCDataset(data_source="path/to/mock_data")
        input_tensor, output_tensor, task_id = dataset[0]
        self.assertEqual(input_tensor.shape, (1, 30, 30), "Input tensor should be padded to (1, 30, 30).")
        self.assertEqual(output_tensor.shape, (1, 30, 30), "Output tensor should be padded to (1, 30, 30).")

    def test_symbol_frequencies(self):
        # Test symbol frequency calculation
        frequencies = self.dataset.get_symbol_frequencies()
        self.assertIsInstance(frequencies, dict, "Symbol frequencies should be a dictionary.")

    def test_dataset_length_and_item_retrieval(self):
        # Test dataset length and item retrieval
        self.assertEqual(len(self.dataset), self.dataset.get_num_samples(), "Dataset length should match number of samples.")
        input_tensor, output_tensor, task_id = self.dataset[0]
        self.assertIsInstance(input_tensor, torch.Tensor, "Input should be a torch.Tensor.")
        self.assertIsInstance(output_tensor, torch.Tensor, "Output should be a torch.Tensor.")
        self.assertIsInstance(task_id, str, "Task ID should be a string.")

if __name__ == '__main__':
    unittest.main()
