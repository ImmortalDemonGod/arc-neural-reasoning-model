import torch

class ArcDataset:
    def __init__(self, data, max_grid_size=(30, 30), num_symbols=10):
        self.data = data
        self.max_grid_size = max_grid_size
        self.num_symbols = num_symbols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_grid = self._preprocess(item['input'])
        output_grid = self._preprocess(item['output'])
        return input_grid, output_grid

    def _preprocess(self, grid):
        # Example preprocessing: convert to tensor and pad to max_grid_size
        tensor_grid = torch.zeros(*self.max_grid_size, self.num_symbols)
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                tensor_grid[i, j, val] = 1
        return tensor_grid
