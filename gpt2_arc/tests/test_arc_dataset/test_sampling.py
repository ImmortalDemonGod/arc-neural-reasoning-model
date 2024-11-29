# test_arc_dataset/test_sampling.py
import torch
from torch.utils.data import DataLoader
from src.data.arc_dataset import ARCDataset

class TestARCDatasetSampling:
    """Test suite for dataset sampling and iteration."""

    def test_weighted_sampling(self, real_arc_tasks):
        """Test weighted sampling with symbol frequencies."""
        symbol_freq = {i: 1/(i+1) for i in range(11)}
        dataset = ARCDataset(
            data_source=real_arc_tasks,
            symbol_freq=symbol_freq
        )
        assert dataset.sampler is not None
        assert isinstance(dataset.sampler, torch.utils.data.WeightedRandomSampler)

    def test_dataloader_integration(self, real_arc_tasks):
        """Test integration with PyTorch DataLoader."""
        dataset = ARCDataset(data_source=real_arc_tasks)
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=dataset.collate_fn,
            num_workers=0
        )
        
        batch = next(iter(loader))
        inputs, outputs, task_ids = batch
        
        assert inputs.shape == (2, 1, 30, 30)
        assert outputs.shape == (2, 1, 30, 30)
        assert len(task_ids) == 2
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches in collate_fn."""
        empty_batch = []
        inputs, outputs, task_ids = ARCDataset.collate_fn(empty_batch)
        
        assert inputs.nelement() == 0
        assert outputs.nelement() == 0
        assert task_ids == []