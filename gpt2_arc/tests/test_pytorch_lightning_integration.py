import unittest
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.data.arc_dataset import ARCDataset

class TestPyTorchLightningIntegration(unittest.TestCase):
    def setUp(self):
        # Define model and training configurations
        model_config = ModelConfig(
            n_embd=16,
            n_head=2,
            n_layer=2,
            mamba_ratio=1,
            d_state=4,
            d_conv=1,
            dropout=0.05
        )
        training_config = TrainingConfig(
            batch_size=2,
            learning_rate=0.001,
            max_epochs=10,
            use_gpu=False,
            log_level="DEBUG",
            use_synthetic_data=False,
            balance_symbols=True,
            balancing_method="weighting",
            synthetic_data_path=None,
            symbol_freq={"0": 0.5, "1": 0.2, "2": 0.1, "3": 0.1, "4": 0.05, "5": 0.05}
        )
        self.config = Config(model=model_config, training=training_config)
        self.model = GPT2ARC(config=self.config, num_classes=6, symbol_freq=self.config.training.symbol_freq)
        self.train_dataset = ARCDataset(data_source="path/to/mock_data")
        self.val_dataset = ARCDataset(data_source="path/to/mock_data")
        self.train_loader = DataLoader(self.train_dataset, batch_size=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=2)

    def test_pytorch_lightning_trainer_initialization(self):
        trainer = pl.Trainer(
            max_epochs=10,
            logger=TensorBoardLogger(save_dir="logs"),
            callbacks=[ModelCheckpoint(monitor="val_loss")],
            accelerator='cpu',
            devices=1
        )
        self.assertEqual(trainer.max_epochs, 10, "Trainer max_epochs mismatch.")
        self.assertIsInstance(trainer.logger, TensorBoardLogger, "Logger is not TensorBoardLogger.")
        self.assertEqual(len(trainer.callbacks), 1, "Unexpected number of callbacks initialized.")

    def test_training_loop_with_mock_data(self):
        trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, accelerator='cpu', devices=1)
        trainer.fit(self.model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)
        # Assert that training completed without errors
        self.assertTrue(True, "Training loop executed successfully.")

if __name__ == '__main__':
    unittest.main()
