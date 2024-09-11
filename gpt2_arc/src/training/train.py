# gpt2_arc/src/training/train.py
import argparse
import sys
import os

# Add the root directory of the project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

print("Current PYTHONPATH:", sys.path)

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.training.trainer import ARCTrainer


def main(args):
    # Load data
    import arckit
    train_set, eval_set = arckit.load_data()
    train_data = ARCDataset(train_set)
    val_data = ARCDataset(eval_set)
    print("Data loaded successfully using arckit")

    print("Initializing model with new configuration")
    model_config = ModelConfig(n_embd=96, n_head=3, n_layer=1)
    model = GPT2ARC(config=model_config)

    print("Initializing trainer with new configuration")
    config = Config(model=model_config, training=TrainingConfig(batch_size=args.batch_size, learning_rate=args.learning_rate, max_epochs=args.max_epochs))
    trainer = ARCTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        config=config
    )

    # Create PyTorch Lightning trainer
    logger = False if args.no_logging else TensorBoardLogger("tb_logs", name="arc_model")
    callbacks = []
    if not args.no_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="arc_model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        callbacks.append(checkpoint_callback)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=7)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=7)

    pl_trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=callbacks if callbacks else None,
        enable_checkpointing=not args.no_checkpointing,
        enable_progress_bar=not args.no_progress_bar,
        gradient_clip_val=1.0,
        accelerator='gpu' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    )

    # Train the model
    pl_trainer.fit(trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ARC Neural Reasoning Model")
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for training if available"
    )

    parser.add_argument(
        "--no_logging", action="store_true", help="Disable logging"
    )
    parser.add_argument(
        "--no_checkpointing", action="store_true", help="Disable checkpointing"
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )

    args = parser.parse_args()
    main(args)
