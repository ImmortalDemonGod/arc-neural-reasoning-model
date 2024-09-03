# gpt2_arc/src/training/train.py
import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.gpt2 import GPT2ARC
from src.data.arc_dataset import ArcDataset
from src.training.trainer import ARCTrainer

def main(args):
    # Load data
    train_data = ArcDataset(args.train_data)
    val_data = ArcDataset(args.val_data)

    # Initialize model
    model = GPT2ARC()

    # Initialize trainer
    trainer = ARCTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        batch_size=args.batch_size,
        lr=args.learning_rate
    )

    # Setup logging and checkpointing
    logger = TensorBoardLogger("tb_logs", name="arc_model")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="arc_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

    use_gpu = args.use_gpu and torch.cuda.is_available()
    pl_trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        gpus=1 if use_gpu else 0
    )

    # Train the model
    pl_trainer.fit(trainer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ARC Neural Reasoning Model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    
    args = parser.parse_args()
    main(args)
