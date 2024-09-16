# gpt2_arc/src/training/train.py
import argparse
import sys
import logging
import os
import json
from unittest.mock import MagicMock

# Add the root directory of the project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

#print("Current PYTHONPATH:", sys.path)

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.utils.experiment_tracker import ExperimentTracker
from gpt2_arc.src.utils.results_collector import ResultsCollector
import os


# Set up logging
logger = logging.getLogger(__name__)

def main(args):
    # Set logging level
    log_level = getattr(logging, args.log_level.upper(), logging.DEBUG)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(log_level)
    # Load data
    import arckit
    train_set, eval_set = arckit.load_data()
    train_data = ARCDataset(train_set)
    val_data = ARCDataset(eval_set)
    logger.info("Data loaded successfully using arckit")

    logger.info("Initializing model with new configuration")
    model_config = ModelConfig(n_embd=96, n_head=3, n_layer=1)
    model = GPT2ARC(config=model_config)

    logger.info("Initializing trainer with new configuration")
    config = Config(model=model_config, training=TrainingConfig(batch_size=args.batch_size, learning_rate=args.learning_rate, max_epochs=args.max_epochs))
    results_collector = ResultsCollector(config)
    try:
        tracker = ExperimentTracker(config, project=args.project)
        tracker.start()
        trainer = ARCTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            config=config,
            results_collector=results_collector
        )

    except Exception as e:
        print(f"Training interrupted: {e}")
        if 'tracker' in locals():
            tracker.log_metric("training_interrupted", 1)
            tracker.log_metric("error_message", str(e))
    finally:
        if 'tracker' in locals():
            tracker.finish()

    # Create PyTorch Lightning trainer
    tb_logger = False if args.no_logging else TensorBoardLogger("tb_logs", name="arc_model")
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

    logger.debug(f"Initializing train DataLoader with batch_size={args.batch_size}")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=7)
    logger.debug(f"Train DataLoader initialized with {len(train_loader)} batches")

    logger.debug(f"Initializing validation DataLoader with batch_size={args.batch_size}")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=7)
    logger.debug(f"Validation DataLoader initialized with {len(val_loader)} batches")

    pl_trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=tb_logger,
        callbacks=callbacks if callbacks else None,
        enable_checkpointing=not args.no_checkpointing,
        enable_progress_bar=not args.no_progress_bar,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=1.0,
        accelerator='gpu' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    )

    global_step = 0
    for epoch in range(args.max_epochs):
        for batch in train_loader:
            # ... (training step)
            loss = trainer.training_step(batch, batch_idx=global_step)
            
            tracker.log_metric("train_loss", loss.item(), step=global_step)
            tracker.update_train_metrics(epoch, {"loss": loss.item()})
            
            # ... (backward pass, optimizer step, etc.)
            global_step += 1

        # Validation loop
        val_loss = trainer.validation_step(val_loader, batch_idx=global_step)
        tracker.log_metric("val_loss", val_loss, step=global_step)
        tracker.update_val_metrics(epoch, {"loss": val_loss})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ARC Neural Reasoning Model")
    parser.add_argument(
        "--train_data", type=str, required=False, help="Path to training data"
    )
    parser.add_argument(
        "--val_data", type=str, required=False, help="Path to validation data"
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

    parser.add_argument(
        "--log_level", type=str, default="ERROR", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    parser.add_argument(
        "--fast_dev_run", type=int, default=0, help="Run a few batches for debugging purposes"
    )

    parser.add_argument("--project", type=str, default="gpt2-arc", help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory to save results")
    
    args = parser.parse_args()
    
    try:
        # Initialize configurations
        model_config = ModelConfig(n_embd=96, n_head=3, n_layer=1)
        config = Config(model=model_config, training=TrainingConfig(batch_size=args.batch_size, learning_rate=args.learning_rate, max_epochs=args.max_epochs))

        # Initialize model
        model = GPT2ARC(config=model_config)

        # Load data
        import arckit
        train_set, eval_set = arckit.load_data()
        train_data = ARCDataset(train_set)
        val_data = ARCDataset(eval_set)

        # Initialize tracker
        tracker = ExperimentTracker(config, project=args.project, use_wandb=not args.no_wandb)
        tracker.start()

        # Initialize trainer
        trainer = ARCTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            config=config
        )

        # Initialize DataLoader
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=7)

        # Create PyTorch Lightning trainer
        tb_logger = False if args.no_logging else TensorBoardLogger("tb_logs", name="arc_model")
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

        pl_trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=tb_logger,
            callbacks=callbacks if callbacks else None,
            enable_checkpointing=not args.no_checkpointing,
            enable_progress_bar=not args.no_progress_bar,
            fast_dev_run=args.fast_dev_run,
            gradient_clip_val=1.0,
            accelerator='gpu' if args.use_gpu and torch.cuda.is_available() else 'cpu'
        )

        # Train the model
        pl_trainer.fit(trainer)

        # After training
        test_results = trainer.test_step(val_loader)
        tracker.set_test_results(test_results)

        tracker.set_final_metrics({
            "best_val_loss": trainer.best_val_loss,
            "best_epoch": trainer.best_epoch,
        })

        # If you're saving a checkpoint
        checkpoint_path = os.path.join(args.results_dir, "model_checkpoint.pth")
        torch.save(trainer.model.state_dict(), checkpoint_path)
        tracker.set_checkpoint_path(checkpoint_path)

        # Save results to JSON
        os.makedirs(args.results_dir, exist_ok=True)
        results_path = os.path.join(args.results_dir, f"results_{args.run_name}.json")
        tracker.save_to_json(results_path)

    except Exception as e:
        print(f"Training interrupted: {e}")
        if 'tracker' in locals():
            tracker.log_metric("training_interrupted", 1)
            tracker.log_metric("error_message", str(e))
    finally:
        if 'tracker' in locals():
            tracker.finish()

