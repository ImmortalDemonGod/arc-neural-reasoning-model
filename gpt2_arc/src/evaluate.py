import argparse

import pytorch_lightning as pl
import torch

from src.data.arc_dataset import ArcDataset
from src.models.gpt2 import GPT2ARC
from src.training.trainer import ARCTrainer


def evaluate(model, test_dataset, batch_size=32):
    trainer = ARCTrainer(model, None, test_dataset, batch_size=batch_size)
    pl_trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)
    results = pl_trainer.test(trainer)
    return results[0]


def main(args):
    # Load the test data
    test_data = ArcDataset(args.test_data)

    # Load the trained model
    model = GPT2ARC.load_from_checkpoint(args.model_checkpoint)
    model.eval()

    # Evaluate the model
    results = evaluate(model, test_data, args.batch_size)

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the ARC Neural Reasoning Model"
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test data"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )

    args = parser.parse_args()
    main(args)
