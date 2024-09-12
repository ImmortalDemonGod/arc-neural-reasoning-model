# gpt2_arc/src/evaluate.py
import argparse

import pytorch_lightning as pl
import torch

import arckit
import logging
from src.data.arc_dataset import ARCDataset
from src.models.gpt2 import GPT2ARC
from src.config import Config
from src.training.trainer import ARCTrainer


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluate(model, test_dataset, batch_size=32):
    trainer = ARCTrainer(model, None, test_dataset, config=Config())
    pl_trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    results = pl_trainer.test(trainer)
    return results[0]


def main(args):
    # Load the test data using arckit
    _, test_set = arckit.load_data()
    test_data = ARCDataset(test_set)

    # Load the trained model
    model = GPT2ARC(Config().model)
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.eval()  # Ensure the model is in evaluation mode
    logger.info("Model set to evaluation mode")

    # Evaluate the model
    results = evaluate(model, test_data, args.batch_size)

    logger.info("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the ARC Neural Reasoning Model"
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
