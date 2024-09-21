# gpt2_arc/src/evaluate.py
import sys
import os

# Add the root directory of the project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import json
import argparse
import pytorch_lightning as pl
import torch

from gpt2_arc.src.config import Config, ModelConfig
import arckit
import logging
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import Config
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluate(model, test_dataset, batch_size=32):
    trainer = ARCTrainer(model, None, test_dataset, config=Config())
    pl_trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    results = pl_trainer.test(trainer)
    # Collect individual task metrics
    individual_metrics = []
    for result in results:
        task_id = result.get('task_id', 'unknown')
        individual_metrics.append((task_id, result))

    # Log individual task metrics
    logger.info("Individual Task Metrics:")
    for task_id, metrics in individual_metrics:
        logger.info(f"Task {task_id}: {metrics}")

    return results[0]


def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['config']

def main(args):
    # Load the test data using arckit
    _, test_set = arckit.load_data()
    test_data = ARCDataset(test_set)

    # Load the checkpoint
    checkpoint = torch.load(args.model_checkpoint)

    # Load the configuration from the JSON file
    config_path = f"results/experiment_{args.model_checkpoint.split('_')[-1].replace('.pth', '.json')}"
    config_dict = load_config_from_json(config_path)
    model_config = ModelConfig(
        n_embd=config_dict['model']['n_embd'],
        n_head=config_dict['model']['n_head'],
        n_layer=config_dict['model']['n_layer'],
        dropout=config_dict['model']['dropout']
    )

    # Initialize the model with the checkpoint configuration
    model = GPT2ARC(model_config)
    checkpoint = torch.load(args.model_checkpoint)
    # Directly load the state dictionary
    model.load_state_dict(checkpoint)
    model.load_state_dict(state_dict)
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
