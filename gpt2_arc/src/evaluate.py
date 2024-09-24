# gpt2_arc/src/evaluate.py
import sys
import os
import json
import argparse
import pytorch_lightning as pl
import torch
import wandb
from datetime import datetime

# Add the root directory of the project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from gpt2_arc.src.config import Config, ModelConfig
import arckit
import logging
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    return results[0], individual_metrics

def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['config']

def save_results(results, individual_metrics, output_dir, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_eval_results_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        json.dump({
            "aggregate_results": results,
            "individual_metrics": individual_metrics
        }, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    return output_path

def main(args):
    # Initialize wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # Load the test data using arckit
    _, test_set = arckit.load_data()
    test_data = ARCDataset(test_set)

    # Load the checkpoint with map_location='cpu'
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')

    # Extract and convert the model configuration from the checkpoint
    if 'model_config' in checkpoint:
        model_config_dict = checkpoint['model_config']
        # Convert dict to ModelConfig object
        model_config = ModelConfig(**model_config_dict)
    else:
        raise ValueError("Model configuration not found in checkpoint")

    # Initialize the model with the checkpoint configuration
    model = GPT2ARC(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model set to evaluation mode")

    # Evaluate the model
    results, individual_metrics = evaluate(model, test_data, args.batch_size)

    logger.info("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
        wandb.log({f"eval/{metric}": value})

    # Save results locally
    model_name = os.path.basename(args.model_checkpoint).split('.')[0]
    results_path = save_results(results, individual_metrics, args.output_dir, model_name)

    # Log results file to wandb
    wandb.save(results_path)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the ARC Neural Reasoning Model")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--wandb_project", type=str, default="arc-evaluation", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
