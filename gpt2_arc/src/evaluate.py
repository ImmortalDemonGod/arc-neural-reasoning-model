# gpt2_arc/src/evaluate.py
import sys
import sys
import os
import json
import argparse
import pytorch_lightning as pl
import os
import torch
import wandb
import numpy as np
from datetime import datetime
from pytorch_lightning.utilities.model_summary import ModelSummary
from torchsummary import summary
from pytorch_lightning.utilities.model_summary import ModelSummary

# Define the base directory for the arc-neural-reasoning-model
arc_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Add the root directory of the project to the PYTHONPATH
project_root = arc_model_dir
sys.path.insert(0, project_root)

from gpt2_arc.src.config import Config, ModelConfig, TrainingConfig, EvaluationConfig
import arckit
import logging
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.training.trainer import ARCTrainer
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model, test_dataset, config, batch_size=32):
    trainer = ARCTrainer(model, None, test_dataset, config=config)
    pl_trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    results = pl_trainer.test(trainer)
    logger.debug(f"DEBUG: Raw results from test: {results}")

    avg_test_loss = pl_trainer.callback_metrics.get('avg_test_loss')
    avg_test_accuracy = pl_trainer.callback_metrics.get('avg_test_accuracy')
    avg_test_diff_accuracy = pl_trainer.callback_metrics.get('avg_test_diff_accuracy')

    # Convert tensors to Python floats if necessary
    if avg_test_loss is not None:
        avg_test_loss = avg_test_loss.item()
    if avg_test_accuracy is not None:
        avg_test_accuracy = avg_test_accuracy.item()
    if avg_test_diff_accuracy is not None:
        avg_test_diff_accuracy = avg_test_diff_accuracy.item()

    aggregated_results = {
        'test_loss': avg_test_loss,
        'test_accuracy': avg_test_accuracy,
        'test_diff_accuracy': avg_test_diff_accuracy,
    }

    print(f"DEBUG: Logged metrics - Avg test loss: {avg_test_loss}, Avg test accuracy: {avg_test_accuracy}, Avg diff accuracy: {avg_test_diff_accuracy}")

    # Collect individual task metrics
    individual_metrics = {}
    for key, value in pl_trainer.callback_metrics.items():
        if '_test_accuracy' in key or '_test_diff_accuracy' in key:
            if isinstance(value, torch.Tensor):
                value = value.item()
            # Key format: 'taskid_test_accuracy' or 'taskid_test_diff_accuracy'
            task_id, metric_name = key.split('_test_')
            if task_id not in individual_metrics:
                individual_metrics[task_id] = {}
            individual_metrics[task_id][f'test_{metric_name}'] = value

    # Compute complete task accuracy (fraction of tasks with perfect accuracy)
    num_tasks = len(individual_metrics)
    perfect_accuracy_threshold = config.evaluation.perfect_accuracy_threshold / 100.0  # Convert percentage to fraction

    num_complete_accuracy = 0
    for task_id, metrics in individual_metrics.items():
        test_accuracy = metrics.get('test_accuracy', 0)
        # Determine if the task is completely solved
        completely_solved = test_accuracy >= perfect_accuracy_threshold
        metrics['completely_solved'] = completely_solved
        if completely_solved:
            num_complete_accuracy += 1

    complete_task_accuracy = num_complete_accuracy / num_tasks if num_tasks > 0 else 0.0
    aggregated_results['complete_task_accuracy'] = complete_task_accuracy

    print(f"DEBUG: Computed complete task accuracy: {complete_task_accuracy}")

    return aggregated_results, individual_metrics

def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['config']

def save_results(results, individual_metrics, output_dir, model_name, model_summary):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_eval_results_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    data_to_save = {
        "aggregate_results": results,
        "individual_metrics": {task_id: metrics for task_id, metrics in individual_metrics.items()},
        "model_summary": str(model_summary)  # Convert ModelSummary to string
    }

    logger.debug(f"DEBUG: Data to be saved: {data_to_save}")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    return output_path

def main(args):
    if args.use_wandb:
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        else:
            print("WARNING: WANDB_API_KEY not found in environment variables.")
            print("Weights & Biases logging is disabled.")
            args.use_wandb = False
    else:
        print("Weights & Biases logging is disabled.")

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
    # Determine the number of classes from the dataset
    def find_max_label(task_set):
        max_label = 0
        for task in task_set.tasks:
            for sample in task.train + task.test:
                input_grid, output_grid = sample
                max_label = max(max_label, np.max(input_grid), np.max(output_grid))
        return max_label

    # Determine the number of classes from the test dataset
    max_label_test = find_max_label(test_set)
    num_classes = max_label_test + 1  # Add 1 because labels start from 0

    model = GPT2ARC(model_config, num_classes=num_classes)
    try:
        # Remove the "model." prefix from state dict keys
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Error while loading state_dict: {e}")
        logger.error(f"Available keys in checkpoint: {list(checkpoint.keys())}")
        raise

    model.eval()

    # Generate model summary
    print("DEBUG: Attempting to generate model summary")
    try:
        model_summary = str(ModelSummary(model, max_depth=-1))
        print("DEBUG: Model summary generated successfully")
    except Exception as e:
        print(f"DEBUG: Error generating model summary - {str(e)}")
        model_summary = "Error generating model summary"

    print("DEBUG: Model summary:")
    print(model_summary)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Define the input size based on your model's expected input
    sequence_length = 100  # Example value; adjust as needed
    input_size = (1, 1, sequence_length)  # Adjusted to match (batch_size, channels, sequence_length)
    logger.info(f"Defined input_size for summary: {input_size}")

    # Extract model name from the checkpoint path and sanitize it
    model_name = os.path.basename(args.model_checkpoint).split('.')[0]
    # Sanitize model_name to contain only valid characters
    model_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in model_name)

    # Debugging statements
    logger.debug(f"Sanitized model_name: {model_name}")
    print(f"DEBUG: Sanitized model_name: {model_name}")

    # Verify that model_name contains only allowed characters
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    invalid_chars = set(model_name) - allowed_chars
    if invalid_chars:
        logger.error(f"Model name contains invalid characters after sanitization: {invalid_chars}")
        print(f"ERROR: Model name contains invalid characters after sanitization: {invalid_chars}")
    else:
        logger.debug("Model name contains only valid characters.")
        print("DEBUG: Model name contains only valid characters.")


    # Create configuration
    config = Config(
        model=model_config,
        training=TrainingConfig(),
        evaluation=EvaluationConfig()
    )

    # Evaluate the model
    results, individual_metrics = evaluate(model, test_data, config, args.batch_size)

    logger.debug(f"DEBUG: Evaluation results: {results}")
    logger.debug(f"DEBUG: Individual metrics: {individual_metrics}")

    logger.info("Evaluation Results:")
    for metric, value in results.items():
        if metric != 'complete_task_accuracy':
            print(f"{metric}: {value}")
            if args.use_wandb:
                wandb.log({f"eval/{metric}": value})

    # Print complete_task_accuracy at the bottom
    if 'complete_task_accuracy' in results:
        print(f"complete_task_accuracy: {results['complete_task_accuracy']}")
        if args.use_wandb:
            wandb.log({"eval/complete_task_accuracy": results['complete_task_accuracy']})

    # Log individual task metrics
    for task_id, metrics in individual_metrics.items():
        # Ensure metrics are not already floats
        if isinstance(metrics['test_accuracy'], list):
            metrics['test_accuracy'] = sum(metrics['test_accuracy']) / len(metrics['test_accuracy'])
        if isinstance(metrics['test_diff_accuracy'], list):
            metrics['test_diff_accuracy'] = sum(metrics['test_diff_accuracy']) / len(metrics['test_diff_accuracy'])
        logger.info(f"Task {task_id}: Accuracy = {metrics['test_accuracy']:.4f}, Diff Accuracy = {metrics['test_diff_accuracy']:.4f}")

    # Save results regardless of wandb usage
    results_path = save_results(results, individual_metrics, args.output_dir, model_name, model_summary)

    if args.use_wandb:
        # Wandb artifact creation and logging
        logger.debug(f"Creating wandb Artifact with name: {model_name}")
        print(f"DEBUG: Creating wandb Artifact with name: {model_name}")

        try:
            artifact = wandb.Artifact(name=model_name, type='evaluation')
            artifact.add_file(results_path)
            wandb.log_artifact(artifact)
            logger.debug("Artifact created and logged successfully.")
            print("DEBUG: Artifact created and logged successfully.")
        except ValueError as ve:
            logger.error(f"Failed to create wandb Artifact: {ve}")
            print(f"ERROR: Failed to create wandb Artifact: {ve}")
            raise ve

        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the ARC Neural Reasoning Model")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--log-level", type=str, default="INFO", help="Set the logging level (e.g., DEBUG, INFO, WARNING)")
    parser.add_argument("--wandb_project", type=str, default="arc-evaluation", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")

    parser.add_argument("--use_wandb", action='store_true', help="Use Weights & Biases for logging")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set logging level
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))
    
    main(args)
