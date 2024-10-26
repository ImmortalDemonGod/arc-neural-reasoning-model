# gpt2_arc/src/evaluate.py
import sys
import sys
import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
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
from gpt2_arc.src.utils.training_helpers import get_num_workers
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(
    model: torch.nn.Module,
    test_dataset: ARCDataset,
    config: Config,
    batch_size: int = 32,
    args: Optional[argparse.Namespace] = None
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    trainer = ARCTrainer(
        model=model,
        train_dataset=None,
        val_dataset=None,
        config=config,
        args=args,
        test_dataset=test_dataset
    )
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
        avg_test_accuracy = avg_test_accuracy.item() if isinstance(avg_test_accuracy, torch.Tensor) else avg_test_accuracy
    if avg_test_diff_accuracy is not None:
        avg_test_diff_accuracy = avg_test_diff_accuracy.item() if isinstance(avg_test_diff_accuracy, torch.Tensor) else avg_test_diff_accuracy

    aggregated_results = {
        'test_loss': avg_test_loss,
        'test_accuracy': avg_test_accuracy,
        'test_diff_accuracy': avg_test_diff_accuracy,
    }

    print(f"DEBUG: Logged metrics - Avg test loss: {avg_test_loss}, Avg test accuracy: {avg_test_accuracy}, Avg diff accuracy: {avg_test_diff_accuracy}")

    # Collect individual task metrics from ResultsCollector
    individual_metrics = trainer.results_collector.get_task_specific_results()

    # Optional: Log individual_metrics for debugging
    logger.debug(f"DEBUG: Individual metrics retrieved: {individual_metrics}")
    print(f"DEBUG: Individual metrics retrieved: {individual_metrics}")

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

def load_config_from_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['config']

import json
import re


def parse_model_summary(model_summary: str, model_checkpoint: str) -> Dict[str, Any]:
    """
    Parses the model summary string into a structured JSON format and adds model file size.

    Args:
        model_summary (str): The raw model summary string.
        model_checkpoint (str): Path to the model checkpoint file.

    Returns:
        Dict[str, Any]: A dictionary containing 'header', 'layers', 'summary', and 'filesize'.
    """
    # Split the model summary into lines
    lines = model_summary.strip().split('\n')

    if len(lines) < 2:
        print("Model summary does not contain sufficient lines.")
        return {"layers": [], "summary": {}, "header": []}

    # Find the header line and the separator line
    header_line = None
    separator_line = None
    for idx, line in enumerate(lines):
        if 'Name' in line and 'Type' in line:
            header_line = line
            separator_line = lines[idx + 1] if idx + 1 < len(lines) else None
            data_start_idx = idx + 2  # Data starts after header and separator
            break

    if header_line is None or separator_line is None:
        print("Header or separator line not found.")
        return {"layers": [], "summary": {}, "header": []}

    # Use the positions of '|' to determine the column boundaries
    positions = [match.start() for match in re.finditer(r'\|', header_line)]

    # Function to parse a line into columns based on '|' positions
    def parse_line(line: str, positions: List[int]) -> List[str]:
        cols = []
        for i in range(len(positions) - 1):
            start = positions[i] + 1
            end = positions[i + 1]
            col = line[start:end].strip()
            cols.append(col)
        # Last column after the last '|'
        start = positions[-1] + 1
        col = line[start:].strip()
        cols.append(col)
        return cols

    # Get the header columns
    header_columns = parse_line(header_line, positions)

    # Initialize list to hold layer details
    layers = []

    # Iterate over the data lines until a non-data line is encountered
    for line in lines[data_start_idx:]:
        # Stop if we reach the separator line (line with dashes)
        if set(line.strip()) == {'-'}:
            # Summary section starts after this line
            summary_start_idx = lines.index(line) + 1
            break

        # Skip empty lines
        if not line.strip():
            continue

        # Parse the line into columns
        cols = parse_line(line, positions)

        # Ensure the number of columns matches the header
        if len(cols) != len(header_columns):
            continue  # Skip lines that don't match the expected format

        # Create a dictionary for the current layer
        layer_dict = dict(zip(header_columns, cols))
        layers.append(layer_dict)
    else:
        # If we didn't break out of the loop, set summary_start_idx to the end
        summary_start_idx = len(lines)

    # Extract summary metrics from the remaining lines
    summary_lines = lines[summary_start_idx:]
    summary_dict = {}
    for line in summary_lines:
        line = line.strip()
        if not line:
            continue
        # Match lines like "195       Trainable params"
        match = re.match(r"(\d+\.?\d*)\s+(.+)", line)
        if match:
            value, key = match.groups()
            # Convert numeric values to float or int where appropriate
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string if conversion fails
            summary_dict[key.strip()] = value
        else:
            # Handle lines that may have the key and value in reverse order
            match = re.match(r"(.+)\s+(\d+\.?\d*)", line)
            if match:
                key, value = match.groups()
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                summary_dict[key.strip()] = value

    # Get model file size
    try:
        filesize = os.path.getsize(model_checkpoint)
        summary_dict["Model File Size (bytes)"] = filesize
    except Exception as e:
        print(f"Error getting model file size: {e}")

    # Combine header, layers, and summary into the final output
    output = {
        "header": header_columns,
        "layers": layers,
        "summary": summary_dict
    }

    return output



def save_results(
    results: Dict[str, Any],
    individual_metrics: Dict[str, Dict[str, Any]],
    output_dir: str,
    model_name: str,
    model_summary: str,
    model_checkpoint: str
) -> str:
    """
    Saves the evaluation results along with the parsed model summary to a JSON file.

    Args:
        results (Dict[str, Any]): Aggregate evaluation metrics.
        individual_metrics (Dict[str, Dict[str, Any]]): Per-task evaluation metrics.
        output_dir (str): Directory to save the results.
        model_name (str): Name of the model for file naming.
        model_summary (str): Raw model summary string.
    
    Returns:
        str: Path to the saved JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_eval_results_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    # Parse the model_summary string into JSON
    parsed_model_summary = parse_model_summary(model_summary, model_checkpoint)

    data_to_save = {
        "aggregate_results": results,
        "individual_metrics": {task_id: metrics for task_id, metrics in individual_metrics.items()},
        "model_summary": parsed_model_summary  # Use the parsed JSON
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

    # Compute symbol frequencies from the test dataset
    symbol_freq_array = test_data.get_symbol_frequencies()
    symbol_freq = {str(i): float(freq) for i, freq in enumerate(symbol_freq_array)}
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')

    # Extract and convert the model configuration from the checkpoint
    if 'model_config' in checkpoint:
        model_config_dict = checkpoint['model_config']
        # Convert dict to ModelConfig object
        model_config = ModelConfig(**model_config_dict)
    else:
        logger.error("Model configuration not found in checkpoint. Please ensure the checkpoint includes 'model_config'.")
        raise ValueError("Model configuration not found in checkpoint. Ensure that the training process includes the ModelConfigSaver callback.")

    # Create configuration
    config = Config(
        model=model_config,
        training=TrainingConfig(),
        evaluation=EvaluationConfig()
    )

    # Determine the number of classes from the test dataset
    max_label_test = max([sample[1].max().item() for sample in test_data])
    num_classes = int(max_label_test) + 1  # Ensure num_classes is an integer
    config.training.symbol_freq = symbol_freq

    # Initialize the model with the complete Config object and symbol frequencies
    model = GPT2ARC(config, num_classes=num_classes, symbol_freq=symbol_freq)
    try:
        # Remove the "model." prefix from state dict keys
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Check if 'loss_fn.weight' is missing and initialize it if necessary
        if "loss_fn.weight" in missing_keys:
            logger.debug("'loss_fn.weight' not found in state_dict. Initializing with default weights.")
            num_classes = config.training.num_classes  # Ensure this is correctly retrieved from your config
            default_weights = torch.ones(num_classes)
            model.loss_fn.weight = default_weights
            logger.debug(f"'loss_fn.weight' initialized with weights: {default_weights}")

        # Optionally, log any unexpected keys for further debugging
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

        # Print all keys in the model's state dictionary
        print("Model state_dict keys:", list(model.state_dict().keys()))
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
    results, individual_metrics = evaluate(model, test_data, config, args.batch_size, args)

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
    results_path = save_results(results, individual_metrics, args.output_dir, model_name, model_summary, args.model_checkpoint)

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
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu"], help="Device accelerator to use (e.g., 'cpu' or 'gpu')")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set logging level
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))
    
    main(args)
