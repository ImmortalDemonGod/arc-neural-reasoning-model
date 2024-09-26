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

    perfect_accuracy_threshold = config.evaluation.perfect_accuracy_threshold
    perfect_tasks = 0
    total_tasks = 0
    individual_metrics = []
    all_accuracies = []
    all_diff_accuracies = []
    total_loss = 0

    for result in results:
        logger.debug(f"Processing result: {result}")
        task_ids = result.get('task_ids', [])
        accuracies = result.get('test_accuracy', [])
        diff_accuracy = result.get('test_diff_accuracy', 0)
        loss = result.get('test_loss', 0)
        
        if not isinstance(task_ids, list):
            task_ids = [task_ids]
        if not isinstance(accuracies, list):
            accuracies = [accuracies]

        logger.debug(f"Task IDs: {task_ids}, Accuracies: {accuracies}, Diff Accuracy: {diff_accuracy}")

        for task_id, accuracy in zip(task_ids, accuracies):
            diff_accuracy = diff_accuracies  # Use the single diff_accuracy value
            if task_id and accuracy is not None and diff_accuracy is not None:
                individual_metrics.append((task_id, {
                    'test_accuracy': accuracy,
                    'test_diff_accuracy': diff_accuracy
                }))
                all_accuracies.append(accuracy)
                all_diff_accuracies.append(diff_accuracy)
                logger.info(f"Task {task_id}: Accuracy = {accuracy:.4f}, Diff Accuracy = {diff_accuracy:.4f}")

                if accuracy >= perfect_accuracy_threshold:
                    perfect_tasks += 1
                total_tasks += 1

        total_loss += loss

    complete_task_accuracy = perfect_tasks / total_tasks if total_tasks > 0 else 0

    logger.debug(f"DEBUG: Individual metrics collected: {individual_metrics}")
    logger.debug(f"DEBUG: Perfect tasks: {perfect_tasks}, Total tasks: {total_tasks}")

    logger.info(f"Complete Task Accuracy: {complete_task_accuracy:.2%}")

    aggregated_results = {
        'test_loss': total_loss / len(results) if results else 0,
        'test_accuracy': sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0,
        'test_diff_accuracy': sum(all_diff_accuracies) / len(all_diff_accuracies) if all_diff_accuracies else 0,
        'complete_task_accuracy': complete_task_accuracy
    }

    logger.debug(f"DEBUG: Aggregated results: {aggregated_results}")

    return aggregated_results, individual_metrics

def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['config']

def save_results(results, individual_metrics, output_dir, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_eval_results_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    data_to_save = {
        "aggregate_results": results,
        "individual_metrics": {task_id: metrics for task_id, metrics in individual_metrics}
    }

    logger.debug(f"DEBUG: Data to be saved: {data_to_save}")

    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)

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
    try:
        # Remove the "model." prefix from state dict keys
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Error while loading state_dict: {e}")
        logger.error(f"Available keys in checkpoint: {list(checkpoint.keys())}")
        raise

    model.eval()
    logger.info("Model set to evaluation mode")

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
        print(f"{metric}: {value}")
        wandb.log({f"eval/{metric}": value})

    # Log individual task metrics
    for task_id, metrics in individual_metrics:
        for metric_name, metric_value in metrics.items():
            wandb.log({f"eval/task_{task_id}/{metric_name}": metric_value})
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
    parser.add_argument("--log-level", type=str, default="INFO", help="Set the logging level (e.g., DEBUG, INFO, WARNING)")
    parser.add_argument("--wandb_project", type=str, default="arc-evaluation", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set logging level
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))
    
    main(args)
