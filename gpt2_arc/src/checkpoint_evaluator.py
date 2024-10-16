#!/usr/bin/env python3
"""
checkpoint_evaluator.py

A script to monitor a directory for new model checkpoints and evaluate them using a specified evaluation script.
Logs are maintained, and resource usage is monitored.

Usage:
    python checkpoint_evaluator.py \
        --output_dir /path/to/output \
        --arc_model_dir /path/to/arc_model \
        --date_folder 2024-10-07 \
        --wandb_project arc-evaluation \
        [optional arguments]
"""

import os
import sys
import time
import subprocess
import threading
import shutil
import logging
import argparse
from datetime import datetime

import psutil  # For resource monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def parse_arguments():
    parser = argparse.ArgumentParser(description="Monitor directories for model checkpoints and evaluate them.")
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store logs and evaluation results.')
    parser.add_argument('--arc_model_dir', type=str, required=True,
                        help='Directory containing the arc model scripts.')
    parser.add_argument('--date_folder', type=str, required=True,
                        help='Date folder name (e.g., 2024-10-07) to organize outputs.')
    parser.add_argument('--wandb_project', type=str, default='arc-evaluation',
                        help='Weights & Biases project name.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation.')
    parser.add_argument('--log_level', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level.')
    parser.add_argument('--resource_monitor_interval', type=int, default=60,
                        help='Interval in seconds for resource monitoring logs.')
    
    return parser.parse_args()

def setup_logging(output_dir, log_level):
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "checkpoint_evaluator.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("CheckpointEvaluator")
    return logger

def load_evaluated_models(evaluated_models_file, logger):
    evaluated_models = set()
    if os.path.exists(evaluated_models_file):
        try:
            with open(evaluated_models_file, "r") as f:
                evaluated_models.update(line.strip() for line in f)
            logger.info(f"Loaded evaluated models from {evaluated_models_file}")
        except Exception as e:
            logger.error(f"Error loading evaluated models from {evaluated_models_file}: {e}")
    else:
        logger.info("No previously evaluated models found. Starting fresh.")
    return evaluated_models

def save_evaluated_model(evaluated_models_file, model_path, logger):
    try:
        with open(evaluated_models_file, "a") as f:
            f.write(model_path + "\n")
        logger.debug(f"Recorded evaluation of {model_path} in {evaluated_models_file}")
    except Exception as e:
        logger.error(f"Error writing to evaluated models file {evaluated_models_file}: {e}")

def wait_for_file_stable(file_path, wait_time=1.0, max_retries=10, logger=None):
    """Wait until the file is stable (not changing size)"""
    previous_size = -1
    retries = 0
    while retries < max_retries:
        if not os.path.exists(file_path):
            if logger:
                logger.warning(f"File {file_path} does not exist.")
            return False
        current_size = os.path.getsize(file_path)
        if current_size == previous_size:
            return True
        else:
            previous_size = current_size
            time.sleep(wait_time)
            retries += 1
    if logger:
        logger.warning(f"File {file_path} is not stable after {max_retries} retries.")
    return False

class CheckpointHandler(FileSystemEventHandler):
    def __init__(self, evaluated_models, temp_checkpoint_dir, evaluate_callback, logger):
        super().__init__()
        self.evaluated_models = evaluated_models
        self.temp_checkpoint_dir = temp_checkpoint_dir
        self.evaluate_callback = evaluate_callback
        self.logger = logger

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.ckpt') or event.src_path.endswith('.pth'):
            self.evaluate_model(event.src_path)

    def evaluate_model(self, model_path):
        model_file = os.path.basename(model_path)
        if model_path in self.evaluated_models:
            self.logger.info(f"Skipping already evaluated model: {model_file}")
            return  # Skip if the model was already evaluated

        # Wait for the file to be stable
        if not wait_for_file_stable(model_path, logger=self.logger):
            self.logger.warning(f"File {model_file} is not stable. Skipping evaluation.")
            return

        # Copy the checkpoint file to temp_checkpoint_dir
        temp_model_path = os.path.join(self.temp_checkpoint_dir, model_file)
        try:
            shutil.copy2(model_path, temp_model_path)
            self.logger.info(f"Copied {model_file} to temporary directory.")
        except Exception as e:
            self.logger.error(f"Error copying {model_file} to temporary directory: {e}")
            return

        # Extract epoch and val_loss from the filename for run_name
        try:
            parts = model_file.replace('.ckpt', '').replace('.pth', '').split('-')
            epoch = None
            val_loss = None
            for part in parts:
                if part.startswith('epoch='):
                    epoch = part.split('=')[1]
                elif part.startswith('val_loss='):
                    val_loss = part.split('=')[1]
            if epoch is not None and val_loss is not None:
                run_name = f"evaluation-epoch{epoch}-val_loss{val_loss}"
            else:
                run_name = f"evaluation-{model_file}"
            self.logger.debug(f"Parsed run name: {run_name}")
        except Exception as e:
            self.logger.error(f"Error parsing run name from filename {model_file}: {e}")
            run_name = f"evaluation-{model_file}"

        # Define the evaluation command
        eval_command = [
            "python", os.path.join(args.arc_model_dir, "gpt2_arc/src/evaluate.py"),
            "--model_checkpoint", temp_model_path,
            "--batch_size", str(args.batch_size),
            "--output_dir", args.output_dir,
            "--wandb_project", args.wandb_project,
            "--wandb_run_name", run_name
        ]
        self.logger.info(f"Evaluating model: {model_file} with run name: {run_name}")

        # Define the log file path
        log_file_path = os.path.join(args.output_dir, f"{model_file}_evaluation.log")
        try:
            with open(log_file_path, "w") as log_file:
                # Run the evaluation command and redirect stdout and stderr to the log file
                subprocess.run(
                    eval_command,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True  # Automatically decode bytes to string
                )
            self.logger.info(f"Successfully evaluated model: {model_file}. Logs at {log_file_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during evaluation of {model_file}. See log at {log_file_path}")
        except Exception as ex:
            self.logger.exception(f"An unexpected error occurred while evaluating {model_file}: {ex}")

        self.evaluated_models.add(model_path)
        save_evaluated_model(os.path.join(args.output_dir, "evaluated_models.txt"), model_path, self.logger)

        # Delete the temp model file
        try:
            os.remove(temp_model_path)
            self.logger.debug(f"Deleted temporary model file: {temp_model_path}")
        except Exception as e:
            self.logger.error(f"Error deleting temp model file {temp_model_path}: {e}")

def get_all_checkpoint_files(directory):
    checkpoint_files = []
    for root, _, files in os.walk(directory):
        checkpoint_files.extend([os.path.join(root, f) for f in files if f.endswith('.ckpt') or f.endswith('.pth')])
    return checkpoint_files

def start_observer(model_dir, handler, logger):
    # Set up and start the watchdog observer
    observer = Observer()
    observer.schedule(handler, model_dir, recursive=True)
    observer.start()

    logger.info("Watching for new checkpoints and final models in all subdirectories...")
    logger.info("This script will continue running in the background.")

    try:
        while True:
            time.sleep(10)
            # Optionally, you can implement additional periodic checks here
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Observer stopped by user.")
    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}")
        logger.error(f"Please ensure that the directory '{model_dir}' exists.")
    except Exception as e:
        logger.exception(f"An error occurred in the observer: {e}")
    finally:
        observer.join()
        logger.info("Checkpoint and final model evaluation completed.")

def monitor_resources(logger, interval=60):
    while True:
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            logger.debug(f"Memory Usage: {memory.percent}%")
            logger.debug(f"CPU Usage: {cpu}%")
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")

def main(args):
    logger = setup_logging(args.output_dir, args.log_level)
    logger.info("Starting Checkpoint Evaluator Script")
    logger.debug(f"Arguments: {args}")

    model_dir = os.path.join(args.date_folder, "checkpoints")
    logger.debug(f"Watching for new models in directory: {model_dir}")

    # Create necessary directories
    os.makedirs(model_dir, exist_ok=True)
    temp_checkpoint_dir = os.path.join(args.output_dir, "temp_checkpoints")
    os.makedirs(temp_checkpoint_dir, exist_ok=True)

    # Load previously evaluated models
    evaluated_models_file = os.path.join(args.output_dir, "evaluated_models.txt")
    evaluated_models = load_evaluated_models(evaluated_models_file, logger)

    # Set up the event handler
    handler = CheckpointHandler(evaluated_models, temp_checkpoint_dir, None, logger)

    # Initialize watchdog event handler with the ability to evaluate models
    event_handler = CheckpointHandler(evaluated_models, temp_checkpoint_dir, None, logger)

    # Start the observer in a separate thread
    observer_thread = threading.Thread(target=start_observer, args=(model_dir, event_handler, logger))
    observer_thread.daemon = True  # Ensures the thread will exit when the main program exits
    observer_thread.start()
    logger.info("Background checkpoint observer started.")

    # Start the resource monitor in a background thread
    resource_monitor_thread = threading.Thread(target=monitor_resources, args=(logger, args.resource_monitor_interval))
    resource_monitor_thread.daemon = True
    resource_monitor_thread.start()
    logger.info("Background resource monitor started.")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Script terminated by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
