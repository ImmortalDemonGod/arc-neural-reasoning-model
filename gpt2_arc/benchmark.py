import argparse
import csv
import uuid
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
import arckit
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.models.gpt2 import GPT2ARC
from gpt2_arc.src.config import ModelConfig
import time
import psutil
import logging
import statistics
import numpy as np
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamically adjustable baseline values for CPU, GPU, and MPS
BASELINES = {
    'cpu': {'total_time': 1.6391, 'grids_per_second': 199.27},
    'gpu': {'total_time': 0.8392, 'grids_per_second': 299.15},
    'mps': {'total_time': 0.9392, 'grids_per_second': 279.10}  # Example baselines for MPS
}

def benchmark_model(model, dataset, batch_size=32, num_batches=10, num_runs=30, device_type='cpu'):
    run_id = str(uuid.uuid4())
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time_runs = []
    grids_per_second_runs = []

    cpu_usages = []
    memory_usages = []

    run_results = []  # Initialize run_results to store each run's data

    # Select device based on the argument (including support for MPS)
    device = torch.device("cuda" if device_type == "gpu" and torch.cuda.is_available() else
                          "mps" if device_type == "gpu" and torch.backends.mps.is_available() else "cpu")
    model.to(device)

    for run in range(num_runs):
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=ARCDataset.collate_fn)
        total_time = 0.0
        total_grids = 0

        for i, (inputs, outputs) in enumerate(dataloader):
            if i >= num_batches:
                break

            # Create a dummy attention mask (all ones)
            attention_mask = torch.ones(inputs.size(0), inputs.size(2) * inputs.size(3), dtype=torch.float32)
            inputs, attention_mask = inputs.to(device), attention_mask.to(device)

            # Log system load and system state before processing the batch
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            cpu_usages.append(cpu_percent)
            memory_usages.append(memory_info.percent)
            logger.info(f"Run {run+1}, Batch {i+1}: CPU Usage: {cpu_percent}%, Memory Usage: {memory_info.percent}%")

            # Measure the time taken to process the batch
            start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                model(inputs, attention_mask)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            total_grids += len(inputs)

        # Average metrics for the run
        grids_per_second = total_grids / total_time

        logger.info(f"Run {run+1}: Total Time: {total_time:.4f} seconds, Grids per Second: {grids_per_second:.2f}")
        
        # Store the results of each run
        run_results.append({
            'run_id': run_id,
            'datetime': current_time,
            'run': run + 1,
            'total_time': total_time,
            'grids_per_second': grids_per_second,
            'cpu_usage': np.mean(cpu_usages),
            'memory_usage': np.mean(memory_usages),
            'batch_size': batch_size,
            'num_batches': num_batches,
            'device': device.type,
            'n_embd': model.config.n_embd,
            'n_head': model.config.n_head,
            'n_layer': model.config.n_layer
        })

        total_time_runs.append(total_time)
        grids_per_second_runs.append(grids_per_second)

    # Calculate average and standard deviation over multiple runs
    avg_total_time = np.mean(total_time_runs)
    avg_grids_per_second = np.mean(grids_per_second_runs)
    std_total_time = np.std(total_time_runs, ddof=1)
    std_grids_per_second = np.std(grids_per_second_runs, ddof=1)

    # Perform statistical analysis (confidence intervals, effect size, etc.)
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    ci_total_time = z_score * (std_total_time / np.sqrt(num_runs))
    ci_grids_per_second = z_score * (std_grids_per_second / np.sqrt(num_runs))

    effect_size_time = (avg_total_time - BASELINES[device.type]['total_time']) / std_total_time
    effect_size_grids = (avg_grids_per_second - BASELINES[device.type]['grids_per_second']) / std_grids_per_second

    # Compare with baseline
    time_improvement = avg_total_time - BASELINES[device.type]['total_time']
    grids_per_second_improvement = avg_grids_per_second - BASELINES[device.type]['grids_per_second']

    time_improvement_percent = (time_improvement / BASELINES[device.type]['total_time']) * 100
    grids_per_second_improvement_percent = (grids_per_second_improvement / BASELINES[device.type]['grids_per_second']) * 100

    if -20 < time_improvement_percent < 20:
        logger.info("No improvement in total time")
    else:
        logger.info(f"Improvement in total time: {time_improvement:.4f} seconds ({time_improvement_percent:.2f}%)")

    if -20 < grids_per_second_improvement_percent < 20:
        logger.info("No improvement in grids per second")
    else:
        logger.info(f"Improvement in grids per second: {grids_per_second_improvement:.2f} ({grids_per_second_improvement_percent:.2f}%)")

    # Perform a one-sample t-test
    t_stat_time, p_value_time = stats.ttest_1samp(total_time_runs, BASELINES[device.type]['total_time'])
    t_stat_grids, p_value_grids = stats.ttest_1samp(grids_per_second_runs, BASELINES[device.type]['grids_per_second'])

    logger.info(f"T-Test for total time: t-statistic = {t_stat_time:.4f}, p-value = {p_value_time:.4f}")
    logger.info(f"T-Test for grids per second: t-statistic = {t_stat_grids:.4f}, p-value = {p_value_grids:.4f}")

    # Log the results including confidence intervals
    logger.info(f"Run Summary:")
    logger.info(f" • Avg Total Time: {avg_total_time:.4f}s (CI 95%: ±{ci_total_time:.4f}s)")
    logger.info(f" • Avg Grids per Second: {avg_grids_per_second:.2f} (CI 95%: ±{ci_grids_per_second:.2f})")
    logger.info(f" • Effect Size (Total Time): {effect_size_time:.4f}, Effect Size (Grids per Second): {effect_size_grids:.4f}")

    # Write results to CSV
    csv_file_path = 'benchmark_results.csv'
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'run_id', 'datetime', 'run', 'total_time', 'grids_per_second', 'cpu_usage', 'memory_usage',
            'batch_size', 'num_batches', 'device', 'n_embd', 'n_head', 'n_layer'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for result in run_results:
            writer.writerow(result)

    # Write statistical summary to CSV
    stats_csv_file_path = 'benchmark_statistics.csv'
    stats_file_exists = os.path.isfile(stats_csv_file_path)
    with open(stats_csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'run_id', 'datetime', 'avg_total_time', 'std_total_time', 'ci_total_time',
            'avg_grids_per_second', 'std_grids_per_second', 'ci_grids_per_second',
            'effect_size_time', 'effect_size_grids', 'percent_change_time', 'percent_change_grids',
            't_stat_time', 'p_value_time', 't_stat_grids', 'p_value_grids'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not stats_file_exists:
            writer.writeheader()
        writer.writerow({
            'run_id': run_id,
            'datetime': current_time,
            'avg_total_time': avg_total_time,
            'std_total_time': std_total_time,
            'ci_total_time': ci_total_time,
            'avg_grids_per_second': avg_grids_per_second,
            'std_grids_per_second': std_grids_per_second,
            'ci_grids_per_second': ci_grids_per_second,
            'effect_size_time': effect_size_time,
            'effect_size_grids': effect_size_grids,
            'percent_change_time': time_improvement_percent,
            'percent_change_grids': grids_per_second_improvement_percent,
            't_stat_time': t_stat_time,
            'p_value_time': p_value_time,
            't_stat_grids': t_stat_grids,
            'p_value_grids': p_value_grids
        })

    return avg_total_time, avg_grids_per_second


def main(args):
    # Load the dataset
    train_set, _ = arckit.load_data()
    full_dataset = ARCDataset(train_set, is_test=False)

    # Create the model configuration
    model_config = ModelConfig(n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)
    model = GPT2ARC(model_config)

    # Run the benchmark for different configurations
    for run_num in range(args.num_full_runs):
        logger.info(f"Starting full benchmark run {run_num + 1}/{args.num_full_runs}")
        avg_time, avg_grids, batch_metrics = benchmark_model(
            model, full_dataset, batch_size=args.batch_size, num_batches=args.num_batches, num_runs=args.num_runs, device_type=args.device
        )
        logger.info(f"Full run {run_num + 1} - Avg Time: {avg_time:.4f}s, Avg Grids per Second: {avg_grids:.2f}")
        logger.info(f"Per-Batch Metrics: {batch_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the GPT2ARC model.")
    parser.add_argument('--num-runs', type=int, default=20, help='Number of runs for each configuration')
    parser.add_argument('--num-full-runs', type=int, default=1, help='Number of full configurations to run')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for each run')
    parser.add_argument('--num-batches', type=int, default=10, help='Number of batches per run')
    parser.add_argument('--n-embd', type=int, default=64, help='Number of embeddings for the model')
    parser.add_argument('--n-head', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n-layer', type=int, default=1, help='Number of layers')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Device to run the benchmark on (cpu, gpu, or mps)')
    
    args = parser.parse_args()
    main(args)