import torch._dynamo
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
from torch.amp import autocast
import psutil
import logging
import argparse
import statistics
import numpy as np
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamically adjustable baseline values for CPU, GPU, and MPS
BASELINES = {
    'cpu': {'total_time': 1.6391, 'grids_per_second': 199.27},
    'cuda': {'total_time': 0.0481, 'grids_per_second': 13774.98},
    'mps': {'total_time': 0.0481, 'grids_per_second': 13774.98}  # Updated baselines for MPS
}

def benchmark_model(model, dataset, batch_size=1, num_batches=1, num_runs=1, device_type='cpu', precision='medium', model_checkpoint=None):
    print(f"Starting benchmark_model with parameters: batch_size={batch_size}, num_batches={num_batches}, num_runs={num_runs}, device_type={device_type}, precision={precision}, model_checkpoint={model_checkpoint}")

    checkpoint_used = False
    checkpoint_info = {}
    if model_checkpoint:
        checkpoint = torch.load(model_checkpoint)
        model_config = ModelConfig(**checkpoint['config'])
        model = GPT2ARC(model_config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device_type)
        model.eval()
        checkpoint_used = True
        checkpoint_info = {
            'checkpoint_path': model_checkpoint,
            'config': checkpoint['config'],
            'state_dict_keys': list(checkpoint['state_dict'].keys())
        }
    run_id = str(uuid.uuid4())
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    practical_threshold = 20.0  # Define a threshold for practical significance
    total_time_runs = []
    grids_per_second_runs = []

    cpu_usages = []
    memory_usages = []

    run_results = []  # Initialize run_results to store each run's data
    gpu_usages = []  # Initialize gpu_usages to store GPU utilization data

    # Select device based on the argument (including support for MPS)
    device = torch.device("cuda" if device_type == "cuda" and torch.cuda.is_available() else
                          "mps" if device_type == "mps" and torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    torch._dynamo.config.suppress_errors = True
    if device.type == "cpu":
        compiled_model = model  # Use the model directly for CPU
    else:
        try:
            if device.type != "mps":
                compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            else:
                compiled_model = model  # Use the model directly for MPS
        except ImportError as e:
            logger.warning(f"Compilation failed with error: {e}. Falling back to eager execution.")
            compiled_model = model

    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}")
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=ARCDataset.collate_fn)
        total_time = 0.0
        total_grids = 0

        for i, (inputs, outputs) in enumerate(dataloader):
            print(f"Processing batch {i + 1}/{num_batches}")
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
            if device.type == 'cuda':
                gpu_utilization = torch.cuda.utilization(device.index)
                gpu_usages.append(gpu_utilization)
                logger.info(f"Run {run+1}, Batch {i+1}: CPU Usage: {cpu_percent}%, Memory Usage: {memory_info.percent}%, GPU Utilization: {gpu_utilization}%")
            else:
                logger.info(f"Run {run+1}, Batch {i+1}: CPU Usage: {cpu_percent}%, Memory Usage: {memory_info.percent}%")

            # Measure the time taken to process the batch
            start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                if device.type == 'cuda':
                    with autocast(device_type=device.type, dtype=torch.float16):
                        compiled_model(inputs, attention_mask)
                else:
                    compiled_model(inputs, attention_mask)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()

            batch_time = end_time - start_time
            print(f"Batch time: {batch_time}")
            if batch_time <= 0:
                print(f"WARNING: Invalid batch time: {batch_time}. Skipping this batch.")
                continue
            total_time += batch_time
            total_grids += len(inputs)

        print(f"Run {run + 1} completed. Total time: {total_time}, Total grids: {total_grids}")
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
            'gpu_usage': np.mean(gpu_usages) if gpu_usages else None,
            'batch_size': batch_size,
            'num_batches': num_batches,
            'device': device.type,
            'n_embd': model.config.n_embd,
            'n_head': model.config.n_head,
            'n_layer': model.config.n_layer,
            'precision': precision,  # Add precision here
            'checkpoint_used': checkpoint_used,
            'checkpoint_info': checkpoint_info
        })

        total_time_runs.append(total_time)
        grids_per_second_runs.append(grids_per_second)

    if total_time <= 0 or total_grids <= 0:
        print(f"ERROR: Invalid total time ({total_time}) or total grids ({total_grids}). Check the benchmark implementation.")
        return 0, 0  # Return sensible defaults instead of negative values
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

    # Calculate improvements and regressions based on averages
    time_improvement = BASELINES[device.type]['total_time'] - avg_total_time
    time_improvement_percent = (time_improvement / BASELINES[device.type]['total_time']) * 100
    time_regression = avg_total_time - BASELINES[device.type]['total_time']
    time_regression_percent = (time_regression / BASELINES[device.type]['total_time']) * 100

    grids_per_second_improvement = avg_grids_per_second - BASELINES[device.type]['grids_per_second']
    grids_per_second_improvement_percent = (grids_per_second_improvement / BASELINES[device.type]['grids_per_second']) * 100
    grids_per_second_regression = BASELINES[device.type]['grids_per_second'] - avg_grids_per_second
    grids_per_second_regression_percent = (grids_per_second_regression / BASELINES[device.type]['grids_per_second']) * 100

    # Determine if there was an improvement
    improvement_time = avg_total_time < BASELINES[device.type]['total_time']
    improvement_grids = avg_grids_per_second > BASELINES[device.type]['grids_per_second']

    # Log improvements or regressions based on averages
    if avg_total_time < BASELINES[device.type]['total_time']:
        logger.info(f"Improvement in average total time: -{time_improvement:.4f} seconds ({time_improvement_percent:.2f}%)")
    else:
        logger.info(f"Regression in average total time: +{time_regression:.4f} seconds ({time_regression_percent:.2f}%)")

    if avg_grids_per_second > BASELINES[device.type]['grids_per_second']:
        logger.info(f"Improvement in average grids per second: +{grids_per_second_improvement:.2f} ({grids_per_second_improvement_percent:.2f}%)")
    else:
        logger.info(f"Regression in average grids per second: -{grids_per_second_regression:.2f} ({grids_per_second_regression_percent:.2f}%)")

    # Update practical significance checks
    practical_significance_time = time_improvement_percent >= practical_threshold
    practical_significance_grids = grids_per_second_improvement_percent >= practical_threshold

    # Log practical significance
    if improvement_time:
        if practical_significance_time:
            logger.info("The improvement in average total time is practically significant.")
        else:
            logger.info("The improvement in average total time is not practically significant.")
    else:
        if practical_significance_time:
            logger.info("The regression in average total time is practically significant.")
        else:
            logger.info("The regression in average total time is not practically significant.")

    if improvement_grids:
        if practical_significance_grids:
            logger.info("The improvement in average grids per second is practically significant.")
        else:
            logger.info("The improvement in average grids per second is not practically significant.")
    else:
        if practical_significance_grids:
            logger.info("The regression in average grids per second is practically significant.")
        else:
            logger.info("The regression in average grids per second is not practically significant.")

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

    # Determine if there was an improvement
    improvement_time = avg_total_time < BASELINES[device.type]['total_time']
    improvement_grids = avg_grids_per_second > BASELINES[device.type]['grids_per_second']
    csv_file_path = 'benchmark_results.csv'
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'run_id', 'datetime', 'run', 'total_time', 'grids_per_second', 'cpu_usage', 'memory_usage',
            'batch_size', 'num_batches', 'device', 'n_embd', 'n_head', 'n_layer', 'gpu_usage', 'precision',
            'checkpoint_used', 'checkpoint_info'
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
            't_stat_time', 'p_value_time', 't_stat_grids', 'p_value_grids',
            'improvement_time', 'improvement_grids',
            'practical_significance_time', 'practical_significance_grids', 'precision'
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
            'percent_change_time': time_improvement_percent if improvement_time else time_regression_percent,
            'percent_change_grids': grids_per_second_improvement_percent if improvement_grids else grids_per_second_regression_percent,
            't_stat_time': t_stat_time,
            'p_value_time': p_value_time,
            't_stat_grids': t_stat_grids,
            'p_value_grids': p_value_grids,
            'improvement_time': improvement_time,
            'improvement_grids': improvement_grids,
            'practical_significance_time': practical_significance_time,
            'practical_significance_grids': practical_significance_grids,
            'precision': precision,  # Add precision here
            'checkpoint_used': checkpoint_used,
            'checkpoint_info': checkpoint_info
        })

    print(f"Benchmark completed. Final results - avg_time: {avg_total_time}, avg_grids: {avg_grids_per_second}")
    return avg_total_time, avg_grids_per_second


def main(args):
    print(f"Starting main function with args: {args}")
    # Set the float32 matmul precision
    torch.set_float32_matmul_precision(args.precision)
    train_set, _ = arckit.load_data()
    full_dataset = ARCDataset(train_set, is_test=False)

    # Create the model configuration
    model_config = ModelConfig(n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)
    model = GPT2ARC(model_config)

    # Run the benchmark for different configurations
    for run_num in range(args.num_full_runs):
        logger.info(f"Starting full benchmark run {run_num + 1}/{args.num_full_runs}")
        avg_time, avg_grids = benchmark_model(
            model, full_dataset, batch_size=args.batch_size, num_batches=args.num_batches, num_runs=args.num_runs, device_type=args.device, precision=args.precision
        )
        logger.info(f"Full run {run_num + 1} - Avg Time: {avg_time:.4f}s, Avg Grids per Second: {avg_grids:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the GPT2ARC model.")
    parser.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--num-runs', type=int, default=20, help='Number of runs for each configuration')
    parser.add_argument('--num-full-runs', type=int, default=1, help='Number of full configurations to run')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for each run')
    parser.add_argument('--num-batches', type=int, default=10, help='Number of batches per run')
    parser.add_argument('--n-embd', type=int, default=64, help='Number of embeddings for the model')
    parser.add_argument('--n-head', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n-layer', type=int, default=1, help='Number of layers')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu', help='Device to run the benchmark on (cpu, cuda, or mps)')
    parser.add_argument('--precision', choices=['highest', 'high', 'medium'], default='highest', help='Precision level for float32 matrix multiplications')
    
    args = parser.parse_args()
    main(args)
