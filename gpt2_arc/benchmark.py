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

# Baseline values for comparison
BASELINE_TOTAL_TIME = 1.6391
BASELINE_GRIDS_PER_SECOND = 199.27

def benchmark_model(model, dataset, batch_size=32, num_batches=10, num_runs=30):
    run_id = str(uuid.uuid4())
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time_runs = []
    grids_per_second_runs = []

    cpu_usages = []
    memory_usages = []

    run_results = []  # Initialize run_results to store each run's data

    for run in range(num_runs):
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=ARCDataset.collate_fn)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        total_time = 0.0
        total_grids = 0

        for i, (inputs, outputs) in enumerate(dataloader):
            if i >= num_batches:
                break

            # Create a dummy attention mask (all ones)
            attention_mask = torch.ones(inputs.size(0), inputs.size(2) * inputs.size(3), dtype=torch.float32)
            inputs, attention_mask = inputs.to(device), attention_mask.to(device)

            # Log system load before processing the batch
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            cpu_usages.append(cpu_percent)
            memory_usages.append(memory_info.percent)
            logger.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_info.percent}%")

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

            logger.info(f"Run {run+1}, Time taken for batch {i+1}: {batch_time:.4f} seconds")

        average_time = total_time / num_batches
        grids_per_second = total_grids / total_time

        logger.info(f"Run {run+1}, Average time per batch: {average_time:.4f} seconds")
        logger.info(f"Run {run+1}, Average grids per second: {grids_per_second:.2f}")
        logger.info(f"Run {run+1}, Total time for {num_batches} batches: {total_time:.4f} seconds")
        logger.info(f"Run {run+1}, Total grids processed: {total_grids}")

        # Store the results of each run
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    # Calculate confidence intervals
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    avg_cpu_usage = np.mean(cpu_usages)
    std_cpu_usage = np.std(cpu_usages, ddof=1)
    ci_cpu_usage = z_score * (std_cpu_usage / np.sqrt(len(cpu_usages)))

    avg_memory_usage = np.mean(memory_usages)
    std_memory_usage = np.std(memory_usages, ddof=1)
    ci_memory_usage = z_score * (std_memory_usage / np.sqrt(len(memory_usages)))
    csv_file_path = 'benchmark_results.csv'

    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(csv_file_path)

    # Append results to the CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'run_id', 'datetime', 'run', 'total_time', 'grids_per_second', 'cpu_usage', 'memory_usage',
            'batch_size', 'num_batches', 'device', 'n_embd', 'n_head', 'n_layer'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        for result in run_results:
            writer.writerow(result)
    avg_total_time = sum(total_time_runs) / num_runs
    avg_grids_per_second = sum(grids_per_second_runs) / num_runs

    std_total_time = statistics.stdev(total_time_runs)
    std_grids_per_second = statistics.stdev(grids_per_second_runs)

    min_total_time = min(total_time_runs)
    max_total_time = max(total_time_runs)
    min_grids_per_second = min(grids_per_second_runs)
    max_grids_per_second = max(grids_per_second_runs)

    # Calculate confidence intervals
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    ci_total_time = z_score * (std_total_time / (num_runs ** 0.5))
    ci_grids_per_second = z_score * (std_grids_per_second / (num_runs ** 0.5))

    logger.info(f"Average total time over {num_runs} runs: {avg_total_time:.4f} seconds (std: {std_total_time:.4f}, min: {min_total_time:.4f}, max: {max_total_time:.4f}, CI: ±{ci_total_time:.4f})")
    logger.info(f"Average grids per second over {num_runs} runs: {avg_grids_per_second:.2f} (std: {std_grids_per_second:.2f}, min: {min_grids_per_second:.2f}, max: {max_grids_per_second:.2f}, CI: ±{ci_grids_per_second:.2f})")

    # Compare with baseline
    time_improvement = avg_total_time - BASELINE_TOTAL_TIME
    grids_per_second_improvement = avg_grids_per_second - BASELINE_GRIDS_PER_SECOND

    time_improvement_percent = (time_improvement / BASELINE_TOTAL_TIME) * 100
    grids_per_second_improvement_percent = (grids_per_second_improvement / BASELINE_GRIDS_PER_SECOND) * 100

    if -20 < time_improvement_percent < 20:
        logger.info("No improvement in total time")
    else:
        logger.info(f"Improvement in total time: {time_improvement:.4f} seconds ({time_improvement_percent:.2f}%)")

    if -20 < grids_per_second_improvement_percent < 20:
        logger.info("No improvement in grids per second")
    else:
        logger.info(f"Improvement in grids per second: {grids_per_second_improvement:.2f} ({grids_per_second_improvement_percent:.2f}%)")

    # Calculate effect size (Cohen's d)
    effect_size_time = (avg_total_time - BASELINE_TOTAL_TIME) / std_total_time
    effect_size_grids = (avg_grids_per_second - BASELINE_GRIDS_PER_SECOND) / std_grids_per_second

    logger.info(f"Effect size for total time: {effect_size_time:.4f}")
    logger.info(f"Effect size for grids per second: {effect_size_grids:.4f}")

    # Perform a one-sample t-test
    t_stat_time, p_value_time = stats.ttest_1samp(total_time_runs, BASELINE_TOTAL_TIME)
    t_stat_grids, p_value_grids = stats.ttest_1samp(grids_per_second_runs, BASELINE_GRIDS_PER_SECOND)

    logger.info(f"T-Test for total time: t-statistic = {t_stat_time:.4f}, p-value = {p_value_time:.4f}")
    logger.info(f"T-Test for grids per second: t-statistic = {t_stat_grids:.4f}, p-value = {p_value_grids:.4f}")

    # Calculate percent change compared to baseline
    percent_change_time = ((avg_total_time - BASELINE_TOTAL_TIME) / BASELINE_TOTAL_TIME) * 100
    percent_change_grids = ((avg_grids_per_second - BASELINE_GRIDS_PER_SECOND) / BASELINE_GRIDS_PER_SECOND) * 100
    stats_csv_file_path = 'benchmark_statistics.csv'
    stats_file_exists = os.path.isfile(stats_csv_file_path)

    with open(stats_csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'run_id', 'datetime', 'avg_total_time', 'std_total_time', 'ci_total_time',
            'avg_grids_per_second', 'std_grids_per_second', 'ci_grids_per_second',
            'effect_size_time', 'effect_size_grids', 'percent_change_time', 'percent_change_grids',
            't_stat_time', 'p_value_time', 't_stat_grids', 'p_value_grids',
            'avg_cpu_usage', 'std_cpu_usage', 'ci_cpu_usage',
            'avg_memory_usage', 'std_memory_usage', 'ci_memory_usage',
            'effect_size_time', 'effect_size_grids',
            'p_value_time', 'p_value_grids'
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
            'percent_change_time': percent_change_time,
            'percent_change_grids': percent_change_grids,
            't_stat_time': t_stat_time,
            'p_value_time': p_value_time,
            't_stat_grids': t_stat_grids,
            'p_value_grids': p_value_grids,
            'avg_cpu_usage': avg_cpu_usage,
            'std_cpu_usage': std_cpu_usage,
            'ci_cpu_usage': ci_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            'std_memory_usage': std_memory_usage,
            'ci_memory_usage': ci_memory_usage,
            'effect_size_time': effect_size_time,
            'effect_size_grids': effect_size_grids,
            'p_value_time': p_value_time,
            'p_value_grids': p_value_grids
        })

    analyze_results(total_time_runs, grids_per_second_runs, cpu_usages, memory_usages)

    return avg_total_time, avg_grids_per_second

def analyze_results(total_time_runs, grids_per_second_runs, cpu_usages, memory_usages):
    # Define confidence level and z-score
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate statistics for CPU and memory usage
    avg_cpu_usage = np.mean(cpu_usages)
    std_cpu_usage = np.std(cpu_usages, ddof=1)
    avg_memory_usage = np.mean(memory_usages)
    std_memory_usage = np.std(memory_usages, ddof=1)

    # Calculate confidence intervals for CPU and memory usage
    ci_cpu_usage = z_score * (std_cpu_usage / np.sqrt(len(cpu_usages)))
    ci_memory_usage = z_score * (std_memory_usage / np.sqrt(len(memory_usages)))

    # Log CPU and memory usage statistics
    logger.info(f" • Average CPU Usage: {avg_cpu_usage:.2f}% (std: {std_cpu_usage:.2f}, CI: ±{ci_cpu_usage:.2f})")
    logger.info(f" • Average Memory Usage: {avg_memory_usage:.2f}% (std: {std_memory_usage:.2f}, CI: ±{ci_memory_usage:.2f})")
    avg_total_time = np.mean(total_time_runs)
    std_total_time = np.std(total_time_runs, ddof=1)
    avg_grids_per_second = np.mean(grids_per_second_runs)
    std_grids_per_second = np.std(grids_per_second_runs, ddof=1)

    # Calculate confidence intervals
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    ci_total_time = z_score * (std_total_time / np.sqrt(len(total_time_runs)))
    ci_grids_per_second = z_score * (std_grids_per_second / np.sqrt(len(grids_per_second_runs)))

    # Calculate effect sizes
    effect_size_time = (avg_total_time - BASELINE_TOTAL_TIME) / std_total_time
    effect_size_grids = (avg_grids_per_second - BASELINE_GRIDS_PER_SECOND) / std_grids_per_second

    # Perform t-tests
    t_stat_time, p_value_time = stats.ttest_1samp(total_time_runs, BASELINE_TOTAL_TIME)
    t_stat_grids, p_value_grids = stats.ttest_1samp(grids_per_second_runs, BASELINE_GRIDS_PER_SECOND)

    # Calculate average CPU and memory usage
    avg_cpu_usage = np.mean(cpu_usages)
    avg_memory_usage = np.mean(memory_usages)

    # Log the results
    logger.info(f" • Average CPU Usage: {avg_cpu_usage:.2f}%")
    logger.info(f" • Average Memory Usage: {avg_memory_usage:.2f}%")
    logger.info("Current Statistics:")
    logger.info(f" • Average Total Time: {avg_total_time:.4f} seconds")
    logger.info(f" • Standard Deviation of Total Time: {std_total_time:.4f} seconds")
    logger.info(f" • Confidence Interval for Total Time (95%): ±{ci_total_time:.4f} seconds")
    logger.info(f" • Average Grids per Second: {avg_grids_per_second:.2f}")
    logger.info(f" • Standard Deviation of Grids per Second: {std_grids_per_second:.2f}")
    logger.info(f" • Confidence Interval for Grids per Second (95%): ±{ci_grids_per_second:.2f}")

    logger.info("Statistical Tests:")
    logger.info(f" • Effect Size for Total Time: {effect_size_time:.4f}")
    logger.info(f" • Effect Size for Grids per Second: {effect_size_grids:.4f}")
    logger.info(f" • T-Test for Total Time: t-statistic = {t_stat_time:.4f}, p-value = {p_value_time:.4f}")
    logger.info(f" • T-Test for Grids per Second: t-statistic = {t_stat_grids:.4f}, p-value = {p_value_grids:.4f}")

    logger.info("Interpretation of Statistical Significance:")
    if abs(effect_size_time) > 0.8:
        logger.info(" • Total Time: Strong practical significance (effect size > 0.8).")
    else:
        logger.info(" • Total Time: Weak practical significance (effect size <= 0.8).")

    if abs(effect_size_grids) > 0.8:
        logger.info(" • Grids per Second: Strong practical significance (effect size > 0.8).")
    else:
        logger.info(" • Grids per Second: Weak practical significance (effect size <= 0.8).")

    if p_value_time < 0.05:
        logger.info(" • Total Time: Statistically significant difference (p-value < 0.05).")
    else:
        logger.info(" • Total Time: No statistically significant difference (p-value >= 0.05).")

    if p_value_grids < 0.05:
        logger.info(" • Grids per Second: Statistically significant difference (p-value < 0.05).")
    else:
        logger.info(" • Grids per Second: No statistically significant difference (p-value >= 0.05).")

    logger.info("Conclusion:")
    if abs(effect_size_time) > 0.8 and p_value_time < 0.05:
        logger.info(" • Total Time: Statistically significant reduction.")
    else:
        logger.info(" • Total Time: No statistically significant reduction.")

    if abs(effect_size_grids) > 0.8 and p_value_grids < 0.05:
        logger.info(" • Grids per Second: Statistically significant improvement.")
    else:
        logger.info(" • Grids per Second: No statistically significant improvement.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the GPT2ARC model.")
    parser.add_argument('--num-runs', type=int, default=20, help='Number of times to run the benchmark')
    args = parser.parse_args()
    # Load data using arckit
    train_set, _ = arckit.load_data()
    
    # Create the ARCDataset
    full_dataset = ARCDataset(train_set, is_test=False)

    # Use the full dataset for benchmarking
    train_dataset = full_dataset
    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)
    model = GPT2ARC(model_config)

    # Run the benchmark with multiple batches
    # Run the benchmark with multiple batches
    total_time, grids_per_second = benchmark_model(model, train_dataset, num_batches=10, num_runs=args.num_runs)

    # Log the results
    logger.info(f"Total time for 10 batches: {total_time:.4f} seconds")
    logger.info(f"Average grids per second: {grids_per_second:.2f}")
