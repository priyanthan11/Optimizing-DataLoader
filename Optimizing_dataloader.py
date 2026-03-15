"""
Optimizing DataLoaders for Performance

Slow model training is a frequent obstacle in machine learning projects. While it's easy
to assume that hardware is the limiting factor, performance issues often stem from a data
bottleneck — where the powerful GPU is left waiting for data to be processed. This
inefficiency can significantly prolong training cycles. Addressing how data is loaded and
delivered to the accelerator is a vital step toward faster and more effective model development.

The Goal: Learning the process, Not the Numbers.

NOTE (Windows users):
    On Windows, Python uses "spawn" (not "fork") to create worker processes.
    This means each worker re-imports this entire file from scratch.
    ALL executable code that should only run once must live inside
    the `if __name__ == "__main__":` block at the bottom of this file.
    Imports and function/class definitions are safe to keep at module level.
"""

import gc
import os
import json
import time
import warnings

import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ── Suppress NumPy 2.x / torchvision pickle warning ──────────────────────────
warnings.filterwarnings(
    "ignore",
    category=np.exceptions.VisibleDeprecationWarning
)


# ── Dataset ───────────────────────────────────────────────────────────────────
def download_and_load_cifar10(data_dir="./data"):
    """Downloads CIFAR-10 and returns (train_set, test_set)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_set, test_set


# ── Timing helper ─────────────────────────────────────────────────────────────
def measure_average_epoch_time(loader, device, epochs=5, warmup=2):
    """
    Iterates through the DataLoader for `epochs` epochs and returns the
    average epoch time (in seconds) over the non-warmup epochs.

    Args:
        loader  (DataLoader):   DataLoader to benchmark.
        device  (torch.device): Target device.
        epochs  (int): Total epochs to run.
        warmup  (int): Initial epochs to discard (OS/CUDA warm-up noise).

    Returns:
        float: Average epoch time in seconds.
    """
    epoch_times = []

    for epoch in tqdm(range(epochs), desc="  Epochs"):
        start = time.perf_counter()

        for batch in loader:
            # CIFAR-10 batches are (images, labels) tuples
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device, non_blocking=True)

            # Sync inside the loop so GPU transfer time is included
            if device.type == "cuda":
                torch.cuda.synchronize()

        epoch_time = time.perf_counter() - start
        epoch_times.append(epoch_time)

        label = "(warm-up)" if epoch < warmup else ""
        print(f"\n  Epoch {epoch + 1}/{epochs} | {epoch_time:.2f}s {label}")

    measured = epoch_times[warmup:]
    avg = sum(measured) / len(measured)
    print(f"\n  -> Average (last {len(measured)} epochs): {avg:.2f}s\n")
    return avg


# ── Single-case experiment ────────────────────────────────────────────────────
def experiment_workers(num_workers, trainset, device):
    """
    Benchmarks the DataLoader for a single `num_workers` value.

    Args:
        num_workers (int):          Number of DataLoader worker processes.
        trainset    (Dataset):      Dataset to load.
        device      (torch.device): Target device.

    Returns:
        float: Average epoch time in seconds (inf on error).
    """
    print(f"\n{'─' * 50}")
    print(f"  Testing num_workers = {num_workers}")
    print(f"{'─' * 50}")

    loader = DataLoader(
        trainset,
        batch_size=256,                        # large enough to amortise worker overhead
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,                       # faster CPU -> GPU transfers
        # keep workers alive between epochs
        persistent_workers=(num_workers > 0),
    )

    try:
        result = measure_average_epoch_time(loader, device)
    except RuntimeError as e:
        print(f"\n  ERROR with {num_workers} workers: {e}")
        result = float("inf")
    finally:
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ── Experiment batch size ────────────────────────────────────────────

def experiment_batch_sizes(batch_sizes_to_test, trainset, device):
    """
      Measures the data loading time for different batch sizes.

      Args:
          batch_sizes_to_test: A list of integers representing the batch sizes to test.
          trainset: The dataset to be loaded.
          device: The device to which the data will be moved (e.g., 'cpu' or 'cuda').
      """

    print(f"\n{'─' * 50}")
    print(f"  Testing batch_size = {batch_sizes_to_test}")
    print(f"{'─' * 50}")

    # Create a new DataLoader instance for each specific test
    loader = DataLoader(
        trainset,
        batch_size=batch_sizes_to_test,  # set the value to the current value in the top
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    # Handle potential runtime errors, especially out-of-memory
    try:
        # Time the data loading for one epoch and save it to the dictionary
        batch_size_times = measure_average_epoch_time(loader, device)
    except RuntimeError as e:
        # If an error occurs (often from running out of GPU memory)
        batch_size_times = float('inf')
    finally:
        # Clean up the loader and call the garbage collector to free up memory
        # ensuring each test runs in a clean environment.
        del loader
        gc.collect()
        # Clear the PyTorch CUDA cache to free up GPU memory.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return batch_size_times

# ── Experiment Prefetch_factor ────────────────────────────────────────────


def experiment_prefetch_factor(prefetch_factors_to_test, trainset, device):
    """
    Measures the data loading time for different prefetch factor settings.

    Args:
        prefetch_factors_to_test: A list of integers representing the prefetch factors to test.
        trainset: The dataset to be loaded.
        device: The device to which the data will be moved (e.g., 'cpu' or 'cuda').
    """
    print(f"\n{'─' * 50}")
    print(f"  Testing prefetch_factor = {prefetch_factors_to_test}")
    print(f"{'─' * 50}")

    # Create a new Dataloader instance for each specific test
    loader = DataLoader(
        trainset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=prefetch_factors_to_test
    )
    # Handle potential run time errors, especially out-of-memory
    try:
        prefetch_factor_times = measure_average_epoch_time(loader, device)
    except RuntimeError as e:
        batch_size_times = float('inf')
    finally:
        del loader
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return prefetch_factor_times


# ── Experiment runner with caching ────────────────────────────────────────────

def run_experiment(
    experiment_name,
    experiment_fcn,
    cases,
    trainset,
    device,
    rerun=False,
    cache_dir="checkpoint_experiments",
):
    """
    Runs an experiment across multiple cases and caches the results as JSON.

    Args:
        experiment_name (str):      Unique identifier; also the cache filename.
        experiment_fcn  (callable): fn(case, trainset, device) -> float
        cases           (list):     Parameter values to iterate over.
        trainset        (Dataset):  Dataset passed to experiment_fcn.
        device          (torch.device): Device passed to experiment_fcn.
        rerun           (bool):     If True, ignore cache and re-run.
        cache_dir       (str):      Directory for cached results.

    Returns:
        dict: {case_value: result}
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{experiment_name}.json")

    if os.path.exists(path) and not rerun:
        print(f"Loading cached results from {path}")
        with open(path, "r") as f:
            return json.load(f)

    print(f"Running experiment '{experiment_name}' ...")
    results = {}

    for case in cases:
        results[case] = experiment_fcn(case, trainset, device)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {path}")
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_performance_summary(worker_times, title="", xlabel="", ylabel=""):
    """
    Line-plots average epoch time (ms) vs. num_workers.

    Args:
        worker_times (dict): {num_workers: avg_epoch_time_seconds}
        title  (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
    """
    # JSON keys are strings — convert to int for correct numeric sorting
    workers = sorted(worker_times.keys(), key=int)
    times_ms = [worker_times[w] * 1000 for w in workers]
    workers = [int(w) for w in workers]

    plt.figure(figsize=(8, 5))
    plt.plot(workers, times_ms, marker="o", linestyle="-",
             color="steelblue", linewidth=2)
    plt.title(title,   fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(workers)
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
# IMPORTANT: On Windows, ALL executable code must be inside this block.
# Worker processes re-import this file, so anything outside this guard
# runs once per worker — causing re-entrant spawning and crashes.
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of CPU cores: {os.cpu_count()}")

    trainset, testset = download_and_load_cifar10()

    workers_to_test = [0, 2, 4, 6, 8, 10]
    # Define the list of batch_size values to test
    batch_sizes_to_test = [16, 32, 64, 128, 256, 512]

    # Define the list of prefetch_factor values to test
    prefetch_factors_to_test = [2, 4, 6, 8, 10, 12]

    # worker_times = run_experiment(
    #     experiment_name="worker_times",
    #     experiment_fcn=experiment_workers,
    #     cases=workers_to_test,
    #     trainset=trainset,
    #     device=device,
    #     rerun=False,  # <- set True to force a fresh run and overwrite cache
    # )

    # plot_performance_summary(
    #     worker_times,
    #     title="DataLoader Performance vs. num_workers",
    #     xlabel="Number of Workers",
    #     ylabel="Average Time per Epoch (ms)",
    # )

    # # Run the experiment to measure the data loading time for different batch sizes.
    # batch_size_times = run_experiment(
    #     # A unique name for this experiment, used as the filename for the cached results.
    #     experiment_name="batch_size_times",
    #     # The actual function that contains the experiment's logic.
    #     experiment_fcn=experiment_batch_sizes,
    #     # The parameters to iterate over; in this case, a list of different batch sizes.
    #     cases=batch_sizes_to_test,
    #     # The dataset required by the experiment function.
    #     trainset=trainset,
    #     # The computation device (e.g., 'cpu' or 'cuda') to be used.
    #     device=device,
    #     # If False, the function will load results from the cache if they exist.
    #     # If True, it will force the experiment to run again and overwrite any old results.
    #     rerun=True
    # )
    # plot_performance_summary(
    #     batch_size_times,
    #     title="DataLoader Performance vs. batch_size",
    #     xlabel="Batch Sizes",
    #     ylabel="Average Time per Epoch (milliseconds)",
    # )

    # Run the experiment to measure the data loading time for different batch sizes.
    prefetch_factor_times = run_experiment(
        # A unique name for this experiment, used as the filename for the cached results.
        experiment_name="prefetch_factor_times",
        # The actual function that contains the experiment's logic.
        experiment_fcn=experiment_prefetch_factor,
        # The parameters to iterate over; in this case, a list of different batch sizes.
        cases=prefetch_factors_to_test,
        # The dataset required by the experiment function.
        trainset=trainset,
        # The computation device (e.g., 'cpu' or 'cuda') to be used.
        device=device,
        # If False, the function will load results from the cache if they exist.
        # If True, it will force the experiment to run again and overwrite any old results.
        rerun=True
    )
    plot_performance_summary(
        prefetch_factor_times,
        title="DataLoader Performance vs. prefetch_factor",
        xlabel="Prefetch Factor",
        ylabel="Average Time per Epoch (milliseconds)",
    )
