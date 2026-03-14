# DataLoader Performance Benchmarking

A benchmarking toolkit for measuring and optimising PyTorch `DataLoader` performance. Runs controlled experiments across different configurations — number of workers, batch size, pinned memory, and prefetch factor — and caches results so experiments don't have to be re-run from scratch every time.

---

## Why this exists

A GPU left waiting for data is wasted money and time. The bottleneck in many training pipelines isn't the model or the hardware — it's how data is loaded and delivered. This project makes the bottleneck measurable so it can be fixed.

Real results from this benchmark on a 32-core CPU + CUDA GPU machine (CIFAR-10):

| num_workers | Avg epoch time |
|-------------|---------------|
| 0           | 12.62s        |
| 2           | 3.50s         |
| 4           | 1.84s         |
| 6           | 1.39s         |
| 8           | 1.24s         |
| 10          | 1.24s *(plateau)* |

That's a **10× speedup** with zero changes to the model.

---

## Project structure

```
├── dataloader_benchmark.py     # Main benchmarking script
├── checkpoint_experiments/     # Cached experiment results (JSON)
│   └── worker_times.json
├── data/                       # CIFAR-10 dataset (auto-downloaded)
└── README.md
```

---

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- torchvision
- tqdm
- matplotlib
- numpy

Install dependencies:

```bash
pip install torch torchvision tqdm matplotlib numpy
```

---

## Usage

Run the benchmark:

```bash
python dataloader_benchmark.py
```

On first run the script downloads CIFAR-10 (~170 MB) into `./data/` and runs all experiments, saving results to `./checkpoint_experiments/`. Subsequent runs load from cache automatically.

To force a fresh run and overwrite cached results, set `rerun=True` in the `run_experiment` call at the bottom of the script:

```python
worker_times = run_experiment(
    experiment_name="worker_times",
    experiment_fcn=experiment_workers,
    cases=workers_to_test,
    trainset=trainset,
    device=device,
    rerun=True,   # <-- change this
)
```

---

## How it works

### Experiment runner

`run_experiment()` is a generic wrapper that:
1. Checks for a cached result in `checkpoint_experiments/<name>.json`
2. If no cache exists (or `rerun=True`), iterates over all test cases and calls the experiment function for each
3. Saves results to JSON for future runs

### Timing

`measure_average_epoch_time()` runs `epochs=5` full passes through the DataLoader. The first `warmup=2` epochs are discarded — they include OS-level and CUDA warm-up noise. The final 3 epochs are averaged to produce a stable measurement.

### Worker experiment

`experiment_workers()` creates a fresh DataLoader for each `num_workers` value and times it. Key settings used:

```python
DataLoader(
    trainset,
    batch_size=256,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
)
```

---

## Key parameters explained

| Parameter | What it does |
|-----------|-------------|
| `num_workers` | Number of parallel CPU processes preloading batches. More workers = less GPU idle time, up to the point of diminishing returns. |
| `batch_size` | Samples processed per forward pass. Larger batches reduce total passes per epoch but are bounded by GPU memory. |
| `pin_memory` | Locks CPU memory for direct GPU access, skipping an intermediate copy. Most beneficial with multiple workers and large datasets. |
| `persistent_workers` | Keeps worker processes alive between epochs. Without this, workers are killed and respawned at the start of each epoch. |
| `prefetch_factor` | Batches each worker preloads ahead of time. Default of 2 is sufficient for most pipelines — increasing it rarely helps once the GPU is fully fed. |

---

## Important: Windows compatibility

On Windows, Python uses **spawn** (not fork) to create worker processes, which means each worker re-imports the script from scratch. All executable code must be inside the `if __name__ == "__main__":` guard at the bottom of the file — otherwise worker spawning triggers an infinite re-import loop and crashes.

```python
if __name__ == "__main__":
    trainset, testset = download_and_load_cifar10()
    worker_times = run_experiment(...)
    plot_performance_summary(...)
```

Function and class definitions are safe at module level. Only the code that *runs the experiment* needs to be guarded.

---

## Extending the benchmark

To add a new experiment (e.g. testing different batch sizes), define a new experiment function and pass it to `run_experiment`:

```python
def experiment_batch_size(batch_size, trainset, device):
    loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,         # fix workers at the optimal value
        pin_memory=True,
        persistent_workers=True,
    )
    try:
        return measure_average_epoch_time(loader, device)
    except RuntimeError:
        return float("inf")
    finally:
        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


batch_times = run_experiment(
    experiment_name="batch_size_times",
    experiment_fcn=experiment_batch_size,
    cases=[32, 64, 128, 256, 512, 1024],
    trainset=trainset,
    device=device,
    rerun=False,
)
```

---

## Results interpretation

- **Times decrease as workers increase** → the pipeline was CPU-bound; workers are helping
- **Times plateau** → the GPU is now fully fed; adding more workers adds overhead with no gain. This is the optimal `num_workers` value for your machine
- **Times increase with more workers** → batch size is too small; worker IPC overhead dominates. Increase `batch_size`
- **First epoch significantly slower than the rest** → expected; this is warm-up noise and is excluded from averages

---

## License

MIT
