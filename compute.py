"""
Hardware utilization strategy for the TDA domain mapper.

Component assignments:
  GPU  — model forward passes, activation extraction (TransformerLens)
  CPU  — TDA (gudhi), pattern analysis (scipy/numpy), PySR symbolic regression
         All parallelized across the 32 available cores
  RAM  — activation store buffer, point cloud staging

TransformerLens generate() is incompatible with RTX 50xx (Blackwell) CUDA
for token sampling, so generation runs on CPU. Cache extraction (run_with_cache)
works fine on CUDA and is used for all non-generation forward passes.
"""

import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

CPU_CORES = os.cpu_count()
TDA_WORKERS = max(1, CPU_CORES - 4)   # leave 4 cores for OS + model overhead
PYSR_WORKERS = max(1, CPU_CORES // 2) # PySR is already multi-threaded internally


def gpu_device() -> str:
    """Returns cuda if available, else cpu. Used for cache extraction only."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def move_for_cache(model, tokens: torch.Tensor):
    """
    Moves tokens to GPU for cache extraction, returns cache on CPU.
    Separates generation (CPU) from activation capture (GPU).
    """
    device = gpu_device()
    tokens = tokens.to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return cache


def parallel_tda(jobs: list[dict], worker_fn, max_workers: int = None) -> list:
    """
    Runs TDA jobs in parallel across CPU cores.
    jobs: list of kwargs dicts passed to worker_fn
    Returns results in same order as jobs.
    """
    max_workers = max_workers or TDA_WORKERS
    results = [None] * len(jobs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, **job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {"error": str(e)}

    return results


def parallel_pattern_analysis(
    activation_matrices: list[np.ndarray],
    worker_fn,
    max_workers: int = None,
) -> list:
    """
    Runs pattern analysis across activation matrices in parallel.
    Uses threads (not processes) since numpy releases the GIL for most ops.
    """
    max_workers = max_workers or TDA_WORKERS
    results = [None] * len(activation_matrices)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_fn, mat): i
            for i, mat in enumerate(activation_matrices)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {"error": str(e)}

    return results


def set_pysr_parallelism():
    """Configure PySR to use all available CPU cores."""
    os.environ["JULIA_NUM_THREADS"] = str(PYSR_WORKERS)


def report():
    print(f"CPU cores available: {CPU_CORES}")
    print(f"TDA workers: {TDA_WORKERS}")
    print(f"PySR workers (Julia threads): {PYSR_WORKERS}")
    print(f"GPU: {gpu_device()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name} — {round(p.total_memory/1e9,1)}GB VRAM, {p.multi_processor_count} SMs")
