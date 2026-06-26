"""
Converts persistence diagrams into fixed-size vectors for storage,
averaging, and comparison.

Persistence images are used: a 2D density map of the diagram in
(birth, lifetime) space. They average cleanly across multiple conversations
and support fast distance computation without rerunning TDA.
"""

import numpy as np


def to_image(diagram: np.ndarray, pixels: int = 20, spread: float = 0.1) -> np.ndarray:
    """
    Converts a persistence diagram to a fixed-size persistence image.
    Replaces persim.PersImage for Python 3.14 compatibility.
    Maps (birth, lifetime) pairs onto a 2D density grid.
    """
    if len(diagram) == 0:
        return np.zeros(pixels * pixels, dtype=np.float32)

    diagram = np.asarray(diagram, dtype=np.float64)
    diagram = diagram[np.isfinite(diagram).all(axis=1)]   # drop non-finite (birth,death)
    if len(diagram) == 0:
        return np.zeros(pixels * pixels, dtype=np.float32)

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    lifetimes = deaths - births

    valid = lifetimes > 0
    births, lifetimes = births[valid], lifetimes[valid]
    if len(births) == 0:
        return np.zeros(pixels * pixels, dtype=np.float32)

    b_min, b_max = births.min(), births.max() + 1e-8
    l_min, l_max = 0.0, lifetimes.max() + 1e-8

    grid = np.zeros((pixels, pixels), dtype=np.float32)
    for b, l in zip(births, lifetimes):
        bi = int(np.clip((b - b_min) / (b_max - b_min) * (pixels - 1), 0, pixels - 1))
        li = int(np.clip((l - l_min) / (l_max - l_min) * (pixels - 1), 0, pixels - 1))
        grid[li, bi] += l  # weight by lifetime — persistent features matter more

    # Gaussian blur approximation
    from scipy.ndimage import gaussian_filter
    grid = gaussian_filter(grid, sigma=spread * pixels)
    return grid.flatten().astype(np.float32)


def from_diagrams(diagrams: list[np.ndarray]) -> np.ndarray:
    """
    Converts a list of persistence diagrams (one per homology dim)
    into a single concatenated fingerprint vector.
    """
    return np.concatenate([to_image(dgm) for dgm in diagrams])


def average(fingerprints: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(fingerprints), axis=0)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
