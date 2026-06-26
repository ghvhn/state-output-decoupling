import numpy as np
from sklearn.metrics import pairwise_distances
import gudhi


def run(cloud: np.ndarray, maxdim: int = 2, metric: str = "cosine") -> list[np.ndarray]:
    """
    Runs Vietoris-Rips persistent homology on a point cloud via gudhi.
    Returns persistence diagrams per homology dimension: [H0, H1, H2].
    Infinite death values are replaced with finite max for downstream compatibility.
    """
    dist_matrix = pairwise_distances(cloud, metric=metric)
    rips = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=2.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=maxdim + 1)
    simplex_tree.compute_persistence()

    diagrams = []
    for dim in range(maxdim + 1):
        pairs = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            diagrams.append(np.empty((0, 2), dtype=np.float32))
        else:
            diagrams.append(np.array(pairs, dtype=np.float32))

    return _clean(diagrams)


def _clean(diagrams: list[np.ndarray]) -> list[np.ndarray]:
    cleaned = []
    for dgm in diagrams:
        if len(dgm) == 0:
            cleaned.append(dgm)
            continue
        finite_vals = dgm[np.isfinite(dgm)]
        cap = finite_vals.max() if len(finite_vals) > 0 else 1.0
        dgm = dgm.copy()
        dgm[~np.isfinite(dgm)] = cap
        cleaned.append(dgm)
    return cleaned


def persistent_features(diagrams: list[np.ndarray], threshold: float = 0.1) -> list[np.ndarray]:
    """
    Filters each diagram to features with lifetime >= threshold.
    These are the real structure — short-lived features are noise.
    """
    return [
        dgm[dgm[:, 1] - dgm[:, 0] >= threshold] if len(dgm) > 0 else dgm
        for dgm in diagrams
    ]
