"""
Discovery mode: let domains emerge from the geometry.
Clusters fingerprint vectors across all conversations without
any predefined domain labels.
"""

import numpy as np
from sklearn.cluster import HDBSCAN
from tda.fingerprint import distance


def cluster(
    fingerprints: dict[str, dict[str, np.ndarray]],
    band: str,
    min_cluster_size: int = 3,
) -> dict[str, int]:
    """
    fingerprints: {conv_id: {band: vector}}
    Returns {conv_id: cluster_label} where -1 = noise/outlier.
    """
    conv_ids = [cid for cid in fingerprints if band in fingerprints[cid]]
    if not conv_ids:
        return {}

    matrix = np.stack([fingerprints[cid][band] for cid in conv_ids])
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(matrix)

    return {cid: int(label) for cid, label in zip(conv_ids, labels)}


def domain_signatures(
    fingerprints: dict[str, dict[str, np.ndarray]],
    cluster_labels: dict[str, int],
    band: str,
) -> dict[str, np.ndarray]:
    """
    Averages fingerprints within each cluster to produce one
    representative signature per emergent domain.
    Returns {domain_id: average_fingerprint}.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for conv_id, label in cluster_labels.items():
        if label == -1:
            continue
        if band in fingerprints[conv_id]:
            groups[label].append(fingerprints[conv_id][band])

    return {
        f"domain_{label}": np.mean(np.stack(vecs), axis=0)
        for label, vecs in groups.items()
    }
