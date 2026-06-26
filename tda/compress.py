import numpy as np
from sklearn.decomposition import PCA


def reduce(cloud: np.ndarray, n_components: int = 50) -> tuple[np.ndarray, PCA]:
    """
    PCA is used only to make distance computation tractable.
    It does not find structure — Vietoris-Rips does.
    Returns compressed cloud and fitted PCA (for later projection of new data).
    """
    n_components = min(n_components, cloud.shape[0] - 1, cloud.shape[1])
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(cloud)
    return compressed, pca
