import numpy as np


def build(band_activation: np.ndarray) -> np.ndarray:
    """
    band_activation: [n_layers_in_band, n_tokens, d_model]
    Returns point cloud: [n_layers * n_tokens, d_model]
    Each point is one (layer, token) observation in activation space.
    """
    n_layers, n_tokens, d_model = band_activation.shape
    return band_activation.reshape(n_layers * n_tokens, d_model).astype(np.float32)
