import numpy as np
import torch
from transformer_lens import HookedTransformer


def extract_bands(
    model: HookedTransformer,
    tokens: torch.Tensor,
    bands: dict[str, list[int]],
    quantize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Returns residual stream activations for each layer band.
    Tokens are moved to GPU for the forward pass; cache is returned to CPU.
    Shape per band: [n_layers_in_band, n_tokens, d_model]
    """
    cache_device = next(model.parameters()).device
    tokens = tokens.to(cache_device)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    dtype = np.float16 if quantize else np.float32
    result = {}

    for band_name, layer_range in bands.items():
        layers = range(layer_range[0], layer_range[1] + 1)
        stacked = torch.stack(
            [cache["resid_post", l].squeeze(0) for l in layers], dim=0
        )
        result[band_name] = stacked.cpu().to(torch.float32).numpy().astype(dtype)

    return result


def extract_full(
    model: HookedTransformer,
    tokens: torch.Tensor,
    quantize: bool = True,
) -> np.ndarray:
    """
    Returns the full residual stream: [n_layers, n_tokens, d_model].
    Stored as float16 by default. Exact values recoverable — language is
    already lossy at the tokenizer; sub-float16 precision carries no signal.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers
    dtype = np.float16 if quantize else np.float32

    stacked = torch.stack(
        [cache["resid_post", l].squeeze(0) for l in range(n_layers)], dim=0
    )
    return stacked.cpu().to(torch.float32).numpy().astype(dtype)
