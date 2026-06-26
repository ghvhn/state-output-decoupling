"""
Tiered activation storage in HDF5.

All data is namespaced under the model that produced it:
  {model_key}/conversations/{conv_id}/bands/{band}
  {model_key}/diagrams/{conv_id}/{band}/H{n}
  {model_key}/fingerprints/{conv_id}/{band}

This allows a single activations.h5 to hold data from multiple models
without collision. Fingerprints from different models are never mixed.

Tier 1: raw residual stream (float16, gzip compressed)
Tier 2: point clouds (PCA-compressed)
Tier 3: persistence diagrams
Tier 4: fingerprint vectors

Each tier is independently readable. Rerunning downstream analysis
never requires re-extracting activations from the model.
"""

import numpy as np
import h5py
from pathlib import Path


def _model_key(model_name: str) -> str:
    """Sanitize model name for use as an HDF5 group key."""
    return model_name.replace("/", "__").replace(".", "_").replace("-", "_")


def save_bands(path: str, model_name: str, conv_id: str, bands: dict[str, np.ndarray]):
    mk = _model_key(model_name)
    with h5py.File(path, "a") as f:
        for band_name, data in bands.items():
            key = f"{mk}/conversations/{conv_id}/bands/{band_name}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=data, compression="gzip", compression_opts=4)


def load_bands(path: str, model_name: str, conv_id: str) -> dict[str, np.ndarray]:
    mk = _model_key(model_name)
    with h5py.File(path, "r") as f:
        base = f[f"{mk}/conversations/{conv_id}/bands"]
        return {name: base[name][:] for name in base}


def save_diagrams(path: str, model_name: str, conv_id: str, band: str, diagrams: list[np.ndarray]):
    mk = _model_key(model_name)
    with h5py.File(path, "a") as f:
        for i, dgm in enumerate(diagrams):
            key = f"{mk}/diagrams/{conv_id}/{band}/H{i}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=dgm, compression="gzip")


def load_diagrams(path: str, model_name: str, conv_id: str, band: str) -> list[np.ndarray]:
    mk = _model_key(model_name)
    with h5py.File(path, "r") as f:
        base = f[f"{mk}/diagrams/{conv_id}/{band}"]
        return [base[f"H{i}"][:] for i in range(len(base))]


def save_fingerprint(path: str, model_name: str, conv_id: str, band: str, vector: np.ndarray):
    mk = _model_key(model_name)
    with h5py.File(path, "a") as f:
        key = f"{mk}/fingerprints/{conv_id}/{band}"
        if key in f:
            del f[key]
        f.create_dataset(key, data=vector)


def load_fingerprint(path: str, model_name: str, conv_id: str, band: str) -> np.ndarray:
    mk = _model_key(model_name)
    with h5py.File(path, "r") as f:
        return f[f"{mk}/fingerprints/{conv_id}/{band}"][:]


def list_conversations(path: str, model_name: str) -> list[str]:
    if not Path(path).exists():
        return []
    mk = _model_key(model_name)
    with h5py.File(path, "r") as f:
        group = f.get(f"{mk}/conversations")
        if group is None:
            return []
        return list(group.keys())


def list_models(path: str) -> list[str]:
    """Returns all model keys present in the store."""
    if not Path(path).exists():
        return []
    with h5py.File(path, "r") as f:
        return list(f.keys())
