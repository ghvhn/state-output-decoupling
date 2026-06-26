import json
import numpy as np
from pathlib import Path


def _model_key(model_name: str) -> str:
    return model_name.replace("/", "__").replace(".", "_").replace("-", "_")


def load(path: str) -> dict:
    """Returns {model_key: {domain: {band: vector}}}"""
    if not Path(path).exists():
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    for model_key in raw:
        for domain in raw[model_key]:
            for band in raw[model_key][domain]:
                raw[model_key][domain][band] = np.array(
                    raw[model_key][domain][band], dtype=np.float32
                )
    return raw


def save(path: str, signatures: dict):
    serializable = {}
    for model_key, domains in signatures.items():
        serializable[model_key] = {}
        for domain, bands in domains.items():
            serializable[model_key][domain] = {
                band: vec.tolist() for band, vec in bands.items()
            }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def upsert(path: str, model_name: str, domain: str, band: str, vector: np.ndarray):
    sigs = load(path)
    mk = _model_key(model_name)
    if mk not in sigs:
        sigs[mk] = {}
    if domain not in sigs[mk]:
        sigs[mk][domain] = {}
    sigs[mk][domain][band] = vector
    save(path, sigs)


def load_for_model(path: str, model_name: str) -> dict:
    """Returns {domain: {band: vector}} for one model."""
    sigs = load(path)
    return sigs.get(_model_key(model_name), {})
