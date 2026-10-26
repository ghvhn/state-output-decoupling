"""A bounded, candidate affine conversion learned from paired raw states.

For centered source rows X and centered change rows Y, the fitted transport is
``delta(x) = mean_delta + (x - mean_x) @ X.T @ coefficients``.  The ridge solve
uses the N by N Gram matrix, so storage and fitting stay low rank in the usually
much wider residual space.  Inputs and outputs retain their original magnitude;
this module does not normalize them or claim semantic accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import torch


RELATIONAL_CONVERSION_VERSION = "relational-conversion-centered-ridge-candidate-v1"
_SCHEMA = 1
_TENSORS = ("center_source", "center_change", "source_relations", "coefficients")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        boundary = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {boundary}")
    return result


def _integer(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _input_tensor(value: Any, label: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{label} must be a floating tensor")
    if value.layout != torch.strided or value.ndim != ndim:
        raise ValueError(f"{label} must be a dense {ndim}-dimensional tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _finite(value: torch.Tensor, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} exceeded the finite numerical range")
    return value


def _stored(value: torch.Tensor, label: str) -> torch.Tensor:
    result = value.to(dtype=torch.float32).contiguous().clone()
    return _finite(result, f"{label} in float32 storage")


def _radius(relations: torch.Tensor) -> float:
    return float(_finite(torch.linalg.vector_norm(relations, dim=1), "source radius").max())


def _pair_rms(rows: torch.Tensor) -> float:
    """RMS Euclidean difference over all unordered pairs, without materializing them."""
    # First remove a row exactly: a large common origin must not create tiny
    # apparent variation solely through mean/sum roundoff.
    differences = rows - rows[0]
    centered = differences - differences.mean(dim=0)
    # sum_{i<j} ||a_i-a_j||^2 = N * sum_i ||a_i-mean(a)||^2.
    mean_pair_square = 2.0 * _finite(centered.square().sum(), "pair variation") / (rows.shape[0] - 1)
    return float(torch.sqrt(mean_pair_square))


def _rank(relations: torch.Tensor) -> int:
    singular = _finite(torch.linalg.svdvals(relations), "source singular values")
    if singular.numel() == 0 or float(singular.max()) == 0:
        return 0
    # Profiles are float32: do not count float32 centering roundoff as a dimension.
    tolerance = max(relations.shape) * torch.finfo(torch.float32).eps * float(singular.max())
    return int((singular > tolerance).sum())


def fit_relational_conversion(
    sources: torch.Tensor,
    changes: torch.Tensor,
    *,
    ridge: float = 0.01,
    extrapolation: float = 1.5,
) -> dict[str, Any]:
    """Fit raw source-to-change pairs, with float64 algebra and float32 storage.

    At least three paired rows are required.  Ridge is scaled by the mean
    squared centered source norm.  Repeated identical sources cannot identify
    different changes and are rejected.  Identical changes remain an explicitly
    constant model, including all-zero changes.
    """
    x = _input_tensor(sources, "sources", 2)
    y = _input_tensor(changes, "changes", 2)
    if x.shape != y.shape:
        raise ValueError("sources and changes must have the same [N, D] shape")
    n, width = x.shape
    if n < 3 or width < 1:
        raise ValueError("paired training data require N >= 3 and D >= 1")
    ridge_value = _number(ridge, "ridge", positive=True)
    extrapolation_value = _number(extrapolation, "extrapolation", positive=True)
    source_constant = bool(torch.equal(x, x[:1].expand_as(x)))
    changes_constant = bool(torch.equal(y, y[:1].expand_as(y)))
    if source_constant and not changes_constant:
        raise ValueError("identical sources with different changes are unidentifiable")

    center_x = x[0].clone() if source_constant else _finite(x.mean(dim=0), "source mean")
    center_y = y[0].clone() if changes_constant else _finite(y.mean(dim=0), "change mean")
    xc = torch.zeros_like(x) if source_constant else _finite(x - center_x, "centered sources")
    yc = torch.zeros_like(y) if changes_constant else _finite(y - center_y, "centered changes")
    gram = _finite(xc @ xc.T, "source Gram matrix")
    source_energy = float(gram.diagonal().mean())
    scaled_ridge = ridge_value * source_energy
    if not math.isfinite(scaled_ridge):
        raise ValueError("scaled ridge exceeded the finite numerical range")
    if source_constant or changes_constant:
        coefficients = torch.zeros_like(y)
    else:
        if source_energy <= 0 or scaled_ridge <= 0:
            raise ValueError("varying sources are too small for a stable ridge solve")
        try:
            coefficients = _finite(
                torch.linalg.solve(gram + scaled_ridge * torch.eye(n, dtype=torch.float64), yc),
                "conversion coefficients",
            )
        except RuntimeError as exc:
            raise ValueError("relational ridge solve failed") from exc

    stored_x = _stored(center_x, "source mean")
    stored_y = _stored(center_y, "change mean")
    stored_relations = _stored(xc, "source relations")
    stored_coefficients = _stored(coefficients, "conversion coefficients")
    relations64 = stored_relations.to(torch.float64)
    coefficients64 = stored_coefficients.to(torch.float64)
    transport = _finite(relations64 @ relations64.T @ coefficients64, "stored training transport")
    fitted = _stored(stored_y.to(torch.float64) + transport, "stored training predictions").to(torch.float64)
    observed_pairs = 0.0 if changes_constant else _pair_rms(y)
    predicted_pairs = 0.0 if changes_constant else _pair_rms(fitted)
    source_rank = _rank(relations64)
    if not source_constant and source_rank == 0:
        raise ValueError("source variation was lost in float32 storage")
    changes_norm_mean = float(_finite(torch.linalg.vector_norm(y, dim=1), "change norms").mean())
    residual_rms = float(torch.sqrt(_finite((fitted - y).square().mean(), "fit residual")))
    profile: dict[str, Any] = {
        "schema": _SCHEMA,
        "version": RELATIONAL_CONVERSION_VERSION,
        "width": width,
        "n_samples": n,
        "center_source": stored_x,
        "center_change": stored_y,
        "source_relations": stored_relations,
        "coefficients": stored_coefficients,
        "training_radius": _radius(relations64),
        "changes_norm_mean": changes_norm_mean,
        "ridge": ridge_value,
        "scaled_ridge": scaled_ridge,
        "extrapolation": extrapolation_value,
        "diagnostics": {
            "adaptive": not changes_constant and predicted_pairs > 0 and source_rank > 0,
            "source_constant": source_constant,
            "changes_constant": changes_constant,
            "observed_pair_change_rms": observed_pairs,
            "predicted_pair_change_rms": predicted_pairs,
            "fit_residual_rms": residual_rms,
            "source_rank": source_rank,
            "diagnostic_only": True,
        },
    }
    return validate_relational_conversion(profile)


def validate_relational_conversion(
    profile: Mapping[str, Any], expected_width: int | None = None
) -> dict[str, Any]:
    """Validate an already safely deserialized profile and return independent data.

    This function never loads pickle files.  A persistence caller should use
    ``torch.load(..., weights_only=True, map_location='cpu')`` first.  Stored
    tensors must already be finite CPU float32 with exact coverage and shape.
    """
    if not isinstance(profile, Mapping):
        raise ValueError("relational profile must be a mapping")
    if type(profile.get("schema")) is not int or profile["schema"] != _SCHEMA:
        raise ValueError("unsupported relational profile schema")
    if profile.get("version") != RELATIONAL_CONVERSION_VERSION:
        raise ValueError("unsupported relational profile version")
    width = _integer(profile.get("width"), "profile width", 1)
    n = _integer(profile.get("n_samples"), "profile n_samples", 3)
    if expected_width is not None and width != _integer(expected_width, "expected width", 1):
        raise ValueError("relational profile width does not match expected width")
    result: dict[str, Any] = {
        "schema": _SCHEMA, "version": RELATIONAL_CONVERSION_VERSION,
        "width": width, "n_samples": n,
    }
    for name in _TENSORS:
        value = profile.get(name)
        shape = (width,) if name.startswith("center_") else (n, width)
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
        ):
            raise ValueError(f"profile {name} must be a CPU float32 tensor with shape {shape}")
        result[name] = _finite(value.detach().contiguous().clone(), f"profile {name}")
    for name in ("training_radius", "changes_norm_mean", "scaled_ridge"):
        result[name] = _number(profile.get(name), f"profile {name}")
    for name in ("ridge", "extrapolation"):
        result[name] = _number(profile.get(name), f"profile {name}", positive=True)
    relations = result["source_relations"].to(torch.float64)
    measured_radius = _radius(relations)
    if not math.isclose(result["training_radius"], measured_radius, rel_tol=1e-6, abs_tol=0.0):
        raise ValueError("profile training_radius does not match stored source relations")
    source_constant = measured_radius == 0
    center_tolerance = float(relations.abs().max()) * 8 * torch.finfo(torch.float32).eps
    if bool((relations.mean(dim=0).abs() > center_tolerance).any()):
        raise ValueError("profile source relations must be centered")
    expected_ridge = result["ridge"] * float(relations.square().sum(dim=1).mean())
    if not math.isclose(result["scaled_ridge"], expected_ridge, rel_tol=1e-5, abs_tol=0.0):
        raise ValueError("profile scaled_ridge does not match stored source energy")
    if not math.isfinite(result["training_radius"] * result["extrapolation"]):
        raise ValueError("profile extrapolation radius exceeded the finite numerical range")

    diagnostic = profile.get("diagnostics")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("profile diagnostics must be a mapping")
    clean_diagnostic: dict[str, Any] = {}
    for name in ("adaptive", "source_constant", "changes_constant", "diagnostic_only"):
        if type(diagnostic.get(name)) is not bool:
            raise ValueError(f"profile diagnostic {name} must be boolean")
        clean_diagnostic[name] = diagnostic[name]
    if not clean_diagnostic["diagnostic_only"]:
        raise ValueError("relational conversion must remain a candidate geometric diagnostic")
    for name in ("observed_pair_change_rms", "predicted_pair_change_rms", "fit_residual_rms"):
        clean_diagnostic[name] = _number(diagnostic.get(name), f"profile diagnostic {name}")
    source_rank = _integer(diagnostic.get("source_rank"), "profile source_rank", 0)
    if source_rank != _rank(relations) or source_rank > min(n - 1, width):
        raise ValueError("profile source_rank does not match stored source relations")
    clean_diagnostic["source_rank"] = source_rank
    if clean_diagnostic["source_constant"] != source_constant:
        raise ValueError("profile source_constant does not match stored source relations")
    coefficients = result["coefficients"].to(torch.float64)
    transport = _finite(relations @ relations.T @ coefficients, "profile transport")
    fitted = _stored(result["center_change"].to(torch.float64) + transport, "profile training predictions")
    predicted_pairs = _pair_rms(fitted.to(torch.float64))
    if not math.isclose(clean_diagnostic["predicted_pair_change_rms"], predicted_pairs, rel_tol=1e-6, abs_tol=0.0):
        raise ValueError("profile predicted variation does not match stored transport")
    if clean_diagnostic["changes_constant"]:
        if clean_diagnostic["observed_pair_change_rms"] != 0 or bool(torch.count_nonzero(coefficients)):
            raise ValueError("constant changes must have zero observed variation and transport")
    elif clean_diagnostic["observed_pair_change_rms"] == 0 or source_constant:
        raise ValueError("varying changes require observed variation and varying sources")
    expected_adaptive = not clean_diagnostic["changes_constant"] and predicted_pairs > 0 and source_rank > 0
    if clean_diagnostic["adaptive"] != expected_adaptive:
        raise ValueError("profile adaptive flag does not match observed and fitted variation")
    baseline_norm = float(torch.linalg.vector_norm(result["center_change"].to(torch.float64)))
    if baseline_norm > result["changes_norm_mean"] * (1 + 4 * torch.finfo(torch.float32).eps):
        raise ValueError("profile changes_norm_mean cannot be smaller than its mean change norm")
    if result["changes_norm_mean"] == 0 and (
        not clean_diagnostic["changes_constant"] or baseline_norm != 0 or bool(torch.count_nonzero(coefficients))
    ):
        raise ValueError("zero changes_norm_mean requires an all-zero constant conversion")
    result["diagnostics"] = clean_diagnostic
    return result


def predict_relational_conversion(
    profile: Mapping[str, Any], source: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Predict a raw change, radially bounding the source relation before transport.

    ``ood_ratio`` is distance / training radius.  It is ``None`` when the ratio
    is undefined or beyond finite range (including a new point with radius 0).
    The separate distance, outside-support and clamp fields remain explicit.
    """
    p = validate_relational_conversion(profile)
    x = _input_tensor(source, "source", 1)
    if tuple(x.shape) != (p["width"],):
        raise ValueError(f"source must have shape ({p['width']},)")
    relation = _finite(x - p["center_source"].to(torch.float64), "source relation")
    distance = float(_finite(torch.linalg.vector_norm(relation), "query source distance"))
    radius = p["training_radius"]
    limit = radius * p["extrapolation"]
    clamped = distance > limit
    source_scale = limit / distance if clamped else 1.0
    relation = relation * source_scale
    transport = _finite(
        relation @ p["source_relations"].to(torch.float64).T @ p["coefficients"].to(torch.float64),
        "predicted conversion transport",
    )
    delta = _stored(p["center_change"].to(torch.float64) + transport, "predicted conversion")
    ratio = distance / radius if radius > 0 else (0.0 if distance == 0 else None)
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    diagnostic = {
        "source_distance": distance,
        "training_radius": radius,
        "max_source_distance": limit,
        "ood_ratio": ratio,
        "outside_training_support": distance > radius,
        "clamped": clamped,
        "source_scale": source_scale,
        "adaptive": p["diagnostics"]["adaptive"],
        "predicted_delta_norm": float(torch.linalg.vector_norm(delta.to(torch.float64))),
        "transport_change_norm": float(torch.linalg.vector_norm(transport)),
        "changes_norm_mean": p["changes_norm_mean"],
        "diagnostic_only": True,
    }
    return delta, diagnostic
