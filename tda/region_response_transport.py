"""Calibration-only regional transports of nonlinear response residuals.

Regions are geometric neighborhoods, not certified semantic categories. A
shared small coordinate map changes how an expert's global residual basis is
used in each region. An excluded expert can test reuse of that map: its global
basis and offsets are still calibrated, but its responses cannot fit the shared
operators. This is a CPU prototype, not a complete compressed-model loader.
"""
from __future__ import annotations

import math

import torch

from .trajectory_compress import _snapshot, _fingerprint, InputStateScope


FORMAT = "regional-nonlinear-response-residual-v1"
VARIANTS = ("global", "region_offsets", "shared_transport", "identity_transport",
            "wrong_region_transport", "independent_local")


def _finite(value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
            value.dtype != torch.float32 or tuple(value.shape) != tuple(shape) or
            not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be finite exact-shape FP32 CPU data")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _solve(x, target, ridge):
    gram = x.double().T @ x.double()
    scale = max(float(gram.trace()) / len(gram), 1e-12)
    return torch.linalg.solve(gram + ridge * scale * torch.eye(len(gram), dtype=torch.float64),
                              x.double().T @ target.double()).float()


def _cluster(z, regions, seed):
    generator = torch.Generator().manual_seed(seed)
    first = int(torch.randint(len(z), (1,), generator=generator))
    centers = [z[first].clone()]
    for _ in range(1, regions):
        distance = torch.cdist(z, torch.stack(centers)).square().amin(dim=1)
        centers.append(z[int(distance.argmax())].clone())
    centers = torch.stack(centers)
    previous = None
    for _ in range(32):
        labels = torch.cdist(z, centers).argmin(dim=1)
        if previous is not None and torch.equal(previous, labels):
            break
        if any(not bool((labels == k).any()) for k in range(regions)):
            raise ValueError("calibration contains an empty geometric region")
        centers = torch.stack([z[labels == k].mean(dim=0) for k in range(regions)])
        previous = labels
    # Recompute centers from the final membership, which is also used in fitting.
    labels = torch.cdist(z, centers).argmin(dim=1)
    if any(not bool((labels == k).any()) for k in range(regions)):
        raise ValueError("calibration contains an empty geometric region")
    centers = torch.stack([z[labels == k].mean(dim=0) for k in range(regions)])
    if not torch.equal(torch.cdist(z, centers).argmin(dim=1), labels):
        raise ValueError("calibration region membership did not converge within32 updates")
    return centers, labels


@torch.no_grad()
def fit_regions(records, residuals, *, expected_scope, heldout_records=(),
                rank=16, regions=4, ridge=0.01, seed=0, operator_excluded_expert=None,
                response_context_sha256=None, max_working_bytes=3 << 30):
    """Fit original-output minus compressed-output, using calibration only.

    The outputs are [expert, record, output-coordinate]. Scope binds them to the
    caller's exact base codec; a GLM runner must additionally verify all three
    projection identities and architecture before producing these residuals.
    """
    for value, name in ((rank, "rank"), (regions, "regions"),
                        (max_working_bytes, "working-byte budget")):
        _positive(value, name)
    if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 or
            isinstance(ridge, bool) or not isinstance(ridge, (float, int)) or
            not math.isfinite(ridge) or ridge <= 0):
        raise ValueError("seed and ridge must be finite bounded learner settings")
    if not isinstance(expected_scope, InputStateScope):
        raise ValueError("an exact input state scope is required")
    n, width = len(records), expected_scope.matrix_input_width
    if n > 2048 or rank > 64 or not 1 <= rank < min(n, width) or not 2 <= regions <= 8:
        raise ValueError("rank, region or calibration count exceeds its fixed bound")
    if not isinstance(residuals, torch.Tensor) or residuals.ndim != 3:
        raise ValueError("residuals require [expert, record, output] dimensions")
    experts, count, output = residuals.shape
    if not 2 <= experts <= 8 or count != n or not 1 <= output <= 8192:
        raise ValueError("residual dimensions exceed the bounded response experiment")
    _finite(residuals, (experts, n, output), "calibration residuals")
    if (isinstance(operator_excluded_expert, bool) or
            not isinstance(operator_excluded_expert, int) or
            not 0 <= operator_excluded_expert < experts):
        raise ValueError("one declared expert must be excluded from shared-operator fitting")
    estimate = (residuals.numel() * 8 + n * width * 16 + n * n * 24 +
                experts * regions * rank * output * 16)
    if estimate > max_working_bytes:
        raise MemoryError("regional fit numerical tensor estimate exceeds its budget")
    frozen, _ = _snapshot(records, (), expected_scope, max_records=2048, max_pairs=0)
    heldout, _ = _snapshot(heldout_records, (), expected_scope, max_records=8192, max_pairs=0)
    if ({r.record_id for r in frozen} & {r.record_id for r in heldout} or
            {r.trajectory_id for r in frozen} & {r.trajectory_id for r in heldout}):
        raise ValueError("heldout records and trajectories cannot enter regional fitting")
    if response_context_sha256 is not None and (
            not isinstance(response_context_sha256, str) or len(response_context_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in response_context_sha256)):
        raise ValueError("response context must be a SHA256 binding of source functions and base codec")
    if any(r.origin == "measured_model_state" for r in frozen) and response_context_sha256 is None:
        raise ValueError("measured response residuals require all-projection/base-codec context")
    point_weights = torch.tensor([r.provenance_weight * r.stimulation_weight for r in frozen])
    # This first diagnostic has a declared uniform-weight capture. Refuse silent
    # substitution if later evidence introduces nonuniform attention strengths.
    if not bool((point_weights == 1).all()):
        raise ValueError("regional prototype requires explicitly uniform provenance/stimulation")
    x = torch.stack([r.state.float() for r in frozen])
    mean = x.mean(dim=0)
    xc = x - mean
    eigenvalues, eigenvectors = torch.linalg.eigh(xc.double() @ xc.double().T)
    values = eigenvalues[-rank:].flip(0)
    if float(values[-1]) <= max(float(values[0]) * 1e-10, 1e-12):
        raise ValueError("calibration does not support the requested input rank")
    basis = (xc.double().T @ eigenvectors[:, -rank:].flip(1) /
             values.sqrt().unsqueeze(0)).float()
    # Fix arbitrary signs to make saved coordinate identities reproducible.
    pivots = basis.abs().argmax(dim=0)
    basis *= torch.sign(basis[pivots, torch.arange(rank)]).unsqueeze(0)
    scale = ((xc @ basis).square().mean(dim=0)).sqrt().clamp(min=1e-8)
    z = xc @ basis / scale
    centers, labels = _cluster(z, regions, seed)
    counts = [int((labels == k).sum()) for k in range(regions)]
    enabled = torch.tensor([float(count >= rank + 2) for count in counts])
    if not bool(enabled.any()):
        raise ValueError("no region has rank+2 calibration records")
    radius = torch.stack([(z[labels == k] - centers[k]).norm(dim=1).amax()
                          for k in range(regions)]).clamp(min=1e-8)
    orthogonal_radius = (xc - (xc @ basis) @ basis.T).norm(dim=1).amax().clamp(min=1e-8)
    response_mean = residuals.mean(dim=1)
    slopes = torch.stack([_solve(z, residuals[e] - response_mean[e], ridge)
                          for e in range(experts)])
    offsets = torch.empty(experts, regions, output)
    local_delta = torch.empty(experts, regions, rank, output)
    operators = torch.empty(regions, rank, rank)
    included = [e for e in range(experts) if e != operator_excluded_expert]
    for k in range(regions):
        if not bool(enabled[k]):
            offsets[:, k] = 0
            local_delta[:, k] = 0
            operators[k] = torch.eye(rank)
            continue
        mask = labels == k
        zk = z[mask] - centers[k]
        region_means = residuals[:, mask].mean(dim=1)
        offsets[:, k] = region_means - response_mean - torch.einsum("r,erd->ed", centers[k], slopes)
        for e in range(experts):
            local_delta[e, k] = _solve(zk, residuals[e, mask] - region_means[e], ridge) - slopes[e]
        # First fit local slopes from included experts, then project their
        # joint changes onto the shared global response bases. Excluded expert
        # residuals do not contribute to either matrix in this shared solve.
        output_gram = torch.zeros(rank, rank, dtype=torch.float64)
        cross = torch.zeros(rank, rank, dtype=torch.float64)
        for e in included:
            b = slopes[e].double()
            local = (local_delta[e, k] + slopes[e]).double()
            output_gram += b @ b.T
            cross += local @ b.T
        penalty = ridge * max(float(output_gram.trace()) / rank, 1e-12)
        operators[k] = torch.linalg.solve(
            output_gram + penalty * torch.eye(rank, dtype=torch.float64),
            (cross + penalty * torch.eye(rank, dtype=torch.float64)).T).T.float()
    metadata = {"format": FORMAT, "scope": expected_scope.to_dict(),
                "response_context_sha256": response_context_sha256,
                "calibration_evidence_sha256": _fingerprint(expected_scope, frozen, ()),
                "calibration_record_ids": [r.record_id for r in frozen],
                "calibration_trajectory_ids": sorted({r.trajectory_id for r in frozen}),
                "rank": rank, "regions": regions, "ridge": float(ridge), "seed": seed,
                "experts": experts, "output_width": output, "region_counts": counts,
                "region_enabled": [bool(value) for value in enabled],
                "minimum_region_support": rank + 2,
                "operator_excluded_expert": operator_excluded_expert,
                "estimated_fit_numerical_bytes": estimate,
                "region_semantics_validated": False, "heldout_used_for_fit": False,
                "unseen_region_transfer_evaluated": False,
                "attention_weights": "uniform provenance and stimulation; all predictions retained",
                "support_policy": "1.5 times max calibration chart and orthogonal radii; global fallback"}
    tensors = {"input_mean": mean, "input_basis": basis, "feature_scale": scale,
               "response_mean": response_mean, "slopes": slopes, "centers": centers,
               "radii": radius, "orthogonal_radius": orthogonal_radius, "region_enabled": enabled,
               "offsets": offsets, "operators": operators, "local_delta": local_delta}
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"regional fit produced nonfinite {name}")
    return {"metadata": metadata, "tensors": tensors}


def variant_bank(bank, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown regional response variant")
    common = ("input_mean", "input_basis", "feature_scale", "response_mean", "slopes")
    regional = ("centers", "radii", "orthogonal_radius", "offsets", "region_enabled")
    fields = common + (() if variant == "global" else regional)
    if variant == "independent_local":
        fields += ("local_delta",)
    elif variant in ("shared_transport", "identity_transport", "wrong_region_transport"):
        fields += ("operators",)
    tensors = {name: bank["tensors"][name].contiguous().clone() for name in fields}
    if variant == "identity_transport":
        k, r = tensors["operators"].shape[:2]
        tensors["operators"] = torch.eye(r).expand(k, r, r).clone()
    elif variant == "wrong_region_transport":
        tensors["operators"] = tensors["operators"].roll(1, dims=0).clone()
    metadata = {**bank["metadata"], "variant": variant,
                "tensor_payload_bytes": sum(t.numel() * t.element_size() for t in tensors.values())}
    return {"metadata": metadata, "tensors": tensors}


@torch.no_grad()
def predict_residual(bank, inputs, *, expected_scope, expected_response_context_sha256=None):
    m, t = bank["metadata"], bank["tensors"]
    if m.get("format") != FORMAT or m.get("scope") != expected_scope.to_dict():
        raise ValueError("regional response source scope differs")
    if m.get("response_context_sha256") != expected_response_context_sha256:
        raise ValueError("regional response function/base-codec context differs")
    variant = m["variant"]
    if variant not in VARIANTS:
        raise ValueError("unknown regional response variant")
    width, r, e, k, d = expected_scope.matrix_input_width, m["rank"], m["experts"], m["regions"], m["output_width"]
    _finite(inputs, (len(inputs), width), "prediction inputs")
    shapes = {"input_mean": (width,), "input_basis": (width, r), "feature_scale": (r,),
              "response_mean": (e, d), "slopes": (e, r, d)}
    if variant != "global":
        shapes.update(centers=(k, r), radii=(k,), orthogonal_radius=(), offsets=(e, k, d), region_enabled=(k,))
    if variant == "independent_local":
        shapes["local_delta"] = (e, k, r, d)
    elif variant in ("shared_transport", "identity_transport", "wrong_region_transport"):
        shapes["operators"] = (k, r, r)
    if set(t) != set(shapes):
        raise ValueError("regional tensor fields differ from the declared variant")
    for name, shape in shapes.items():
        _finite(t[name], shape, name)
    if bool((t["feature_scale"] <= 0).any()):
        raise ValueError("regional feature scales must be positive")
    if variant != "global" and not bool(((t["region_enabled"] == 0) | (t["region_enabled"] == 1)).all()):
        raise ValueError("regional support flags must be binary")
    xc = inputs - t["input_mean"]
    projected = xc @ t["input_basis"]
    z = projected / t["feature_scale"]
    predicted = t["response_mean"][:, None, :] + torch.einsum("nr,erd->end", z, t["slopes"])
    labels = torch.full((len(inputs),), -1, dtype=torch.long)
    supported = torch.ones(len(inputs), dtype=torch.bool)
    if variant != "global":
        distance = torch.cdist(z, t["centers"])
        labels = distance.argmin(dim=1)
        orthogonal = (xc - projected @ t["input_basis"].T).norm(dim=1)
        supported = ((t["region_enabled"][labels] == 1) &
                     (distance[torch.arange(len(inputs)), labels] <= 1.5 * t["radii"][labels]) &
                     (orthogonal <= 1.5 * t["orthogonal_radius"]))
        for region in range(k):
            mask = (labels == region) & supported
            if not bool(mask.any()):
                continue
            zk = z[mask] - t["centers"][region]
            predicted[:, mask] += t["offsets"][:, region, None, :]
            if variant == "independent_local":
                predicted[:, mask] += torch.einsum("nr,erd->end", zk, t["local_delta"][:, region])
            elif "operators" in t:
                transformed = zk @ (t["operators"][region] - torch.eye(r))
                predicted[:, mask] += torch.einsum("nr,erd->end", transformed, t["slopes"])
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("regional residual prediction overflowed")
    return predicted, {"region_ids": labels, "supported": supported,
                       "global_fallback_count": int((~supported).sum())}
