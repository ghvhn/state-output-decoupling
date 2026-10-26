"""Affine matrix laws of radial response-error change in calibrated charts.

Shell labels are calibration-only radial quantile bands, not electron physics
or certified meanings. Their inputs may define geometry; one shell's responses
are excluded from every response fit. A designated expert's responses cannot
fit shared maps, although its own global basis and offsets remain calibrated.
"""
from __future__ import annotations

import hashlib
import math

import torch

from .trajectory_compress import InputStateScope, _snapshot, _fingerprint
from .region_response_transport import _cluster, _solve, _finite


FORMAT = "radial-shell-affine-polynomial-response-v1"
VARIANTS = ("global", "region_offsets", "constant_transport", "linear_transport",
            "generated_transport", "identity_transport", "wrong_shell_transport",
            "separate_shell_transport")
_COMMON = ("input_mean", "input_basis", "feature_scale", "response_mean", "slopes")
_GEOMETRY = ("centers", "radii", "orthogonal_radius", "shell_boundaries",
             "shell_coordinate_ranges", "region_enabled")


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def _context(value):
    if value is not None and (not isinstance(value, str) or len(value) != 64 or
                             any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("response context must bind the exact functions/base codec with SHA256")


def _psd_eigh(matrix):
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * .5)
    tolerance = max(float(values.abs().amax()) * 1e-10, 1e-12)
    if float(values.amin()) < -tolerance:
        raise ValueError("shared response Gram is not positive semidefinite")
    values = values.clamp(min=0)
    return values, vectors, int((values > tolerance).sum())


def _design(v, s, degree):
    q = torch.cat((v, torch.ones(len(v), 1, dtype=v.dtype)), dim=1)
    return torch.cat([q * s[:, None].pow(power) for power in range(degree + 1)], dim=1)


def _shared_fit(v, s, z, mask, residuals, intercept, slopes, offset, included, degree, ridge):
    """Exact separable shared ridge objective, using actual per-record radii.

    min_H sum_e ||D H B_e - Y_e||_F² + penalty ||H||_F².
    G_D H G_B + penalty H = sum_e D.T Y_e B_e.T is solved by
    the two Gram eigensystems, avoiding a huge Kronecker matrix.
    """
    r = slopes.shape[1]
    design = _design(v, s, degree).double()
    gd = design.T @ design
    gb = torch.zeros(r, r, dtype=torch.float64)
    cross = torch.zeros(design.shape[1], r, dtype=torch.float64)
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    for expert in included:
        basis = slopes[expert].double()
        gb += basis @ basis.T
        # Bound dense record/output intermediates independently of expert count.
        for start in range(0, len(selected), 128):
            stop = min(start + 128, len(selected))
            target = residuals[expert, selected[start:stop]].double()
            target -= (intercept[expert].double() +
                       z[start:stop].double() @ basis + offset[expert].double())
            cross += (design[start:stop].T @ target) @ basis.T
    a, u, design_rank = _psd_eigh(gd)
    b, w, basis_rank = _psd_eigh(gb)
    penalty = ridge * max(float(gd.trace()) / len(gd) * float(gb.trace()) / r, 1e-12)
    h = u @ ((u.T @ cross @ w) / (a[:, None] * b[None, :] + penalty)) @ w.T
    if not bool(torch.isfinite(h).all()):
        raise ValueError("shared shell fit produced nonfinite coefficients")
    return h.float().reshape(degree + 1, r + 1, r), {
        "degree": degree, "records": len(selected), "design_columns": design.shape[1],
        "effective_design_rank": design_rank, "effective_response_basis_rank": basis_rank,
        "full_design_rank": design_rank == design.shape[1], "ridge_penalty": penalty}


@torch.no_grad()
def fit_shells(records, residuals, *, expected_scope, heldout_records=(), rank=16,
               regions=4, shells=4, withheld_shell=1, ridge=.01, seed=0,
               operator_excluded_expert=7, response_context_sha256=None,
               max_working_bytes=3 << 30):
    """Fit charts and shared affine polynomials without withheld-shell targets.

    Residuals are [expert slot, calibration record, output coordinate]. The
    context supplied by the runner binds exact source identities/expert order.
    Geometry sees all calibration inputs; this is a partial response holdout.
    """
    _integer(rank, "rank", 1, 64)
    _integer(regions, "regions", 2, 8)
    _integer(shells, "shells", 2, 8)
    _integer(withheld_shell, "withheld_shell", 0, shells - 1)
    _integer(seed, "seed", 0, 2**32 - 1)
    _integer(max_working_bytes, "working-byte budget", 1, 16 << 30)
    if (isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or
            not math.isfinite(ridge) or ridge <= 0):
        raise ValueError("ridge must be finite and positive")
    if not isinstance(expected_scope, InputStateScope):
        raise ValueError("an exact input state scope is required")
    if not isinstance(records, (list, tuple)) or not isinstance(heldout_records, (list, tuple)):
        raise ValueError("records and heldout records must be bounded sequences")
    n, width = len(records), expected_scope.matrix_input_width
    if not rank < min(n, width) or n > 2048 or len(heldout_records) > 8192 or width > 8192:
        raise ValueError("shell input dimensions exceed fixed bounds")
    if not isinstance(residuals, torch.Tensor) or residuals.ndim != 3:
        raise ValueError("residuals need [expert,record,output] dimensions")
    experts, count, output = residuals.shape
    if not 2 <= experts <= 8 or count != n or not 1 <= output <= 8192:
        raise ValueError("shell response dimensions exceed fixed bounds")
    _integer(operator_excluded_expert, "operator_excluded_expert", 0, experts - 1)
    _context(response_context_sha256)
    # Counts are fixed before snapshots, double-precision eigensystems and any
    # response selection. These are numerical arrays, not a process RSS bound.
    p = 3 * (rank + 1)
    estimate = (residuals.numel() * 8 + n * width * 24 + len(heldout_records) * width * 8 +
                n * n * 32 + 128 * output * 48 + experts * (regions + rank + 2) * output * 16 +
                regions * (9 + shells) * (rank + 1) * rank * 16 + p * p * 64)
    if estimate > max_working_bytes:
        raise MemoryError("shell fit numerical arrays exceed their declared budget")
    _finite(residuals, (experts, n, output), "calibration residuals")
    frozen, _ = _snapshot(records, (), expected_scope, max_records=2048, max_pairs=0)
    heldout, _ = _snapshot(heldout_records, (), expected_scope, max_records=8192, max_pairs=0)
    if ({r.record_id for r in frozen} & {r.record_id for r in heldout} or
            {r.trajectory_id for r in frozen} & {r.trajectory_id for r in heldout}):
        raise ValueError("heldout records/trajectories cannot enter shell fitting")
    if heldout and frozen[0].origin != heldout[0].origin:
        raise ValueError("heldout and calibration origins must agree")
    if frozen[0].origin == "measured_model_state" and response_context_sha256 is None:
        raise ValueError("measured residuals require an exact response/base-codec context")
    if any(r.provenance_weight * r.stimulation_weight != 1 for r in frozen):
        raise ValueError("shell prototype requires uniform provenance and stimulation")
    x = torch.stack([r.state for r in frozen])
    mean = x.mean(dim=0)
    xc = x - mean
    values, vectors = torch.linalg.eigh(xc.double() @ xc.double().T)
    top = values[-rank:].flip(0)
    if float(top[-1]) <= max(float(top[0]) * 1e-10, 1e-12):
        raise ValueError("calibration inputs do not support requested PCA rank")
    basis = (xc.double().T @ vectors[:, -rank:].flip(1) / top.sqrt()[None, :]).float()
    pivots = basis.abs().argmax(dim=0)
    basis *= torch.sign(basis[pivots, torch.arange(rank)])[None, :]
    scale = (xc @ basis).square().mean(dim=0).sqrt().clamp(min=1e-8)
    z = xc @ basis / scale
    centers, labels = _cluster(z, regions, seed)
    distance = (z - centers[labels]).norm(dim=1)
    radii = torch.stack([distance[labels == k].amax() for k in range(regions)]).clamp(min=1e-8)
    coordinate = distance / radii[labels]
    boundaries = torch.stack([torch.quantile(coordinate[labels == k],
        torch.arange(1, shells).float() / shells) for k in range(regions)])
    ranges = torch.stack([torch.stack((coordinate[labels == k].amin(), coordinate[labels == k].amax()))
                          for k in range(regions)])
    shell_ids = torch.empty(n, dtype=torch.long)
    for k in range(regions):
        mask = labels == k
        shell_ids[mask] = torch.bucketize(coordinate[mask], boundaries[k], right=False)
    permitted = shell_ids != withheld_shell
    if not bool((~permitted).any()) or int(permitted.sum()) < rank + 2:
        raise ValueError("the withheld shell or remaining responses lack actual support")
    za = z[permitted]
    zmean = za.mean(dim=0)
    averages = residuals[:, permitted].mean(dim=1)
    slopes = torch.stack([_solve(za - zmean, residuals[e, permitted] - averages[e], ridge)
                          for e in range(experts)])
    intercept = averages - torch.einsum("r,erd->ed", zmean, slopes)
    offsets = torch.zeros(experts, regions, output)
    constant = torch.zeros(regions, 3, rank + 1, rank)
    linear = torch.zeros_like(constant)
    generated = torch.zeros_like(constant)
    table = torch.zeros(regions, shells, rank + 1, rank)
    table_enabled = torch.zeros(regions, shells)
    enabled = torch.zeros(regions)
    counts, allowed_counts, shell_counts, diagnostics = [], [], [], []
    included = [e for e in range(experts) if e != operator_excluded_expert]
    for k in range(regions):
        region_mask = labels == k
        allowed = region_mask & permitted
        cell_counts = [int((region_mask & (shell_ids == s)).sum()) for s in range(shells)]
        counts.append(int(region_mask.sum()))
        allowed_counts.append(int(allowed.sum()))
        shell_counts.append(cell_counts)
        valid_shells = [s for s in range(shells) if s != withheld_shell and cell_counts[s] >= rank + 2]
        # A quadratic design has3*(rank+1) columns, not merely rank columns.
        # Effective ranks are recorded separately from this sample-count guard.
        if int(allowed.sum()) < p + 1 or len(valid_shells) != shells - 1:
            diagnostics.append({"region": k, "enabled": False, "reason": "insufficient permitted response support"})
            continue
        enabled[k] = 1
        for e in range(experts):
            offsets[e, k] = (residuals[e, allowed] - intercept[e] - z[allowed] @ slopes[e]).mean(dim=0)
        v, sk, zk = z[allowed] - centers[k], coordinate[allowed], z[allowed]
        d = {"region": k, "enabled": True, "degrees": []}
        for degree, destination in ((0, constant), (1, linear), (2, generated)):
            coefficients, detail = _shared_fit(v, sk, zk, allowed, residuals, intercept,
                                               slopes, offsets[:, k], included, degree, ridge)
            destination[k, :degree + 1] = coefficients
            d["degrees"].append(detail)
        d["table"] = []
        for s in valid_shells:
            cell = allowed & (shell_ids == s)
            coefficients, detail = _shared_fit(z[cell] - centers[k], coordinate[cell], z[cell],
                cell, residuals, intercept, slopes, offsets[:, k], included, 0, ridge)
            table[k, s] = coefficients[0]
            table_enabled[k, s] = 1
            d["table"].append({"shell": s, **detail})
        diagnostics.append(d)
    if not bool(enabled.any()):
        raise ValueError("no region supports the permitted quadratic response design")
    orthogonal_radius = (xc - (xc @ basis) @ basis.T).norm(dim=1).amax().clamp(min=1e-8)
    response_digest = hashlib.sha256()
    response_indices = torch.nonzero(permitted, as_tuple=False).flatten()
    for expert in range(experts):
        for start in range(0, len(response_indices), 128):
            response_digest.update(residuals[expert, response_indices[start:start + 128]].detach().contiguous().numpy().tobytes())
    metadata = {"format": FORMAT, "scope": expected_scope.to_dict(),
        "response_context_sha256": response_context_sha256,
        "calibration_evidence_sha256": _fingerprint(expected_scope, frozen, ()),
        "response_fit_sha256": response_digest.hexdigest(),
        "response_digest_order": "expert slots ascending, then permitted calibration records in archive order; FP32 bytes",
        "calibration_record_ids": [r.record_id for r in frozen],
        "calibration_trajectory_ids": sorted({r.trajectory_id for r in frozen}),
        "response_fit_record_ids": [r.record_id for r, use in zip(frozen, permitted) if bool(use)],
        "response_withheld_record_ids": [r.record_id for r, use in zip(frozen, permitted) if not bool(use)],
        "calibration_region_ids": labels.tolist(), "calibration_shell_ids": shell_ids.tolist(),
        "rank": rank, "regions": regions, "shells": shells, "withheld_shell": withheld_shell,
        "experts": experts, "output_width": output, "ridge": float(ridge), "seed": seed,
        "operator_excluded_expert": operator_excluded_expert,
        "operator_excluded_expert_is_slot": True,
        "region_counts": counts, "permitted_response_region_counts": allowed_counts,
        "shell_counts": shell_counts, "region_enabled": [bool(v) for v in enabled],
        "fit_diagnostics": diagnostics, "minimum_region_response_records": p + 1,
        "minimum_table_response_records": rank + 2, "estimated_fit_numerical_bytes": estimate,
        "coefficient_layout": "degree-major [local coordinates, affine1] -> response coordinates",
        "shared_objective": "sum included-expert ||D H B_e-Y_e||F² + scaled-ridge ||H||F²; exact two-Gram eigensolve",
        "shell_coordinate": "distance from regional center / max calibration regional distance",
        "shell_labels": "ordered within-region calibration radial quantiles; boundary ties choose lower shell",
        "wrong_shell_policy": "reflect continuous coordinate about its regional calibration min+max",
        "table_fallback": "nearest fitted ordered shell index; equal-distance ties choose lower index",
        "radial_boundary_tolerance": 1e-5,
        "support_policy": "all nonwithheld bins>=rank+2 and pooled records>=3*(rank+1)+1; enabled chart, measured radial min..max (1e-5 tolerance), orthogonal radius<=1.5x; global fallback",
        "fit_weighting": "uniform records; uniform provenance/stimulation; predictions retained",
        "withheld_level_inputs_used_for_geometry": True, "withheld_level_responses_used_for_fit": False,
        "heldout_used_for_fit": False, "partial_response_holdout": True,
        "region_semantics_validated": False, "electron_shell_physics_modeled": False,
        "unseen_region_transfer_evaluated": False, "whole_model_compression_established": False}
    tensors = {"input_mean": mean, "input_basis": basis, "feature_scale": scale,
        "response_mean": intercept, "slopes": slopes, "centers": centers, "radii": radii,
        "orthogonal_radius": orthogonal_radius, "shell_boundaries": boundaries,
        "shell_coordinate_ranges": ranges, "region_enabled": enabled, "offsets": offsets,
        "constant_coefficients": constant, "linear_coefficients": linear,
        "generated_coefficients": generated, "table_coefficients": table, "table_enabled": table_enabled}
    if any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise ValueError("shell fit produced nonfinite tensors")
    return {"metadata": metadata, "tensors": tensors}


def variant_bank(bank, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown shell response variant")
    fields = _COMMON + (() if variant == "global" else _GEOMETRY + ("offsets",))
    tensors = {name: bank["tensors"][name].clone().contiguous() for name in fields}
    extra_metadata = {}
    if variant == "separate_shell_transport":
        ids = [s for s in range(bank["metadata"]["shells"]) if s != bank["metadata"]["withheld_shell"]]
        for name in ("table_coefficients", "table_enabled"):
            tensors[name] = bank["tensors"][name][:, ids].clone().contiguous()
        extra_metadata["table_shell_ids"] = ids
    elif variant not in ("global", "region_offsets"):
        source = {"constant_transport": "constant_coefficients", "linear_transport": "linear_coefficients"}.get(
                  variant, "generated_coefficients")
        tensors["coefficients"] = bank["tensors"][source].clone().contiguous()
        if variant == "identity_transport":
            tensors["coefficients"].zero_()
    metadata = {**bank["metadata"], **extra_metadata, "variant": variant,
                "tensor_payload_bytes": sum(t.numel() * t.element_size() for t in tensors.values())}
    return {"metadata": metadata, "tensors": tensors}


def _validate(bank, scope, context, *, geometry_only=False):
    if not isinstance(scope, InputStateScope) or not isinstance(bank, dict):
        raise ValueError("prediction requires an exact scoped shell bank")
    m, t = bank["metadata"], bank["tensors"]
    if m.get("format") != FORMAT or m.get("scope") != scope.to_dict():
        raise ValueError("shell source scope differs")
    _context(context)
    if m.get("response_context_sha256") != context:
        raise ValueError("shell response/base-codec context differs")
    for key, low, high in (("rank", 1, 64), ("regions", 2, 8), ("shells", 2, 8),
                           ("experts", 2, 8), ("output_width", 1, 8192)):
        _integer(m.get(key), key, low, high)
    r, k, s, e, d = (m[key] for key in ("rank", "regions", "shells", "experts", "output_width"))
    shapes = {"input_mean": (scope.matrix_input_width,), "input_basis": (scope.matrix_input_width, r),
              "feature_scale": (r,)}
    variant = m.get("variant")
    if not geometry_only:
        if variant not in VARIANTS:
            raise ValueError("unknown shell response variant")
        shapes.update(response_mean=(e, d), slopes=(e, r, d))
    if geometry_only or variant != "global":
        shapes.update(centers=(k, r), radii=(k,), orthogonal_radius=(), shell_boundaries=(k, s - 1),
                      shell_coordinate_ranges=(k, 2), region_enabled=(k,))
    if not geometry_only and variant != "global":
        shapes["offsets"] = (e, k, d)
        if variant == "separate_shell_transport":
            shapes.update(table_coefficients=(k, s - 1, r + 1, r), table_enabled=(k, s - 1))
        elif variant != "region_offsets":
            shapes["coefficients"] = (k, 3, r + 1, r)
    if not geometry_only and set(t) != set(shapes):
        raise ValueError("shell tensor fields differ from the variant")
    for name, shape in shapes.items():
        _finite(t[name], shape, name)
    if bool((t["feature_scale"] <= 0).any()):
        raise ValueError("shell feature scales must be positive")
    if geometry_only or variant != "global":
        if (bool((t["radii"] <= 0).any()) or float(t["orthogonal_radius"]) <= 0 or
                bool((t["shell_boundaries"].diff(dim=1) < 0).any()) or
                bool((t["shell_coordinate_ranges"][:, 0] > t["shell_coordinate_ranges"][:, 1]).any())):
            raise ValueError("shell radii/boundaries/ranges are invalid")
        if not bool(((t["region_enabled"] == 0) | (t["region_enabled"] == 1)).all()):
            raise ValueError("shell region support flags must be binary")
    if not geometry_only and variant == "separate_shell_transport":
        _integer(m.get("withheld_shell"), "withheld_shell", 0, s - 1)
        if (m.get("table_shell_ids") != [slot for slot in range(s) if slot != m["withheld_shell"]] or
                not bool(((t["table_enabled"] == 0) | (t["table_enabled"] == 1)).all()) or
                not torch.equal(t["table_enabled"].sum(dim=1), t["region_enabled"] * (s - 1))):
            raise ValueError("compact table slots/support must exclude the withheld shell")
    return m, t


@torch.no_grad()
def predict_geometry(bank, inputs, *, expected_scope, expected_response_context_sha256=None,
                     max_working_bytes=3 << 30):
    """Select charts and ordered shells without allocating expert outputs."""
    m, t = _validate(bank, expected_scope, expected_response_context_sha256, geometry_only=True)
    _integer(max_working_bytes, "geometry working-byte budget", 1, 16 << 30)
    if len(inputs) > 8192:
        raise ValueError("shell prediction record count exceeds fixed bound")
    estimate = (sum(v.untyped_storage().nbytes() for v in t.values()) +
                len(inputs) * expected_scope.matrix_input_width * 16 + len(inputs) * m["rank"] * 64)
    if estimate > max_working_bytes:
        raise MemoryError("shell geometry numerical arrays exceed their declared budget")
    _finite(inputs, (len(inputs), expected_scope.matrix_input_width), "prediction inputs")
    xc = inputs - t["input_mean"]
    projected = xc @ t["input_basis"]
    z = projected / t["feature_scale"]
    distances = torch.cdist(z, t["centers"])
    labels = distances.argmin(dim=1)
    # Match fitting's direct local norm exactly; cdist only chooses the chart.
    # Its matrix-multiplication path can shift boundary radii by a few ULPs.
    selected_distance = (z - t["centers"][labels]).norm(dim=1)
    coordinate = selected_distance / t["radii"][labels]
    orthogonal = (xc - projected @ t["input_basis"].T).norm(dim=1)
    ranges = t["shell_coordinate_ranges"][labels]
    supported = ((t["region_enabled"][labels] == 1) &
                 (coordinate >= ranges[:, 0] - 1e-5) & (coordinate <= ranges[:, 1] + 1e-5) &
                 (orthogonal <= 1.5 * t["orthogonal_radius"]))
    if not bool(torch.isfinite(coordinate).all() & torch.isfinite(orthogonal).all()):
        raise ValueError("shell geometry overflowed")
    shell_ids = torch.empty(len(inputs), dtype=torch.long)
    for k in range(m["regions"]):
        mask = labels == k
        shell_ids[mask] = torch.bucketize(coordinate[mask], t["shell_boundaries"][k], right=False)
    return {"region_ids": labels, "shell_ids": shell_ids, "shell_coordinates": coordinate,
            "supported": supported, "global_fallback_count": int((~supported).sum())}


@torch.no_grad()
def predict_residual(bank, inputs, *, expected_scope, expected_response_context_sha256=None,
                     max_working_bytes=3 << 30):
    m, t = _validate(bank, expected_scope, expected_response_context_sha256)
    _integer(max_working_bytes, "prediction working-byte budget", 1, 16 << 30)
    n, width, r, e, d = len(inputs), expected_scope.matrix_input_width, m["rank"], m["experts"], m["output_width"]
    if n > 8192:
        raise ValueError("shell prediction record count exceeds fixed bound")
    estimate = (sum(v.untyped_storage().nbytes() for v in t.values()) +
                n * width * 16 + e * n * d * 16 + n * r * 64)
    if estimate > max_working_bytes:
        raise MemoryError("shell prediction numerical arrays exceed their declared budget")
    _finite(inputs, (n, width), "prediction inputs")
    z = (inputs - t["input_mean"]) @ t["input_basis"] / t["feature_scale"]
    predicted = t["response_mean"][:, None, :] + torch.einsum("nr,erd->end", z, t["slopes"])
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("shell prediction overflowed")
    if m["variant"] == "global":
        return predicted, {"region_ids": torch.full((n,), -1, dtype=torch.long),
            "shell_ids": torch.full((n,), -1, dtype=torch.long), "shell_coordinates": torch.zeros(n),
            "supported": torch.ones(n, dtype=torch.bool), "global_fallback_count": 0}
    diagnostic = predict_geometry(bank, inputs, expected_scope=expected_scope,
                                  expected_response_context_sha256=expected_response_context_sha256,
                                  max_working_bytes=max_working_bytes)
    for k in range(m["regions"]):
        mask = (diagnostic["region_ids"] == k) & diagnostic["supported"]
        if not bool(mask.any()):
            continue
        predicted[:, mask] += t["offsets"][:, k, None, :]
        if m["variant"] == "region_offsets":
            continue
        v = z[mask] - t["centers"][k]
        s = diagnostic["shell_coordinates"][mask]
        # All methods share the same measured range. Canonicalize only accepted
        # floating-point boundary tolerance before evaluating or reflecting s.
        s = s.clamp(min=t["shell_coordinate_ranges"][k, 0], max=t["shell_coordinate_ranges"][k, 1])
        if m["variant"] == "separate_shell_transport":
            valid_slots = torch.nonzero(t["table_enabled"][k], as_tuple=False).flatten()
            if not len(valid_slots):
                raise ValueError("an enabled chart lacks any fitted table shell")
            valid = torch.tensor(m["table_shell_ids"], dtype=torch.long)[valid_slots]
            observed = diagnostic["shell_ids"][mask]
            nearest = (observed[:, None] - valid[None, :]).abs().argmin(dim=1)
            selected = valid[nearest]
            q = torch.cat((v, torch.ones(len(v), 1)), dim=1)
            delta = torch.einsum("ni,nij->nj", q, t["table_coefficients"][k, valid_slots[nearest]])
            diagnostic.setdefault("selected_table_shell_ids", torch.full((n,), -1, dtype=torch.long))[mask] = selected
        else:
            if m["variant"] == "wrong_shell_transport":
                s = t["shell_coordinate_ranges"][k].sum() - s
                s = s.clamp(min=t["shell_coordinate_ranges"][k, 0], max=t["shell_coordinate_ranges"][k, 1])
            delta = _design(v, s, 2) @ t["coefficients"][k].reshape(3 * (r + 1), r)
        predicted[:, mask] += torch.einsum("nr,erd->end", delta, t["slopes"])
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("shell prediction overflowed")
    return predicted, diagnostic
