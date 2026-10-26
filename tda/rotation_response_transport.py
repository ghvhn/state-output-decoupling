"""Calibration-only angular laws of nonlinear response-error transport.

Each chart has a fixed measured two-plane in PCA coordinates. Static maps are
periodic in that plane: they do not infer how many revolutions a trajectory
made. One angular sector's response targets are excluded from *every* fit.
Regions and angles describe measured geometry, not certified semantic meaning.
"""
from __future__ import annotations

import hashlib
import math

import torch

from .trajectory_compress import InputStateScope, _snapshot, _fingerprint
from .region_response_transport import _cluster, _solve, _finite
from .shell_response_transport import _integer, _context, _psd_eigh, _design
from .rotation_geometry import fit_rotation_plane, project_directions, embedded_rotations


FORMAT = "angular-covariant-affine-polynomial-response-v1"
VARIANTS = ("global", "region_offsets", "radial_transport", "rotation_transport",
            "wrong_rotation_transport", "fourier_transport", "wrong_fourier_transport",
            "separate_angle_transport")
_COMMON = ("input_mean", "input_basis", "feature_scale", "response_mean", "slopes")
_GEOMETRY = ("centers", "radii", "orthogonal_radius", "rotation_planes",
             "rotation_plane_defined", "radial_coordinate_ranges", "region_enabled")


def _rotation_matrices(phase, plane):
    """R(theta) canonicalizes a row direction: v @ R(theta) has phase zero.

    Forward rotation of a row vector is v @ R(alpha).T. The orthogonal
    complement is retained exactly by the embedded two-plane rotation.
    """
    return embedded_rotations(plane, phase).to(plane.dtype)


def _plane_geometry(directions, plane):
    if not len(directions):
        return torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.bool)
    geometry = project_directions(directions, plane, min_plane_fraction=.05)
    valid = geometry["supported"]
    phase = torch.remainder(geometry["phase_rad"], 2 * math.pi).float()
    phase = torch.where(valid, phase, torch.zeros_like(phase))
    return phase, geometry["plane_fraction"].float(), valid


def _fit_plane(directions):
    """Fit an oriented plane from unit calibration directions only."""
    nonzero = directions.norm(dim=1) > 1e-8
    return fit_rotation_plane(directions[nonzero]).float()


def _fourier_design(v, rho, phase):
    affine = torch.cat((v, torch.ones(len(v), 1, dtype=v.dtype)), dim=1)
    harmonics = (torch.ones_like(phase), phase.cos(), phase.sin(),
                 (2 * phase).cos(), (2 * phase).sin())
    return torch.cat([affine * (rho.pow(degree) * angular)[:, None]
                      for degree in range(3) for angular in harmonics], dim=1)


def _projected_cross(design, z, selected, residuals, intercept, slopes, offset, included,
                     rotations=None):
    rank = slopes.shape[1]
    cross = torch.zeros(design.shape[1], rank, dtype=torch.float64)
    for expert in included:
        basis = slopes[expert].double()
        for start in range(0, len(selected), 128):
            stop = min(start + 128, len(selected))
            target = residuals[expert, selected[start:stop]].double()
            target -= intercept[expert].double() + z[start:stop].double() @ basis + offset[expert].double()
            projected = target @ basis.T
            if rotations is not None:
                projected = torch.einsum("nr,nrs->ns", projected, rotations[start:stop])
            cross += design[start:stop].T @ projected
    return cross


def _linear_shared_fit(design, z, selected, residuals, intercept, slopes, offset, included, ridge):
    """Exact sum_e ||D H B_e - Y_e||² + scaled ridge ||H||²."""
    design = design.double()
    gd = design.T @ design
    gb = sum((slopes[e].double() @ slopes[e].double().T for e in included),
             torch.zeros(slopes.shape[1], slopes.shape[1], dtype=torch.float64))
    cross = _projected_cross(design, z, selected, residuals, intercept, slopes, offset, included)
    a, u, design_rank = _psd_eigh(gd)
    b, w, basis_rank = _psd_eigh(gb)
    penalty = ridge * max(float(gd.trace()) / len(gd) * float(gb.trace()) / len(gb), 1e-12)
    h = u @ ((u.T @ cross @ w) / (a[:, None] * b[None] + penalty)) @ w.T
    if not bool(torch.isfinite(h).all()):
        raise ValueError("shared angular fit produced nonfinite coefficients")
    return h.float(), {"records": len(selected), "design_columns": len(gd),
        "effective_design_rank": design_rank, "effective_response_basis_rank": basis_rank,
        "full_design_rank": design_rank == len(gd), "ridge_penalty": penalty,
        "solver": "exact two-Gram eigensolve"}


def _covariant_shared_fit(v, rho, phase, plane, z, selected, residuals, intercept,
                          slopes, offset, included, ridge):
    """Fit the exact rotating-output objective, not a constant-basis shortcut.

    For record n, canonical D_n is [v_n R(theta_n),1] times radial powers;
    the output basis is B_n,e = R(theta_n).T B_e. Thus the full normal is
    sum_n (D_n.T D_n) kron (R_n.T sum_e B_e B_e.T R_n).
    The finite p*r normal is assembled in blocks of32 records and solved in
    FP64. At native rank16 it has816 unknowns, occupying5.3MB per matrix.
    """
    rank = slopes.shape[1]
    rotations = _rotation_matrices(phase.double(), plane.double())
    canonical = torch.einsum("nr,nrs->ns", v.double(), rotations)
    design = _design(canonical, rho.double(), 2)
    p = design.shape[1]
    if p * rank > 1024:
        raise ValueError("covariant exact normal exceeds the1024-unknown bound")
    gb = sum((slopes[e].double() @ slopes[e].double().T for e in included),
             torch.zeros(rank, rank, dtype=torch.float64))
    normal = torch.zeros(p * rank, p * rank, dtype=torch.float64)
    for start in range(0, len(selected), 32):
        stop = min(start + 32, len(selected))
        rotation = rotations[start:stop]
        local_gb = rotation.transpose(1, 2) @ gb @ rotation
        dn = design[start:stop]
        normal += torch.einsum("np,nq,nrs->prqs", dn, dn, local_gb).reshape(p * rank, p * rank)
    cross = _projected_cross(design, z, selected, residuals, intercept, slopes, offset,
                             included, rotations)
    penalty = ridge * max(float(normal.trace()) / len(normal), 1e-12)
    normal.diagonal().add_(penalty)
    h = torch.linalg.solve(normal, cross.flatten()).reshape(p, rank)
    solve_error = (normal @ h.flatten() - cross.flatten()).norm() / cross.norm().clamp(min=1e-30)
    if not bool(torch.isfinite(h).all()) or float(solve_error) > 1e-8:
        raise ValueError("covariant exact response objective did not solve accurately")
    _, _, basis_rank = _psd_eigh(gb)
    # Canonical rotation removes one plane coordinate. Saved FP32 planes and
    # phases leave tiny roundoff there; FP64's default matrix-rank tolerance
    # would count that numerical residue as evidence of an extra feature.
    _, _, design_rank = _psd_eigh(design.T @ design)
    return h.float().reshape(3, rank + 1, rank), {"records": len(selected),
        "design_columns": p, "effective_design_rank": design_rank,
        "effective_response_basis_rank": basis_rank, "full_design_rank": design_rank == p,
        "ridge_penalty": penalty, "normal_unknowns": p * rank,
        "solver": "exact per-record rotated-basis FP64 normal solve",
        "design_rank_threshold": "Gram eigenvalue > max(max(abs eigenvalues)*1e-10,1e-12)",
        "normal_residual_relative": float(solve_error)}


@torch.no_grad()
def fit_rotations(records, residuals, *, expected_scope, heldout_records=(), rank=16,
                  regions=4, angle_bins=4, withheld_angle_bin=1, ridge=.01, seed=0,
                  operator_excluded_expert=7, response_context_sha256=None,
                  max_working_bytes=3 << 30):
    """Fit angular maps without any withheld-sector response targets.

    Calibration inputs alone define PCA, regions, planes and radial support.
    The partial response holdout permits those inputs but excludes every
    withheld-sector target from global slopes, offsets and every shared map.
    """
    _integer(rank, "rank", 2, 16)
    _integer(regions, "regions", 2, 8)
    _integer(angle_bins, "angle_bins", 4, 4)
    _integer(withheld_angle_bin, "withheld_angle_bin", 0, angle_bins - 1)
    _integer(seed, "seed", 0, 2**32 - 1)
    _integer(max_working_bytes, "working-byte budget", 1, 16 << 30)
    if (isinstance(ridge, bool) or not isinstance(ridge, (float, int)) or not math.isfinite(ridge) or ridge <= 0):
        raise ValueError("ridge must be finite and positive")
    if not isinstance(expected_scope, InputStateScope):
        raise ValueError("an exact input state scope is required")
    if not isinstance(records, (list, tuple)) or not isinstance(heldout_records, (list, tuple)):
        raise ValueError("records must be bounded sequences")
    n, width = len(records), expected_scope.matrix_input_width
    if not rank < min(n, width) or n > 2048 or len(heldout_records) > 8192 or width > 8192:
        raise ValueError("angular input dimensions exceed fixed bounds")
    if not isinstance(residuals, torch.Tensor) or residuals.ndim != 3:
        raise ValueError("residuals require [expert,record,output] dimensions")
    experts, count, output = residuals.shape
    if not 2 <= experts <= 8 or count != n or not 1 <= output <= 8192:
        raise ValueError("angular response dimensions exceed fixed bounds")
    _integer(operator_excluded_expert, "operator_excluded_expert", 0, experts - 1)
    _context(response_context_sha256)
    p = 3 * (rank + 1)
    unknowns = p * rank
    estimate = (residuals.numel() * 8 + n * width * 24 + len(heldout_records) * width * 8 +
        n * n * 32 + 128 * output * 48 + experts * (regions + rank + 2) * output * 16 +
        regions * 24 * (rank + 1) * rank * 16 + unknowns * unknowns * 48 +
        (15 * (rank + 1)) ** 2 * 64 + n * rank * rank * 32 +
        32 * p * p * rank * rank * 8)
    if estimate > max_working_bytes:
        raise MemoryError("angular fit numerical arrays exceed declared budget")
    _finite(residuals, (experts, n, output), "calibration residuals")
    frozen, _ = _snapshot(records, (), expected_scope, max_records=2048, max_pairs=0)
    heldout, _ = _snapshot(heldout_records, (), expected_scope, max_records=8192, max_pairs=0)
    if ({r.record_id for r in frozen} & {r.record_id for r in heldout} or
            {r.trajectory_id for r in frozen} & {r.trajectory_id for r in heldout}):
        raise ValueError("heldout records/trajectories cannot enter angular fitting")
    if heldout and frozen[0].origin != heldout[0].origin:
        raise ValueError("heldout and calibration origins must agree")
    if frozen[0].origin == "measured_model_state" and response_context_sha256 is None:
        raise ValueError("measured residuals require exact response/base-codec context")
    if any(r.provenance_weight * r.stimulation_weight != 1 for r in frozen):
        raise ValueError("angular prototype requires uniform provenance and stimulation")
    x = torch.stack([r.state for r in frozen])
    mean = x.mean(dim=0)
    xc = x - mean
    values, vectors = torch.linalg.eigh(xc.double() @ xc.double().T)
    top = values[-rank:].flip(0)
    if float(top[-1]) <= max(float(top[0]) * 1e-10, 1e-12):
        raise ValueError("calibration does not support requested PCA rank")
    basis = (xc.double().T @ vectors[:, -rank:].flip(1) / top.sqrt()[None]).float()
    pivots = basis.abs().argmax(dim=0)
    basis *= torch.sign(basis[pivots, torch.arange(rank)])[None]
    scale = (xc @ basis).square().mean(dim=0).sqrt().clamp(min=1e-8)
    z = xc @ basis / scale
    centers, labels = _cluster(z, regions, seed)
    local = z - centers[labels]
    distance = local.norm(dim=1)
    radii = torch.stack([distance[labels == k].amax() for k in range(regions)]).clamp(min=1e-8)
    rho = distance / radii[labels]
    ranges = torch.stack([torch.stack((rho[labels == k].amin(), rho[labels == k].amax())) for k in range(regions)])
    # A sparse chart can genuinely have only one measured direction. Keep an
    # orthonormal storage placeholder, explicitly marked undefined: it never
    # supplies phase, support or angular response coefficients.
    planes = torch.eye(rank)[:, :2].expand(regions, rank, 2).clone()
    plane_defined = torch.zeros(regions)
    plane_reasons = [None] * regions
    phase = torch.zeros(n)
    fraction = torch.zeros(n)
    valid = torch.zeros(n, dtype=torch.bool)
    for k in range(regions):
        mask = labels == k
        if int((local[mask].norm(dim=1) > 1e-8).sum()) < 2:
            plane_reasons[k] = "fewer than two nonzero calibration directions"
            continue
        try:
            planes[k] = _fit_plane(local[mask])
        except ValueError as error:
            # Only the declared measured-span degeneracy is a local fallback.
            # Invalid arrays, budget failures and other errors remain visible.
            if str(error) != "calibration direction span does not support a 2-plane":
                raise
            plane_reasons[k] = str(error)
            continue
        plane_defined[k] = 1
        phase[mask], fraction[mask], valid[mask] = _plane_geometry(local[mask], planes[k])
    if not bool(plane_defined.any()):
        raise ValueError("no calibration chart supports a defined rotation plane")
    angle_ids = torch.floor(phase / (2 * math.pi / angle_bins)).long().clamp(max=angle_bins - 1)
    angle_ids[~valid] = -1
    permitted = angle_ids != withheld_angle_bin
    if not bool((~permitted).any()) or int(permitted.sum()) < rank + 2:
        raise ValueError("withheld sector or remaining responses lack actual support")
    za = z[permitted]
    zmean = za.mean(dim=0)
    averages = residuals[:, permitted].mean(dim=1)
    slopes = torch.stack([_solve(za - zmean, residuals[e, permitted] - averages[e], ridge) for e in range(experts)])
    intercept = averages - torch.einsum("r,erd->ed", zmean, slopes)
    offsets = torch.zeros(experts, regions, output)
    radial = torch.zeros(regions, 3, rank + 1, rank)
    covariant = torch.zeros_like(radial)
    fourier = torch.zeros(regions, 15, rank + 1, rank)
    table = torch.zeros(regions, angle_bins, rank + 1, rank)
    table_enabled = torch.zeros(regions, angle_bins)
    enabled = torch.zeros(regions)
    counts, allowed_counts, sector_counts, diagnostics = [], [], [], []
    included = [e for e in range(experts) if e != operator_excluded_expert]
    for k in range(regions):
        region = labels == k
        allowed = region & permitted & valid
        cell_counts = [int((region & valid & (angle_ids == s)).sum()) for s in range(angle_bins)]
        counts.append(int(region.sum()))
        allowed_counts.append(int(allowed.sum()))
        sector_counts.append(cell_counts)
        if not bool(plane_defined[k]):
            diagnostics.append({"region": k, "enabled": False, "reason": plane_reasons[k],
                                "rotation_plane_defined": False})
            continue
        valid_sectors = [s for s in range(angle_bins) if s != withheld_angle_bin and cell_counts[s] >= rank + 2]
        if int(allowed.sum()) < p + 1 or len(valid_sectors) != angle_bins - 1:
            diagnostics.append({"region": k, "enabled": False, "reason": "insufficient permitted angular response support"})
            continue
        enabled[k] = 1
        selected = torch.nonzero(allowed, as_tuple=False).flatten()
        for e in range(experts):
            offsets[e, k] = (residuals[e, selected] - intercept[e] - z[selected] @ slopes[e]).mean(dim=0)
        vk, sk, tk, zk = local[selected], rho[selected], phase[selected], z[selected]
        h, dr = _linear_shared_fit(_design(vk, sk, 2), zk, selected, residuals,
            intercept, slopes, offsets[:, k], included, ridge)
        radial[k] = h.reshape(3, rank + 1, rank)
        covariant[k], dc = _covariant_shared_fit(vk, sk, tk, planes[k], zk, selected,
            residuals, intercept, slopes, offsets[:, k], included, ridge)
        h, df = _linear_shared_fit(_fourier_design(vk, sk, tk), zk, selected, residuals,
            intercept, slopes, offsets[:, k], included, ridge)
        fourier[k] = h.reshape(15, rank + 1, rank)
        detail = {"region": k, "enabled": True, "radial": dr, "rotation": dc, "fourier": df, "table": []}
        for s in valid_sectors:
            selected = torch.nonzero(allowed & (angle_ids == s), as_tuple=False).flatten()
            h, dt = _linear_shared_fit(_design(local[selected], rho[selected], 0), z[selected],
                selected, residuals, intercept, slopes, offsets[:, k], included, ridge)
            table[k, s] = h.reshape(rank + 1, rank)
            table_enabled[k, s] = 1
            detail["table"].append({"angle_bin": s, **dt})
        diagnostics.append(detail)
    if not bool(enabled.any()):
        raise ValueError("no region supports permitted quadratic angular response design")
    orthogonal_radius = (xc - (xc @ basis) @ basis.T).norm(dim=1).amax().clamp(min=1e-8)
    digest = hashlib.sha256()
    selected = torch.nonzero(permitted, as_tuple=False).flatten()
    for expert in range(experts):
        for start in range(0, len(selected), 128):
            digest.update(residuals[expert, selected[start:start + 128]].contiguous().numpy().tobytes())
    metadata = {"format": FORMAT, "scope": expected_scope.to_dict(), "response_context_sha256": response_context_sha256,
        "calibration_evidence_sha256": _fingerprint(expected_scope, frozen, ()), "response_fit_sha256": digest.hexdigest(),
        "response_digest_order": "expert slots ascending; permitted records in archive order; FP32 bytes",
        "calibration_record_ids": [r.record_id for r in frozen],
        "calibration_trajectory_ids": sorted({r.trajectory_id for r in frozen}),
        "response_fit_record_ids": [r.record_id for r, use in zip(frozen, permitted) if bool(use)],
        "response_withheld_record_ids": [r.record_id for r, use in zip(frozen, permitted) if not bool(use)],
        "calibration_region_ids": labels.tolist(), "calibration_angle_bin_ids": angle_ids.tolist(),
        "rank": rank, "regions": regions, "angle_bins": angle_bins, "withheld_angle_bin": withheld_angle_bin,
        "experts": experts, "output_width": output, "ridge": float(ridge), "seed": seed,
        "operator_excluded_expert": operator_excluded_expert, "operator_excluded_expert_is_slot": True,
        "region_counts": counts, "permitted_response_region_counts": allowed_counts, "angle_bin_counts": sector_counts,
        "region_enabled": [bool(value) for value in enabled], "fit_diagnostics": diagnostics,
        "rotation_plane_defined": [bool(value) for value in plane_defined],
        "rotation_plane_fallback_reasons": plane_reasons,
        "undefined_plane_policy": "inactive orthonormal storage placeholder; no phase, sector or local response map; global fallback",
        "minimum_region_response_records": p + 1, "minimum_table_response_records": rank + 2,
        "estimated_fit_numerical_bytes": estimate, "plane_fraction_minimum": .05,
        "radial_boundary_tolerance": 1e-5, "angular_boundary_policy": "four equal half-open sectors in [0,2pi); undefined phase is-1",
        "rotation_law": "([v R(theta),1] H(rho)) R(theta).T; H quadratic in full local radius",
        "covariance": "delta(v R(alpha).T,rho,theta+alpha)=delta(v,rho,theta) R(alpha).T",
        "shared_objective": "sum included-expert ||D_n H R(theta_n).T B_e-Y_n,e||F² + scaled ridge ||H||F²",
        "effective_rank_policy": "Gram eigenvalue > max(max(abs eigenvalues)*1e-10,1e-12); identical threshold for every shared fit",
        "fourier_layout": "radial degree-major; angular[1,cos(theta),sin(theta),cos(2theta),sin(2theta)]; local affine coordinates",
        "wrong_rotation_policy": "negate phase, preserving geometry and fitted coefficients",
        "wrong_fourier_policy": "phase plus pi/2, preserving geometry and fitted coefficients",
        "table_fallback": "nearest fitted sector center by circular angular distance; exact ties choose lower actual sector ID",
        "support_policy": "every nonwithheld sector>=rank+2; pooled>=3*(rank+1)+1; plane fraction>=.05; measured radial min..max plus1e-5; orthogonal<=1.5x; global fallback",
        "fit_weighting": "uniform records; uniform provenance/stimulation; all predictions retained",
        "withheld_sector_inputs_used_for_geometry": True, "withheld_sector_responses_used_for_fit": False,
        "heldout_used_for_fit": False, "partial_response_holdout": True,
        "static_laws_are_2pi_periodic": True, "winding_encoded_by_static_map": False,
        "region_semantics_validated": False, "whole_model_compression_established": False}
    tensors = {"input_mean": mean, "input_basis": basis, "feature_scale": scale, "response_mean": intercept,
        "slopes": slopes, "centers": centers, "radii": radii, "orthogonal_radius": orthogonal_radius,
        "rotation_planes": planes, "rotation_plane_defined": plane_defined,
        "radial_coordinate_ranges": ranges, "region_enabled": enabled, "offsets": offsets,
        "radial_coefficients": radial, "rotation_coefficients": covariant, "fourier_coefficients": fourier,
        "table_coefficients": table, "table_enabled": table_enabled}
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("angular fit produced nonfinite tensors")
    return {"metadata": metadata, "tensors": tensors}


def variant_bank(bank, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown angular response variant")
    fields = _COMMON + _GEOMETRY + (() if variant == "global" else ("offsets",))
    tensors = {name: bank["tensors"][name].clone().contiguous() for name in fields}
    extra = {}
    if variant == "separate_angle_transport":
        ids = [s for s in range(bank["metadata"]["angle_bins"]) if s != bank["metadata"]["withheld_angle_bin"]]
        for name in ("table_coefficients", "table_enabled"):
            tensors[name] = bank["tensors"][name][:, ids].clone().contiguous()
        extra["table_angle_bin_ids"] = ids
    elif variant not in ("global", "region_offsets"):
        source = ("fourier_coefficients" if "fourier" in variant else
                  "rotation_coefficients" if "rotation" in variant else "radial_coefficients")
        tensors["coefficients"] = bank["tensors"][source].clone().contiguous()
    metadata = {**bank["metadata"], **extra, "variant": variant,
        "tensor_payload_bytes": sum(value.numel() * value.element_size() for value in tensors.values())}
    return {"metadata": metadata, "tensors": tensors}


def _validate(bank, scope, context, *, geometry_only=False):
    if not isinstance(scope, InputStateScope) or not isinstance(bank, dict):
        raise ValueError("prediction requires an exact scoped angular bank")
    m, t = bank["metadata"], bank["tensors"]
    if m.get("format") != FORMAT or m.get("scope") != scope.to_dict():
        raise ValueError("angular source scope differs")
    _context(context)
    if m.get("response_context_sha256") != context:
        raise ValueError("angular response/base-codec context differs")
    for key, low, high in (("rank", 2, 16), ("regions", 2, 8), ("angle_bins", 4, 4),
                           ("experts", 2, 8), ("output_width", 1, 8192)):
        _integer(m.get(key), key, low, high)
    r, k, s, e, d = (m[key] for key in ("rank", "regions", "angle_bins", "experts", "output_width"))
    _integer(m.get("withheld_angle_bin"), "withheld_angle_bin", 0, s - 1)
    shapes = {"input_mean": (scope.matrix_input_width,), "input_basis": (scope.matrix_input_width, r),
        "feature_scale": (r,), "centers": (k, r), "radii": (k,), "orthogonal_radius": (),
        "rotation_planes": (k, r, 2), "rotation_plane_defined": (k,),
        "radial_coordinate_ranges": (k, 2), "region_enabled": (k,)}
    variant = m.get("variant")
    if not geometry_only:
        if variant not in VARIANTS:
            raise ValueError("unknown angular response variant")
        shapes.update(response_mean=(e, d), slopes=(e, r, d))
        if variant != "global":
            shapes["offsets"] = (e, k, d)
        if variant == "separate_angle_transport":
            shapes.update(table_coefficients=(k, s - 1, r + 1, r), table_enabled=(k, s - 1))
        elif variant not in ("global", "region_offsets"):
            shapes["coefficients"] = (k, 15 if "fourier" in variant else 3, r + 1, r)
        if set(t) != set(shapes):
            raise ValueError("angular tensor fields differ from declared variant")
    for name, shape in shapes.items():
        _finite(t[name], shape, name)
    if (bool((t["feature_scale"] <= 0).any()) or bool((t["radii"] <= 0).any()) or
            float(t["orthogonal_radius"]) <= 0 or
            bool((t["radial_coordinate_ranges"][:, 0] > t["radial_coordinate_ranges"][:, 1]).any())):
        raise ValueError("angular feature scales/radii/ranges are invalid")
    if not bool(((t["region_enabled"] == 0) | (t["region_enabled"] == 1)).all()):
        raise ValueError("angular support flags must be binary")
    if (not bool(((t["rotation_plane_defined"] == 0) | (t["rotation_plane_defined"] == 1)).all()) or
            bool((t["region_enabled"] > t["rotation_plane_defined"]).any())):
        raise ValueError("angular charts cannot be enabled without a defined rotation plane")
    if not torch.allclose(t["rotation_planes"].transpose(1, 2) @ t["rotation_planes"],
                          torch.eye(2).expand(k, 2, 2), atol=2e-5, rtol=2e-5):
        raise ValueError("rotation planes must be orthonormal")
    if not geometry_only and variant == "separate_angle_transport":
        if (m.get("table_angle_bin_ids") != [slot for slot in range(s) if slot != m["withheld_angle_bin"]] or
                not bool(((t["table_enabled"] == 0) | (t["table_enabled"] == 1)).all()) or
                not torch.equal(t["table_enabled"].sum(dim=1), t["region_enabled"] * (s - 1))):
            raise ValueError("compact angular table must exclude withheld sector")
    return m, t


def _geometry(m, t, inputs):
    xc = inputs - t["input_mean"]
    projected = xc @ t["input_basis"]
    z = projected / t["feature_scale"]
    labels = torch.cdist(z, t["centers"]).argmin(dim=1)
    local = z - t["centers"][labels]
    rho = local.norm(dim=1) / t["radii"][labels]
    orthogonal = (xc - projected @ t["input_basis"].T).norm(dim=1)
    phase, fraction = torch.zeros(len(inputs)), torch.zeros(len(inputs))
    valid = torch.zeros(len(inputs), dtype=torch.bool)
    for k in range(m["regions"]):
        mask = labels == k
        if not bool(t["rotation_plane_defined"][k]):
            continue
        phase[mask], fraction[mask], valid[mask] = _plane_geometry(local[mask], t["rotation_planes"][k])
    angle_ids = torch.floor(phase / (2 * math.pi / m["angle_bins"])).long().clamp(max=m["angle_bins"] - 1)
    angle_ids[~valid] = -1
    ranges = t["radial_coordinate_ranges"][labels]
    supported = ((t["region_enabled"][labels] == 1) & valid &
        (rho >= ranges[:, 0] - 1e-5) & (rho <= ranges[:, 1] + 1e-5) &
        (orthogonal <= 1.5 * t["orthogonal_radius"]))
    if not bool(torch.isfinite(z).all() & torch.isfinite(rho).all() & torch.isfinite(orthogonal).all()):
        raise ValueError("angular geometry overflowed")
    diagnostic = {"region_ids": labels, "angle_bin_ids": angle_ids,
        "phase_rad": torch.where(valid, phase, torch.full_like(phase, float("nan"))),
        "radial_coordinates": rho, "plane_fraction": fraction, "phase_valid": valid,
        "supported": supported, "global_fallback_count": int((~supported).sum())}
    return z, local, diagnostic


@torch.no_grad()
def predict_geometry(bank, inputs, *, expected_scope, expected_response_context_sha256=None,
                     max_working_bytes=3 << 30):
    m, t = _validate(bank, expected_scope, expected_response_context_sha256, geometry_only=True)
    _integer(max_working_bytes, "geometry working-byte budget", 1, 16 << 30)
    if len(inputs) > 8192:
        raise ValueError("angular prediction count exceeds bound")
    estimate = sum(value.untyped_storage().nbytes() for value in t.values()) + len(inputs) * expected_scope.matrix_input_width * 16 + len(inputs) * m["rank"] * 64
    if estimate > max_working_bytes:
        raise MemoryError("angular geometry numerical arrays exceed budget")
    _finite(inputs, (len(inputs), expected_scope.matrix_input_width), "prediction inputs")
    return _geometry(m, t, inputs)[2]


def _transport_delta(v, rho, phase, plane, coefficients):
    rotations = _rotation_matrices(phase, plane)
    canonical = torch.einsum("nr,nrs->ns", v, rotations)
    delta = _design(canonical, rho, 2) @ coefficients.reshape(3 * (v.shape[1] + 1), v.shape[1])
    return torch.einsum("nr,nsr->ns", delta, rotations)


@torch.no_grad()
def predict_residual(bank, inputs, *, expected_scope, expected_response_context_sha256=None,
                     max_working_bytes=3 << 30):
    m, t = _validate(bank, expected_scope, expected_response_context_sha256)
    _integer(max_working_bytes, "prediction working-byte budget", 1, 16 << 30)
    n, width, r, e, d = len(inputs), expected_scope.matrix_input_width, m["rank"], m["experts"], m["output_width"]
    if n > 8192:
        raise ValueError("angular prediction count exceeds bound")
    estimate = (sum(value.untyped_storage().nbytes() for value in t.values()) + n * width * 16 +
                e * n * d * 16 + n * r * r * 32 + n * 15 * (r + 1) * 16)
    if estimate > max_working_bytes:
        raise MemoryError("angular prediction numerical arrays exceed budget")
    _finite(inputs, (n, width), "prediction inputs")
    z, local, diagnostic = _geometry(m, t, inputs)
    predicted = t["response_mean"][:, None] + torch.einsum("nr,erd->end", z, t["slopes"])
    variant = m["variant"]
    if variant != "global":
        for k in range(m["regions"]):
            mask = (diagnostic["region_ids"] == k) & diagnostic["supported"]
            if not bool(mask.any()):
                continue
            predicted[:, mask] += t["offsets"][:, k, None]
            if variant == "region_offsets":
                continue
            v, rho, phase = local[mask], diagnostic["radial_coordinates"][mask], diagnostic["phase_rad"][mask]
            if variant in ("rotation_transport", "wrong_rotation_transport"):
                phase = -phase if variant == "wrong_rotation_transport" else phase
                delta = _transport_delta(v, rho, phase, t["rotation_planes"][k], t["coefficients"][k])
            elif variant in ("fourier_transport", "wrong_fourier_transport"):
                phase = phase + math.pi / 2 if variant == "wrong_fourier_transport" else phase
                delta = _fourier_design(v, rho, phase) @ t["coefficients"][k].reshape(15 * (r + 1), r)
            elif variant == "separate_angle_transport":
                ids = torch.tensor(m["table_angle_bin_ids"])
                centers = (ids.float() + .5) * (2 * math.pi / m["angle_bins"])
                difference = torch.remainder(phase[:, None] - centers[None] + math.pi, 2 * math.pi) - math.pi
                # Slots are stored in actual-sector-ID order, so argmin's first
                # exact minimum provides the declared circular tie policy.
                nearest = difference.abs().argmin(dim=1)
                affine = torch.cat((v, torch.ones(len(v), 1)), dim=1)
                delta = torch.einsum("nr,nrs->ns", affine, t["table_coefficients"][k, nearest])
            else:
                delta = _design(v, rho, 2) @ t["coefficients"][k].reshape(3 * (r + 1), r)
            predicted[:, mask] += torch.einsum("nr,erd->end", delta, t["slopes"])
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("angular residual prediction overflowed")
    return predicted, diagnostic
