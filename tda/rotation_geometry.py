"""Oriented-plane angles and winding of explicitly ordered sampled paths.

An endpoint relationship supplies a principal angle, never an omitted turn.
Path accumulation uses principal increments on consecutive observed positions;
gaps, unsupported phases and ambiguous antipodal steps terminate continuity.
Returned turns are relative to each valid segment, not an absolute past state.
"""
from __future__ import annotations

import math

import torch


_DTYPES = (torch.float32, torch.float64)
_TAU = 2 * math.pi


def _integer(value, name, low, high=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < low or
            high is not None and value > high):
        raise ValueError(f"invalid bounded integer {name}")


def _number(value, name, low, high=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < low or high is not None and value > high):
        raise ValueError(f"invalid finite parameter {name}")


def _directions(value, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in _DTYPES or
            value.ndim not in (1, 2) or not 2 <= value.shape[-1] <= 128):
        raise ValueError(f"{name} must be CPU FP32/FP64 direction vectors of rank 2..128")
    rows = value.unsqueeze(0) if value.ndim == 1 else value
    if not 1 <= len(rows) <= 8192 or not bool(torch.isfinite(rows).all()):
        raise ValueError(f"{name} must contain bounded nonempty finite vectors")
    return rows


def _budget(values, temporary_bytes, maximum):
    _integer(maximum, "max_working_bytes", 1)
    # Backing storage is counted, including small views of much larger arrays.
    persistent = sum(value.untyped_storage().nbytes() for value in values)
    if persistent + temporary_bytes > maximum:
        raise MemoryError("rotation geometry arrays exceed their declared working budget")


def _units(rows):
    x = rows.double()
    scale = x.abs().amax(dim=1)
    scaled = x / torch.where(scale > 0, scale, torch.ones_like(scale))[:, None]
    length = torch.linalg.vector_norm(scaled, dim=1)
    unit = scaled / torch.where(length > 0, length, torch.ones_like(length))[:, None]
    radius = scale * length
    if not bool(torch.isfinite(radius).all()):
        raise ValueError("direction magnitude exceeds FP64 geometry range")
    return unit, radius


def validate_rotation_plane(plane, *, orthonormal_tolerance=2e-5):
    """Require an explicit oriented orthonormal[R,2] basis, not a projector."""
    _number(orthonormal_tolerance, "orthonormal_tolerance", 0, .001)
    if (not isinstance(plane, torch.Tensor) or plane.device.type != "cpu" or plane.dtype not in _DTYPES or
            plane.ndim != 2 or not 2 <= plane.shape[0] <= 128 or plane.shape[1] != 2 or
            not bool(torch.isfinite(plane).all())):
        raise ValueError("plane must be finite CPU FP32/FP64[R,2] data")
    gram = plane.double().T @ plane.double()
    if not torch.allclose(gram, torch.eye(2, dtype=torch.float64), atol=orthonormal_tolerance, rtol=0):
        raise ValueError("rotation plane must have orthonormal oriented columns")
    return plane


@torch.no_grad()
def fit_rotation_plane(calibration_directions, *, svd_rtol=1e-8,
                       max_working_bytes=128 << 20):
    """Uncentered PCA of unit calibration directions; never add heldout rows.

    The caller determines the calibrated source scope. Signs are canonicalized
    for reproducibility; nearly degenerate eigenspaces need not be uniquely
    identified, so saved plane coordinates must accompany later measurements.
    """
    rows = _directions(calibration_directions, "calibration directions")
    _number(svd_rtol, "svd_rtol", 0, .1)
    n, rank = rows.shape
    _budget((calibration_directions,), n * rank * 64 + rank * rank * 64, max_working_bytes)
    unit, radii = _units(rows)
    if bool((radii == 0).any()):
        raise ValueError("zero calibration vectors cannot define a direction plane")
    values, vectors = torch.linalg.eigh(unit.T @ unit)
    if float(values[-2]) <= max(float(values[-1]) * svd_rtol, 1e-12):
        raise ValueError("calibration direction span does not support a 2-plane")
    plane = vectors[:, -2:].flip(1)
    pivots = plane.abs().argmax(dim=0)
    plane *= torch.sign(plane[pivots, torch.arange(2)])[None, :]
    plane = plane.to(calibration_directions.dtype).contiguous()
    return validate_rotation_plane(plane)


@torch.no_grad()
def project_directions(directions, plane, *, min_projection_norm=1e-8,
                       min_plane_fraction=0., max_working_bytes=128 << 20):
    """Measure phase and radius in one explicit plane, retaining support flags.

    ``plane_fraction`` is the projected/full norm ratio. Phase is undefined for
    a zero vector or negligible projected radius; those entries receive NaN.
    All measured coordinates/radii are FP64, without overflowing FP32 norms.
    """
    rows = _directions(directions, "directions")
    validate_rotation_plane(plane)
    if rows.shape[1] != len(plane):
        raise ValueError("direction rank differs from rotation plane")
    _number(min_projection_norm, "min_projection_norm", 0)
    _number(min_plane_fraction, "min_plane_fraction", 0, 1)
    _budget((directions, plane), rows.numel() * 64 + len(rows) * 128, max_working_bytes)
    unit, radius = _units(rows)
    coordinates = unit @ plane.double()
    fraction = torch.linalg.vector_norm(coordinates, dim=1)
    projected_radius = radius * fraction
    supported = ((radius > 0) & (projected_radius > min_projection_norm) &
                 (fraction > 1e-12) & (fraction >= min_plane_fraction))
    plane_unit = coordinates / torch.where(fraction > 0, fraction, torch.ones_like(fraction))[:, None]
    phase = torch.atan2(plane_unit[:, 1], plane_unit[:, 0])
    phase = torch.where(supported, phase, torch.full_like(phase, float("nan")))
    # Measure the orthogonal residual directly: sqrt(1-fraction**2) cancels
    # small but real components when the direction lies almost in the plane.
    orthogonal_radius = radius * torch.linalg.vector_norm(unit - coordinates @ plane.double().T, dim=1)
    report = {"phase_rad": phase, "unit_coordinates": plane_unit, "total_radius": radius,
              "projected_radius": projected_radius, "orthogonal_radius": orthogonal_radius,
              "plane_fraction": fraction, "supported": supported}
    if directions.ndim == 1:
        report = {key: value[0] for key, value in report.items()}
    return report


@torch.no_grad()
def principal_relative_angle(source, target, plane, *, ambiguity_tolerance=1e-6,
                             min_projection_norm=1e-8, min_plane_fraction=0.,
                             max_working_bytes=128 << 20):
    """Pointwise signed principal angle in [-π,π]; no winding from endpoints.

    At antipodal projections the angle magnitude is π but its direction is
    ambiguous. Unsupported endpoints yield NaN; ambiguity is a separate flag.
    """
    if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor) or source.shape != target.shape:
        raise ValueError("principal-angle endpoints must have identical vector shapes")
    _number(ambiguity_tolerance, "ambiguity_tolerance", 0, .01)
    rows = _directions(source, "source direction")
    _directions(target, "target direction")
    validate_rotation_plane(plane)
    _budget((source, target, plane), rows.numel() * 128 + len(rows) * 256, max_working_bytes)
    a = project_directions(source, plane, min_projection_norm=min_projection_norm,
        min_plane_fraction=min_plane_fraction, max_working_bytes=max_working_bytes)
    b = project_directions(target, plane, min_projection_norm=min_projection_norm,
        min_plane_fraction=min_plane_fraction, max_working_bytes=max_working_bytes)
    pa, pb = a["unit_coordinates"], b["unit_coordinates"]
    dot = (pa * pb).sum(dim=-1).clamp(-1, 1)
    cross = pa[..., 0] * pb[..., 1] - pa[..., 1] * pb[..., 0]
    supported = a["supported"] & b["supported"]
    angle = torch.atan2(cross, dot)
    ambiguous = supported & ((math.pi - angle.abs()).abs() <= ambiguity_tolerance)
    angle = torch.where(supported, angle, torch.full_like(angle, float("nan")))
    return {"angle_rad": angle, "supported": supported, "antipodal_ambiguous": ambiguous,
            "dot": dot, "cross": cross, "winding_inferable_from_endpoints": False}


def _positions(actual_positions, n):
    if isinstance(actual_positions, torch.Tensor):
        if (actual_positions.device.type != "cpu" or actual_positions.ndim != 1 or
                actual_positions.dtype not in (torch.int16, torch.int32, torch.int64)):
            raise ValueError("actual positions require a 1D CPU integer tensor")
        values = actual_positions.tolist()
    elif isinstance(actual_positions, (list, tuple)):
        values = list(actual_positions)
    else:
        raise ValueError("actual positions require an explicit ordered integer sequence")
    if len(values) != n or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > 2**63 - 1 for x in values):
        raise ValueError("actual positions must match phase count and be nonnegative integers")
    return values


@torch.no_grad()
def unwrap_ordered_path(phases, actual_positions, *, supported=None, trajectory_ids=None,
                        ambiguity_tolerance=1e-6, closure_tolerance=1e-6,
                        max_records=8192, max_working_bytes=128 << 20):
    """Lift an observed path using explicit consecutive-position connections.

    Every segment starts with accumulated rotation zero. Angles are accumulated
    only across supported, consecutive points of the same trajectory and
    unambiguous steps smaller than π. Unobserved extra rotations between samples
    are not recoverable. Integer winding is reported only for observed closure.
    Open-path branch excess is distinct from a count of completed loops.
    """
    _integer(max_records, "max_records", 1, 8192)
    _number(ambiguity_tolerance, "ambiguity_tolerance", 0, .01)
    _number(closure_tolerance, "closure_tolerance", 0, .01)
    if (not isinstance(phases, torch.Tensor) or phases.device.type != "cpu" or
            phases.dtype not in _DTYPES or phases.ndim != 1 or len(phases) > max_records):
        raise ValueError("path phases require a bounded 1D CPU FP32/FP64 tensor")
    n = len(phases)
    values = (phases,) + (() if not isinstance(supported, torch.Tensor) else (supported,))
    _budget(values, n * 512, max_working_bytes)
    positions = _positions(actual_positions, n)
    if trajectory_ids is None:
        trajectory_ids = ["path"] * n
    if (not isinstance(trajectory_ids, (list, tuple)) or len(trajectory_ids) != n or
            any(not isinstance(x, str) or not x or len(x) > 256 or "\x00" in x for x in trajectory_ids)):
        raise ValueError("trajectory IDs must be explicit nonempty strings aligned with phases")
    for i in range(1, n):
        if trajectory_ids[i] == trajectory_ids[i - 1] and positions[i] <= positions[i - 1]:
            raise ValueError("actual positions must strictly increase within an ordered trajectory")
    if supported is None:
        valid = torch.isfinite(phases)
    else:
        if (not isinstance(supported, torch.Tensor) or supported.device.type != "cpu" or
                supported.dtype != torch.bool or supported.shape != phases.shape):
            raise ValueError("phase support requires an aligned CPU Boolean mask")
        valid = supported & torch.isfinite(phases)
    raw = phases.double()
    principal = torch.atan2(torch.sin(raw), torch.cos(raw))
    principal[~valid] = float("nan")
    segment_ids = torch.full((n,), -1, dtype=torch.long)
    connection = torch.zeros(n, dtype=torch.bool)
    gaps = torch.zeros(n, dtype=torch.bool)
    ambiguous = torch.zeros(n, dtype=torch.bool)
    increments = torch.full((n,), float("nan"), dtype=torch.float64)
    rotation = torch.full_like(increments, float("nan"))
    travel = torch.full_like(increments, float("nan"))
    unwrapped = torch.full_like(increments, float("nan"))
    relative_principal = torch.full_like(increments, float("nan"))
    endpoint_ambiguous = torch.zeros(n, dtype=torch.bool)
    branch = torch.zeros(n, dtype=torch.long)
    branch_valid = torch.zeros(n, dtype=torch.bool)
    reasons = [None] * n
    segment = -1
    first = None
    previous = None
    summaries = []

    def finish(last):
        if first is None:
            return
        closed = last > first and abs(float(relative_principal[last])) <= closure_tolerance
        summaries.append({"segment_id": segment, "trajectory_id": trajectory_ids[first],
            "start_index": first, "end_index": last, "start_position": positions[first],
            "end_position": positions[last], "samples": last - first + 1,
            "accumulated_rotation_rad": float(rotation[last]), "net_turns": float(rotation[last]) / _TAU,
            "absolute_angular_travel_rad": float(travel[last]), "observed_closed": closed,
            "signed_winding": round(float(rotation[last]) / _TAU) if closed else None})

    for i in range(n):
        if i:
            gaps[i] = trajectory_ids[i] == trajectory_ids[i - 1] and positions[i] != positions[i - 1] + 1
        if not bool(valid[i]):
            if previous is not None:
                finish(previous)
            reasons[i] = "unsupported_phase"
            first = previous = None
            continue
        reason = None
        delta = None
        if previous is None:
            reason = "start" if i == 0 else "after_unsupported_phase"
        elif trajectory_ids[i] != trajectory_ids[previous]:
            reason = "trajectory_change"
        elif positions[i] != positions[previous] + 1:
            reason = "position_gap"
        else:
            difference = float(principal[i] - principal[previous])
            delta = math.atan2(math.sin(difference), math.cos(difference))
            if abs(math.pi - abs(delta)) <= ambiguity_tolerance:
                ambiguous[i] = True
                reason = "ambiguous_pi_step"
        if reason is not None:
            if previous is not None:
                finish(previous)
            segment += 1
            first = i
            rotation[i] = travel[i] = 0
            unwrapped[i] = principal[i]
            reasons[i] = reason
        else:
            connection[i] = True
            increments[i] = delta
            rotation[i] = rotation[previous] + delta
            travel[i] = travel[previous] + abs(delta)
            unwrapped[i] = principal[first] + rotation[i]
        segment_ids[i] = segment
        difference = float(principal[i] - principal[first])
        endpoint = math.atan2(math.sin(difference), math.cos(difference))
        relative_principal[i] = endpoint
        endpoint_ambiguous[i] = abs(math.pi - abs(endpoint)) <= ambiguity_tolerance
        branch_valid[i] = not bool(endpoint_ambiguous[i])
        if bool(branch_valid[i]):
            branch[i] = round((float(rotation[i]) - endpoint) / _TAU)
        previous = i
    if previous is not None:
        finish(previous)
    turns = rotation / _TAU
    features = torch.stack((torch.sin(principal), torch.cos(principal), turns), dim=1)
    features[~valid] = 0
    return {"principal_phase_rad": principal, "supported": valid, "segment_ids": segment_ids,
        "connection_valid": connection, "gap_before": gaps, "ambiguous_pi_step": ambiguous,
        "wrapped_increment_rad": increments, "accumulated_rotation_rad": rotation,
        "unwrapped_phase_rad": unwrapped, "net_turns": turns,
        "absolute_angular_travel_rad": travel, "principal_from_segment_start_rad": relative_principal,
        "endpoint_principal_ambiguous": endpoint_ambiguous, "branch_excess": branch,
        "branch_excess_valid": branch_valid, "periodic_and_linear_features": features,
        "break_reasons": reasons, "segments": summaries, "absolute_prior_turns_known": False,
        "sampling_convention": "sum signed principal increments only on explicit consecutive observed positions",
        "feature_columns": ["sin_principal_phase", "cos_principal_phase", "segment_net_turns"]}


@torch.no_grad()
def embedded_rotations(plane, angle_rad, *, max_working_bytes=128 << 20):
    """Embed standard column rotations of an oriented plane into full rank.

    R=I+(cosθ−1)(pp.T+qq.T)+sinθ(qp.T−pq.T). A row vector v at
    phase θ is canonicalized by v@Rθ; @Rθ.T restores that direction.
    The matrix is periodic: observed winding belongs to separate path features.
    Always returns FP64[N,R,R], including one matrix for a scalar angle.
    """
    validate_rotation_plane(plane)
    if isinstance(angle_rad, (int, float)) and not isinstance(angle_rad, bool):
        angles = torch.tensor([angle_rad], dtype=torch.float64)
    elif isinstance(angle_rad, torch.Tensor):
        if angle_rad.device.type != "cpu" or angle_rad.dtype not in _DTYPES or angle_rad.ndim not in (0, 1):
            raise ValueError("rotation angles must be scalar or 1D CPU FP32/FP64 data")
        angles = angle_rad.reshape(-1)
    else:
        raise ValueError("rotation angles require explicit numeric radians")
    if not 1 <= len(angles) <= 8192 or not bool(torch.isfinite(angles).all()):
        raise ValueError("rotation angles must be bounded, nonempty and finite")
    rank = len(plane)
    _budget((plane, angles), len(angles) * rank * rank * 64 + rank * rank * 64, max_working_bytes)
    p, q = plane.double().unbind(dim=1)
    projector = torch.outer(p, p) + torch.outer(q, q)
    generator = torch.outer(q, p) - torch.outer(p, q)
    angles = angles.double()
    return (torch.eye(rank, dtype=torch.float64)[None, :, :] +
            (torch.cos(angles) - 1)[:, None, None] * projector +
            torch.sin(angles)[:, None, None] * generator)


@torch.no_grad()
def minimal_proper_rotation(source, target, *, unit_tolerance=2e-5,
                            max_working_bytes=128 << 20):
    """Minimal full-vector proper rotations, distinct from signed plane phases.

    Antipodal vectors do not identify a unique rotation plane: use a declared
    deterministic orthogonal axis and mark that case. No path winding is known.
    """
    a, b = _directions(source, "source unit directions"), _directions(target, "target unit directions")
    if source.shape != target.shape:
        raise ValueError("full-vector rotation endpoints must have identical shapes")
    _number(unit_tolerance, "unit_tolerance", 0, .001)
    n, rank = a.shape
    _budget((source, target), n * rank * rank * 64 + n * rank * 128, max_working_bytes)
    u, an = _units(a)
    v, bn = _units(b)
    if bool(((an - 1).abs() > unit_tolerance).any() | ((bn - 1).abs() > unit_tolerance).any()):
        raise ValueError("minimal proper rotation requires full unit vectors")
    dot = (u * v).sum(dim=1).clamp(-1, 1)
    orthogonal = v - dot[:, None] * u
    sine = torch.linalg.vector_norm(orthogonal, dim=1)
    antipodal = (sine <= 1e-12) & (dot < 0)
    # Small nonzero rotations still carry a measured direction. The antipodal
    # ambiguity threshold does not justify discarding a tiny parallel rotation.
    w = orthogonal / torch.where(sine > 0, sine, torch.ones_like(sine))[:, None]
    for i in torch.nonzero(antipodal, as_tuple=False).flatten().tolist():
        axis = torch.zeros(rank, dtype=torch.float64)
        axis[int(u[i].abs().argmin())] = 1
        direction = axis - (axis @ u[i]) * u[i]
        w[i] = direction / direction.norm()
    uu = u[:, :, None] * u[:, None, :]
    ww = w[:, :, None] * w[:, None, :]
    skew = w[:, :, None] * u[:, None, :] - u[:, :, None] * w[:, None, :]
    matrices = (torch.eye(rank, dtype=torch.float64)[None, :, :] +
                (dot - 1)[:, None, None] * (uu + ww) + sine[:, None, None] * skew)
    report = {"matrix": matrices, "unsigned_angle_rad": torch.atan2(sine, dot),
              "antipodal_plane_underdetermined": antipodal, "winding_inferable_from_endpoints": False}
    if source.ndim == 1:
        report = {key: value[0] if isinstance(value, torch.Tensor) else value for key, value in report.items()}
    return report
