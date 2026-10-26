"""Observable plane rotations and explicit path winding, without hidden turns."""
import math

import pytest
import torch

from tda.rotation_geometry import (fit_rotation_plane, validate_rotation_plane,
    project_directions, principal_relative_angle, unwrap_ordered_path,
    embedded_rotations, minimal_proper_rotation)


def vectors(degrees, rank=2):
    phase = torch.tensor(degrees, dtype=torch.float64) * math.pi / 180
    result = torch.zeros(len(phase), rank, dtype=torch.float64)
    result[:, 0], result[:, 1] = torch.cos(phase), torch.sin(phase)
    return result


def path(degrees, positions=None, **kwargs):
    direction = vectors(degrees)
    measured = project_directions(direction, torch.eye(2, dtype=torch.float64))
    if positions is None:
        positions = list(range(len(degrees)))
    return unwrap_ordered_path(measured["phase_rad"], positions, supported=measured["supported"], **kwargs)


def test_90_and450_same_endpoint_different_ordered_observed_rotation():
    short = path([0, 90])
    long = path([0, 90, 180, 270, 360, 450])
    torch.testing.assert_close(vectors([90]), vectors([450]), atol=1e-15, rtol=0)
    assert short["accumulated_rotation_rad"][-1] == pytest.approx(math.pi / 2)
    assert long["accumulated_rotation_rad"][-1] == pytest.approx(5 * math.pi / 2)
    assert short["branch_excess"][-1] == 0
    assert long["branch_excess"][-1] == 1
    assert short["net_turns"][-1] == pytest.approx(.25)
    assert long["net_turns"][-1] == pytest.approx(1.25)
    assert long["segments"][0]["signed_winding"] is None  # An open path.
    torch.testing.assert_close(short["periodic_and_linear_features"][-1, :2],
                               long["periodic_and_linear_features"][-1, :2], atol=1e-15, rtol=0)
    assert short["periodic_and_linear_features"][-1, 2] != long["periodic_and_linear_features"][-1, 2]
    assert long["absolute_prior_turns_known"] is False


def test_endpoints_and_sparse_positions_never_invent_an_extra_turn():
    relation = principal_relative_angle(vectors([0]), vectors([450]), torch.eye(2, dtype=torch.float64))
    assert relation["angle_rad"][0] == pytest.approx(math.pi / 2)
    assert relation["winding_inferable_from_endpoints"] is False
    sparse = path([0, 450])
    assert sparse["net_turns"][-1] == pytest.approx(.25)
    gap = path([0, 450], positions=[0, 5])
    assert gap["segment_ids"].tolist() == [0, 1]
    assert gap["connection_valid"].tolist() == [False, False]
    assert gap["gap_before"].tolist() == [False, True]
    assert gap["net_turns"].tolist() == [0., 0.]
    assert gap["break_reasons"][-1] == "position_gap"


def test_reversal_closed_winding_and_cancelled_net_rotation():
    clockwise = path([0, -90, -180, -270, -360, -450])
    assert clockwise["accumulated_rotation_rad"][-1] == pytest.approx(-5 * math.pi / 2)
    assert clockwise["branch_excess"][-1] == -1
    assert path([0, 90, 180, 270, 360])["segments"][0]["signed_winding"] == 1
    assert path([0, -90, -180, -270, -360])["segments"][0]["signed_winding"] == -1
    cancel = path([0, 90, 180, 270, 360, 270, 180, 90, 0])
    assert cancel["segments"][0]["signed_winding"] == 0
    assert cancel["net_turns"][-1] == pytest.approx(0., abs=1e-15)
    assert cancel["absolute_angular_travel_rad"][-1] == pytest.approx(4 * math.pi)


@pytest.mark.parametrize("degrees,expected", [([170, 190], 20), ([190, 170], -20),
                                               ([-179, 179], -2), ([179, -179], 2)])
def test_wrap_boundary_is_continuous_and_signed(degrees, expected):
    result = path(degrees)
    assert result["accumulated_rotation_rad"][-1] == pytest.approx(expected * math.pi / 180)
    assert result["connection_valid"][-1]


def test_antipodal_steps_reset_without_dropping_supported_points():
    result = path([0, 180, 270])
    assert result["segment_ids"].tolist() == [0, 1, 1]
    assert result["ambiguous_pi_step"].tolist() == [False, True, False]
    assert result["supported"].tolist() == [True, True, True]
    assert result["accumulated_rotation_rad"].tolist() == pytest.approx([0., 0., math.pi / 2])
    assert result["break_reasons"][1] == "ambiguous_pi_step"
    assert math.isnan(float(result["wrapped_increment_rad"][1]))
    near = path([0, 180 - .00001], ambiguity_tolerance=1e-5)
    assert near["ambiguous_pi_step"][-1]
    endpoint = path([0, 90, 180])
    assert endpoint["connection_valid"][-1]
    assert endpoint["endpoint_principal_ambiguous"][-1]
    assert not endpoint["branch_excess_valid"][-1]


def test_gaps_unsupported_phases_and_scope_chart_changes_never_bridge():
    result = path([0, 90, 180, 270], trajectory_ids=["a/chart0", "a/chart0", "a/chart1", "a/chart0"])
    assert result["segment_ids"].tolist() == [0, 0, 1, 2]
    assert result["break_reasons"][2:] == ["trajectory_change", "trajectory_change"]
    phases = torch.tensor([0., float("nan"), math.pi / 2, math.pi], dtype=torch.float64)
    result = unwrap_ordered_path(phases, [0, 1, 2, 3])
    assert result["segment_ids"].tolist() == [0, -1, 1, 1]
    assert result["connection_valid"].tolist() == [False, False, False, True]
    assert result["periodic_and_linear_features"][1].tolist() == [0., 0., 0.]
    assert result["break_reasons"][2] == "after_unsupported_phase"
    result = path([0, 90, 180, 270], positions=[10, 11, 0, 1], trajectory_ids=["a", "a", "b", "b"])
    assert result["segment_ids"].tolist() == [0, 0, 1, 1]


def test_higher_rank_oriented_plane_and_out_of_plane_support():
    plane = torch.eye(4, dtype=torch.float64)[:, :2]
    direction = vectors([0, 90, 180, 270, 360, 450], rank=4)
    direction[:, 2:] = torch.tensor([[2., -5.], [1000., -1000.], [4., -3.], [1., 2.], [0., 8.], [1., 1.]])
    measured = project_directions(direction, plane)
    result = unwrap_ordered_path(measured["phase_rad"], list(range(6)))
    assert result["net_turns"][-1] == pytest.approx(1.25)
    unsupported = project_directions(torch.tensor([[0., 0., 4., 3.], [0., 0., 0., 0.]], dtype=torch.float64), plane)
    assert unsupported["supported"].tolist() == [False, False]
    assert bool(torch.isnan(unsupported["phase_rad"]).all())
    fractional = project_directions(direction, plane, min_plane_fraction=.1)
    assert not fractional["supported"][1]
    reflected = plane.clone()
    reflected[:, 1] *= -1
    reverse = project_directions(direction, reflected)
    assert unwrap_ordered_path(reverse["phase_rad"], list(range(6)))["net_turns"][-1] == pytest.approx(-1.25)
    rotated = plane @ torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    shifted = project_directions(direction, rotated)
    assert unwrap_ordered_path(shifted["phase_rad"], list(range(6)))["net_turns"][-1] == pytest.approx(1.25)


def test_fit_plane_calibrated_span_signs_and_degenerate_rejection():
    direction = vectors([0, 30, 45, 90, 120, 170], rank=5).float()
    plane = fit_rotation_plane(direction)
    assert plane.shape == (5, 2) and plane.dtype == torch.float32
    torch.testing.assert_close(plane.T @ plane, torch.eye(2), atol=2e-6, rtol=0)
    assert torch.equal(plane, fit_rotation_plane(direction))
    assert not bool(plane[2:].any())
    with pytest.raises(ValueError, match="span"):
        fit_rotation_plane(vectors([0, 0, 0]))
    with pytest.raises(ValueError, match="zero"):
        fit_rotation_plane(torch.zeros(4, 3))
    with pytest.raises(ValueError, match="orthonormal"):
        validate_rotation_plane(torch.tensor([[1., 1.], [0., 0.]]))


def test_large_finite_fp32_directions_normalize_without_overflow():
    measured = project_directions(torch.tensor([[1e38, 1e38], [-1e38, 1e38]]), torch.eye(2))
    assert bool(torch.isfinite(measured["total_radius"]).all())
    assert measured["phase_rad"].tolist() == pytest.approx([math.pi / 4, 3 * math.pi / 4])
    angle = principal_relative_angle(torch.tensor([1e38, 1e38]), torch.tensor([-1e38, 1e38]), torch.eye(2))
    assert angle["angle_rad"] == pytest.approx(math.pi / 2)


def test_tiny_orthogonal_component_is_retained_without_subtractive_cancellation():
    measured = project_directions(torch.tensor([1., 0., 1e-8], dtype=torch.float64),
                                  torch.eye(3, dtype=torch.float64)[:, :2])
    assert measured["orthogonal_radius"] == pytest.approx(1e-8, rel=1e-12, abs=0)


def test_embedded_rotation_row_convention_periodicity_and_properness():
    plane = torch.eye(4, dtype=torch.float64)[:, :2]
    angles = torch.tensor([0., math.pi / 2, 5 * math.pi / 2], dtype=torch.float64)
    matrices = embedded_rotations(plane, angles)
    direction = vectors([0, 90, 450], rank=4)
    direction[:, 2:] = torch.tensor([4., -3.])
    canonical = torch.einsum("nr,nrs->ns", direction, matrices)
    torch.testing.assert_close(canonical[:, :2], torch.tensor([[1., 0.]]).double().expand(3, 2), atol=1e-15, rtol=0)
    assert torch.equal(canonical[:, 2:], direction[:, 2:])
    restored = torch.einsum("nr,nrs->ns", canonical, matrices.transpose(-1, -2))
    torch.testing.assert_close(restored, direction, atol=1e-15, rtol=0)
    torch.testing.assert_close(matrices[1], matrices[2], atol=1e-15, rtol=0)
    torch.testing.assert_close(torch.linalg.det(matrices), torch.ones(3, dtype=torch.float64), atol=1e-15, rtol=0)


def test_minimal_full_vector_rotation_maps_units_and_marks_antipodal_ambiguity():
    source = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    target = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    result = minimal_proper_rotation(source, target)
    matrices = result["matrix"]
    mapped = torch.einsum("nrs,ns->nr", matrices, source)
    torch.testing.assert_close(mapped, target, atol=1e-15, rtol=0)
    torch.testing.assert_close(matrices.transpose(-1, -2) @ matrices, torch.eye(3).double().expand(3, 3, 3), atol=1e-15, rtol=0)
    torch.testing.assert_close(torch.linalg.det(matrices), torch.ones(3, dtype=torch.float64), atol=1e-15, rtol=0)
    assert result["antipodal_plane_underdetermined"].tolist() == [False, True, False]
    assert result["winding_inferable_from_endpoints"] is False


def test_tiny_nonzero_full_rotation_retains_its_measured_direction():
    source = torch.tensor([1., 0., 0.], dtype=torch.float64)
    target = torch.tensor([1., 1e-13, 0.], dtype=torch.float64)
    result = minimal_proper_rotation(source, target)
    torch.testing.assert_close(result["matrix"] @ source, target, atol=1e-26, rtol=0)
    assert result["unsigned_angle_rad"] == pytest.approx(1e-13, abs=1e-26)
    assert not result["antipodal_plane_underdetermined"]


def test_input_order_shape_dtype_and_budget_guards():
    for positions in ([1, 0], [1, 1], [0, True], [0, .5], [0], [-1, 0]):
        with pytest.raises(ValueError):
            unwrap_ordered_path(torch.zeros(2), positions)
    with pytest.raises(ValueError):
        unwrap_ordered_path(torch.zeros(2), torch.tensor([0., 1.]))
    with pytest.raises(ValueError):
        unwrap_ordered_path(torch.zeros(2), [0, 1], supported=torch.ones(2))
    with pytest.raises(ValueError):
        project_directions(torch.zeros(2, 3), torch.eye(2))
    with pytest.raises(ValueError):
        principal_relative_angle(torch.ones(2), torch.ones(1, 2), torch.eye(2))
    with pytest.raises(ValueError):
        minimal_proper_rotation(torch.tensor([2., 0.]), torch.tensor([0., 1.]))
    for operation in (lambda: fit_rotation_plane(vectors([0, 90]), max_working_bytes=1),
                      lambda: project_directions(vectors([0, 90]), torch.eye(2), max_working_bytes=1),
                      lambda: unwrap_ordered_path(torch.zeros(2), [0, 1], max_working_bytes=1),
                      lambda: embedded_rotations(torch.eye(2), torch.tensor([0.]), max_working_bytes=1),
                      lambda: minimal_proper_rotation(torch.tensor([1., 0.]), torch.tensor([0., 1.]), max_working_bytes=1)):
        with pytest.raises(MemoryError):
            operation()


def test_empty_path_no_fabricated_segment():
    result = unwrap_ordered_path(torch.empty(0), [])
    assert result["segments"] == []
    assert result["periodic_and_linear_features"].shape == (0, 3)


def test_relative_angle_budget_counts_both_endpoint_backing_storages():
    source = torch.ones(1000, dtype=torch.float64)[:2]
    target = torch.ones(1000, dtype=torch.float64)[:2]
    # Each endpoint fits independently, but their combined backing storage does not.
    with pytest.raises(MemoryError):
        principal_relative_angle(source, target, torch.eye(2, dtype=torch.float64), max_working_bytes=14000)
