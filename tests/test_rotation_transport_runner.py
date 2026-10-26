import math
from dataclasses import replace

import pytest
import torch

from scripts.rotation_transport_glm import angular_metrics, winding_audit
from tda.expert_compress import ExpertGroupIdentity
from tda.trajectory_compress import CalibrationRecord, InputStateScope, TrajectoryPair


def fixture(n=6, *, trajectory="trajectory"):
    identity = ExpertGroupIdentity("fixture", "a" * 40, 3, "gate_proj", "fixture layout",
                                  ("0", "1"), (2, 4), "float32")
    scope = InputStateScope.from_identity(identity, state_site="post_attention_norm_expert_gate_up_input")
    records = [CalibrationRecord(f"r{i}", "environment", trajectory, scope, torch.zeros(4),
                                 origin="synthetic_fixture") for i in range(n)]
    evidence = {"records": [{"record_id": r.record_id, "split": "heldout", "token_position": i}
                            for i, r in enumerate(records)]}
    diagnostic = {"phase_rad": torch.zeros(n, dtype=torch.float64),
                  "angle_bin_ids": torch.zeros(n, dtype=torch.long),
                  "region_ids": torch.zeros(n, dtype=torch.long),
                  "supported": torch.ones(n, dtype=torch.bool)}
    return records, evidence, diagnostic


def test_angular_metrics_score_measured_endpoints_and_mark_empty_sectors():
    records, _, diagnostic = fixture()
    diagnostic["angle_bin_ids"] = torch.tensor([0, 1, 1, 1, 2, 3])
    diagnostic["supported"][3] = False
    pairs = [TrajectoryPair(f"p{i}", f"r{2*i}", f"r{2*i+1}") for i in range(3)]
    reference = torch.arange(1, 13, dtype=torch.float32).reshape(2, 6, 1)
    proposed = reference.clone()
    proposed[:, 1] += 2
    report = angular_metrics(reference, proposed, records, pairs, diagnostic, 1, 1)
    assert report["withheld_angle_records"] == 3
    assert report["supported_withheld_angle_records"] == 2
    assert report["withheld_angle_crossing_pairs"] == 1
    assert report["withheld_angle_both_endpoint_pairs"] == 1
    assert report["withheld_angle_crossing_difference"]["relative_rmse"] == pytest.approx(2)
    assert report["withheld_angle_both_endpoints_difference"]["relative_rmse"] == 0
    assert report["per_expert"][1]["excluded_from_operator_fit"] is True
    diagnostic["angle_bin_ids"].zero_()
    empty = angular_metrics(reference, proposed, records, pairs, diagnostic, 1, 1)
    assert empty["withheld_angle_output"]["relative_rmse"] is None


def test_winding_retains_multi_turn_order_even_with_identical_endpoints():
    records, evidence, diagnostic = fixture(11)
    raw = torch.arange(11, dtype=torch.float64) * math.pi / 4
    diagnostic["phase_rad"] = torch.atan2(raw.sin(), raw.cos())
    # Record array order is not token order. The evidence determines the path.
    order = torch.tensor([10, 2, 6, 0, 3, 9, 1, 8, 7, 5, 4])
    reordered = [records[i] for i in order.tolist()]
    diagnostic = {k: v[order] for k, v in diagnostic.items()}
    report = winding_audit(reordered, evidence, diagnostic, "heldout")
    assert report["valid_connections"] == 10
    assert report["longest_supported_segment"] == 11
    assert report["trajectories_with_observed_segment_over_360"] == 1
    assert report["max_abs_accumulated_degrees"] == pytest.approx(450)
    last = report["observations"][-1]
    assert last["net_turns"] == pytest.approx(1.25)
    assert last["branch_excess"] == 1
    assert report["absolute_prior_winding_known"] is False
    assert report["winding_used_to_fit_response_maps"] is False


@pytest.mark.parametrize("change", ["gap", "chart", "environment", "undefined"])
def test_winding_resets_instead_of_joining_missing_or_incompatible_path(change):
    records, evidence, diagnostic = fixture(11)
    raw = torch.arange(11, dtype=torch.float64) * math.pi / 4
    diagnostic["phase_rad"] = torch.atan2(raw.sin(), raw.cos())
    if change == "gap":
        for row in evidence["records"][5:]:
            row["token_position"] += 10
    elif change == "chart":
        diagnostic["region_ids"][5:] = 1
    elif change == "environment":
        records[5:] = [replace(r, environment_id="other") for r in records[5:]]
    else:
        diagnostic["phase_rad"][5] = float("nan")
    report = winding_audit(records, evidence, diagnostic, "heldout")
    assert report["trajectories_with_observed_segment_over_360"] == 0
    assert report["longest_supported_segment"] <= 6
    assert report["max_abs_accumulated_degrees"] <= 225 + 1e-6


def test_winding_requires_matching_position_evidence():
    records, evidence, diagnostic = fixture()
    evidence["records"].pop()
    with pytest.raises(ValueError, match="evidence"):
        winding_audit(records, evidence, diagnostic, "heldout")
