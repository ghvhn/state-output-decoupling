from dataclasses import replace

import pytest
import torch

from tda.expert_compress import ExpertGroupIdentity
from tda.trajectory_compress import InputStateScope, CalibrationRecord, TrajectoryPair
from scripts.shell_transport_glm import shell_metrics


def example():
    identity = ExpertGroupIdentity("fixture", "a" * 40, 3, "gate_proj", "fixture layout",
                                  ("0", "1"), (2, 4), "float32")
    scope = InputStateScope.from_identity(identity, state_site="post_attention_norm_expert_gate_up_input")
    records = [CalibrationRecord(f"r{i}", "environment", f"trajectory{i // 2}", scope,
                                 torch.zeros(4), origin="synthetic_fixture") for i in range(6)]
    pairs = [TrajectoryPair(f"p{i}", f"r{2 * i}", f"r{2 * i + 1}") for i in range(3)]
    # First pair enters withheld level, second stays there, third avoids it.
    diagnostic = {"shell_ids": torch.tensor([0, 1, 1, 1, 2, 3]),
                  "region_ids": torch.tensor([0, 0, 0, 0, 1, 1]),
                  "supported": torch.tensor([True, True, True, False, True, True])}
    reference = torch.arange(1, 13, dtype=torch.float32).reshape(2, 6, 1)
    return records, pairs, diagnostic, reference


def test_shell_subsets_score_real_endpoints_and_keep_full_split_weights():
    records, pairs, diagnostic, reference = example()
    metrics = shell_metrics(reference, reference + 1, records, pairs, diagnostic, 1, 1)
    assert metrics["output"]["relative_rmse"] > 0
    assert metrics["withheld_shell_records"] == 3
    assert metrics["supported_withheld_shell_records"] == 2
    assert metrics["withheld_shell_touching_pairs"] == 2
    assert metrics["withheld_shell_crossing_pairs"] == 1
    assert metrics["withheld_shell_both_endpoint_pairs"] == 1
    for name in ("output_difference", "withheld_shell_touching_difference",
                 "withheld_shell_crossing_difference", "withheld_shell_both_endpoints_difference"):
        assert metrics[name]["relative_rmse"] == 0
    assert metrics["per_expert"][1]["excluded_from_operator_fit"] is True


def test_shell_crossing_difference_subtracts_each_predicted_endpoint():
    records, pairs, diagnostic, reference = example()
    proposed = reference.clone()
    proposed[:, 1] += 2
    metrics = shell_metrics(reference, proposed, records, pairs, diagnostic, 1, 1)
    assert metrics["withheld_shell_crossing_difference"]["relative_rmse"] == pytest.approx(2)
    assert metrics["withheld_shell_both_endpoints_difference"]["relative_rmse"] == 0


def test_empty_withheld_subsets_are_unavailable_not_zero_error():
    records, pairs, diagnostic, reference = example()
    diagnostic["shell_ids"] = torch.zeros(6, dtype=torch.long)
    metrics = shell_metrics(reference, reference, records, pairs, diagnostic, 1, 1)
    assert metrics["withheld_shell_output"]["relative_rmse"] is None
    assert metrics["withheld_shell_touching_difference"]["relative_rmse"] is None


def test_pair_cannot_cross_semantic_environment_or_trajectory():
    records, pairs, diagnostic, reference = example()
    records[1] = replace(records[1], environment_id="different")
    with pytest.raises(ValueError, match="trajector"):
        shell_metrics(reference, reference, records, pairs, diagnostic, 1, 1)


def test_metric_shapes_and_missing_endpoints_are_checked():
    records, pairs, diagnostic, reference = example()
    with pytest.raises(ValueError, match="shapes"):
        shell_metrics(reference, reference[:, :-1], records, pairs, diagnostic, 1, 1)
    with pytest.raises(ValueError, match="absent"):
        shell_metrics(reference, reference, records, [TrajectoryPair("bad", "r0", "missing")], diagnostic, 1, 1)


def test_reversed_pairs_cannot_be_counted_twice():
    records, pairs, diagnostic, reference = example()
    with pytest.raises(ValueError, match="counted once"):
        shell_metrics(reference, reference, records, [pairs[0], TrajectoryPair("reverse", "r1", "r0")], diagnostic, 1, 1)
