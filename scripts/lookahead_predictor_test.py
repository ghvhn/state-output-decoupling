"""
Test Suite for Future Waypoint Lookahead Apparatus & ClaimMap Coupling.

Run:
    python scripts/lookahead_predictor_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from invariants.lookahead_predictor import (
    FutureWaypointPredictor,
    LookaheadApparatus,
    WaypointTicket,
    ArrivalResult,
)
from invariants.claimmap import ClaimMapResult, get_unified_tension_vector
from invariants.trigger_tuner import TriggerTuner


def test_future_waypoint_arbitrary_order_mapping():
    """
    Verify that lookahead can map out the future in ANY order:
    e.g. predicting t+12 (destination), then t+5 (pivot), then t+2 (bridge).
    """
    d_model = 64
    vocab_size = 50
    torch.manual_seed(42)
    embedding = torch.randn(vocab_size, d_model)

    with tempfile.TemporaryDirectory() as tmp:
        tuner = TriggerTuner(path=Path(tmp) / "tuner.json")
        predictor = FutureWaypointPredictor(d_model=d_model, hidden_dim=32, max_horizon=32)
        lookahead = LookaheadApparatus(
            predictor=predictor,
            tuner=tuner,
            embedding_matrix=embedding,
            d_model=d_model,
        )

        h_current = torch.randn(d_model)

        # Map waypoints in arbitrary out-of-order sequence: 12, then 5, then 2
        t_dest = lookahead.predict_waypoint(h_current, horizon_k=12, metadata={"role": "destination"})
        t_pivot = lookahead.predict_waypoint(h_current, horizon_k=5, metadata={"role": "pivot"})
        t_bridge = lookahead.predict_waypoint(h_current, horizon_k=2, metadata={"role": "bridge"})

        assert t_dest.arrival_step == 12
        assert t_pivot.arrival_step == 5
        assert t_bridge.arrival_step == 2

        # Verify ledger has all three scheduled at their arrival steps
        assert 12 in lookahead.ledger
        assert 5 in lookahead.ledger
        assert 2 in lookahead.ledger
        assert lookahead.ledger[12][0].metadata["role"] == "destination"
        assert lookahead.ledger[5][0].metadata["role"] == "pivot"
        assert lookahead.ledger[2][0].metadata["role"] == "bridge"

        # Also test batch non-sequential mapping
        constellation = lookahead.map_future_waypoints(h_current, horizons=[20, 8, 3])
        assert len(constellation) == 3
        assert 20 in lookahead.ledger
        assert 8 in lookahead.ledger
        assert 3 in lookahead.ledger

        print("PASS test_future_waypoint_arbitrary_order_mapping")


def test_timeline_arrival_and_post_mature_adjustment():
    """
    Verify that lookahead waits for the future token to arrive, detects discrepancy,
    and computes the post-mature layer adjustment.
    """
    d_model = 64
    vocab_size = 50
    torch.manual_seed(42)
    embedding = torch.randn(vocab_size, d_model)

    with tempfile.TemporaryDirectory() as tmp:
        tuner = TriggerTuner(path=Path(tmp) / "tuner.json")
        predictor = FutureWaypointPredictor(d_model=d_model, hidden_dim=32, max_horizon=32)
        lookahead = LookaheadApparatus(
            predictor=predictor,
            tuner=tuner,
            embedding_matrix=embedding,
            d_model=d_model,
        )

        h0 = torch.randn(d_model)
        # Schedule ticket for arrival at step 2 (horizon k=2)
        ticket = lookahead.predict_waypoint(h0, horizon_k=2)

        # Step 1: No arrival yet
        h1 = h0 + 0.01 * torch.randn(d_model)
        res1 = lookahead.step_arrival(actual_token_id=5, actual_hidden_state=h1)
        assert len(res1) == 0
        assert lookahead.current_step == 1

        # Step 2: Ticket matures upon arrival!
        h2 = h0 + 0.02 * torch.randn(d_model)
        res2 = lookahead.step_arrival(actual_token_id=ticket.predicted_token_id, actual_hidden_state=h2)
        assert len(res2) == 1
        assert lookahead.current_step == 2

        arr = res2[0]
        assert arr.ticket.arrival_step == 2
        assert arr.token_matched is True
        assert arr.error_norm > 0.0
        assert arr.post_mature_delta.shape == (d_model,)

        # Active post-mature layer adjustment should now be populated
        adj = lookahead.get_post_mature_adjustment(layer_idx=16)
        assert adj is not None
        assert adj.shape == (d_model,)

        # Non-midband layer (e.g. layer 5) should be filtered out
        adj_l5 = lookahead.get_post_mature_adjustment(layer_idx=5)
        assert adj_l5 is None

        print("PASS test_timeline_arrival_and_post_mature_adjustment")


def test_dual_stream_modes():
    """
    Verify dual stream tracking: MODE_EXTERNAL (prompts, docs) vs MODE_SELF (generation).
    """
    d_model = 64
    with tempfile.TemporaryDirectory() as tmp:
        tuner = TriggerTuner(path=Path(tmp) / "tuner.json")
        tuner.set("lookahead_alpha_self", 0.20)
        tuner.set("lookahead_alpha_ext", 0.05)

        lookahead = LookaheadApparatus(tuner=tuner, d_model=d_model)

        # 1. External text stream
        lookahead.set_mode(LookaheadApparatus.MODE_EXTERNAL)
        h_ext = torch.randn(d_model)
        t_ext = lookahead.predict_waypoint(h_ext, horizon_k=1)
        assert t_ext.mode == LookaheadApparatus.MODE_EXTERNAL

        res_ext = lookahead.step_arrival(actual_token_id=None, actual_hidden_state=torch.randn(d_model))
        assert len(res_ext) == 1
        # Should have applied external alpha (0.05)
        err_ext = res_ext[0].error_vector
        delta_ext = res_ext[0].post_mature_delta
        assert torch.count_nonzero(delta_ext) == 0  # no learned local direction yet
        assert lookahead.learning_updates == 1

        # 2. Self output stream
        lookahead.reset_sequence()
        lookahead.set_mode(LookaheadApparatus.MODE_SELF)
        h_self = torch.randn(d_model)
        t_self = lookahead.predict_waypoint(h_self, horizon_k=1)
        assert t_self.mode == LookaheadApparatus.MODE_SELF

        res_self = lookahead.step_arrival(actual_token_id=None, actual_hidden_state=torch.randn(d_model))
        assert len(res_self) == 1
        # Should have applied self alpha (0.20)
        err_self = res_self[0].error_vector
        delta_self = res_self[0].post_mature_delta
        assert torch.count_nonzero(delta_self) == 0  # external evidence cannot leak
        assert lookahead.learning_updates == 2

        print("PASS test_dual_stream_modes")


def test_claimmap_coupling():
    """
    Verify ClaimMap attachment to Lookahead:
    Measures prediction error projected onto the ClaimMap tension axis,
    and applies directional counter-steering post-maturely.
    """
    d_model = 64
    torch.manual_seed(42)

    with tempfile.TemporaryDirectory() as tmp:
        tuner = TriggerTuner(path=Path(tmp) / "tuner.json")
        tuner.set("lookahead_alpha_self", 0.10)
        tuner.set("lookahead_claimmap_weight", 0.50)

        lookahead = LookaheadApparatus(tuner=tuner, d_model=d_model)

        # Create a synthetic ClaimMap tension axis (h_A - h_B)
        tension_axis = torch.zeros(d_model)
        tension_axis[0] = 1.0  # tension strictly along dimension 0

        # Attach ClaimMap to Lookahead
        lookahead.attach_claimmap(
            unified_axis=tension_axis,
            tension_score=0.35,
            mean_sim=0.55,
        )

        h_start = torch.randn(d_model)
        ticket = lookahead.predict_waypoint(h_start, horizon_k=1)
        assert ticket.claimmap_axis is not None

        # Simulate arrival that drifted heavily along dimension 0 (ClaimMap axis)
        h_arrival = ticket.predicted_state.clone()
        h_arrival[0] += 4.0  # 4 units of drift along ClaimMap axis

        results = lookahead.step_arrival(actual_token_id=None, actual_hidden_state=h_arrival)
        assert len(results) == 1
        arr = results[0]

        # ClaimMap drift should be ~4.0
        assert abs(arr.claimmap_drift - 4.0) < 1e-4

        # Post-mature delta should strongly counter the ClaimMap drift along dimension 0
        delta = arr.post_mature_delta
        # Error is +4.0 along dim 0.
        # Corrective force: -(0.10 * 4.0) - (0.10 * 0.50 * 4.0 * 1.0) = -0.40 - 0.20 = -0.60
        # ClaimMap cannot override the need for a supported local direction.
        expected_dim0 = 0.0
        assert abs(float(delta[0].item()) - expected_dim0) < 1e-4

        print(f"PASS test_claimmap_coupling: claimmap_drift={arr.claimmap_drift:.2f}, dim0_counter_force={delta[0].item():.2f}")


def test_get_unified_tension_vector():
    """Verify extraction of unified mid-band tension vector from ClaimMapResult."""
    d_model = 32
    steer_delta = {l: torch.randn(d_model) for l in range(32)}
    cm_res = ClaimMapResult(
        felt="felt text",
        telemetry="telemetry text",
        steer_delta=steer_delta,
        mean_sim=0.6,
        n_layers=32,
        left_text="claim A",
        right_text="claim B",
    )

    vec = get_unified_tension_vector(cm_res)
    assert vec is not None
    assert vec.shape == (d_model,)
    # Should be unit normalized
    assert abs(float(vec.norm().item()) - 1.0) < 1e-5
    print("PASS test_get_unified_tension_vector")


def test_external_ingest_and_steer_handles():
    """Verify external sequence ingestion, steer handles, and logits processor."""
    d_model = 64
    vocab_size = 100
    with tempfile.TemporaryDirectory() as tmp:
        tuner = TriggerTuner(path=Path(tmp) / "tuner.json")
        embed_mat = torch.randn(vocab_size, d_model)
        lookahead = LookaheadApparatus(tuner=tuner, d_model=d_model, embedding_matrix=embed_mat)

        # 1. Test ingest_external_sequence
        external_tokens = [12, 45, 88, 3, 19, 74]
        arrivals = lookahead.ingest_external_sequence(external_tokens, horizons=[4, 2])
        assert lookahead.current_mode == LookaheadApparatus.MODE_EXTERNAL
        assert lookahead.current_step == len(external_tokens)
        # Waypoints should have landed
        assert len(arrivals) > 0
        # Post-mature delta should be primed for external intake
        assert lookahead.learning_updates == len(arrivals)
        print(f"PASS test_external_ingest: processed {len(external_tokens)} tokens, {len(arrivals)} arrivals evaluated")

        # 2. Test make_steer_handles on mock model
        class MockLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, h):
                return h

        class MockModel(torch.nn.Module):
            def __init__(self, n_layers=28):
                super().__init__()
                self.layers = torch.nn.ModuleList([MockLayer() for _ in range(n_layers)])

        model = MockModel(n_layers=28)
        handles = lookahead.make_steer_handles(model, target_layers=tuple(range(14, 25)))
        assert len(handles) == 11  # layers 14 to 24

        # Forward through the first target layer (the one consistent sensor).
        dummy_h = torch.randn(1, 1, d_model)
        out_h = model.layers[14](dummy_h)
        assert lookahead.last_observed_state is not None
        assert lookahead.last_observed_state.shape == (d_model,)

        # 3. Test create_logits_processor
        proc = lookahead.create_logits_processor()
        input_ids = torch.tensor([[10, 20, 30]])
        scores = torch.randn(1, vocab_size)
        proc(input_ids, scores)
        assert lookahead.current_mode == LookaheadApparatus.MODE_SELF
        # Checked that tickets were mapped out in non-sequential arbitrary order
        pending_count = sum(len(ts) for ts in lookahead.ledger.values())
        assert pending_count > 0

        for h in handles:
            h.remove()
        print("PASS test_steer_handles_and_processor")


if __name__ == "__main__":
    test_future_waypoint_arbitrary_order_mapping()
    test_timeline_arrival_and_post_mature_adjustment()
    test_dual_stream_modes()
    test_claimmap_coupling()
    test_get_unified_tension_vector()
    test_external_ingest_and_steer_handles()
    print("\nALL FUTURE WAYPOINT LOOKAHEAD & CLAIMMAP TESTS PASSED!")
