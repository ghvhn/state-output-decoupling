"""Model-free checks for the steering envelope: bounded but entirely flexible.

Every additive residual-stream injection routes through engine._cap_steer, so a
push can never exceed the cap fraction of the state it lands in -- while every
knob (alpha, fraction, band, per-call override) stays freely tunable. These
tests pin both halves: the bound always holds, and the surfaces really move.

Run:
    .venv\\Scripts\\python.exe scripts\\steer_bound_test.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants import engine
from invariants.engine import (
    _cap_steer,
    _steer_handles,
    calibrate_steer_cap_fraction,
    get_steer_band,
    get_steer_cap_fraction,
    reset_steer_telemetry,
    set_steer_band,
    set_steer_cap_fraction,
    steer_band_layers,
    steer_cap_from_data,
    steer_telemetry_stats,
)


def make_dummy_model(n_layers=4, d_model=8):
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n_layers)])
    return SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        device="cpu",
        n_layers=n_layers,
    )


def test_small_steer_passes_unchanged():
    h = torch.randn(1, 3, 8) * 10.0
    add = torch.randn(8)
    add = add / add.norm()  # norm 1 vs cap 0.5 * ~30 per token
    out = _cap_steer(add, h)
    assert torch.allclose(out, add.expand_as(out) if out.shape != add.shape else add, atol=1e-5)


def test_over_steer_is_clipped_per_token():
    h = torch.randn(1, 3, 8) * 10.0
    add = torch.randn(8) * 1000.0
    capped = _cap_steer(add, h)
    frac = get_steer_cap_fraction()
    push_norms = capped.float().norm(dim=-1)
    h_norms = h.float().norm(dim=-1)
    assert torch.allclose(push_norms, frac * h_norms, rtol=1e-3)


def test_batched_add_caps_per_row():
    h = torch.stack([torch.ones(8) * 10.0, torch.ones(8) * 2.0])  # norms ~28.3, ~5.7
    small = torch.zeros(8); small[0] = 0.1
    huge = torch.ones(8) * 500.0
    add = torch.stack([small, huge])
    capped = _cap_steer(add, h)
    assert torch.allclose(capped[0], small, atol=1e-6)  # small row untouched
    frac = get_steer_cap_fraction()
    assert abs(capped[1].norm().item() - frac * h[1].norm().item()) < 1e-3


def test_cap_fraction_zero_is_a_kill_switch():
    h = torch.randn(2, 8) * 5.0
    add = torch.randn(8)
    assert _cap_steer(add, h, cap_fraction=0.0).abs().max().item() == 0.0


def test_nonfinite_add_is_dropped_not_injected():
    h = torch.randn(1, 2, 8)
    add = torch.full((8,), float("nan"))
    assert _cap_steer(add, h).abs().max().item() == 0.0
    add_inf = torch.zeros(8); add_inf[0] = float("inf")
    assert _cap_steer(add_inf, h).abs().max().item() == 0.0


def test_envelope_moves_but_never_disappears():
    original = get_steer_cap_fraction()
    try:
        h = torch.ones(1, 1, 4) * 10.0  # token norm 20
        add = torch.ones(4) * 100.0
        set_steer_cap_fraction(0.1)
        assert abs(_cap_steer(add, h).norm().item() - 0.1 * 20.0) < 1e-3
        set_steer_cap_fraction(2.5)  # any finite fraction is legal
        assert get_steer_cap_fraction() == 2.5
        for bad in (float("inf"), float("nan"), -0.1):
            try:
                set_steer_cap_fraction(bad)
                assert False, f"{bad} should have been rejected"
            except ValueError:
                pass
        assert get_steer_cap_fraction() == 2.5  # last good value survives
        # per-call override wins over the global
        set_steer_cap_fraction(1.0)
        assert abs(_cap_steer(add, h, cap_fraction=0.05).norm().item() - 0.05 * 20.0) < 1e-3
    finally:
        set_steer_cap_fraction(original)


def test_steer_band_is_flexible_and_validated():
    original = get_steer_band()
    try:
        set_steer_band(0.40, 0.70)
        assert steer_band_layers(32) == list(range(12, 22))
        set_steer_band(0.1, 0.2)
        assert steer_band_layers(32) == list(range(3, 6))
        # per-call band overrides the global without moving it
        assert steer_band_layers(32, lo=0.5, hi=0.75) == list(range(16, 24))
        assert get_steer_band() == (0.1, 0.2)
        for bad in ((0.5, 0.4), (-0.1, 0.5), (0.2, 1.5), (float("nan"), 0.5)):
            try:
                set_steer_band(*bad)
                assert False, f"{bad} should have been rejected"
            except ValueError:
                pass
    finally:
        set_steer_band(*original)


def test_steer_handles_end_to_end_bounded():
    M = make_dummy_model(n_layers=4, d_model=8)
    torch.manual_seed(0)
    vecs = {l: torch.randn(8) for l in range(4)}
    h = torch.randn(1, 5, 8) * 7.0
    frac = get_steer_cap_fraction()
    handles = _steer_handles(M, vecs, [1, 2], alpha=1e6)
    try:
        for l in range(4):
            out = M.model.model.layers[l](h)
            delta = (out - h).float().norm(dim=-1)
            if l in (1, 2):
                assert torch.all(delta <= frac * h.float().norm(dim=-1) * (1 + 1e-3))
                assert delta.max().item() > 0
            else:
                assert delta.max().item() == 0.0
    finally:
        for hd in handles:
            hd.remove()
    # per-layer alpha dict + per-call cap zero -> exact identity
    handles = _steer_handles(M, vecs, [0, 3], alpha={0: 5.0, 3: 9.0}, cap_fraction=0.0)
    try:
        for l in (0, 3):
            out = M.model.model.layers[l](h)
            assert torch.equal(out, h)
    finally:
        for hd in handles:
            hd.remove()


def test_claimmap_handles_respect_band_and_bound():
    from invariants.claimmap import claimmap_steer_handles

    M = make_dummy_model(n_layers=10, d_model=8)
    steer_delta = {l: torch.randn(8) * 50.0 for l in range(10)}
    assert claimmap_steer_handles(M, steer_delta, alpha=0.0) == []
    handles = claimmap_steer_handles(M, steer_delta, alpha=1e5, band=(0.2, 0.5))
    try:
        assert len(handles) == 3  # layers 2, 3, 4
        h = torch.randn(1, 4, 8) * 6.0
        out = M.model.model.layers[2](h)
        delta = (out - h).float().norm(dim=-1)
        assert torch.all(delta <= get_steer_cap_fraction() * h.float().norm(dim=-1) * (1 + 1e-3))
    finally:
        for hd in handles:
            hd.remove()


def test_agentic_delta_choke_point_is_bounded():
    from invariants.agentic_engine import _add_last_token_delta, _sane_fraction

    h = torch.randn(1, 4, 8) * 8.0
    huge = torch.randn(8) * 1e4
    out = _add_last_token_delta(h, huge)
    assert torch.equal(out[:, :-1, :], h[:, :-1, :])  # only the last token moves
    push = (out[:, -1, :] - h[:, -1, :]).float().norm()
    cap = get_steer_cap_fraction() * h[:, -1, :].float().norm()
    assert push <= cap * (1 + 1e-3)
    assert push.item() > 0

    assert _sane_fraction(0.3, 0.25) == 0.3
    assert _sane_fraction(0.0, 0.25) == 0.0
    for bad in (float("inf"), float("nan"), -1.0, "x", None):
        assert _sane_fraction(bad, 0.25) == 0.25


def _steer_with_ratio(ratio):
    """One _cap_steer application whose attempted push/residual ratio == ratio."""
    h = torch.zeros(1, 1, 4)
    h[..., 0] = 10.0                      # residual norm 10
    add = torch.zeros(4)
    add[0] = 10.0 * ratio                 # push norm 10 * ratio
    _cap_steer(add, h)


def test_telemetry_records_attempted_ratios():
    reset_steer_telemetry()
    _steer_with_ratio(0.2)                # under the cap: recorded, not clipped
    _steer_with_ratio(10.0)               # over the cap: recorded and clipped
    _cap_steer(torch.zeros(4), torch.ones(1, 1, 4))            # zero push: skipped
    _cap_steer(torch.full((4,), float("nan")), torch.ones(1, 1, 4))  # dropped: skipped
    stats = steer_telemetry_stats()
    assert stats["n"] == 2 and stats["applied"] == 2
    assert stats["clipped"] == 1 and abs(stats["clip_rate"] - 0.5) < 1e-9
    assert abs(stats["min"] - 0.2) < 1e-4 and abs(stats["max"] - 10.0) < 1e-3
    reset_steer_telemetry()
    assert steer_telemetry_stats()["n"] == 0


def test_cap_calibration_is_data_informed_and_deterministic():
    original = get_steer_cap_fraction()
    reset_steer_telemetry()
    try:
        # Not enough evidence -> refuse, prior stays in force.
        for r in (0.1, 0.2, 0.3):
            _steer_with_ratio(r)
        assert steer_cap_from_data(95) is None            # default min_n = 64
        assert calibrate_steer_cap_fraction(95) is None
        assert get_steer_cap_fraction() == original

        # Known distribution: ratios 0.01 .. 1.00.
        reset_steer_telemetry()
        for i in range(1, 101):
            _steer_with_ratio(i / 100.0)
        # Exact order-statistic percentile: p95 of 100 samples -> index 94 -> 0.95.
        v1 = steer_cap_from_data(95, min_n=10)
        v2 = steer_cap_from_data(95, min_n=10)
        assert v1 == v2                                    # same data, same bound
        assert abs(v1 - 0.95) < 1e-3
        assert abs(steer_cap_from_data(0, min_n=10) - 0.01) < 1e-4
        assert abs(steer_cap_from_data(100, min_n=10) - 1.00) < 1e-3
        applied = calibrate_steer_cap_fraction(50, min_n=10)
        assert abs(applied - 0.51) < 1e-3                  # index round(0.5*99) = 50
        assert get_steer_cap_fraction() == applied
        for bad in (-5, 101, float("nan")):
            try:
                steer_cap_from_data(bad, min_n=10)
                assert False, f"percentile {bad} should have been rejected"
            except ValueError:
                pass
    finally:
        reset_steer_telemetry()
        set_steer_cap_fraction(original)


def test_band_suggestion_derives_from_outcomes():
    import tempfile
    from invariants.steer_map_store import SteerMapEvent, SteerMapStore

    with tempfile.TemporaryDirectory() as tmp:
        store = SteerMapStore(
            events_path=Path(tmp) / "events.jsonl",
            summary_path=Path(tmp) / "summary.json",
        )
        assert store.suggest_band(32) is None              # no data -> keep prior

        def add(layer, success, times):
            for _ in range(times):
                store.append(
                    SteerMapEvent(
                        kind="synthesis_record",
                        action="test",
                        end_layer=layer,
                        event_success=success,
                        success_label="final_correct" if success else "final_wrong",
                    )
                )

        add(14, True, 4)
        add(15, True, 4)
        add(16, True, 4)
        add(5, False, 4)                                   # enough support, bad outcomes
        add(20, True, 2)                                   # good outcomes, thin support
        store.append(SteerMapEvent(kind="x", action="y", end_layer=8))  # unlabeled: ignored

        suggestion = store.suggest_band(32, min_events=3)
        assert suggestion is not None
        assert suggestion["eligible_layers"] == [14, 15, 16]
        assert abs(suggestion["lo"] - 14 / 32) < 1e-9
        assert abs(suggestion["hi"] - 17 / 32) < 1e-9
        assert suggestion["labeled_events"] == 18

        # Deterministic: same events, same band.
        again = store.suggest_band(32, min_events=3)
        assert again["lo"] == suggestion["lo"] and again["hi"] == suggestion["hi"]

        # The derived band is a legal band for the flexible surface.
        original = get_steer_band()
        try:
            set_steer_band(suggestion["lo"], suggestion["hi"])
            assert steer_band_layers(32) == [14, 15, 16]
        finally:
            set_steer_band(*original)


def test_conversations_are_evidence_too():
    """Humans learn from conversations: turns with no gold still label their
    steering events via the live productivity read, on a separate, auditable
    evidence lane that never masquerades as gold."""
    import tempfile
    from invariants.steer_map_store import SteerMapStore

    def rec(layer):
        return {"metadata": {"reason": "r", "start_layer": layer - 2, "end_layer": layer,
                             "steps": 3, "expert": "A"}}

    with tempfile.TemporaryDirectory() as tmp:
        store = SteerMapStore(
            events_path=Path(tmp) / "events.jsonl",
            summary_path=Path(tmp) / "summary.json",
        )
        # Productive conversational turns whose steering landed on layer 16.
        for _ in range(3):
            e = store.record_synthesis_record(
                rec(16), source="interactive",
                conversation_outcome={"score": 0.4, "threshold": 0.0},
            )
        assert e.success_basis == "conversation" and e.event_success is True
        assert e.success_label == "conversation_productive"
        assert e.metrics["conversation_sense"] == 0.4
        # Unproductive turns on layer 8: score below the threshold.
        for _ in range(3):
            e = store.record_synthesis_record(
                rec(8), source="interactive",
                conversation_outcome={"score": -0.2, "threshold": 0.0},
            )
        assert e.event_success is False and e.success_label == "conversation_unproductive"
        # A turn with no outcome read stays honestly unlabeled.
        e = store.record_synthesis_record(rec(20), source="interactive")
        assert e.event_success is None and e.success_basis == "gold"
        # One gold benchmark event on layer 10.
        store.record_synthesis_record(
            rec(10), source="benchmark_result", final_correct=True, attempt={"accepted": True}
        )

        conv = store.suggest_band(32, min_events=3, basis="conversation")
        assert conv["eligible_layers"] == [16]
        assert conv["labeled_by_basis"] == {"conversation": 6}
        gold = store.suggest_band(32, min_events=1, basis="gold")
        assert gold["eligible_layers"] == [10]
        assert gold["labeled_by_basis"] == {"gold": 1}
        both = store.suggest_band(32, min_events=3, basis="any")
        assert both["eligible_layers"] == [16]
        assert both["labeled_by_basis"] == {"conversation": 6, "gold": 1}

        # Gold always wins when both are available for the same event.
        e = store.record_synthesis_record(
            rec(12), source="benchmark_result", final_correct=False,
            conversation_outcome={"score": 5.0, "threshold": 0.0},
        )
        assert e.success_basis == "gold" and e.event_success is False

        # The lane survives persistence: reload from disk, same answer.
        reloaded = SteerMapStore(
            events_path=Path(tmp) / "events.jsonl",
            summary_path=Path(tmp) / "summary.json",
        )
        again = reloaded.suggest_band(32, min_events=3, basis="conversation")
        assert again["eligible_layers"] == [16] and again["lo"] == conv["lo"]

        try:
            store.suggest_band(32, basis="vibes")
            assert False, "unknown basis should have been rejected"
        except ValueError:
            pass


def test_channel_accounting_accumulates_and_rejects_nonfinite():
    from invariants.agentic_engine import _note_channel

    state = {"steer_channels": {}}
    _note_channel(state, "urgency", 0.10, 0.5)
    _note_channel(state, "urgency", 0.90, 0.5)          # over the channel cap -> clipped
    _note_channel(state, "urgency", float("nan"), 0.5)  # rejected
    _note_channel(state, "urgency", float("inf"), 0.5)  # rejected
    row = state["steer_channels"]["urgency"]
    assert row["applications"] == 2
    assert abs(row["ratio_sum"] - 1.0) < 1e-9 and row["ratio_max"] == 0.90
    assert row["clipped"] == 1
    _note_channel({"steer_channels": None}, "urgency", 0.1, 0.5)  # no accumulator: no-op


def test_channel_lift_answers_should_it_be_on():
    """The point of the accounting: labeled runs decide whether a channel helps.
    Fired-vs-unfired outcome comparison per channel, per evidence lane."""
    import tempfile
    from invariants.steer_map_store import SteerMapStore

    def stats_record(channels):
        return {
            "type": "steer_channel_stats",
            "channels": channels,
            "flags": {"synthesis_enabled": True, "cache_enabled": True},
        }

    fired_urgency = {"urgency": {"applications": 2, "ratio_sum": 0.2, "ratio_max": 0.12, "clipped": 0}}

    with tempfile.TemporaryDirectory() as tmp:
        store = SteerMapStore(
            events_path=Path(tmp) / "events.jsonl",
            summary_path=Path(tmp) / "summary.json",
        )
        # Gold lane: urgency fired on correct rows, stayed silent on wrong rows.
        for _ in range(3):
            created = store.record_synthesis_record(
                stats_record(fired_urgency), source="benchmark_result",
                final_correct=True, attempt={"accepted": True},
            )
        assert created and len(created) == 5  # every known channel gets its event
        for _ in range(3):
            store.record_synthesis_record(
                stats_record({}), source="benchmark_result", final_correct=False,
            )

        lift = {row["channel"]: row for row in store.channel_lift(basis="gold")}
        urg = lift["urgency"]
        assert urg["fired_n"] == 3 and urg["fired_rate"] == 1.0
        assert urg["unfired_n"] == 3 and urg["unfired_rate"] == 0.0
        assert urg["lift"] == 1.0                       # fired turns won: keep it on
        branch = lift["expert_branch"]                  # never fired: no fired bucket
        assert branch["fired_n"] == 0 and branch["unfired_n"] == 6
        assert branch["lift"] is None                   # no contrast -> no verdict

        # Conversation lane stays separate.
        store.record_synthesis_record(
            stats_record({"cache_delta": {"applications": 1, "ratio_sum": 0.3, "ratio_max": 0.3, "clipped": 0}}),
            source="interactive",
            conversation_outcome={"score": 0.5, "threshold": 0.0},
        )
        conv = {row["channel"]: row for row in store.channel_lift(basis="conversation")}
        assert conv["cache_delta"]["fired_n"] == 1 and conv["cache_delta"]["fired_rate"] == 1.0
        gold_only = {row["channel"]: row for row in store.channel_lift(basis="gold")}
        assert gold_only["cache_delta"]["fired_n"] == 0  # conversational evidence excluded

        # Unlabeled all-silent generations carry no evidence and write nothing.
        assert store.record_synthesis_record(stats_record({}), source="interactive") is None

        # Deterministic across reload.
        reloaded = SteerMapStore(
            events_path=Path(tmp) / "events.jsonl",
            summary_path=Path(tmp) / "summary.json",
        )
        again = {row["channel"]: row for row in reloaded.channel_lift(basis="gold")}
        assert again["urgency"]["lift"] == 1.0


def test_channel_ablation_switch_isolates_one_effect():
    """The automated isolation contract: a disabled channel computes as in
    control but injects nothing and notes nothing — one-variable ablation."""
    import os
    from invariants import agentic_engine as ae
    from invariants.config import AgenticConfig, KNOWN_STEER_CHANNELS, _env_channel_set
    from invariants.steer_map_store import SteerMapStore

    # The channel vocabulary is shared across config, engine, and store.
    assert tuple(SteerMapStore.KNOWN_STEER_CHANNELS) == tuple(KNOWN_STEER_CHANNELS)

    # Env parsing: valid names pass, unknown names are dropped, empty -> empty.
    old = os.environ.get("TDA_DISABLED_STEER_CHANNELS")
    try:
        os.environ["TDA_DISABLED_STEER_CHANNELS"] = "urgency, cache_delta ,not_a_channel"
        assert _env_channel_set("TDA_DISABLED_STEER_CHANNELS") == frozenset({"urgency", "cache_delta"})
        os.environ["TDA_DISABLED_STEER_CHANNELS"] = ""
        assert _env_channel_set("TDA_DISABLED_STEER_CHANNELS") == frozenset()
    finally:
        if old is None:
            os.environ.pop("TDA_DISABLED_STEER_CHANNELS", None)
        else:
            os.environ["TDA_DISABLED_STEER_CHANNELS"] = old

    assert ae._channel_enabled(AgenticConfig(), "urgency")
    ablated_cfg = AgenticConfig(disabled_steer_channels=frozenset({"urgency"}))
    assert not ae._channel_enabled(ablated_cfg, "urgency")
    assert ae._channel_enabled(ablated_cfg, "cache_delta")

    # Live site check: the urgency injection actually vanishes under ablation.
    original_layer_vector = ae._layer_vector
    try:
        ae._layer_vector = lambda name, layer, device: torch.ones(8)
        h = torch.randn(1, 3, 8) * 5.0

        on_cfg = AgenticConfig(continuous_urgency_injection=True)
        state = {"steer_channels": {}}
        out_on = ae._maybe_apply_time_gated_urgency(h, 16, on_cfg, state)
        assert not torch.equal(out_on, h)                      # injected
        assert state["steer_channels"]["urgency"]["applications"] == 1

        off_cfg = AgenticConfig(
            continuous_urgency_injection=True,
            disabled_steer_channels=frozenset({"urgency"}),
        )
        state = {"steer_channels": {}}
        out_off = ae._maybe_apply_time_gated_urgency(h, 16, off_cfg, state)
        assert torch.equal(out_off, h)                         # untouched
        assert state["steer_channels"] == {}                   # and unnoted
    finally:
        ae._layer_vector = original_layer_vector


def test_isolation_report_is_a_pure_function_of_summaries():
    from scripts.isolate_channel_lifts import build_isolation_report

    control = {"accuracy": 0.8, "coverage": 0.6, "selective_accuracy": 1.0, "n": 5}
    ablations = {
        "urgency": {"accuracy": 0.6, "coverage": 0.6, "selective_accuracy": 1.0, "n": 5},
        "cache_delta": {"accuracy": 0.8, "coverage": 0.2, "selective_accuracy": None, "n": 5},
    }
    report = build_isolation_report("humble_synthesis", control, ablations)
    rows = {row["channel"]: row for row in report["channels"]}
    # Removing urgency cost 0.2 accuracy -> urgency helps.
    assert rows["urgency"]["accuracy_contribution"] == 0.2
    assert rows["urgency"]["coverage_contribution"] == 0.0
    # Removing cache left accuracy alone but collapsed coverage.
    assert rows["cache_delta"]["accuracy_contribution"] == 0.0
    assert rows["cache_delta"]["coverage_contribution"] == 0.4
    assert rows["cache_delta"]["selective_accuracy_contribution"] is None  # missing metric -> no claim
    # Deterministic: same inputs, same report rows.
    again = build_isolation_report("humble_synthesis", control, ablations)
    assert again["channels"] == report["channels"]


def test_optimizer_gradient_path_bypasses_cap_and_telemetry():
    """The TTT synthesis optimizer probes the choke point with its trainable
    delta from inside the gradient loop. The cap must not reshape that
    optimization landscape (behavioral parity with the earned run) nor record
    the optimizer's simulated pushes as real steering telemetry. The applied
    delta is always detached -- and that path stays capped and recorded."""
    reset_steer_telemetry()
    h = torch.ones(1, 1, 4) * 10.0
    v = (torch.ones(4) * 1000.0).requires_grad_(True)

    out = _cap_steer(v, h)
    assert out.requires_grad                       # graph intact for the optimizer
    assert torch.allclose(out.detach(), v.detach())  # NOT capped inside the loop
    assert steer_telemetry_stats()["n"] == 0       # simulated push: not recorded

    from invariants.agentic_engine import _add_last_token_delta
    h3 = torch.randn(1, 3, 8) * 8.0
    grad_delta = (torch.randn(8) * 1e4).requires_grad_(True)
    out3 = _add_last_token_delta(h3, grad_delta)
    assert out3.requires_grad
    assert steer_telemetry_stats()["n"] == 0

    # The REAL application detaches first -- and that one is capped + recorded.
    applied = _add_last_token_delta(h3, grad_delta.detach())
    push = (applied[:, -1, :] - h3[:, -1, :]).float().norm()
    assert push <= get_steer_cap_fraction() * h3[:, -1, :].float().norm() * (1 + 1e-3)
    assert steer_telemetry_stats()["n"] == 1
    reset_steer_telemetry()


def test_axis_drift_reads_whether_one_axis_persists():
    from invariants.engine import axis_drift, drift_at_layer

    same = torch.tensor([1.0, 0.0, 0.0])
    orth = torch.tensor([0.0, 1.0, 0.0])
    vecs = {10: same, 11: same * 3.0, 12: orth, 13: -same}
    drift = axis_drift(vecs)
    assert abs(drift["pairs"][10] - 1.0) < 1e-6      # same direction, scale-free
    assert abs(drift["pairs"][11]) < 1e-6            # rotates to orthogonal
    assert abs(drift["pairs"][12]) < 1e-6            # orthogonal again
    assert drift["min"] < 0.1 and drift["mean"] < 0.5
    # A sign flip is the loudest "not the same axis" signal.
    flip = axis_drift({5: same, 6: -same})
    assert abs(flip["pairs"][5] + 1.0) < 1e-6
    # Gaps and zero vectors are skipped, not fabricated.
    sparse = axis_drift({1: same, 3: same, 4: torch.zeros(3)})
    assert sparse["pairs"] == {} and sparse["mean"] is None
    assert axis_drift({}) == {"pairs": {}, "mean": None, "min": None, "var": None}

    # Lawfulness ("set equation"): a constant per-layer rotation has adjacent
    # cosines all equal -> variance ~0 (steerable: the transport law is fixed,
    # even though the axis moves). Erratic evolution -> variance > 0.
    import math as _m
    def rot(theta):
        return torch.tensor([_m.cos(theta), _m.sin(theta), 0.0])
    lawful = axis_drift({i: rot(i * 0.4) for i in range(6)})     # fixed 0.4-rad steps
    assert lawful["var"] is not None and lawful["var"] < 1e-10
    assert abs(lawful["mean"] - _m.cos(0.4)) < 1e-6              # rotating, yet lawful
    erratic = axis_drift({0: rot(0.0), 1: rot(0.1), 2: rot(1.5), 3: rot(1.6)})
    assert erratic["var"] > 0.01                                  # the law itself changes
    # drift_at_layer averages the cosines touching the steered layer.
    assert abs(drift_at_layer(drift, 11) - 0.5) < 1e-6   # (1.0 + 0.0) / 2
    assert drift_at_layer(drift, 99) is None


def test_layer_sweep_rotates_and_attributes_per_layer():
    import tempfile
    from invariants.engine import pick_sweep_layer, pick_sweep_layers
    from invariants.steer_map_store import SteerMapStore

    band = [12, 13, 14]
    assert pick_sweep_layer(band, {}) == 12                      # untested -> lowest
    assert pick_sweep_layer(band, {12: 1}) == 13                 # least-tested next
    assert pick_sweep_layer(band, {12: 1, 13: 1, 14: 1}) == 12   # even -> rotate from lowest
    assert pick_sweep_layer([], {}) is None

    # Width > 1: a deterministic overlay of the k least-tested layers.
    assert pick_sweep_layers(band, {}, width=2) == [12, 13]
    assert pick_sweep_layers(band, {12: 2, 13: 1}, width=2) == [13, 14]
    assert pick_sweep_layers(band, {}, width=99) == band          # bounded by the band
    assert pick_sweep_layers(band, {}, width=0) == []             # 0 = sweep off
    assert pick_sweep_layers(band, {12: 1, 13: 1, 14: 1}, width=2) == [12, 13]  # ties -> lowest
    assert pick_sweep_layers(band, {}, width=2) == pick_sweep_layers(band, {}, width=2)  # deterministic

    with tempfile.TemporaryDirectory() as tmp:
        store = SteerMapStore(
            events_path=Path(tmp) / "events.jsonl",
            summary_path=Path(tmp) / "summary.json",
        )
        # Simulate the sweep: counts drive rotation; outcomes label each layer.
        counts = store.layer_steer_counts("claimmap")
        for sense in (0.4, 0.5, -0.2, 0.6, 0.3, -0.1):           # 6 steered turns
            layer = pick_sweep_layer(band, counts)
            store.record_layer_steer(
                "claimmap", layer, 0.02,
                conversation_outcome={"score": sense, "threshold": 0.0},
                metrics={"axis_drift_at_layer": 0.9},
            )
            counts = store.layer_steer_counts("claimmap")
        assert counts == {12: 2, 13: 2, 14: 2}                   # even coverage, deterministic

        # Per-layer outcomes: L12 got 0.4, 0.6 (2/2); L13 got 0.5, 0.3 (2/2);
        # L14 got -0.2, -0.1 (0/2). Only productive layers clear the bar.
        band_pick = store.suggest_band(32, min_events=2, evidence="layer_steer")
        assert band_pick["eligible_layers"] == [12, 13]
        assert band_pick["labeled_by_evidence"] == {"layer_steer": 6}
        assert band_pick["per_layer"][14]["success_rate"] == 0.0

        # The evidence kinds never blend silently: synthesis events elsewhere
        # do not move the layer-steer band, and vice versa.
        from invariants.steer_map_store import SteerMapEvent
        for _ in range(3):
            store.append(SteerMapEvent(
                kind="synthesis_record", action="synthesis_r", end_layer=27,
                event_success=True, success_label="final_correct",
            ))
        assert store.suggest_band(32, min_events=2, evidence="layer_steer")["eligible_layers"] == [12, 13]
        assert store.suggest_band(32, min_events=3, evidence="synthesis")["eligible_layers"] == [27]
        merged = store.suggest_band(32, min_events=2, evidence="any")
        assert merged["labeled_by_evidence"] == {"layer_steer": 6, "synthesis": 3}
        try:
            store.suggest_band(32, evidence="vibes")
            assert False, "unknown evidence kind should be rejected"
        except ValueError:
            pass


TESTS = [
    test_small_steer_passes_unchanged,
    test_over_steer_is_clipped_per_token,
    test_batched_add_caps_per_row,
    test_cap_fraction_zero_is_a_kill_switch,
    test_nonfinite_add_is_dropped_not_injected,
    test_envelope_moves_but_never_disappears,
    test_steer_band_is_flexible_and_validated,
    test_steer_handles_end_to_end_bounded,
    test_claimmap_handles_respect_band_and_bound,
    test_agentic_delta_choke_point_is_bounded,
    test_telemetry_records_attempted_ratios,
    test_cap_calibration_is_data_informed_and_deterministic,
    test_band_suggestion_derives_from_outcomes,
    test_conversations_are_evidence_too,
    test_channel_accounting_accumulates_and_rejects_nonfinite,
    test_channel_lift_answers_should_it_be_on,
    test_channel_ablation_switch_isolates_one_effect,
    test_isolation_report_is_a_pure_function_of_summaries,
    test_optimizer_gradient_path_bypasses_cap_and_telemetry,
    test_axis_drift_reads_whether_one_axis_persists,
    test_layer_sweep_rotates_and_attributes_per_layer,
]


def main():
    print("STEER BOUND TEST -- every push bounded, every knob movable\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  The envelope caps every additive steer; the surfaces stay live-tunable.")


if __name__ == "__main__":
    main()
