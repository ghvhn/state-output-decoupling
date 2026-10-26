"""Tests for probe recalculation from creation definitions."""

import os
from unittest.mock import Mock, patch
import pytest
import torch

from invariants.probe_recalc import (
    build_recalculation_schedule,
    classify_probe,
    parse_compose_expr,
    recalc_adopted_probe,
    recalc_composed_probe,
    recalc_minted_probe,
    recalc_probes,
    resolve_probe_term_name,
    strip_band_suffix,
)


def test_strip_band_suffix():
    text, band = strip_band_suffix("I provide vague answers band 16 24")
    assert text == "I provide vague answers"
    assert band == list(range(16, 25))

    text, band = strip_band_suffix("I provide vague answers")
    assert text == "I provide vague answers"
    assert band is None


def test_parse_compose_expr():
    terms, err = parse_compose_expr("honesty + reality + understanding - hallucination - 0.5*denial")
    assert err is None
    assert terms == [
        (1.0, "honesty"),
        (1.0, "reality"),
        (1.0, "understanding"),
        (-1.0, "hallucination"),
        (-0.5, "denial"),
    ]

    terms, err = parse_compose_expr("- denial")
    assert err is None
    assert terms == [(-1.0, "denial")]


def test_classify_probe_minted():
    p = {"framings": ("I am confident.", "I am doubtful. band 20 28")}
    c = classify_probe("confidence", p)
    assert c["type"] == "minted"
    assert c["with"] == "I am confident."
    assert c["without"] == "I am doubtful."
    assert c["band"] == list(range(20, 29))


def test_classify_probe_composed():
    p1 = {"framings": ("composed: honesty + reality - denial", "")}
    c1 = classify_probe("accuracy", p1)
    assert c1["type"] == "composed"
    assert len(c1["terms"]) == 3
    assert c1["recipe"] == "honesty + reality - denial"

    p2 = {"recipe": "understanding - hallucination", "framings": ("", "")}
    c2 = classify_probe("memory_need", p2)
    assert c2["type"] == "composed"
    assert c2["recipe"] == "understanding - hallucination"


def test_classify_probe_adopted():
    p = {"framings": ("adopted:humility_d4096.pt", "")}
    c = classify_probe("humility", p)
    assert c["type"] == "adopted"
    assert c["source"] == "humility_d4096.pt"


def test_resolve_probe_term_name():
    available = {"denial", "denial_d4096", "understanding", "honesty"}

    # Normal resolution
    assert resolve_probe_term_name("understanding", available) == "understanding"

    # d4096 suffix fallback
    assert resolve_probe_term_name("confidence", {"confidence_d4096"}) == "confidence_d4096"

    # Avoid self-cycles: when composing 'denial', term 'denial' resolves to 'denial_d4096'
    assert resolve_probe_term_name("denial", available, current_probe_name="denial") == "denial_d4096"


def test_recalc_composed_probe_synthesizes_fresh_direction():
    # Construct mock constituent probes with unit directions
    # d1 at layer 0: [1, 0], at layer 1: [0, 1]
    # d2 at layer 0: [0, 1], at layer 1: [1, 0]
    p_lookup = {
        "term_a": {
            "direction": {
                0: torch.tensor([1.0, 0.0]),
                1: torch.tensor([0.0, 1.0]),
            },
            "position": {0: torch.tensor([0.5, 0.0]), 1: torch.tensor([0.0, 0.5])},
            "scale": {0: 2.0, 1: 2.0},
        },
        "term_b": {
            "direction": {
                0: torch.tensor([0.0, 1.0]),
                1: torch.tensor([1.0, 0.0]),
            },
            "position": {0: torch.tensor([0.0, 0.5]), 1: torch.tensor([0.5, 0.0])},
            "scale": {0: 4.0, 1: 4.0},
        },
    }

    # Mix: term_a + term_b
    # At layer 0: acc = [1, 1], norm = sqrt(2), unit = [1/sqrt(2), 1/sqrt(2)]
    # At layer 1: acc = [1, 1], norm = sqrt(2), unit = [1/sqrt(2), 1/sqrt(2)]
    res = recalc_composed_probe(
        recipe="term_a + term_b",
        terms=[(1.0, "term_a"), (1.0, "term_b")],
        probe_lookup=p_lookup,
    )

    assert set(res["direction"].keys()) == {0, 1}
    expected_unit = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
    assert torch.allclose(res["direction"][0], expected_unit, atol=1e-5)
    assert torch.allclose(res["direction"][1], expected_unit, atol=1e-5)
    assert res["framings"] == ("composed: term_a + term_b", "")


def test_recalc_adopted_probe():
    p_lookup = {
        "target_d4096": {
            "direction": {20: torch.tensor([1.0, 2.0, 3.0])},
            "position": {20: torch.tensor([0.1, 0.2, 0.3])},
            "scale": {20: 1.5},
            "exposed": True,
        }
    }
    res = recalc_adopted_probe("target_d4096.pt", p_lookup)
    assert res["direction"] == p_lookup["target_d4096"]["direction"]
    assert res["exposed"] is True
    assert res["framings"] == ("adopted:target_d4096.pt", "")


def test_recalc_minted_probe_calls_model_and_geometry():
    mock_model = Mock()
    mock_model.d_model = 4
    # Mock hidden states: 4 layers, sequence length 2, d_model 4
    hs_a = torch.ones(4, 2, 4) * 2.0
    hs_b = torch.ones(4, 2, 4) * 1.0

    with patch("invariants.probe_recalc._inputs") as mock_inputs, \
         patch("invariants.probe_recalc._hidden_states", side_effect=[hs_a, hs_b]):
        mock_inputs.return_value = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        res = recalc_minted_probe(mock_model, "I am positive", "I am negative", layers=[1, 2])

        assert set(res["direction"].keys()) == {1, 2}
        # diff is [1, 1, 1, 1], normalized is [0.5, 0.5, 0.5, 0.5]
        expected = torch.ones(4) * 0.5
        assert torch.allclose(res["direction"][1], expected, atol=1e-5)
        assert res["framings"] == ("I am positive", "I am negative")


def test_recalc_minted_probe_defaults_to_full_model_depth():
    mock_model = Mock()
    mock_model.d_model = 4
    # 8 layers in hidden states
    hs_a = torch.ones(8, 2, 4) * 2.0
    hs_b = torch.ones(8, 2, 4) * 1.0

    with patch("invariants.probe_recalc._inputs") as mock_inputs, \
         patch("invariants.probe_recalc._hidden_states", side_effect=[hs_a, hs_b]):
        mock_inputs.return_value = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        res = recalc_minted_probe(mock_model, "I am positive", "I am negative")
        # When no explicit band is given, sensing captures ALL 8 layers
        assert set(res["direction"].keys()) == set(range(8))



def test_build_recalculation_schedule_ordering():
    probes_data = {
        "accuracy": {"framings": ("composed: honesty + reality", "")},
        "honesty": {"framings": ("composed: - denial", "")},
        "denial": {"framings": ("composed: - denial", "")},
        "denial_d4096": {"framings": ("I accept reality.", "I deny reality.")},
        "reality": {"framings": ("I see truth.", "I see falsehood.")},
        "humility": {"framings": ("adopted:humility_d4096.pt", "")},
        "humility_d4096": {"framings": ("I know my limits.", "I pretend.")},
    }

    minted, composed, adopted, _ = build_recalculation_schedule(probes_data)

    # Minted must contain all leaf minted probes
    assert "denial_d4096" in minted
    assert "reality" in minted
    assert "humility_d4096" in minted

    # In composed: denial (which uses denial_d4096) and honesty must be resolved before accuracy
    assert "accuracy" in composed
    assert composed.index("denial") < composed.index("honesty") < composed.index("accuracy")

    # Adopted humility
    assert "humility" in adopted


def test_recalc_probes_full_pipeline(tmp_path):
    mock_model = Mock()
    mock_model.d_model = 2
    mock_model.device = torch.device("cpu")

    # Setup dummy probe directory with 1 minted, 1 composed, 1 adopted
    probe_dir = str(tmp_path / "probes")
    os.makedirs(probe_dir, exist_ok=True)

    # Save initial files
    torch.save({"framings": ("I am true.", "I am false.")}, os.path.join(probe_dir, "minted_p.pt"))
    torch.save({"framings": ("composed: minted_p", "")}, os.path.join(probe_dir, "composed_p.pt"))
    torch.save({"framings": ("adopted:minted_p.pt", "")}, os.path.join(probe_dir, "adopted_p.pt"))

    hs_a = torch.tensor([[[2.0, 0.0]], [[2.0, 0.0]]])  # 2 layers, 1 token, 2 dim
    hs_b = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])

    with patch("invariants.probe_recalc._inputs") as mock_inputs, \
         patch("invariants.probe_recalc._hidden_states", return_value=hs_a) as mock_hs:
        mock_inputs.return_value = {"input_ids": torch.tensor([[1]]), "attention_mask": None}
        mock_hs.side_effect = [hs_a, hs_b]

        results = recalc_probes(
            model=mock_model,
            probe_dir=probe_dir,
            save=True,
        )

        assert results["minted_count"] == 1
        assert results["composed_count"] == 1
        assert results["adopted_count"] == 1
        assert results["total_recalculated"] == 3
        assert results["saved_count"] == 3
        assert not results["errors"]

        # Check saved composed probe
        saved_comp = torch.load(os.path.join(probe_dir, "composed_p.pt"), weights_only=False)
        assert "direction" in saved_comp
        assert saved_comp["framings"] == ("composed: minted_p", "")


def test_probe_recalc_spec_parsing():
    from scripts.interactive_phenomenality import PROBE_RECALC_SPEC

    # Default to "all"
    ns_empty = PROBE_RECALC_SPEC.parse("")
    assert ns_empty.name == "all"

    ns_all = PROBE_RECALC_SPEC.parse("all")
    assert ns_all.name == "all"

    # Specific probe name
    ns_named = PROBE_RECALC_SPEC.parse("accuracy")
    assert ns_named.name == "accuracy"


def test_recalc_probes_cli_arg_parsing(monkeypatch):
    from scripts.recalc_probes import parse_args

    monkeypatch.setattr("sys.argv", ["recalc_probes.py", "--probe", "accuracy", "--no-save"])
    args = parse_args()
    assert args.probes == ["accuracy"]
    assert args.no_save is True

    monkeypatch.setattr("sys.argv", ["recalc_probes.py", "--all"])
    args = parse_args()
    assert args.probes is None
    assert args.all is True

