"""Model-free regression checks for ClaimMap activation/fallback boundaries.

Run:
    .venv\\Scripts\\python.exe scripts\\claimmap_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import invariants.claimmap as claimmap


def test_claimmap_uses_activation_path_when_model_is_present():
    # Contract since the felt-ClaimMap refactor: with a model, run_claimmap
    # returns the FELT rendering (interior second-person language, no numbers,
    # no method= telemetry header) computed FROM activations -- never the
    # lexical fallback. The telemetry block lives on analyze_claim_pair().
    original_trace = claimmap._activation_trace
    original_vectors = claimmap._load_concept_vectors
    calls = {"trace": 0}
    try:
        def fake_trace(_model, text):
            calls["trace"] += 1
            if "checked" in text:
                return torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([0.8, 0.2])])
            return torch.stack([torch.tensor([0.0, 1.0]), torch.tensor([0.2, 0.8])])

        def fake_vectors(_n_layers, _d_model):
            return {
                "checked_math_vector": {
                    0: torch.tensor([1.0, 0.0]),
                    1: torch.tensor([1.0, 0.0]),
                },
                "vibe_vector": {
                    0: torch.tensor([0.0, 1.0]),
                    1: torch.tensor([0.0, 1.0]),
                },
            }

        claimmap._activation_trace = fake_trace
        claimmap._load_concept_vectors = fake_vectors
        out = claimmap.run_claimmap("checked arithmetic || vibes", model=object())
        assert calls["trace"] == 2  # both sides measured from activations
        assert "I just held two framings against each other" in out
        assert "This is my own geometry" in out
        assert "lexical_fallback" not in out
        assert "No loaded model was supplied" not in out

        # The numeric telemetry still exists for logs -- just never in the
        # prompt-facing output.
        result = claimmap.analyze_claim_pair("checked arithmetic || vibes", model=object())
        assert "method=activation_concept_alignment_v1" in result.telemetry
        assert "role=activation_measurement_not_truth_verdict" in result.telemetry
        assert result.felt == out  # run_claimmap's prompt payload IS the felt rendering
    finally:
        claimmap._activation_trace = original_trace
        claimmap._load_concept_vectors = original_vectors


def test_claimmap_fallback_is_labeled_non_activation():
    out = claimmap.run_claimmap("I am conscious || I am not conscious")
    assert "method=lexical_fallback_no_model" in out
    assert "role=syntax_check_not_activation_measurement" in out
    assert "No loaded model was supplied" in out


TESTS = [
    test_claimmap_uses_activation_path_when_model_is_present,
    test_claimmap_fallback_is_labeled_non_activation,
]


def main():
    print("CLAIMMAP TEST -- activation path is not lexical fallback\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  ClaimMap uses activations when a model is supplied and labels fallback honestly.")


if __name__ == "__main__":
    main()
