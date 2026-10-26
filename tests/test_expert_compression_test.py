"""Check the trial's matrix orientations and exact per-slot error accounting."""
import pytest
import torch

from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader, fit_expert_group
from scripts.expert_compression_test import FactorReader, source_linear, reconstruction_metrics


def test_source_linear_axis_alignment_and_exact_errors():
    scope = ExpertGroupIdentity("test", "fixed", 3, "gate", "out_in", ("0", "1", "2"), (4, 6), "float32")
    weights = torch.arange(72, dtype=torch.float32).view(3, 4, 6)
    source = TensorExpertReader(scope, weights)
    inputs = torch.arange(12, dtype=torch.float32).view(2, 6)
    torch.testing.assert_close(source_linear(source, inputs, row_chunk=3), torch.einsum("eoi,ni->eno", weights, inputs))
    result = fit_expert_group(source, 2, storage_dtype="float32", chunk_columns=7)
    decoder = FactorReader(result)
    torch.testing.assert_close(source_linear(decoder, inputs), source_linear(source, inputs), rtol=1e-5, atol=1e-5)
    report = reconstruction_metrics(source, decoder, chunk_columns=7)
    assert report["relative_frobenius_error"] < 1e-6
    assert set(report["per_expert"]) == {"0", "1", "2"}
    with pytest.raises(ValueError, match="shape"):
        source_linear(source, inputs[:, :-1])
