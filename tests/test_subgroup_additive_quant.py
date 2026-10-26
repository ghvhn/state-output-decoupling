"""Tests for Subgroup Additive Subvalue / Simplicial Quantization."""
from dataclasses import replace
import pytest
import torch

from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader
from tda.block_quant import quantize_reader
from tda.subgroup_additive_quant import (
    fit_subgroup_additive_reader,
    save_subgroup_additive_quantization,
    load_subgroup_additive_quantization,
)


def source_reader():
    scope = ExpertGroupIdentity("test-model", "0" * 40, 3, "gate_proj", "separate gate projection; row-major; FP8 times 128x128 FP32 block scales", ("0", "1"), (4, 16), "float32")
    # Gaussian-like weights matching real neural network weight distribution
    torch.manual_seed(42)
    weights = torch.randn(2, 4, 16) * 0.05
    return TensorExpertReader(scope, weights)


def test_roundtrip_and_columns_read(tmp_path):
    reader = source_reader()
    quant = fit_subgroup_additive_reader(reader, block_size=16, num_subvalues=4, iterations=3)

    original = reader.weights.flatten(1)
    restored = quant.read_columns(0, 64)

    assert restored.shape == (2, 64)
    # Reconstructed columns should be close to original
    err = (restored - original).abs().max()
    assert err < 0.05

    # Check slicing
    sliced = quant.read_columns(5, 35, expert_ids=("1",))
    torch.testing.assert_close(sliced, restored[1:2, 5:35])

    # Check save and load roundtrip
    out_dir = tmp_path / "additive_quant"
    save_subgroup_additive_quantization(quant, out_dir)
    loaded = load_subgroup_additive_quantization(out_dir, expected_identity=reader.identity)
    torch.testing.assert_close(loaded.read_columns(0, 64), restored)

    # Test scope mismatch rejects
    wrong_id = replace(reader.identity, revision="1" * 40)
    with pytest.raises(ValueError, match="scope mismatch"):
        load_subgroup_additive_quantization(out_dir, expected_identity=wrong_id)


def test_additive_subvalues_outperforms_uniform_int4_on_gaussian():
    # Real neural weights are Gaussian/bell-shaped
    scope = ExpertGroupIdentity("test-gaussian", "0" * 40, 3, "gate_proj", "separate gate projection; row-major; FP8 times 128x128 FP32 block scales", ("0",), (16, 64), "float32")
    torch.manual_seed(123)
    weights = torch.randn(1, 16, 64) * 0.02
    reader = TensorExpertReader(scope, weights)

    int4_res = quantize_reader(reader, bits=4, block_size=32)
    additive_res = fit_subgroup_additive_reader(reader, block_size=32, num_subvalues=4, iterations=4)

    int4_err = int4_res.metadata["relative_frobenius_error"]
    add_err = additive_res.metadata["relative_frobenius_error"]

    # Additive subvalues should achieve strictly lower error than uniform INT4
    assert add_err < int4_err, f"Expected additive error ({add_err}) < INT4 error ({int4_err})"
