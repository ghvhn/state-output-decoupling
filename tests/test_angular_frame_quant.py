"""Unit tests for Angular Frame (Phasor / Simplicial) Quantization."""
from dataclasses import replace
import pytest
import torch

from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader
from tda.angular_frame_quant import (
    fit_angular_frame_reader,
    save_angular_frame_quantization,
    load_angular_frame_quantization,
)


def source_reader():
    scope = ExpertGroupIdentity(
        "test-model", "0" * 40, 3, "gate_proj",
        "separate gate projection; row-major; FP8 times 128x128 FP32 block scales",
        ("0", "1"), (4, 16), "float32"
    )
    torch.manual_seed(42)
    weights = torch.randn(2, 4, 16) * 0.05
    return TensorExpertReader(scope, weights)


def test_angular_frame_roundtrip_and_columns_read(tmp_path):
    reader = source_reader()
    quant = fit_angular_frame_reader(
        reader, block_size=16, angles_deg=(67.0, 35.0), levels=(-2.0, -1.0, 0.0, 1.0, 2.0), iterations=3
    )

    original = reader.weights.flatten(1)
    restored = quant.read_columns(0, 64)

    assert restored.shape == (2, 64)
    # Check that error is bounded and finite
    err = (restored - original).abs().max()
    assert err < 0.15

    # Check slicing
    sliced = quant.read_columns(4, 28, expert_ids=("1",))
    torch.testing.assert_close(sliced, restored[1:2, 4:28])

    # Save and load roundtrip
    out_dir = tmp_path / "angular_quant"
    save_angular_frame_quantization(quant, out_dir)
    loaded = load_angular_frame_quantization(out_dir, expected_identity=reader.identity)
    torch.testing.assert_close(loaded.read_columns(0, 64), restored)

    # Check scope mismatch rejects
    wrong_id = replace(reader.identity, revision="1" * 40)
    with pytest.raises(ValueError, match="source scope mismatch"):
        load_angular_frame_quantization(out_dir, expected_identity=wrong_id)


def test_angular_frame_custom_angles():
    reader = source_reader()
    # Test custom angles: e.g. 45 deg and 90 deg
    quant = fit_angular_frame_reader(
        reader, block_size=16, angles_deg=(45.0, 90.0), levels=(-1.0, 0.0, 1.0)
    )
    restored = quant.read_columns(0, 64)
    assert bool(torch.isfinite(restored).all())
    assert quant.metadata["num_angular_states"] == 27
