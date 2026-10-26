"""Packing and scoped source-storage checks for the explicit quantization control."""
from dataclasses import replace
import hashlib
import json
import zipfile

import pytest
import torch

from tda.block_quant import quantize_reader, save_block_quantization, load_block_quantization
from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader
from tda.expert_compress import _metadata_bytes


def source():
    scope = ExpertGroupIdentity("test", "fixed", 3, "gate", "out_in", ("0", "1"), (3, 5), "float32")
    weight = torch.arange(-15., 15.).reshape(2, 3, 5)
    return TensorExpertReader(scope, weight)


@pytest.mark.parametrize("bits", [4, 8])
def test_packing_padding_scope_and_roundtrip(bits, tmp_path):
    reader = source()
    result = quantize_reader(reader, bits=bits, block_size=4, chunk_blocks=1)
    original = reader.weights.flatten(1)
    restored = result.read_columns(0, 15)
    assert (restored - original).abs().max() <= 15 / (2**(bits-1)-1) / 2 + 1e-6
    torch.testing.assert_close(result.read_columns(3, 12, expert_ids=("1", "0")), restored[[1, 0], 3:12])
    assert result.metadata["padding_values"] == 1
    assert result.metadata["factor_tensor_bytes"] == 2*4*(4*bits//8+4)
    output = tmp_path / "quant"
    report = save_block_quantization(result, output)
    loaded = load_block_quantization(output, expected_identity=reader.identity)
    torch.testing.assert_close(loaded.read_columns(0, 15), restored, rtol=0, atol=0)
    assert report["serialized_total_bytes"] == sum(p.stat().st_size for p in output.iterdir())
    with pytest.raises(ValueError, match="scope"):
        load_block_quantization(output, expected_identity=replace(reader.identity, revision="other"))
    with pytest.raises(FileExistsError):
        save_block_quantization(result, output)


def test_zero_blocks_finite_and_unchanged():
    reader = source()
    reader.weights.zero_()
    result = quantize_reader(reader, block_size=4)
    assert result.scales.count_nonzero() == 0
    assert result.read_columns(0, 15).count_nonzero() == 0
    assert result.metadata["relative_frobenius_error"] == 0
    assert reader.weights.count_nonzero() == 0


def test_fp8_source_bytes_and_budget_before_reads():
    reader = source()
    reader.identity = replace(reader.identity, source_dtype="float8_e4m3fn", source_storage_bytes=46)
    result = quantize_reader(reader, block_size=4)
    assert result.metadata["source_tensor_bytes"] == 46
    assert result.metadata["tensor_storage_ratio"] == 48/46
    with pytest.raises(MemoryError):
        quantize_reader(reader, block_size=4, max_working_bytes=1)


def test_nonfinite_and_invalid_options():
    reader = source()
    with pytest.raises(ValueError):
        quantize_reader(reader, bits=True)
    with pytest.raises(ValueError):
        quantize_reader(reader, block_size=3)
    reader.weights[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        quantize_reader(reader)


def test_corrupt_checksum_and_accounting(tmp_path):
    reader = source()
    output = tmp_path / "quant"
    save_block_quantization(quantize_reader(reader), output)
    path = output / "metadata.json"
    data = json.loads(path.read_text())
    data["factors_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_block_quantization(output, expected_identity=reader.identity)


@pytest.mark.parametrize("bits", [True, 4.0, 8.0, 0, 2, "4"])
def test_bits_are_exact_integers(bits):
    with pytest.raises(ValueError, match="integer"):
        quantize_reader(source(), bits=bits)


def _rewrite_metadata(output, **updates):
    path = output / "metadata.json"
    metadata = json.loads(path.read_bytes())
    metadata.update(updates)
    tensor_size = (output / "factors.pt").stat().st_size
    for _ in range(16):
        encoded = _metadata_bytes(metadata)
        total = tensor_size + len(encoded)
        if metadata["metadata_json_bytes"] == len(encoded) and metadata["serialized_total_bytes"] == total:
            break
        metadata.update(metadata_json_bytes=len(encoded), serialized_total_bytes=total,
                        serialized_tensor_file_bytes=tensor_size,
                        serialized_is_smaller=total < metadata["source_tensor_bytes"],
                        serialized_storage_ratio=total / metadata["source_tensor_bytes"])
    path.write_bytes(encoded)


@pytest.mark.parametrize("updates", [
    {"tensor_storage_ratio": 0.}, {"tensor_storage_is_smaller": False},
    {"level_range": [-8, 7]}, {"scale_dtype": "float16"}, {"padding_values": 0},
    {"method": "GPTQ"}, {"evaluation_scope": "verified_language_quality"},
    {"relative_frobenius_error": -1.}, {"maximum_absolute_error": -1.},
    {"semantic_quality_evaluated": True}, {"runtime_savings_evaluated": True},
])
def test_invalid_codec_metadata_cannot_be_loaded_before_tensor_load(tmp_path, monkeypatch, updates):
    reader = source()
    output = tmp_path / "quant"
    save_block_quantization(quantize_reader(reader, block_size=4), output)
    _rewrite_metadata(output, **updates)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("bad metadata reached tensor loading"))
    with pytest.raises(ValueError):
        load_block_quantization(output, expected_identity=reader.identity)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("kind", ["unused", "padding", "negative_scale", "nonfinite_scale", "noncanonical_max", "positive_zero"])
def test_code_scale_and_padding_integrity_on_export(bits, kind, tmp_path):
    value = quantize_reader(source(), bits=bits, block_size=4)
    qmax = 2 ** (bits - 1) - 1
    if kind == "unused":
        value.codes[0, 0, 0] = 255
    elif kind == "padding":
        if bits == 4:
            value.codes[0, -1, -1] = (int(value.codes[0, -1, -1]) & 15) | ((qmax + 1) << 4)
        else:
            value.codes[0, -1, -1] = qmax + 1
    elif kind == "negative_scale":
        value.scales[0, 0] = -1
    elif kind == "nonfinite_scale":
        value.scales[0, 0] = float("nan")
    else:
        code = qmax if kind == "positive_zero" else qmax + 1
        value.codes[0, 0].fill_(code | (code << 4) if bits == 4 else code)
    with pytest.raises(ValueError):
        save_block_quantization(value, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_zero_scale_with_nonzero_code_is_rejected(tmp_path):
    reader = source()
    reader.weights.zero_()
    value = quantize_reader(reader, block_size=4)
    value.codes[0, 0, 0] = 0
    with pytest.raises(ValueError, match="zero block"):
        save_block_quantization(value, tmp_path / "invalid")


def test_quantized_archive_compression_or_extra_storage_is_rejected_before_load(tmp_path, monkeypatch):
    reader = source()
    output = tmp_path / "quant"
    save_block_quantization(quantize_reader(reader, block_size=4), output)
    tensor = output / "factors.pt"
    with zipfile.ZipFile(tensor) as old:
        entries = [(info.filename, old.read(info.filename)) for info in old.infolist()]
    with zipfile.ZipFile(tensor, "w", compression=zipfile.ZIP_DEFLATED) as new:
        for name, value in entries:
            new.writestr(name, value)
    _rewrite_metadata(output, factors_sha256=hashlib.sha256(tensor.read_bytes()).hexdigest())
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("compressed tensor container was loaded"))
    with pytest.raises(ValueError, match="storage container"):
        load_block_quantization(output, expected_identity=reader.identity)


def test_huge_declared_block_allocation_is_bounded_before_torch_load(tmp_path, monkeypatch):
    reader = source()
    output = tmp_path / "quant"
    save_block_quantization(quantize_reader(reader, block_size=4), output)
    size = 10**10
    payload = 2 * (size // 2 + 4)
    _rewrite_metadata(output, block_size=size, padding_values=size - 15,
                      factor_tensor_bytes=payload, tensor_storage_ratio=payload/reader.identity.actual_source_bytes,
                      tensor_storage_is_smaller=False)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("oversized declared allocation was loaded"))
    with pytest.raises(MemoryError, match="declared quantization"):
        load_block_quantization(output, expected_identity=reader.identity)


def test_quantized_metadata_json_object_and_size_limits(tmp_path, monkeypatch):
    reader = source()
    output = tmp_path / "quant"
    save_block_quantization(quantize_reader(reader, block_size=4), output)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("invalid metadata was loaded"))
    with pytest.raises(MemoryError):
        load_block_quantization(output, expected_identity=reader.identity, max_metadata_bytes=5)
    (output / "metadata.json").write_text("[]")
    with pytest.raises(ValueError, match="object"):
        load_block_quantization(output, expected_identity=reader.identity)


def test_actual_serialized_savings_ratios_are_verified(tmp_path, monkeypatch):
    reader = source()
    output = tmp_path / "quant"
    save_block_quantization(quantize_reader(reader, block_size=4), output)
    path = output / "metadata.json"
    metadata = json.loads(path.read_bytes())
    metadata["serialized_storage_ratio"] = 1.0
    encoded = _metadata_bytes(metadata)
    # This exact replacement keeps the padded metadata byte length unchanged.
    assert len(encoded) == path.stat().st_size
    path.write_bytes(encoded)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("false ratio reached tensor loading"))
    with pytest.raises(ValueError, match="serialized savings"):
        load_block_quantization(output, expected_identity=reader.identity)


def test_export_cannot_leak_extra_backing_storage_and_failed_save_is_atomic(tmp_path):
    reader = source()
    value = quantize_reader(reader, block_size=4)
    original = value.codes.clone()
    unrelated = torch.full((10000,), 254, dtype=torch.uint8)
    unrelated[10:10+original.numel()] = original.flatten()
    value.codes = unrelated[10:10+original.numel()].view_as(original)
    with pytest.raises(MemoryError):
        save_block_quantization(value, tmp_path / "too-small", max_artifact_bytes=value.metadata["factor_tensor_bytes"])
    assert list(tmp_path.iterdir()) == []
    report = save_block_quantization(value, tmp_path / "safe")
    assert report["serialized_tensor_file_bytes"] < 10000
    loaded = load_block_quantization(tmp_path / "safe", expected_identity=reader.identity)
    torch.testing.assert_close(loaded.codes, original, rtol=0, atol=0)
    assert loaded.codes.untyped_storage().nbytes() == original.numel()


def test_failed_quantized_export_removes_only_generated_staging(tmp_path):
    reader = source()
    value = quantize_reader(reader, block_size=4)
    neighbor = tmp_path / "source-checkpoint"
    neighbor.mkdir()
    (neighbor / "keep.txt").write_text("preserve source")
    with pytest.raises(MemoryError):
        save_block_quantization(value, tmp_path / "failed", max_artifact_bytes=value.metadata["factor_tensor_bytes"])
    assert not (tmp_path / "failed").exists()
    assert list(tmp_path.iterdir()) == [neighbor]
    assert (neighbor / "keep.txt").read_text() == "preserve source"


def test_tiny_nonzero_weights_do_not_underflow_energy_diagnostics():
    reader = source()
    reader.weights.mul_(1e-30)
    result = quantize_reader(reader, bits=4, block_size=4)
    assert 0 < result.metadata["relative_frobenius_error"] < 1


def test_reader_scope_change_on_its_last_chunk_is_rejected():
    original = source()

    class MovingReader:
        identity = original.identity

        def read_columns(self, start, stop, expert_ids=None):
            self.identity = replace(original.identity, layer=99)
            return original.read_columns(start, stop, expert_ids)

    with pytest.raises(ValueError, match="scope changed"):
        quantize_reader(MovingReader(), block_size=4)
