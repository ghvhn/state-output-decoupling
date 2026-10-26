"""Lossless angular index storage; no large fitting or behavioral claims."""
from dataclasses import replace
import hashlib
import json
import zipfile

import pytest
import torch

from tda.angular_frame_quant import (AngularFrameQuantization, _build_angular_grid,
                                    fit_angular_frame_reader)
from tda.angular_frame_packing import (DEFAULT_LEVELS, pack_7bit_codes, unpack_7bit_codes,
    pack_angular_frame_quantization, save_packed_angular_frame, load_packed_angular_frame)
from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader, _metadata_bytes


def angular_fixture(columns=64, block_size=32, levels=DEFAULT_LEVELS, angles=(67., 35.)):
    identity = ExpertGroupIdentity("fixture", "a" * 40, 3, "gate_proj", "fixture row-major",
                                   ("0", "1"), (1, columns), "float32")
    blocks = (columns + block_size - 1) // block_size
    count = 2 * blocks * (block_size // 2)
    codes = torch.arange(count).remainder(len(levels) ** 3).to(torch.uint8).reshape(2, blocks, block_size // 2)
    scales = torch.arange(1, 2 * blocks + 1).float().reshape(2, blocks) / 32
    metadata = {"identity": identity.to_dict(), "identity_sha256": identity.fingerprint,
                "block_size": block_size, "angles_deg": list(angles), "levels": list(levels),
                "num_angular_states": len(levels) ** 3}
    return AngularFrameQuantization(identity, block_size, angles, codes, scales, metadata)


@pytest.mark.parametrize("count", [0, 1, 2, 7, 8, 9, 15, 16, 17, 125, 129])
@pytest.mark.parametrize("chunk", [1, 3, 8, 65536])
def test_all_byte_and_chunk_boundaries_roundtrip(count, chunk):
    codes = torch.arange(6 * count).remainder(125).to(torch.uint8).reshape(2, 3, count)
    packed = pack_7bit_codes(codes, chunk_codes=chunk)
    assert packed.shape == (2, 3, (count * 7 + 7) // 8)
    restored = unpack_7bit_codes(packed, count=count, expected_shape=codes.shape, chunk_codes=chunk)
    assert torch.equal(restored, codes)


def test_known_lsb_first_encoding_and_independent_row_padding():
    assert pack_7bit_codes(torch.tensor([1, 2], dtype=torch.uint8)).tolist() == [1, 1]
    assert pack_7bit_codes(torch.tensor([124, 124], dtype=torch.uint8)).tolist() == [124, 62]
    codes = torch.tensor([[1], [124]], dtype=torch.uint8)
    assert torch.equal(pack_7bit_codes(codes), codes)
    assert torch.equal(unpack_7bit_codes(codes, count=1), codes)


def test_noncontiguous_logical_rows_and_custom128_alphabet():
    codes = torch.arange(128, dtype=torch.uint8).reshape(8, 16).T
    assert not codes.is_contiguous()
    packed = pack_7bit_codes(codes, num_states=128, chunk_codes=5)
    assert torch.equal(unpack_7bit_codes(packed, count=8, num_states=128, chunk_codes=3), codes)


@pytest.mark.parametrize("bad", [125, 126, 127, 255])
def test_default_reserved_indices_rejected_in_both_directions(bad):
    codes = torch.tensor([bad], dtype=torch.uint8)
    with pytest.raises(ValueError, match="alphabet"):
        pack_7bit_codes(codes)
    if bad < 128:
        with pytest.raises(ValueError, match="alphabet"):
            unpack_7bit_codes(codes, count=1)
    else:
        with pytest.raises(ValueError, match="padding"):
            unpack_7bit_codes(codes, count=1)


@pytest.mark.parametrize("kwargs", [{"num_states": 0}, {"num_states": 129}, {"num_states": True},
                                   {"chunk_codes": 0}, {"chunk_codes": 1 << 21}, {"max_working_bytes": 0}])
def test_integer_and_alphabet_guards(kwargs):
    with pytest.raises(ValueError):
        pack_7bit_codes(torch.tensor([1], dtype=torch.uint8), **kwargs)


def test_exact_shapes_padding_and_dtypes_rejected():
    for packed, count in ((torch.tensor([0, 0], dtype=torch.uint8), 1),
                          (torch.tensor([], dtype=torch.uint8), 1),
                          (torch.tensor([128], dtype=torch.uint8), 1)):
        with pytest.raises(ValueError):
            unpack_7bit_codes(packed, count=count)
    with pytest.raises(ValueError, match="shape"):
        unpack_7bit_codes(torch.zeros(2, 1, dtype=torch.uint8), count=1, expected_shape=(1, 2))
    for bad in (torch.tensor([1.]), torch.tensor([1]), torch.tensor(True), torch.zeros(0, 1, dtype=torch.uint8)):
        with pytest.raises(ValueError):
            pack_7bit_codes(bad)
    with pytest.raises(ValueError):
        unpack_7bit_codes(torch.tensor([0], dtype=torch.uint8), count=True)


def test_budget_rejection_precedes_output_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("output allocated before the budget guard")
    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises(MemoryError):
        pack_7bit_codes(torch.tensor([1], dtype=torch.uint8), max_working_bytes=1)
    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(MemoryError):
        unpack_7bit_codes(torch.tensor([1], dtype=torch.uint8), count=1, max_working_bytes=1)


def test_hidden_backing_storage_is_included_in_packing_budget():
    codes = torch.zeros(1000, dtype=torch.uint8)[:1]
    with pytest.raises(MemoryError):
        pack_7bit_codes(codes, max_working_bytes=400)


def test_scale_views_cannot_hide_large_backing_storage():
    quant = angular_fixture()
    quant.scales = torch.ones(20000).reshape(2, 10000)[:, :2]
    with pytest.raises(MemoryError, match="scale"):
        pack_angular_frame_quantization(quant, expected_identity=quant.identity, max_working_bytes=65536)


@pytest.mark.parametrize("columns", [1, 2, 31, 32, 33, 63, 64, 65])
def test_default_decoder_exact_for_odd_lengths_cross_block_and_expert_order(columns):
    quant = angular_fixture(columns)
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    expected = quant.read_columns(0, columns)
    assert torch.equal(packed.read_columns(0, columns), expected)
    assert torch.equal(packed.read_columns(0, columns, expert_ids=("1", "0")), expected.flip(0))
    chunks = [packed.read_columns(start, min(columns, start + 7)) for start in range(0, columns, 7)]
    assert torch.equal(torch.cat(chunks, dim=1), expected)
    for start, stop in ((0, min(1, columns)), (columns - 1, columns), (columns // 3, columns)):
        assert torch.equal(packed.read_columns(start, stop, expert_ids=("1",)), expected[1:2, start:stop])


def test_tiny_actual_default_fit_repacking_changes_no_decoded_weights():
    identity = angular_fixture(64).identity
    weights = torch.randn(2, 1, 64, generator=torch.Generator().manual_seed(11)) / 32
    quant = fit_angular_frame_reader(TensorExpertReader(identity, weights), block_size=32,
                                    chunk_blocks=1, max_working_bytes=1 << 20)
    packed = pack_angular_frame_quantization(quant, expected_identity=identity)
    assert torch.equal(quant.read_columns(0, 64), packed.read_columns(0, 64))
    assert torch.equal(quant.scales, packed.scales)


def test_custom_levels_use_declared27_state_grid_instead_of_legacy_default():
    quant = angular_fixture(33, levels=(-1., 0., 1.), angles=(45., 90.))
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    _, _, grid = _build_angular_grid(quant.angles_deg, quant.metadata["levels"])
    expected = (grid[quant.codes.long()] * quant.scales[..., None, None]).reshape(2, -1)[:, :33]
    assert torch.equal(packed.read_columns(0, 33), expected)
    assert not torch.equal(quant.read_columns(0, 33), expected)
    assert packed.metadata["configuration"]["num_angular_states"] == 27


def test_actual_packed_scales_configuration_and_block_overhead_counts():
    quant = angular_fixture(64)
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    metadata = packed.metadata
    assert packed.packed_codes.shape == (2, 2, 14)
    assert metadata["packed_index_bytes"] == packed.packed_codes.numel() == 56
    assert metadata["scale_tensor_bytes"] == packed.scales.numel() * 4 == 16
    assert metadata["factor_tensor_bytes"] == 72
    assert metadata["full_block_bits_per_weight_excluding_shared_configuration"] == 4.5
    config = json.dumps(metadata["configuration"], sort_keys=True, separators=(",", ":")).encode()
    assert metadata["configuration_json_bytes"] == len(config)
    assert metadata["tensor_and_configuration_bytes"] == 72 + len(config)
    assert metadata["effective_bits_per_source_weight_including_configuration"] > 4.5


@pytest.mark.parametrize("wrong", [{"revision": "b" * 40}, {"projection": "up_proj"},
                                  {"layout": "another matrix layout"}, {"expert_ids": ("1", "0")},
                                  {"source_storage_bytes": 1024}, {"matrix_shape": (2, 32)}])
def test_exact_source_identity_required_for_repacking_and_load(tmp_path, wrong):
    quant = angular_fixture()
    identity = replace(quant.identity, **wrong)
    with pytest.raises(ValueError, match="identity"):
        pack_angular_frame_quantization(quant, expected_identity=identity)
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    save_packed_angular_frame(packed, tmp_path / "packed")
    with pytest.raises(ValueError, match="identity"):
        load_packed_angular_frame(tmp_path / "packed", expected_identity=identity)


def test_configuration_conflicts_oversized_alphabet_scales_and_shapes_rejected():
    for field, value in (("num_angular_states", 124), ("angles_deg", [0., 90.]),
                         ("block_size", 16), ("identity_sha256", "b" * 64)):
        quant = angular_fixture()
        quant.metadata[field] = value
        with pytest.raises(ValueError):
            pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    quant = angular_fixture(levels=(-3., -2., -1., 0., 1., 2.))
    with pytest.raises(ValueError, match="levels"):
        pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    quant = angular_fixture()
    quant.scales[0, 0] = float("nan")
    with pytest.raises(ValueError, match="scales"):
        pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    quant = angular_fixture()
    quant.codes = quant.codes[:, :, :-1]
    with pytest.raises(ValueError, match="shape"):
        pack_angular_frame_quantization(quant, expected_identity=quant.identity)


def test_source_scales_and_codes_are_frozen_at_repacking():
    quant = angular_fixture()
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    expected = packed.read_columns(0, 64).clone()
    quant.codes.zero_()
    quant.scales.zero_()
    assert torch.equal(packed.read_columns(0, 64), expected)


def test_read_budgets_and_unknown_or_duplicate_experts():
    quant = angular_fixture()
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity, max_read_bytes=1)
    with pytest.raises(MemoryError):
        packed.read_columns(0, 64)
    for ids in ((), ("missing",), ("0", "0")):
        with pytest.raises(ValueError, match="expert"):
            packed.read_columns(0, 1, expert_ids=ids)
    for start, stop in ((0, 0), (-1, 1), (1, 0), (0, 65), (True, 2), (0, True)):
        with pytest.raises(ValueError):
            packed.read_columns(start, stop)


def test_custom_configuration_saved_roundtrip_and_actual_final_file_bytes(tmp_path):
    quant = angular_fixture(33, levels=(-1., 0., 1.), angles=(45., 90.))
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    output = tmp_path / "packed"
    receipt = save_packed_angular_frame(packed, output)
    loaded = load_packed_angular_frame(output, expected_identity=quant.identity)
    assert loaded.levels == (-1., 0., 1.)
    assert torch.equal(loaded.read_columns(0, 33), packed.read_columns(0, 33))
    assert receipt["serialized_total_bytes"] == sum(path.stat().st_size for path in output.iterdir())
    assert receipt["metadata_json_bytes"] == (output / "metadata.json").stat().st_size
    assert receipt["configuration_json_bytes"] == (output / "configuration.json").stat().st_size
    assert receipt["serialized_total_bytes"] > receipt["tensor_and_configuration_bytes"]
    with pytest.raises(FileExistsError):
        save_packed_angular_frame(packed, output)
    with pytest.raises(MemoryError):
        load_packed_angular_frame(output, expected_identity=quant.identity, max_artifact_bytes=1)


def test_load_rejects_undeclared_storage_and_working_budget(tmp_path):
    quant = angular_fixture()
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    output = tmp_path / "packed"
    save_packed_angular_frame(packed, output)
    with pytest.raises(MemoryError, match="workspace"):
        load_packed_angular_frame(output, expected_identity=quant.identity, max_working_bytes=1)
    (output / "unaccounted.bin").write_bytes(b"unaccounted artifact bytes")
    with pytest.raises(ValueError, match="undeclared"):
        load_packed_angular_frame(output, expected_identity=quant.identity)


@pytest.mark.parametrize("filename", ["configuration.json", "factors.pt"])
def test_serialized_corruption_rejected(tmp_path, filename):
    quant = angular_fixture()
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    output = tmp_path / "packed"
    save_packed_angular_frame(packed, output)
    path = output / filename
    raw = bytearray(path.read_bytes())
    if filename == "configuration.json":
        raw = bytearray(bytes(raw).replace(b"67.0", b"68.0"))
    else:
        raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_packed_angular_frame(output, expected_identity=quant.identity)


def test_large_nontensor_archive_member_rejected_before_torch_load(tmp_path, monkeypatch):
    quant = angular_fixture()
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    output = tmp_path / "packed"
    save_packed_angular_frame(packed, output)
    factors_path = output / "factors.pt"
    with zipfile.ZipFile(factors_path, "a") as archive:
        archive.writestr("factors/large_control_blob", b"x" * ((1 << 20) + 1))
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata.update(factors_sha256=hashlib.sha256(factors_path.read_bytes()).hexdigest(),
                    serialized_tensor_file_bytes=factors_path.stat().st_size)
    for _ in range(16):
        encoded = _metadata_bytes(metadata)
        total = len(encoded) + factors_path.stat().st_size + (output / "configuration.json").stat().st_size
        if metadata["metadata_json_bytes"] == len(encoded) and metadata["serialized_total_bytes"] == total:
            break
        metadata.update(metadata_json_bytes=len(encoded), serialized_total_bytes=total)
    metadata_path.write_bytes(encoded)
    def forbidden(*args, **kwargs):
        raise AssertionError("restricted tensor load began before the container bound")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="container"):
        load_packed_angular_frame(output, expected_identity=quant.identity)


def test_failure_cleans_only_new_staging_and_retains_source(tmp_path, monkeypatch):
    quant = angular_fixture()
    packed = pack_angular_frame_quantization(quant, expected_identity=quant.identity)
    sentinel = tmp_path / "source.txt"
    sentinel.write_text("preserve")
    def fail(*args, **kwargs):
        raise RuntimeError("controlled write failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="write failure"):
        save_packed_angular_frame(packed, tmp_path / "packed")
    assert not (tmp_path / "packed").exists()
    assert not list(tmp_path.glob(".packed-*"))
    assert sentinel.read_text() == "preserve"
