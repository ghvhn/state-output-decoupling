"""CPU storage/algebra checks; none establish GLM language or runtime quality."""

from dataclasses import replace
import hashlib
import json
import math
import zipfile

import pytest
import torch

from tda.expert_compress import (
    ExpertCompression,
    ExpertGroupIdentity,
    TensorExpertReader,
    expert_linear,
    fit_expert_group,
    load_expert_compression,
    reconstruct_expert,
    save_expert_compression,
    _cleanup_staging,
)


def identity(e=5, shape=(8, 12), dtype="float32", **kwargs):
    return ExpertGroupIdentity(
        model_id="synthetic/test", revision="immutable-test-revision", layer=4,
        projection="down_proj", layout="out_in_row_major", expert_ids=tuple(map(str, range(e))),
        matrix_shape=shape, source_dtype=dtype, **kwargs,
    )


def known_weights():
    generator = torch.Generator().manual_seed(931)
    mean = torch.randn(96, generator=generator)
    first = torch.randn(96, generator=generator)
    second = torch.randn(96, generator=generator)
    coefficients = torch.tensor([[-2., 0.], [-1., -1.], [0., 2.], [1., -1.], [2., 0.]])
    weights = mean + coefficients @ torch.stack((first, second))
    return weights.reshape(5, 8, 12)


def fit(weights=None, rank=2, dtype="float32", chunk=13):
    weights = known_weights() if weights is None else weights
    scope = identity(weights.shape[0], tuple(weights.shape[1:]))
    return fit_expert_group(TensorExpertReader(scope, weights), rank, chunk_columns=chunk, storage_dtype=dtype)


def test_known_shared_basis_exact_matrix_and_linear_roundtrip(tmp_path):
    original = known_weights()
    before = original.clone()
    result = fit(original)
    x = torch.randn(2, 3, 12, generator=torch.Generator().manual_seed(5))
    bias = torch.arange(8, dtype=torch.float32)
    for index in range(5):
        restored = reconstruct_expert(result, str(index))
        torch.testing.assert_close(restored, original[index], rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(expert_linear(result, str(index), x, bias, column_chunk=3),
                                   x @ original[index].T + bias, rtol=3e-5, atol=5e-6)
    torch.testing.assert_close(original, before, rtol=0, atol=0)
    assert result.metadata["diagnostics"]["relative_frobenius_error"] < 1e-6
    assert result.metadata["diagnostics"]["retained_centered_energy_fraction"] > .99999
    report = save_expert_compression(result, tmp_path / "encoded")
    loaded = load_expert_compression(tmp_path / "encoded", expected_identity=result.identity)
    for name in ("mean", "basis", "coefficients"):
        torch.testing.assert_close(getattr(loaded, name), getattr(result, name), rtol=0, atol=0)
    assert report["serialized_total_bytes"] == report["serialized_tensor_file_bytes"] + report["metadata_json_bytes"]


def test_mean_only_rank_zero_and_zero_variation(tmp_path):
    weights = known_weights()
    result = fit(weights, rank=0)
    assert tuple(result.basis.shape) == (0, 96)
    assert tuple(result.coefficients.shape) == (5, 0)
    for index in range(5):
        torch.testing.assert_close(reconstruct_expert(result, str(index)), weights.mean(0))
    assert result.metadata["diagnostics"]["retained_centered_energy_fraction"] == 0
    save_expert_compression(result, tmp_path / "zero")
    loaded = load_expert_compression(tmp_path / "zero", expected_identity=result.identity)
    torch.testing.assert_close(loaded.mean, result.mean)
    identical = weights[:1].expand(5, -1, -1).clone()
    constant = fit(identical, rank=0)
    assert constant.metadata["diagnostics"]["retained_centered_energy_fraction"] == 1
    assert constant.metadata["diagnostics"]["reconstruction_error_frobenius_sq"] == 0


def test_independent_experts_have_loss_and_full_centered_rank_restores_them():
    weights = torch.randn(5, 8, 12, generator=torch.Generator().manual_seed(842))
    compressed = fit(weights, rank=1)
    complete = fit(weights, rank=4)
    assert compressed.metadata["diagnostics"]["relative_frobenius_error"] > .5
    assert complete.metadata["diagnostics"]["relative_frobenius_error"] < 1e-6
    assert not complete.metadata["tensor_storage_is_smaller"]
    assert complete.metadata["diagnostics"]["evaluation_scope"] == "fitted_weight_reconstruction_only"
    assert not complete.metadata["diagnostics"]["semantic_quality_evaluated"]
    assert not complete.metadata["diagnostics"]["runtime_savings_evaluated"]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_explicit_storage_precision_and_actual_bytes(dtype, tmp_path):
    result = fit(dtype=dtype)
    b = torch.tensor([], dtype=getattr(torch, dtype)).element_size()
    assert result.mean.dtype == getattr(torch, dtype)
    assert result.basis.dtype == getattr(torch, dtype)
    assert result.coefficients.dtype == torch.float32
    assert result.metadata["source_tensor_bytes"] == 5 * 96 * 4
    assert result.metadata["factor_tensor_bytes"] == 3 * 96 * b + 5 * 2 * 4
    report = save_expert_compression(result, tmp_path / dtype)
    actual = (tmp_path / dtype / "factors.pt").stat().st_size + (tmp_path / dtype / "metadata.json").stat().st_size
    assert actual == report["serialized_total_bytes"]
    assert actual > report["factor_tensor_bytes"]
    # Tiny synthetic tensors save raw payload bytes but container overhead can
    # make the complete artifact larger. Never claim coefficient count savings.
    assert report["serialized_is_smaller"] == (actual < report["source_tensor_bytes"])
    assert report["serialized_storage_ratio"] == actual / report["source_tensor_bytes"]


def test_fp8_byte_baseline_includes_scales_and_is_not_decoded_fp32():
    weights = known_weights()
    fp8_scope = identity(dtype="float8_e4m3fn", source_storage_bytes=5 * 96 + 40)
    result = fit_expert_group(TensorExpertReader(fp8_scope, weights), 2, chunk_columns=13,
                             storage_dtype="bfloat16")
    assert result.metadata["source_tensor_bytes"] == 520
    assert result.metadata["factor_tensor_bytes"] == 616
    assert not result.metadata["tensor_storage_is_smaller"]
    with pytest.raises(ValueError, match="including scales"):
        identity(dtype="float8_e4m3fn")


def test_streamed_reads_are_bounded_two_passes_and_preserve_order():
    scope = identity()
    source = known_weights()

    class BoundedReader:
        identity = scope
        calls = []

        def read_columns(self, start, stop, expert_ids=None):
            assert stop - start <= 13
            assert stop - start < scope.column_count
            assert tuple(expert_ids) == scope.expert_ids
            self.calls.append((start, stop))
            return source.reshape(5, -1)[:, start:stop]

    reader = BoundedReader()
    result = fit_expert_group(reader, 2, chunk_columns=13, storage_dtype="float32")
    assert len(reader.calls) == 2 * math.ceil(96 / 13)
    assert reader.calls[:8] == reader.calls[8:]
    torch.testing.assert_close(reconstruct_expert(result, "4"), source[4], rtol=3e-5, atol=3e-6)


def test_impossible_memory_plan_fails_before_any_source_read():
    class NoRead:
        identity = identity()

        def read_columns(self, *args, **kwargs):
            pytest.fail("source read happened before the memory guard")

    with pytest.raises(MemoryError, match="workspace"):
        fit_expert_group(NoRead(), 2, chunk_columns=13, max_working_bytes=50)


def test_chunk_partition_does_not_change_reconstruction():
    small = fit(chunk=1)
    larger = fit(chunk=17)
    for index in range(5):
        torch.testing.assert_close(reconstruct_expert(small, str(index)), reconstruct_expert(larger, str(index)),
                                   rtol=3e-5, atol=5e-6)


def test_expert_permutation_and_signed_column_transforms_preserve_geometry():
    weights = known_weights()
    original = fit(weights)
    order = [3, 0, 4, 1, 2]
    scope = replace(original.identity, expert_ids=tuple(map(str, order)))
    permuted = fit_expert_group(TensorExpertReader(scope, weights[order].contiguous()), 2,
                                storage_dtype="float32", chunk_columns=13)
    for expert in scope.expert_ids:
        torch.testing.assert_close(reconstruct_expert(permuted, expert), reconstruct_expert(original, expert),
                                   rtol=3e-5, atol=4e-6)
    signs = torch.where(torch.arange(96) % 3 == 0, -1., 1.).reshape(8, 12)
    signed = fit((weights * signs).contiguous())
    for index in range(5):
        torch.testing.assert_close(reconstruct_expert(signed, str(index)), weights[index] * signs,
                                   rtol=3e-5, atol=4e-6)


@pytest.mark.parametrize("field,value", [("layer", 5), ("projection", "gate_up_proj"),
                                        ("layout", "gate_up_interleaved"), ("revision", "other-revision"),
                                        ("model_id", "other/model"), ("expert_ids", ("4", "3", "2", "1", "0")),
                                        ("matrix_shape", (12, 8))])
def test_other_scope_is_rejected_for_fit_and_load(field, value, tmp_path):
    result = fit()
    wrong = replace(result.identity, **{field: value})
    reader = TensorExpertReader(result.identity, known_weights())
    with pytest.raises(ValueError, match="expected identity"):
        fit_expert_group(reader, 2, expected_identity=wrong)
    save_expert_compression(result, tmp_path / "scope")
    with pytest.raises(ValueError, match="scope"):
        load_expert_compression(tmp_path / "scope", expected_identity=wrong)


def test_scope_change_mid_stream_is_rejected():
    scope = identity()

    class MovingReader:
        identity = scope

        def read_columns(self, start, stop, expert_ids=None):
            self.identity = replace(scope, layer=99)
            return known_weights().reshape(5, -1)[:, start:stop]

    with pytest.raises(ValueError, match="changed"):
        fit_expert_group(MovingReader(), 2, chunk_columns=13)


def test_scope_change_during_final_projection_read_is_rejected():
    scope = identity()
    source = known_weights().flatten(1)
    expected_calls = 2 * math.ceil(scope.column_count / 13)

    class FinalReadMutation:
        identity = scope
        calls = 0

        def read_columns(self, start, stop, expert_ids=None):
            self.calls += 1
            if self.calls == expected_calls:
                assert stop == scope.column_count
                self.identity = replace(scope, layout="different_projection_layout")
            return source[:, start:stop]

    reader = FinalReadMutation()
    with pytest.raises(ValueError, match="changed during"):
        fit_expert_group(reader, 2, chunk_columns=13, storage_dtype="float32")
    assert reader.calls == expected_calls


@pytest.mark.parametrize("options", [dict(rank=True), dict(rank=-1), dict(rank=5), dict(rank=1.5),
                                     dict(rank=1, chunk_columns=True), dict(rank=1, chunk_columns=0),
                                     dict(rank=1, max_working_bytes=True), dict(rank=1, max_working_bytes=0),
                                     dict(rank=1, storage_dtype="int8")])
def test_invalid_fit_controls_are_rejected(options):
    with pytest.raises(ValueError):
        fit_expert_group(TensorExpertReader(identity(), known_weights()), **options)


@pytest.mark.parametrize("kind", ["nan", "wrongshape", "integer"])
def test_invalid_source_chunks_are_rejected(kind):
    class BrokenReader:
        identity = identity()

        def read_columns(self, start, stop, expert_ids=None):
            if kind == "wrongshape":
                return torch.zeros(4, stop - start)
            if kind == "integer":
                return torch.zeros(5, stop - start, dtype=torch.int64)
            return torch.full((5, stop - start), float("nan"))

    with pytest.raises(ValueError):
        fit_expert_group(BrokenReader(), 2)


def test_storage_overflow_and_fp32_range_are_rejected():
    huge = torch.full((5, 8, 12), 100000.)
    with pytest.raises(ValueError, match="storage precision"):
        fit(huge, rank=0, dtype="float16")
    with pytest.raises(ValueError, match="FP32 fitting"):
        fit(torch.full((5, 8, 12), 1e30), rank=0)


def test_save_is_atomic_no_overwrite_and_checks_full_container_budget(tmp_path):
    result = fit()
    destination = tmp_path / "model-part"
    with pytest.raises(MemoryError):
        save_expert_compression(result, destination, max_artifact_bytes=result.metadata["factor_tensor_bytes"])
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
    save_expert_compression(result, destination)
    original = (destination / "factors.pt").read_bytes()
    with pytest.raises(FileExistsError):
        save_expert_compression(result, destination)
    assert (destination / "factors.pt").read_bytes() == original


def test_corrupted_factor_file_and_metadata_limits_fail_before_torch_load(tmp_path, monkeypatch):
    result = fit()
    destination = tmp_path / "artifact"
    save_expert_compression(result, destination)

    def never(*args, **kwargs):
        pytest.fail("unsafe or over-limit artifact reached Torch loading")

    monkeypatch.setattr(torch, "load", never)
    with pytest.raises(MemoryError):
        load_expert_compression(destination, expected_identity=result.identity, max_metadata_bytes=5)
    with pytest.raises(MemoryError):
        load_expert_compression(destination, expected_identity=result.identity, max_artifact_bytes=5)
    path = destination / "factors.pt"
    contents = bytearray(path.read_bytes())
    contents[len(contents) // 2] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum"):
        load_expert_compression(destination, expected_identity=result.identity)


def _rewrite_metadata(destination, **updates):
    path = destination / "metadata.json"
    metadata = json.loads(path.read_bytes())
    metadata.update(updates)
    tensor_size = (destination / "factors.pt").stat().st_size
    for _ in range(16):
        encoded = (json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        encoded += b" " * ((-len(encoded)) % 512)
        total = tensor_size + len(encoded)
        if metadata["metadata_json_bytes"] == len(encoded) and metadata["serialized_total_bytes"] == total:
            break
        metadata.update(metadata_json_bytes=len(encoded), serialized_total_bytes=total,
                        serialized_tensor_file_bytes=tensor_size,
                        serialized_is_smaller=total < metadata["source_tensor_bytes"],
                        serialized_storage_ratio=total / metadata["source_tensor_bytes"])
    path.write_bytes(encoded)


def test_compressed_zip_cannot_hide_large_storages(tmp_path, monkeypatch):
    result = fit()
    destination = tmp_path / "artifact"
    save_expert_compression(result, destination)
    factor_path = destination / "factors.pt"
    with zipfile.ZipFile(factor_path) as old:
        entries = [(info.filename, old.read(info.filename)) for info in old.infolist()]
    with zipfile.ZipFile(factor_path, "w", compression=zipfile.ZIP_DEFLATED) as new:
        for name, value in entries:
            new.writestr(name, value)
    _rewrite_metadata(destination, factors_sha256=hashlib.sha256(factor_path.read_bytes()).hexdigest())
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("compressed archive was loaded"))
    with pytest.raises(ValueError, match="storage container"):
        load_expert_compression(destination, expected_identity=result.identity)


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan", "unsupportedclaim", "byteclaim"])
def test_invalid_factor_arrays_or_claims_cannot_be_saved(kind, tmp_path):
    result = fit()
    if kind == "dtype":
        result.coefficients = result.coefficients.half()
    elif kind == "shape":
        result.basis = result.basis[:, :-1].contiguous()
    elif kind == "nan":
        result.mean[0] = float("nan")
    elif kind == "unsupportedclaim":
        result.metadata["diagnostics"]["semantic_quality_evaluated"] = True
    else:
        result.metadata["factor_tensor_bytes"] -= 1
    with pytest.raises(ValueError):
        save_expert_compression(result, tmp_path / kind)
    assert not (tmp_path / kind).exists()


def test_reconstruction_and_linear_do_not_silently_accept_shape_or_expert_changes():
    result = fit()
    with pytest.raises(ValueError):
        reconstruct_expert(result, "unknown")
    with pytest.raises(MemoryError):
        reconstruct_expert(result, "0", max_output_bytes=10)
    with pytest.raises(ValueError):
        expert_linear(result, "0", torch.zeros(2, 11))
    with pytest.raises(ValueError):
        expert_linear(result, "0", torch.full((2, 12), float("inf")))
    with pytest.raises(ValueError):
        expert_linear(result, "0", torch.zeros(2, 12), torch.zeros(7))


def test_tensor_reader_subset_order_and_noncontiguous_source_rejection():
    weights = known_weights()
    reader = TensorExpertReader(identity(), weights)
    torch.testing.assert_close(reader.read_columns(7, 20, ["4", "1"]), weights.reshape(5, -1)[[4, 1], 7:20])
    with pytest.raises(ValueError):
        reader.read_columns(0, 5, ["0", "0"])
    with pytest.raises(ValueError):
        TensorExpertReader(identity(shape=(12, 8)), weights.transpose(1, 2))


def _external_objective(scope):
    return {"method": "trajectory_response_shared_basis_v1", "identity_sha256": scope.fingerprint,
            "semantic_quality_evaluated": False, "calibration": {"synthetic": True}}


def test_external_expert_gram_selects_subspace_without_relabelling_functional_energy(tmp_path):
    source = known_weights()
    scope = identity()
    direction = torch.tensor([0., -1., 2., -1., 0.])
    functional = direction[:, None] @ direction[None, :] * 1e8
    result = fit_expert_group(TensorExpertReader(scope, source), 1, chunk_columns=13,
                             storage_dtype="float32", expert_gram=functional,
                             objective_metadata=_external_objective(scope))
    center = source.flatten(1) - source.flatten(1).mean(0)
    retained_weight_energy = float(torch.trace(result.coefficients.T @ (center @ center.T) @ result.coefficients))
    assert result.metadata["diagnostics"]["selected_centered_energy_frobenius_sq"] == pytest.approx(retained_weight_energy, rel=1e-5)
    assert retained_weight_energy < 1e5
    assert result.metadata["objective"]["method"] == "trajectory_response_shared_basis_v1"
    assert result.metadata["objective"]["calibration"]["synthetic"]
    save_expert_compression(result, tmp_path / "external")
    loaded = load_expert_compression(tmp_path / "external", expected_identity=scope)
    assert loaded.metadata["objective"] == result.metadata["objective"]


@pytest.mark.parametrize("kind", ["nonfinite", "nonsymmetric", "noncentered", "negative", "float64", "wrongscope", "qualityclaim", "missingmetadata"])
def test_invalid_external_gram_or_provenance_fail_before_weight_reads(kind):
    scope = identity()

    class NoRead:
        identity = scope

        def read_columns(self, *args, **kwargs):
            pytest.fail("invalid external objective reached source weights")

    centered = torch.eye(5) - torch.ones(5, 5) / 5
    objective = _external_objective(scope)
    if kind == "nonfinite":
        centered[0, 0] = float("nan")
    elif kind == "nonsymmetric":
        centered[0, 1] = 20
    elif kind == "noncentered":
        centered = torch.eye(5)
    elif kind == "negative":
        centered.neg_()
    elif kind == "float64":
        centered = centered.double()
    elif kind == "wrongscope":
        objective["identity_sha256"] = "wrong"
    elif kind == "qualityclaim":
        objective["semantic_quality_evaluated"] = True
    else:
        objective = None
    with pytest.raises(ValueError):
        fit_expert_group(NoRead(), 1, expert_gram=centered, objective_metadata=objective)


def test_supplied_objective_without_gram_is_rejected_and_baseline_is_explicit():
    result = fit()
    assert result.metadata["objective"]["method"] == "weight_pca_shared_basis_v1"
    with pytest.raises(ValueError, match="external expert_gram"):
        fit_expert_group(TensorExpertReader(identity(), known_weights()), 1,
                         objective_metadata=_external_objective(identity()))


def test_factor_storage_backing_cannot_export_unrelated_values(tmp_path):
    result = fit()
    before = result.mean.clone()
    unrelated = torch.full((10000,), 90000.)
    unrelated[10:106] = result.mean
    result.mean = unrelated[10:106]
    assert result.mean.is_contiguous()
    report = save_expert_compression(result, tmp_path / "slice")
    loaded = load_expert_compression(tmp_path / "slice", expected_identity=result.identity)
    assert loaded.mean.untyped_storage().nbytes() == before.numel() * before.element_size()
    torch.testing.assert_close(loaded.mean, before, rtol=0, atol=0)
    assert report["serialized_tensor_file_bytes"] < 10000 * 4


def test_staging_cleanup_requires_exact_parent_prefix_and_preserves_neighbors(tmp_path):
    intended = tmp_path / "exports"
    intended.mkdir()
    protected = intended / "source-checkpoint"
    protected.mkdir()
    (protected / "keep.txt").write_text("source bytes")
    outside = tmp_path / ".artifact-outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("unrelated bytes")
    with pytest.raises(ValueError, match="refusing staging cleanup"):
        _cleanup_staging(outside, intended, ".artifact-")
    with pytest.raises(ValueError, match="refusing staging cleanup"):
        _cleanup_staging(protected, intended, ".artifact-")
    with pytest.raises(ValueError, match="refusing staging cleanup"):
        _cleanup_staging(intended, intended, ".artifact-")
    staging = intended / ".artifact-generated123"
    staging.mkdir()
    (staging / "temporary.txt").write_text("partial artifact")
    _cleanup_staging(staging, intended, ".artifact-")
    assert not staging.exists()
    assert (protected / "keep.txt").read_text() == "source bytes"
    assert (outside / "keep.txt").read_text() == "unrelated bytes"
