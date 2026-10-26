"""Saved-candidate nonlinear integration, immutable local CPU fixtures only."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import weakref

import pytest
import torch

from scripts import calibrate_glm_compression as calibration
from scripts import evaluate_glm_moe_compression as trial
from tda.expert_compress import TensorExpertReader
from tda import glm_moe_sample_eval as evaluator
from tda.glm_capture_archive import CaptureArchiveWriter
from tda.trajectory_compress import InputStateScope


_spec = importlib.util.spec_from_file_location("native_calibration_fixture",
    Path(__file__).with_name("test_calibrate_glm_compression.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


@pytest.fixture
def saved(tmp_path):
    gate, archive = _fixtures.fixture(tmp_path, probe_seed=True, routers=True, token_gap=3)
    comparison = tmp_path / "comparison"
    reports = calibration.compare(gate, archive, comparison, ranks=(0, 1, 2),
        storage_dtype="float32", max_train=2, max_heldout=2, row_chunk=1,
        include_probe_span=True, include_probe_residual=True,
        include_probe_residual_controls=True, include_probe_hybrid=True)
    assert len(reports) == 16
    weights = torch.eye(2).repeat(gate.identity.expert_count, 1, 1)
    up = TensorExpertReader(replace(gate.identity, projection="up_proj"), weights.clone())
    down = TensorExpertReader(replace(gate.identity, projection="down_proj"), weights.clone())
    return gate, up, down, archive, comparison, tmp_path / "nonlinear"


def evaluate(saved, **kwargs):
    return trial.evaluate_saved_comparison(*saved, expected_origin="synthetic_fixture",
        row_chunk=1, batch_records=2, **kwargs)


def digest_tree(directory):
    return {p.relative_to(directory).as_posix(): trial.sha_file(p)
        for p in directory.rglob("*") if p.is_file()}


def test_saved_seven_candidates_run_without_refitting_and_preserve_sources(saved, monkeypatch):
    gate, up, down, archive, comparison, out = saved
    old_artifacts, old_archive = digest_tree(comparison), digest_tree(archive)
    old_weights = [value.weights.clone() for value in (gate, up, down)]
    def fitting_forbidden(*args, **kwargs):
        pytest.fail("saved candidate evaluation attempted to fit or quantize")
    from tda import expert_compress, trajectory_compress, probe_span_compress, probe_residual_compress, block_quant
    for module, names in ((expert_compress, ("fit_expert_group",)),
        (trajectory_compress, ("fit_trajectory_group",)),
        (probe_span_compress, ("fit_probe_span_group",)),
        (probe_residual_compress, ("fit_probe_residual_group", "fit_scoped_residual_basis",
            "make_calibration_pca_basis", "make_random_basis", "make_probe_pca_complement_basis")),
        (block_quant, ("quantize_reader",))):
        for name in names:
            monkeypatch.setattr(module, name, fitting_forbidden)
    reports = evaluate(saved)
    assert [r["variant"] for r in reports] == list(trial.PRIMARY_VARIANTS)
    originals = {r["variant"]: r for r in trial.read_json(comparison / "results.json")}
    for report in reports:
        assert report["candidate_refit"] is False and report["fitting_performed"] is False
        assert report["method"] == evaluator.METHOD
        assert report["compression_method"] == originals[report["variant"]]["method"]
        assert report["serialized_total_bytes"] == originals[report["variant"]]["serialized_total_bytes"]
        assert report["factor_tensor_bytes"] == originals[report["variant"]]["factor_tensor_bytes"]
        assert sum(f["bytes"] for f in report["candidate_artifact_verification"]["files"]) == report["serialized_total_bytes"]
        assert report["splits"]["heldout"]["records"] == 2
        assert report["splits"]["heldout"]["pairs"] == 1
        assert report["seed_pole_response"]["pole_records"] == 2
        assert report["seed_pole_response"]["inputs"].startswith("exact raw measured")
        assert report["seed_pole_response"]["semantic_meanings_validated"] is False
        for field in ("native_dynamic_fp8_kernel_evaluated", "native_router_mixture_evaluated",
            "modified_decoder_evaluated", "logit_intervention_evaluated", "generation_evaluated",
            "semantic_quality_evaluated", "runtime_savings_evaluated"):
            assert report[field] is False
    evidence = trial.read_json(out / "evidence.json")
    assert evidence["source_archive_status"] == "partial"
    assert evidence["original_comparison_evidence"]["pairs"]["heldout"][0]["token_gap"] == 3
    assert evidence["routing_coverage"]["available"] is True
    assert evidence["routing_used_to_weight_responses"] is False
    assert [row["record_id"] for row in evidence["raw_seed_poles"]["records"]] == ["seed:0", "seed:1"]
    settings = trial.read_json(out / "settings.json")
    assert settings["record_origin"] == "synthetic_fixture"
    assert all(r["source_hashes_verified"] is False for r in settings["source_receipts"].values())
    assert trial.read_json(out / "status.json")["status"] == "finished"
    assert trial.read_json(out / "results.json") == reports
    assert not list(out.parent.glob(f".{out.name}-*"))
    assert digest_tree(comparison) == old_artifacts and digest_tree(archive) == old_archive
    for reader, weights in zip((gate, up, down), old_weights):
        torch.testing.assert_close(reader.weights, weights, rtol=0, atol=0)


def test_all_sixteen_saved_formats_are_loaded_one_at_a_time(saved, monkeypatch):
    original = trial.load_gate_candidate
    references = []
    def load(*args, **kwargs):
        assert not references or references[-1]() is None, "previous candidate remained resident"
        candidate, metadata = original(*args, **kwargs)
        references.append(weakref.ref(candidate))
        return candidate, metadata
    monkeypatch.setattr(trial, "load_gate_candidate", load)
    reports = evaluate(saved, variants="all")
    assert len(reports) == 16 and len(references) == 16
    assert all(reference() is None for reference in references)
    assert {r["rank"] for r in reports if r["compression_method"] in trial.FACTOR_METHODS} == {0, 1, 2}


@pytest.mark.parametrize("variants", [(), ("unknown",), ("probe_residual", "probe_residual")])
def test_unknown_empty_and_duplicate_variant_lists_reject_before_candidate_loading(saved, variants, monkeypatch):
    monkeypatch.setattr(trial, "load_gate_candidate", lambda *a, **k: pytest.fail("candidate loaded"))
    with pytest.raises(ValueError, match="unique saved"):
        evaluate(saved, variants=variants)
    assert not saved[-1].exists()


@pytest.mark.parametrize("field,value", [("projection", "gate_proj"), ("revision", "d" * 40),
    ("expert_ids", ("3", "2", "1", "0")), ("matrix_shape", (3, 2))])
def test_original_up_scope_must_match_gate_before_loading_archive(saved, field, value, monkeypatch):
    gate, up, down, archive, comparison, out = saved
    wrong = TensorExpertReader(replace(up.identity, **{field: value}),
        torch.zeros(4, *value) if field == "matrix_shape" else up.weights)
    monkeypatch.setattr(trial, "load_archive_manifest", lambda *a, **k: pytest.fail("archive loaded"))
    with pytest.raises(ValueError, match="scope or nonlinear matrix geometry"):
        evaluate((gate, wrong, down, archive, comparison, out))


def test_changed_archive_states_do_not_silently_reuse_the_saved_split(saved, tmp_path, monkeypatch):
    gate, up, down, _, comparison, out = saved
    _, altered_archive = _fixtures.fixture(tmp_path / "altered", heldout_scale=17,
        probe_seed=True, routers=True, token_gap=3)
    monkeypatch.setattr(trial, "load_gate_candidate", lambda *a, **k: pytest.fail("candidate loaded"))
    with pytest.raises(ValueError, match="archive evidence or whole-trajectory selection differs"):
        evaluate((gate, up, down, altered_archive, comparison, out))
    assert not out.exists()


def test_altered_artifact_is_rejected_and_failed_staging_retained(saved, monkeypatch):
    gate, up, down, archive, comparison, out = saved
    correction = comparison / "probe_residual/correction.pt"
    contents = bytearray(correction.read_bytes())
    contents[len(contents) // 2] ^= 1
    correction.write_bytes(contents)
    monkeypatch.setattr(evaluator, "evaluate_sampled_moe_candidate", lambda *a, **k: pytest.fail("corrupt artifact evaluated"))
    with pytest.raises(ValueError) as error:
        evaluate(saved, variants=("probe_residual",))
    staging = Path(error.value.evaluation_staging_directory)
    assert staging.parent.resolve() == out.parent.resolve()
    assert not out.exists()
    status = trial.read_json(staging / "status.json")
    assert status["status"] == "failed" and status["completed_variants"] == []


def test_second_candidate_failure_keeps_completed_result_without_publishing(saved, monkeypatch):
    original = evaluator.evaluate_sampled_moe_candidate
    calls = []
    def evaluate_one(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected fixture evaluation failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluator, "evaluate_sampled_moe_candidate", evaluate_one)
    with pytest.raises(RuntimeError, match="injected fixture") as error:
        evaluate(saved, variants=("symmetric_int4", "probe_residual"))
    staging = Path(error.value.evaluation_staging_directory)
    assert not saved[-1].exists()
    assert trial.read_json(staging / "status.json")["completed_variants"] == ["symmetric_int4"]
    assert [r["variant"] for r in trial.read_json(staging / "results.json")] == ["symmetric_int4"]


def test_existing_output_is_preserved_before_any_load(saved, monkeypatch):
    out = saved[-1]
    out.mkdir()
    (out / "keep").write_text("unrelated")
    monkeypatch.setattr(trial, "load_archive_manifest", lambda *a, **k: pytest.fail("archive loaded"))
    with pytest.raises(FileExistsError):
        evaluate(saved)
    assert (out / "keep").read_text() == "unrelated"


def test_fixture_cannot_be_presented_as_measured_evidence(saved):
    with pytest.raises(ValueError, match="source identity or evidence origin"):
        trial.evaluate_saved_comparison(*saved)


@pytest.mark.parametrize("file,mutate", [
    ("settings.json", lambda value: value.update(heldout_used_for_fit=True)),
    ("status.json", lambda value: value.update(status="running")),
    ("results.json", lambda value: value[0].update(heldout_used_for_fit=True)),
])
def test_stale_or_heldout_fitted_comparison_metadata_rejected_before_loading(saved, file, mutate, monkeypatch):
    comparison = saved[-2]
    value = trial.read_json(comparison / file)
    mutate(value)
    (comparison / file).write_text(json.dumps(value))
    monkeypatch.setattr(trial, "load_archive_manifest", lambda *a, **k: pytest.fail("archive loaded"))
    with pytest.raises(ValueError):
        evaluate(saved)
    assert not saved[-1].exists()


@pytest.mark.parametrize("option,value", [("row_chunk", 0), ("row_chunk", 33), ("batch_records", 257),
    ("max_working_bytes", True), ("max_working_bytes", (1 << 30) + 1),
    ("expert_chunk", 1.0), ("expert_chunk", 2)])
def test_invalid_count_and_working_bounds_reject_before_source_or_archive_load(saved, option, value, monkeypatch):
    monkeypatch.setattr(trial, "source_scopes", lambda *a: pytest.fail("sources inspected"))
    kwargs = {"row_chunk": 1, "batch_records": 2, "expected_origin": "synthetic_fixture", option: value}
    with pytest.raises(ValueError):
        trial.evaluate_saved_comparison(*saved, **kwargs)


def test_raw_seed_loader_keeps_recorded_values_dtype_ids_and_preflights_budget(saved, monkeypatch):
    gate, _, _, archive, _, _ = saved
    manifest = trial.load_archive_manifest(archive)
    scope = trial.load_samples(archive, gate.identity, max_train=2, max_heldout=2)[0]
    bank = trial.native_probe_bank(archive, tensor_name="layer_3_expert_input")
    records, evidence = trial.load_seed_records(archive, manifest, scope, bank, max_working_bytes=1 << 20)
    torch.testing.assert_close(records["p_seed"]["negative"].state, torch.tensor([0., -1.]), rtol=0, atol=0)
    torch.testing.assert_close(records["p_seed"]["positive"].state, torch.tensor([0., 1.]), rtol=0, atol=0)
    assert records["p_seed"]["positive"].record_id == "seed:1"
    assert records["p_seed"]["positive"].origin == "synthetic_fixture"
    assert all(row["dtype"] == "torch.float32" for row in evidence["records"])
    monkeypatch.setattr(trial, "load_archive_entry", lambda *a, **k: pytest.fail("payload read before budget guard"))
    with pytest.raises(MemoryError, match="chunk buffers"):
        trial.load_seed_records(archive, manifest, scope, bank, max_working_bytes=1)


def test_raw_bfloat16_asymmetric_seed_endpoints_are_preserved_exactly(tmp_path):
    identity = replace(_fixtures.fixture(tmp_path / "identity")[0].identity, source_dtype="bfloat16")
    teacher = {"model_id": "tiny", "revision": "a" * 40, "hidden_size": 2,
        "vocab_size": 4, "num_layers": 4, "source_precision": "bfloat16",
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64}
    archive = tmp_path / "raw_bf16"
    writer = CaptureArchiveWriter(archive, teacher, origin="synthetic_fixture")
    states = torch.tensor([[0.28125, -1.359375], [1.7421875, 0.40625]], dtype=torch.bfloat16)
    rows = [{"id": f"raw:{i}", "trajectory_id": f"raw-trajectory-{i}", "environment_id": "seed",
        "split": "calibration", "kind": "probe_framing", "probe_id": "p_raw",
        "polarity": -1 if i == 0 else 1, "token_position": 0,
        "next_token_id": -1, "next_token_observed": False} for i in range(2)]
    writer.write_chunk("raw", {"final_hidden": states.clone(), "layer_3_expert_input": states.clone(),
        "positions": torch.tensor([0, 0]), "next_token_ids": torch.tensor([-1, -1])},
        {"rows": rows, "state_sites": {"final_hidden": "post_final_norm_lm_head_input",
            "layer_3_expert_input": trial.SITE}})
    writer.finalize("partial")
    bank = trial.native_probe_bank(archive, tensor_name="layer_3_expert_input")
    scope = InputStateScope.from_identity(identity, state_site=trial.SITE)
    records, evidence = trial.load_seed_records(archive, trial.load_archive_manifest(archive), scope, bank,
        max_working_bytes=1 << 20)
    for i, side in enumerate(("negative", "positive")):
        actual = records["p_raw"][side].state
        assert actual.dtype == torch.bfloat16
        assert actual.view(torch.uint8).numpy().tobytes() == states[i].view(torch.uint8).numpy().tobytes()
    assert all(row["dtype"] == "torch.bfloat16" for row in evidence["records"])


def test_pinned_silu_config_requires_all_three_source_receipt_hashes(saved, tmp_path):
    gate = saved[0]
    metadata = tmp_path / "model_metadata"
    metadata.mkdir()
    config = {"hidden_act": "silu", "model_type": "glm_moe_dsa", "hidden_size": 2,
        "moe_intermediate_size": 2, "num_hidden_layers": 4}
    path = metadata / "config.json"
    path.write_text(json.dumps(config))
    receipts = {role: {"source_metadata_sha256_from_receipt": {"config.json": trial.sha_file(path)}}
        for role in ("gate", "up", "down")}
    teacher = {"num_layers": 4}
    result = trial.architecture_receipt(metadata, gate, receipts, teacher, "measured_model_state")
    assert result["hidden_act"] == "silu" and result["config_file_sha256"] == trial.sha_file(path)
    receipts["down"]["source_metadata_sha256_from_receipt"]["config.json"] = "0" * 64
    with pytest.raises(ValueError, match="config hash differs"):
        trial.architecture_receipt(metadata, gate, receipts, teacher, "measured_model_state")
    config["hidden_act"] = "gelu"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="SiLU expert geometry"):
        trial.architecture_receipt(metadata, gate, receipts, teacher, "measured_model_state")


def cli_arguments(tmp_path):
    return [argument for name in ("sample", "up-sample", "down-sample", "comparison", "archive", "out")
        for argument in (f"--{name}", str(tmp_path / name))]


@pytest.mark.parametrize("option,value", [("--threads", "17"), ("--budget", "1024"),
    ("--row-chunk", "33"), ("--batch-records", "257"), ("--expert-chunk", "2")])
def test_cli_bad_bounds_reject_before_opening_sources(tmp_path, option, value, monkeypatch):
    monkeypatch.setattr(trial, "GLMWeightSampleReader", lambda *a: pytest.fail("source opened"))
    with pytest.raises(SystemExit) as error:
        trial.main(cli_arguments(tmp_path) + [option, value])
    assert error.value.code == 2


def test_cli_defaults_and_source_contexts_close_on_evaluation_failure(tmp_path, monkeypatch):
    events, calls = [], []
    class Reader:
        def __init__(self, path):
            self.path = path
        def __enter__(self):
            events.append(("open", self.path.name))
            return self
        def __exit__(self, *exc):
            events.append(("close", self.path.name))
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("fixture guard failure")
    monkeypatch.setattr(trial, "GLMWeightSampleReader", Reader)
    monkeypatch.setattr(trial, "evaluate_saved_comparison", fail)
    old_threads = torch.get_num_threads()
    try:
        with pytest.raises(SystemExit) as error:
            trial.main(cli_arguments(tmp_path))
        assert error.value.code == 2
        assert calls[0][1]["row_chunk"] == 16 and calls[0][1]["batch_records"] == 128
        assert calls[0][1]["variants"] is None
        assert calls[0][1]["metadata"] == trial.ROOT / "invariants/out/glm_5_3_preflight"
        assert events == [("open", "sample"), ("open", "up-sample"), ("open", "down-sample"),
            ("close", "down-sample"), ("close", "up-sample"), ("close", "sample")]
    finally:
        torch.set_num_threads(old_threads)
