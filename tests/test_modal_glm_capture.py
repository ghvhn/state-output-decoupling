from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest


SPEC = importlib.util.spec_from_file_location(
    "modal_glm_capture_under_test", Path(__file__).resolve().parents[1] / "scripts/modal_glm_capture.py")
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)
NOW = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)


def evidence(**changes):
    result = {
        "protocol": capture.PROTOCOL, "provider": "modal", "workspace": "test-workspace",
        "billing_cycle": "2026-10", "observed_at": NOW.isoformat(),
        "verified_by_operator": True, "active_profile_workspace_verified": True,
        "other_workloads_stopped": True, "workspace_usage_usd": "4.50",
        "workspace_usage_limit_usd": "30", "workspace_spend_limit_usd": "0",
        "credits_remaining_usd": "25.50", "free_volume_bytes_remaining": 1024 ** 4,
        "gpu_concurrency_limit": 10,
    }
    result.update(changes)
    return result


def test_fresh_credit_is_clamped_to_gross_workspace_remainder():
    value = evidence(credits_remaining_usd="100", workspace_usage_limit_usd="20")
    assert capture.verify_budget_evidence(value, "test-workspace", now=NOW, needs_gpu=True) == Decimal("15.50")


@pytest.mark.parametrize("changes", [
    {"verified_by_operator": False}, {"active_profile_workspace_verified": False},
    {"other_workloads_stopped": False}, {"billing_cycle": "2026-09"},
    {"observed_at": (NOW - timedelta(minutes=16)).isoformat()},
    {"observed_at": (NOW + timedelta(seconds=1)).isoformat()},
    {"observed_at": "2026-10-02T12:00:00"}, {"workspace_spend_limit_usd": "0.01"},
    {"authorized_paid_backup_usd": "5.01"}, {"workspace_paid_spend_usd": "1"}, {"workspace_usage_usd": "30"},
    {"credits_remaining_usd": "0"}, {"credits_remaining_usd": "NaN"},
    {"credits_remaining_usd": True}, {"gpu_concurrency_limit": 7},
])
def test_no_cloud_budget_is_inferred_from_stale_incomplete_or_paid_configuration(changes):
    with pytest.raises(capture.CaptureLaunchError):
        capture.verify_budget_evidence(evidence(**changes), "test-workspace", now=NOW, needs_gpu=True)


def test_workspace_must_be_the_operator_verified_active_profile():
    with pytest.raises(capture.CaptureLaunchError, match="workspace"):
        capture.verify_budget_evidence(evidence(), "another-workspace", now=NOW)


def test_explicit_five_dollar_backup_preserves_trial_ceiling():
    value = evidence(workspace_usage_usd="0", workspace_usage_limit_usd="42.50",
        credits_remaining_usd="30", workspace_spend_limit_usd="5", authorized_paid_backup_usd="5")
    assert capture.verify_budget_evidence(value, "test-workspace", now=NOW) == Decimal("35")
    value.update(workspace_usage_usd="31", credits_remaining_usd="0", workspace_paid_spend_usd="1")
    assert capture.verify_budget_evidence(value, "test-workspace", now=NOW) == Decimal("4")
    value["workspace_spend_limit_usd"] = "6"
    with pytest.raises(capture.CaptureLaunchError, match="authorized backup"):
        capture.verify_budget_evidence(value, "test-workspace", now=NOW)


def test_volume_remaining_is_phase_specific_after_checkpoint_staging():
    staged = evidence(free_volume_bytes_remaining=200 * 1024 ** 3)
    assert capture.verify_budget_evidence(staged, "test-workspace", now=NOW,
                                         required_volume_bytes=capture.MAX_CAPTURE_BYTES) > 0
    with pytest.raises(capture.CaptureLaunchError, match="Volume"):
        capture.verify_budget_evidence(staged, "test-workspace", now=NOW,
                                       required_volume_bytes=755_000_000_000)


@pytest.mark.parametrize("phase", ["build", "stage"])
def test_all_preparation_is_cpu_only_and_never_warms_or_retries(phase):
    policy = capture.resource_policy(phase, 600)
    assert policy["gpu"] is None
    assert policy["cpu"] == (2, 2)
    assert policy["memory"] == (8192, 8192)
    assert policy["min_containers"] == policy["buffer_containers"] == policy["retries"] == 0
    assert policy["max_containers"] == 1
    assert policy["scaledown_window"] == 2
    assert policy["single_use_containers"]
    assert not policy["enable_memory_snapshot"]


def test_capture_bounds_include_startup_ram_cpu_idle_and_reserve():
    remaining = Decimal("14")
    seconds = capture.capture_seconds(remaining, max_seconds=5000)
    policy = capture.resource_policy("capture", seconds)
    assert policy["gpu"] == "H200:8"
    assert capture.cost_ceiling(policy, seconds) + Decimal("2") <= remaining
    assert capture.cost_ceiling(policy, seconds + 1) + Decimal("2") > remaining
    expected_rate = Decimal("0.001261") * 8 + Decimal("0.0000131") * 16 + Decimal("0.00000222") * 128
    assert capture.cost_ceiling(policy, seconds) == expected_rate * (seconds + 62)


def test_insufficient_credit_refuses_instead_of_shortening_below_save_envelope():
    with pytest.raises(capture.CaptureLaunchError):
        capture.capture_seconds("2.50")


def test_cost_bound_cannot_omit_actual_burst_limits():
    policy = capture.resource_policy("capture", 600)
    policy["cpu"] = (16, 32)
    with pytest.raises(capture.CaptureLaunchError, match="hard"):
        capture.cost_ceiling(policy, 600)


def payload():
    content = b"original native fp8 bytes"
    return {
        "protocol": capture.PROTOCOL, "repository": capture.MODEL_ID, "revision": capture.REVISION,
        "files": [{"name": "model-00001-of-00001.safetensors", "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}],
        "manifest": {"id": capture.MODEL_ID, "sha": capture.REVISION},
        "identity": {"revision": capture.REVISION}, "total_bytes": len(content),
    }, content


def test_cpu_staging_uses_only_pinned_public_files_and_atomic_completion(tmp_path):
    request, content = payload()
    downloads = []
    commits = []

    def download(**kwargs):
        downloads.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(content)
        return str(path)

    receipt = capture.stage_checkpoint(request, tmp_path, deadline_epoch=time.time() + 60,
                                       download_file=download, commit=lambda: commits.append(True))
    assert receipt["all_sha256_verified"]
    assert (tmp_path / capture.REVISION / "stage_receipt.json").is_file()
    assert not (tmp_path / (capture.REVISION + ".incomplete")).exists()
    assert downloads[0]["revision"] == capture.REVISION
    assert downloads[0]["repo_id"] == capture.MODEL_ID
    assert downloads[0]["token"] is False
    assert len(commits) == 2
    capture.stage_checkpoint(request, tmp_path, deadline_epoch=time.time() + 60,
                             download_file=lambda **_: pytest.fail("complete source was downloaded again"))


def test_corrupt_download_does_not_publish_a_complete_checkpoint(tmp_path):
    request, content = payload()

    def corrupt(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(b"X" * len(content))
        return path

    with pytest.raises(capture.CaptureLaunchError, match="digest"):
        capture.stage_checkpoint(request, tmp_path, deadline_epoch=time.time() + 60, download_file=corrupt)
    assert not (tmp_path / capture.REVISION).exists()
    assert (tmp_path / (capture.REVISION + ".incomplete")).exists()


def test_staging_stops_before_any_download_when_deadline_has_passed(tmp_path):
    request, _ = payload()
    with pytest.raises(capture.CaptureLaunchError, match="deadline"):
        capture.stage_checkpoint(request, tmp_path, deadline_epoch=time.time() - 1,
                                 download_file=lambda **_: pytest.fail("download after deadline"))


def test_model_code_is_never_a_stage_target(tmp_path):
    request, _ = payload()
    request["files"][0]["name"] = "modeling_glm.py"
    with pytest.raises(capture.CaptureLaunchError, match="source code"):
        capture.stage_checkpoint(request, tmp_path, deadline_epoch=time.time() + 60,
                                 download_file=lambda **_: pytest.fail("model code download"))


def test_selected_corpus_is_bounded_tokenized_and_provenance_checked(tmp_path):
    identity = {"vocab_size": 10, "hidden_size": 4, "revision": capture.REVISION,
                "tokenizer_sha256": "a" * 64, "chat_template_sha256": "b" * 64}
    path = tmp_path / "invariants/out/selected.json"
    path.parent.mkdir(parents=True)
    corpus = {"format": "glm_native_capture_corpus_v1", "model_id": capture.MODEL_ID,
              "revision": capture.REVISION, "tokenizer_identity": {
                  "vocab_size": 10, "tokenizer_sha256": "a" * 64, "chat_template_sha256": "b" * 64,
                  "tokenizer_config_sha256": "c" * 64, "config_sha256": "d" * 64},
              "examples": [{"token_ids": [1, 2, 3], "selected_positions": [0, 1], "pairs": [[0, 1]],
               "environment_id": "conversation", "trajectory_id": "contrast-1", "conversation_id": "dialogue-1",
               "source_sha256": "e" * 64, "source_record_ids": ["record-1"], "kind": "conversation", "split": "calibration",
               "context": {"retained_token_range": [0, 3], "rendered_token_count": 3, "rendered_sha256": "f" * 64}}]}
    path.write_text(json.dumps(corpus))
    selected, data, digest = capture.load_corpus(path, identity, workspace_root=tmp_path)
    assert selected == corpus
    assert digest == hashlib.sha256(data).hexdigest()
    corpus["examples"][0]["token_ids"] = [1, True]
    path.write_text(json.dumps(corpus))
    with pytest.raises(capture.CaptureLaunchError, match="token IDs"):
        capture.load_corpus(path, identity, workspace_root=tmp_path)


def test_corpus_cannot_upload_an_arbitrary_local_json_file(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text("{}")
    with pytest.raises(capture.CaptureLaunchError, match="prepared"):
        capture.load_corpus(path, {}, workspace_root=tmp_path)


def test_oversize_corpus_is_rejected_before_reading(tmp_path, monkeypatch):
    path = tmp_path / "invariants/out/selected.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 11)
    monkeypatch.setattr(capture, "MAX_CORPUS_BYTES", 10)
    with pytest.raises(capture.CaptureLaunchError, match="at most"):
        capture.load_corpus(path, {}, workspace_root=tmp_path)


class Image:
    def __init__(self):
        self.uploads = []

    def add_local_file(self, local, remote, **kwargs):
        self.uploads.append((local, remote))
        return self

    def env(self, values):
        assert not self.uploads, "build layers must precede local source mounts"
        return self


def test_source_upload_is_a_small_exact_allowlist_never_a_directory(tmp_path):
    for relative in capture.SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# allowed source\n")
    (tmp_path / ".env").write_text("PRIVATE=value")
    image = capture.upload_sources(Image(), root=tmp_path)
    assert len(image.uploads) == 4
    assert all(remote.startswith("/opt/capture/") and Path(local).suffix == ".py" for local, remote in image.uploads)
    assert all(".env" not in local for local, _ in image.uploads)


def test_cloud_guards_run_before_importing_or_initializing_modal(monkeypatch):
    monkeypatch.setattr(capture.importlib, "import_module", lambda _: pytest.fail("Modal initialized before budget check"))
    request, _ = payload()
    with pytest.raises(capture.CaptureLaunchError):
        capture.launch_cloud("stage", request, evidence={}, workspace="test-workspace",
                             volume_name="test-volume", execution_seconds=600)


def test_default_cli_plan_is_offline_even_when_modal_is_missing(monkeypatch, capsys):
    request, _ = payload()
    monkeypatch.setattr(capture, "load_stage_payload", lambda _: request)
    monkeypatch.setattr(capture, "launch_cloud", lambda *a, **k: pytest.fail("default launched cloud"))
    assert capture.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cloud_started"] is False
    assert result["remaining_credit_verified"] is False


def test_failed_capture_returns_failure_exit_code_and_retains_its_receipt(tmp_path, monkeypatch, capsys):
    request, _ = payload()
    monkeypatch.setattr(capture, "load_stage_payload", lambda _: request)
    monkeypatch.setattr(capture, "read_bounded_json", lambda *a, **k: ({}, b"{}"))
    monkeypatch.setattr(capture, "verify_budget_evidence", lambda *a, **k: Decimal("30"))
    failed = {"protocol": capture.PROTOCOL, "return_code": 1, "run_id": "failed-fixture"}
    monkeypatch.setattr(capture, "launch_cloud", lambda *a, **k: failed)
    target = tmp_path / "failed-result.json"
    assert capture.main(["--mode", "capture", "--budget-evidence", "fixture.json", "--output", str(target)]) == 2
    assert json.loads(target.read_text()) == failed
    assert json.loads(capsys.readouterr().out)["return_code"] == 1


def test_elastic_worker_cannot_restart_or_interpret_shell_text(monkeypatch):
    calls = []

    class Process:
        def wait(self, timeout):
            assert timeout == 600
            return 0

    monkeypatch.setattr(capture.subprocess, "Popen", lambda command, **kwargs: calls.append((command, kwargs)) or Process())
    assert capture.run_runtime_process("/capture/data/quote;not-a-shell.json", timeout_seconds=600) == 0
    command, options = calls[0]
    assert "--max-restarts=0" in command
    assert "--standalone" not in command
    assert "--rdzv-backend=static" in command
    assert "--rdzv-conf=timeout=45" in command
    assert "--master-addr=127.0.0.1" in command
    assert "--node-rank=0" in command
    assert command[-1] == "/capture/data/quote;not-a-shell.json"
    assert "shell" not in options
    assert options["start_new_session"]


def test_cpu_rendezvous_uses_identical_local_transport_without_model_config(monkeypatch):
    calls = []
    class Process:
        def wait(self, timeout):
            return 0
    monkeypatch.setattr(capture.subprocess, "Popen", lambda command, **kwargs: calls.append(command) or Process())
    assert capture.run_runtime_process(None, timeout_seconds=120, check_rendezvous=True) == 0
    assert "--master-addr=127.0.0.1" in calls[0]
    assert calls[0][-1] == "--check-rendezvous"
    assert "--config" not in calls[0]


def test_image_dependencies_are_exact_and_model_metadata_compatible():
    assert capture.validate_image_config(capture.DEFAULT_IMAGE)["validated_on_glm_h200"] is False
    bad = dict(capture.DEFAULT_IMAGE, packages=["transformers>=5.15"])
    with pytest.raises(capture.CaptureLaunchError, match="exact"):
        capture.validate_image_config(bad)


def test_image_rejects_fp8_kernel_version_that_failed_cloud_cpu_guard():
    config = dict(capture.DEFAULT_IMAGE)
    config["packages"] = [p.replace("kernels==0.16.0", "kernels==0.17.1") for p in config["packages"]]
    with pytest.raises(capture.CaptureLaunchError, match="kernels0.16"):
        capture.validate_image_config(config)
