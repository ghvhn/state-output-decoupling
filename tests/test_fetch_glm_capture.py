import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("glm_fetch", Path(__file__).resolve().parents[1] / "scripts/fetch_glm_capture.py")
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)
RUN = "20261002T150000-abcdef123456"


class Volume:
    def __init__(self):
        payload = b"a measured data-only artifact"
        entry = {"file": "chunk-first.pt", "role": "capture", "bytes": len(payload),
                 "sha256": hashlib.sha256(payload).hexdigest()}
        self.manifest = {"format": "glm-native-capture-v1", "origin": "synthetic_fixture",
                         "status": "partial", "entries": [entry], "artifact_bytes": len(payload), "head_rows": 0}
        self.payload, self.calls = payload, []
    def read_file(self, path):
        self.calls.append(path)
        data = json.dumps(self.manifest).encode() if path.endswith("manifest.json") else self.payload
        return iter([data[:3], data[3:]])


def test_fetch_streams_only_declared_capture_artifacts(tmp_path):
    volume = Volume()
    report = fetch.fetch_archive(volume, RUN, tmp_path / "capture")
    assert report["downloaded_bytes"] == len(volume.payload)
    assert all(path.startswith(f"/runs/{RUN}/archive/") for path in volume.calls)
    assert (tmp_path / "capture/chunk-first.pt").read_bytes() == volume.payload
    assert json.loads((tmp_path / "capture/manifest.json").read_text()) == volume.manifest


@pytest.mark.parametrize("case", ["path", "duplicate", "size", "checksum", "accounting"])
def test_fetch_preflight_refuses_invalid_manifest_before_local_writes(tmp_path, case):
    volume = Volume()
    entry = volume.manifest["entries"][0]
    if case == "path": entry["file"] = "../checkpoint.safetensors"
    elif case == "duplicate": volume.manifest["entries"].append(entry)
    elif case == "size": entry["bytes"] = fetch.MAX_ARTIFACT + 1
    elif case == "checksum": entry["sha256"] = "bad"
    else: volume.manifest["artifact_bytes"] += 1
    with pytest.raises(ValueError): fetch.fetch_archive(volume, RUN, tmp_path / "capture")
    assert not (tmp_path / "capture").exists()
    assert len(volume.calls) == 1


@pytest.mark.parametrize("payload", [b"changed artifact", b"X" * 100])
def test_fetch_corruption_keeps_a_partial_index_and_no_invalid_artifact(tmp_path, payload):
    volume = Volume()
    volume.payload = payload
    with pytest.raises(ValueError): fetch.fetch_archive(volume, RUN, tmp_path / "capture")
    local = json.loads((tmp_path / "capture/manifest.json").read_text())
    assert local["status"] == "partial" and local["entries"] == []
    assert not (tmp_path / "capture/chunk-first.pt").exists()
    assert not list((tmp_path / "capture").glob(".download-*"))


def test_download_limit_and_existing_output_cannot_be_bypassed(tmp_path):
    volume = Volume()
    with pytest.raises(ValueError, match="budget"):
        fetch.fetch_archive(volume, RUN, tmp_path / "capture", max_bytes=1)
    with pytest.raises(ValueError, match="new directory"):
        fetch.fetch_archive(volume, RUN, tmp_path)
    with pytest.raises(ValueError, match="run ID"):
        fetch.fetch_archive(volume, "../other-run", tmp_path / "capture")
