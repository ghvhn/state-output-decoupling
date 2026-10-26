"""Fetch one bounded GLM capture archive from its cloud Volume, never weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import warnings

MAX_MANIFEST = 32 * 1024**2
MAX_ARCHIVE = 8 * 1024**3
MAX_ARTIFACT = 64 * 1024**2


def _atomic_json(path, value):
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(data) > MAX_MANIFEST:
        raise ValueError("capture manifest exceeds byte bound")
    temporary = path.with_name("." + path.name + ".incomplete")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_archive(volume, run_id, destination, *, max_bytes=MAX_ARCHIVE):
    if not isinstance(run_id, str) or not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{12}", run_id):
        raise ValueError("capture run ID must name one concrete capture")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 0 < max_bytes <= MAX_ARCHIVE:
        raise ValueError("capture download budget must fit the trial byte limit")
    base = f"/runs/{run_id}/archive/"
    data = bytearray()
    for block in volume.read_file(base + "manifest.json"):
        if len(data) + len(block) > MAX_MANIFEST:
            raise ValueError("remote capture manifest exceeds byte bound")
        data.extend(block)
    source = json.loads(data)
    if (not isinstance(source, dict) or source.get("format") != "glm-native-capture-v1" or
            source.get("origin") not in ("measured_model_state", "synthetic_fixture") or
            source.get("status") not in ("complete", "partial", "failed")):
        raise ValueError("remote artifact is not a capture archive")
    entries = source.get("entries")
    if not isinstance(entries, list) or len(entries) > 100000:
        raise ValueError("remote capture has an invalid artifact count")
    total, head_rows, names = 0, 0, set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("remote capture entry must be an object")
        filename = entry.get("file")
        if (not isinstance(filename, str) or not re.fullmatch(r"(?:head-[0-9]{9}|chunk-[A-Za-z0-9][A-Za-z0-9_-]{0,63})\.pt", filename) or
                filename in names or entry.get("role") not in ("head", "capture")):
            raise ValueError("remote capture entry has an unsafe or duplicate path")
        names.add(filename)
        size = entry.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_ARTIFACT:
            raise ValueError("remote capture artifact exceeds byte bound")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("remote capture artifact lacks a content checksum")
        total += size
        if entry["role"] == "head":
            shape = entry.get("tensors", {}).get("weight", {}).get("shape")
            if (not isinstance(shape, list) or len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape) or
                    entry.get("metadata", {}).get("start_row") != head_rows):
                raise ValueError("remote output head lost its vocabulary ordering")
            head_rows += shape[0]
    if total != source.get("artifact_bytes") or total > max_bytes or head_rows != source.get("head_rows"):
        raise ValueError("remote capture accounting exceeds its download budget")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("capture download requires a new directory")
    destination.mkdir(parents=True)
    local = dict(source, entries=[], artifact_bytes=0, head_rows=0, status="partial")
    _atomic_json(destination / "manifest.json", local)
    for entry in entries:
        temporary = None
        try:
            digest, size = hashlib.sha256(), 0
            with tempfile.NamedTemporaryFile(dir=destination, prefix=".download-", delete=False) as stream:
                temporary = Path(stream.name)
                for block in volume.read_file(base + entry["file"]):
                    size += len(block)
                    if size > entry["bytes"]:
                        raise ValueError("remote capture stream exceeds declared size")
                    digest.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError("remote capture failed its size/checksum check")
            temporary.replace(destination / entry["file"])
            local["entries"].append(entry)
            local["artifact_bytes"] += size
            if entry["role"] == "head":
                local["head_rows"] += entry["tensors"]["weight"]["shape"][0]
            _atomic_json(destination / "manifest.json", local)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    _atomic_json(destination / "manifest.json", source)
    return {"run_id": run_id, "downloaded_bytes": total, "artifacts": len(entries),
            "source_status": source["status"], "local_archive": str(destination),
            "source_manifest_sha256": hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--volume", default="glm-5-3-credit-capture")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", message="Your Modal token .* expires in .*")
    import modal
    try:
        report = fetch_archive(modal.Volume.from_name(args.volume), args.run_id, args.out)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Capture fetch stopped: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
