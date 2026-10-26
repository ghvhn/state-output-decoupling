"""Bounded, resumable evidence from a pinned teacher, not a compressed model.

Native final readout inputs and the original output head permit offline logit
reconstruction. Spot checks remain necessary: different matrix kernels need not
produce bit-identical logits. Every artifact is data-only, scoped and hashed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import torch

FORMAT = "glm-native-capture-v1"
IDENTITY_KEYS = {"model_id", "revision", "hidden_size", "vocab_size", "num_layers",
                 "source_precision", "tokenizer_sha256", "chat_template_sha256"}
FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
DTYPE_NAMES = {str(value).removeprefix("torch."): value for value in
               (*FLOAT_DTYPES, torch.int64, torch.int32, torch.bool)}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{name} must be nonblank text")
    return value


def validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        raise ValueError("capture identity requires the exact source and tokenizer fields")
    for field in ("model_id", "source_precision"):
        _text(identity[field], field)
    if not re.fullmatch(r"[0-9a-f]{40}", str(identity["revision"])):
        raise ValueError("capture model revision must be a pinned SHA")
    for field in ("tokenizer_sha256", "chat_template_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(identity[field])):
            raise ValueError(f"{field} must identify measured source formatting")
    for field in ("hidden_size", "vocab_size", "num_layers"):
        _integer(identity[field], field, 1)
    return dict(identity)


MAX_MANIFEST_BYTES = 32 * 1024**2


def _json(value, limit=4 * 1024**2):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    if len(data) > limit:
        raise ValueError("capture metadata exceeds its byte bound")
    return data


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024**2):
            digest.update(data)
    return digest.hexdigest()


def _atomic_json(path, value, *, limit=4 * 1024**2):
    path = Path(path)
    data = _json(value, limit)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".capture-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _tensor_info(tensors, byte_limit):
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("capture requires a nonempty named tensor mapping")
    description, total = {}, 0
    for name, value in tensors.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,95}", name):
            raise ValueError("invalid capture tensor name")
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
                value.layout != torch.strided or value.dtype not in DTYPE_NAMES.values() or
                not value.is_contiguous() or value.requires_grad):
            raise ValueError("capture tensors must be contiguous detached CPU native floats or integer IDs")
        if not value.numel():
            raise ValueError("empty capture tensors do not establish evidence")
        # Avoid serializing an entire backing allocation for a narrow view.
        payload = value.numel() * value.element_size()
        if value.storage_offset() != 0 or value.untyped_storage().nbytes() != payload:
            raise ValueError("capture tensors require compact independent storage")
        total += payload
        if total > byte_limit:
            raise ValueError("capture tensor payload exceeds its byte bound")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("capture tensors contain nonfinite observations")
        description[name] = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
                             "bytes": payload}
    return description, total


def _validate_rows(metadata, identity):
    _json(metadata)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("rows"), list):
        raise ValueError("capture chunk requires explicit measured row identities")
    rows = metadata["rows"]
    if not rows:
        raise ValueError("capture requires measured rows")
    sites = metadata.get("state_sites")
    if not isinstance(sites, dict) or sites.get("final_hidden") != "post_final_norm_lm_head_input":
        raise ValueError("final_hidden must be captured after the native final normalization")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("invalid capture row")
        for field in ("id", "trajectory_id", "environment_id"):
            _text(row.get(field), field)
        if row["id"] in seen:
            raise ValueError("capture row IDs must be unique")
        seen.add(row["id"])
        if row.get("split") not in ("calibration", "heldout"):
            raise ValueError("capture row needs a sealed calibration or heldout split")
        _integer(row.get("token_position"), "token_position")
        observed = row.get("next_token_observed", True)
        if not isinstance(observed, bool):
            raise ValueError("next-token observation flag must be boolean")
        if observed:
            target = _integer(row.get("next_token_id"), "next_token_id")
        else:
            if (row.get("next_token_id") != -1 or row.get("kind") != "probe_framing" or
                    not isinstance(row.get("probe_id"), str) or not row["probe_id"].strip() or
                    isinstance(row.get("polarity"), bool) or row.get("polarity") not in (-1, 1)):
                raise ValueError("unobserved targets require an explicit signed probe framing and -1 sentinel")
            target = -1
        if target >= identity["vocab_size"]:
            raise ValueError("observed token is outside the teacher vocabulary")
        for field in ("provenance_weight", "stimulation_weight"):
            value = row.get(field, 1.0)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"invalid {field}")
    pairs = metadata.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("capture pairs must be row endpoint lists")
    pair_seen = set()
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2 or any(isinstance(i, bool) or not isinstance(i, int) for i in pair) or
                not 0 <= pair[0] < pair[1] < len(rows)):
            raise ValueError("capture pairs require ordered row endpoints")
        a, b = rows[pair[0]], rows[pair[1]]
        if (a["trajectory_id"] != b["trajectory_id"] or a["environment_id"] != b["environment_id"] or
                a["split"] != b["split"] or a["token_position"] >= b["token_position"] or tuple(pair) in pair_seen):
            raise ValueError("capture pairs must preserve one measured trajectory and environment")
        pair_seen.add(tuple(pair))
    return rows


def _validate_chunk(tensors, metadata, identity):
    rows = _validate_rows(metadata, identity)
    sites = metadata["state_sites"]
    states = tensors.get("final_hidden")
    if (states is None or states.dtype not in FLOAT_DTYPES or states.ndim != 2 or
            tuple(states.shape) != (len(rows), identity["hidden_size"])):
        raise ValueError("final_hidden must match the teacher readout input width and row count")
    for name, row_field in (("positions", "token_position"), ("next_token_ids", "next_token_id")):
        value = tensors.get(name)
        if value is None or value.dtype != torch.int64 or tuple(value.shape) != (len(rows),):
            raise ValueError(f"{name} must retain the original row timeline")
        if value.tolist() != [row[row_field] for row in rows]:
            raise ValueError(f"{name} disagrees with measured metadata")
    for name, value in tensors.items():
        match = re.fullmatch(r"layer_(\d+)_(hidden|expert_input|router_ids|router_weights)", name)
        if match:
            layer = int(match[1])
            if layer >= identity["num_layers"]:
                raise ValueError("capture layer is outside the original decoder")
            kind = match[2]
            if value.ndim != 2 or value.shape[0] != len(rows):
                raise ValueError("layer captures must share the measured row timeline")
            if kind in ("hidden", "expert_input"):
                if value.dtype not in FLOAT_DTYPES or value.shape[1] != identity["hidden_size"]:
                    raise ValueError("layer state shape differs from native hidden width")
                expected_site = "decoder_block_output" if kind == "hidden" else "post_attention_norm_expert_gate_up_input"
                if sites.get(name) != expected_site:
                    raise ValueError("layer state must declare its native measurement site")
            elif kind == "router_ids":
                if value.dtype not in (torch.int32, torch.int64) or not bool((value >= 0).all()):
                    raise ValueError("router choices require nonnegative native expert IDs")
            elif value.dtype not in FLOAT_DTYPES or not bool((value >= 0).all()):
                raise ValueError("router weights require finite nonnegative floating observations")
    router_layers = {int(m[1]) for name in tensors if (m := re.fullmatch(r"layer_(\d+)_router_(ids|weights)", name))}
    for layer in router_layers:
        ids, weights = tensors.get(f"layer_{layer}_router_ids"), tensors.get(f"layer_{layer}_router_weights")
        if ids is None or weights is None or ids.shape != weights.shape:
            raise ValueError("native router IDs and weights must form matching pairs")
        if ids.shape[1] > 1 and bool((ids.sort(dim=1).values.diff(dim=1) == 0).any()):
            raise ValueError("native router choices cannot duplicate an expert in one row")
    if "spotcheck_logits" in tensors or "spotcheck_rows" in tensors:
        logits, selected = tensors.get("spotcheck_logits"), tensors.get("spotcheck_rows")
        if (logits is None or selected is None or selected.dtype != torch.int64 or selected.ndim != 1 or
                logits.dtype not in FLOAT_DTYPES or tuple(logits.shape) != (selected.numel(), identity["vocab_size"]) or
                not bool(((selected >= 0) & (selected < len(rows))).all()) or
                selected.unique().numel() != selected.numel()):
            raise ValueError("spot checks must preserve the full original vocabulary and row mapping")


class CaptureArchiveWriter:
    """Commit each artifact before adding it to the durable archive index."""
    def __init__(self, root, identity, *, max_total_bytes=8 * 1024**3,
                 max_chunk_bytes=64 * 1024**2, origin="measured_model_state", run_metadata=None):
        self.identity = validate_identity(identity)
        self.max_total_bytes = _integer(max_total_bytes, "max_total_bytes", 1)
        self.max_chunk_bytes = _integer(max_chunk_bytes, "max_chunk_bytes", 1)
        if origin not in ("measured_model_state", "synthetic_fixture"):
            raise ValueError("capture origin must distinguish measured and synthetic observations")
        self.root = Path(root).resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError("refusing to overwrite an existing capture")
        self.root.mkdir(parents=True, exist_ok=True)
        self.origin, self.entries, self.bytes_written = origin, [], 0
        self.next_head_row, self.closed = 0, False
        self.row_ids, self.trajectory_splits = set(), {}
        self.run_metadata = {} if run_metadata is None else run_metadata
        _json(self.run_metadata)
        self._index("partial", {})

    def _index(self, status, details):
        self.manifest = {"format": FORMAT, "identity": self.identity, "origin": self.origin,
                         "status": status, "entries": self.entries, "artifact_bytes": self.bytes_written,
                         "head_rows": self.next_head_row, "run_metadata": self.run_metadata,
                         "details": details}
        _atomic_json(self.root / "manifest.json", self.manifest, limit=MAX_MANIFEST_BYTES)

    def _save(self, filename, tensors, metadata, role):
        if self.closed:
            raise ValueError("capture archive is already finalized")
        if any(entry["file"] == filename for entry in self.entries):
            raise ValueError("capture artifacts cannot overwrite earlier measurements")
        description, payload = _tensor_info(tensors, self.max_chunk_bytes)
        _json(metadata)
        if self.bytes_written + payload >= self.max_total_bytes:
            raise ValueError("capture archive exhausted its total byte budget")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".capture-", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(tensors, stream)
                stream.flush()
                os.fsync(stream.fileno())
            size = temporary.stat().st_size
            if size > self.max_chunk_bytes or self.bytes_written + size > self.max_total_bytes:
                raise ValueError("serialized capture exceeds its byte budget")
            entry = {"file": filename, "role": role, "bytes": size, "sha256": _hash(temporary),
                     "tensors": description, "metadata": metadata}
            os.replace(temporary, self.root / filename)
            self.entries.append(entry)
            self.bytes_written += size
            return entry
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def write_head_block(self, start_row, weight):
        start_row = _integer(start_row, "head start row")
        if (start_row != self.next_head_row or not isinstance(weight, torch.Tensor) or weight.ndim != 2 or
                weight.shape[1] != self.identity["hidden_size"] or weight.dtype not in FLOAT_DTYPES or
                start_row + weight.shape[0] > self.identity["vocab_size"]):
            raise ValueError("output head blocks must preserve native shape and contiguous vocabulary order")
        if any(e["role"] == "head" and e["tensors"]["weight"]["dtype"] != str(weight.dtype).removeprefix("torch.") for e in self.entries):
            raise ValueError("output head blocks cannot change native dtype")
        entry = self._save(f"head-{start_row:09d}.pt", {"weight": weight}, {"start_row": start_row}, "head")
        self.next_head_row += weight.shape[0]
        self._index("partial", {})
        return entry

    def write_chunk(self, chunk_id, tensors, metadata):
        if not isinstance(chunk_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", chunk_id):
            raise ValueError("invalid capture chunk ID")
        _validate_chunk(tensors, metadata, self.identity)
        rows = metadata["rows"]
        new_splits = dict(self.trajectory_splits)
        for row in rows:
            if row["id"] in self.row_ids:
                raise ValueError("capture archive cannot duplicate measured row IDs")
            scope = row.get("conversation_id", row["trajectory_id"])
            _text(scope, "split scope")
            if scope in new_splits and new_splits[scope] != row["split"]:
                raise ValueError("a trajectory cannot leak across calibration and heldout sets")
            new_splits[scope] = row["split"]
        entry = self._save(f"chunk-{chunk_id}.pt", tensors, metadata, "capture")
        self.row_ids.update(row["id"] for row in rows)
        self.trajectory_splits = new_splits
        self._index("partial", {})
        return entry

    def finalize(self, status="complete", details=None):
        if self.closed or status not in ("complete", "partial", "failed"):
            raise ValueError("invalid capture finalization")
        details = {} if details is None else details
        _json(details)
        if status == "complete" and (self.next_head_row != self.identity["vocab_size"] or
                                     not any(entry["role"] == "capture" for entry in self.entries)):
            raise ValueError("a complete capture requires the entire output head and measured rows")
        self._index(status, details)
        self.closed = True
        return self.manifest


def _path(root, filename):
    if not isinstance(filename, str) or not re.fullmatch(r"(?:head-[0-9]{9}|chunk-[A-Za-z0-9][A-Za-z0-9_-]{0,63})\.pt", filename):
        raise ValueError("invalid archive artifact path")
    path = root / filename
    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
        raise ValueError("capture artifact must be a regular file inside its archive")
    return path


def load_archive_manifest(root, *, expected_identity=None, max_total_bytes=8 * 1024**3,
                          max_chunk_bytes=64 * 1024**2):
    root = Path(root).resolve()
    path = root / "manifest.json"
    if path.is_symlink() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("invalid capture manifest size or path")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("origin") not in ("measured_model_state", "synthetic_fixture"):
        raise ValueError("unsupported capture archive")
    identity = validate_identity(manifest.get("identity"))
    if expected_identity is not None and identity != validate_identity(expected_identity):
        raise ValueError("capture archive differs from the requested teacher scope")
    if manifest.get("status") not in ("complete", "partial", "failed"):
        raise ValueError("invalid capture status")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) > 100_000:
        raise ValueError("capture index exceeds its artifact count bound")
    total, next_head_row, names, row_ids, splits = 0, 0, set(), set(), {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in ("head", "capture"):
            raise ValueError("invalid capture artifact entry")
        artifact = _path(root, entry.get("file"))
        if entry["file"] in names:
            raise ValueError("duplicate capture artifact")
        names.add(entry["file"])
        size = _integer(entry.get("bytes"), "artifact bytes", 1)
        if size > max_chunk_bytes or artifact.stat().st_size != size:
            raise ValueError("capture artifact size differs from its bound")
        total += size
        if total > max_total_bytes:
            raise ValueError("capture archive exceeds its total byte bound")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))):
            raise ValueError("invalid capture checksum")
        info = entry.get("tensors")
        if not isinstance(info, dict) or not info:
            raise ValueError("missing capture tensor schema")
        tensor_bytes = 0
        for field in info.values():
            if not isinstance(field, dict) or field.get("dtype") not in DTYPE_NAMES:
                raise ValueError("invalid capture tensor dtype")
            shape = field.get("shape")
            if not isinstance(shape, list) or len(shape) > 4:
                raise ValueError("invalid capture tensor shape")
            for dimension in shape:
                _integer(dimension, "capture dimension", 1)
            expected_bytes = math.prod(shape) * torch.empty((), dtype=DTYPE_NAMES[field["dtype"]]).element_size()
            if field.get("bytes") != expected_bytes:
                raise ValueError("capture tensor schema has an invalid byte count")
            tensor_bytes += expected_bytes
        if tensor_bytes > max_chunk_bytes:
            raise ValueError("declared capture tensors exceed their allocation bound")
        if entry["role"] == "head":
            weight = info.get("weight")
            if (set(info) != {"weight"} or weight["shape"][1:] != [identity["hidden_size"]] or
                    len(weight["shape"]) != 2 or entry.get("metadata", {}).get("start_row") != next_head_row):
                raise ValueError("invalid vocabulary order or head shape")
            next_head_row += weight["shape"][0]
            if next_head_row > identity["vocab_size"]:
                raise ValueError("output head exceeds teacher vocabulary")
        else:
            rows = _validate_rows(entry.get("metadata"), identity)
            if info.get("final_hidden", {}).get("shape") != [len(rows), identity["hidden_size"]]:
                raise ValueError("capture schema differs from its measured rows")
            for row in rows:
                if row["id"] in row_ids:
                    raise ValueError("archive index duplicates measured rows")
                row_ids.add(row["id"])
                # New capture metadata carries conversation_id; older archives
                # retain the narrower trajectory identity as their split scope.
                group = row.get("conversation_id", row["trajectory_id"])
                _text(group, "split scope")
                if group in splits and splits[group] != row["split"]:
                    raise ValueError("archive index leaks a conversation across splits")
                splits[group] = row["split"]
    if total != manifest.get("artifact_bytes") or next_head_row != manifest.get("head_rows"):
        raise ValueError("capture manifest accounting differs from its artifacts")
    if manifest["status"] == "complete" and (next_head_row != identity["vocab_size"] or
                                             not any(e["role"] == "capture" for e in entries)):
        raise ValueError("complete archive is missing original teacher evidence")
    return manifest


def load_archive_entry(root, entry, *, max_chunk_bytes=64 * 1024**2, identity=None):
    root = Path(root).resolve()
    path = _path(root, entry["file"])
    if path.stat().st_size != entry["bytes"] or path.stat().st_size > max_chunk_bytes or _hash(path) != entry["sha256"]:
        raise ValueError("capture artifact failed its size or checksum check")
    # Bound raw storages before torch can allocate any tensors.
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if len(items) > 4096 or len({item.filename for item in items}) != len(items):
            raise ValueError("invalid capture storage index")
        if any(item.compress_type != zipfile.ZIP_STORED for item in items):
            raise ValueError("compressed capture storages are unsupported")
        raw_storages = sum(item.file_size for item in items if "/data/" in item.filename)
        declared = sum(info["bytes"] for info in entry["tensors"].values())
        if raw_storages != declared or raw_storages > max_chunk_bytes:
            raise ValueError("capture storage allocation differs from its schema")
        if any(item.file_size > max_chunk_bytes for item in items):
            raise ValueError("capture archive entry exceeds its read bound")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    description, _ = _tensor_info(tensors, max_chunk_bytes)
    if description != entry["tensors"]:
        raise ValueError("loaded capture tensors differ from their stored schema")
    if identity is not None and entry["role"] == "capture":
        _validate_chunk(tensors, entry["metadata"], validate_identity(identity))
    return tensors


def load_output_head(root, manifest, *, max_head_bytes=2 * 1024**3):
    identity = manifest["identity"]
    entries = [entry for entry in manifest["entries"] if entry["role"] == "head"]
    if not entries or manifest["head_rows"] != identity["vocab_size"]:
        raise ValueError("full vocabulary reconstruction needs a complete original output head")
    dtype = entries[0]["tensors"]["weight"]["dtype"]
    if any(entry["tensors"]["weight"]["dtype"] != dtype for entry in entries):
        raise ValueError("output head dtype changed between blocks")
    item_size = torch.empty((), dtype=DTYPE_NAMES[dtype]).element_size()
    if identity["hidden_size"] * identity["vocab_size"] * item_size > max_head_bytes:
        raise ValueError("output head exceeds its allocation bound")
    result = torch.empty((identity["vocab_size"], identity["hidden_size"]), dtype=DTYPE_NAMES[dtype])
    for entry in entries:
        start = entry["metadata"]["start_row"]
        weight = load_archive_entry(root, entry)["weight"]
        result[start:start + weight.shape[0]].copy_(weight)
    return result


def reconstruct_logits(final_hidden, output_head, *, max_working_bytes=2 * 1024**3, validate_output_head=True):
    """Native readout; bound tensors, excluding the matrix library's workspace.

    Callers may skip scanning the unchanged head after load_output_head already
    checked every bounded source block. New states and logits are always checked.
    """
    if (not isinstance(final_hidden, torch.Tensor) or not isinstance(output_head, torch.Tensor) or
            final_hidden.ndim != 2 or output_head.ndim != 2 or
            final_hidden.shape[1] != output_head.shape[1] or
            final_hidden.dtype not in FLOAT_DTYPES or final_hidden.dtype != output_head.dtype or
            final_hidden.device != output_head.device):
        raise ValueError("readout requires matching native width, dtype and device")
    required = (final_hidden.numel() + output_head.numel() +
                final_hidden.shape[0] * output_head.shape[0]) * final_hidden.element_size()
    if required > _integer(max_working_bytes, "readout working bytes", 1):
        raise ValueError("readout exceeds its declared tensor working budget")
    if not bool(torch.isfinite(final_hidden).all()) or (validate_output_head and
            any(not bool(torch.isfinite(block).all()) for block in output_head.reshape(-1).split(1024**2))):
        raise ValueError("readout contains nonfinite teacher data")
    with torch.inference_mode():
        logits = final_hidden @ output_head.T
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("reconstructed logits exceed native precision")
    return logits


def verify_spotchecks(final_hidden, output_head, original_logits, *, atol=0.03125, rtol=0.01,
                      max_working_bytes=2 * 1024**3):
    for value in (atol, rtol):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("spotcheck tolerances must be finite and nonnegative")
    reconstructed = reconstruct_logits(final_hidden, output_head, max_working_bytes=max_working_bytes).float()
    if (not isinstance(original_logits, torch.Tensor) or original_logits.shape != reconstructed.shape or
            original_logits.device != reconstructed.device or not bool(torch.isfinite(original_logits).all())):
        raise ValueError("spotcheck targets differ from original full-vocabulary observations")
    original = original_logits.float()
    difference = (reconstructed - original).abs()
    passed = bool(torch.allclose(reconstructed, original, atol=atol, rtol=rtol))
    return {"passed": passed, "atol": float(atol), "rtol": float(rtol),
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "argmax_agreement": float((reconstructed.argmax(-1) == original.argmax(-1)).float().mean()),
            "scope": "captured_teacher_final_readout_only"}
