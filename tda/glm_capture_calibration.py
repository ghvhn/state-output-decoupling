"""Native probe contrasts and scoped gate/up calibration from capture evidence.

Seed contrasts define experimental axes, not established semantic meanings.
Measured gate/up inputs cannot be reused as down-projection calibration.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re

import torch

from .glm_capture_archive import load_archive_manifest, load_archive_entry
from .expert_compress import ExpertGroupIdentity
from .trajectory_compress import InputStateScope, CalibrationRecord, TrajectoryPair


def native_probe_bank_fingerprint(bank):
    """Bind derived axes to their source poles; this hash is not anonymization."""
    digest = hashlib.sha256()
    fields = {name: bank[name] for name in ("format", "scope", "origin", "probe_ids",
        "pole_record_ids", "source_capture_sha256", "unpaired_probe_ids", "zero_contrast_probe_ids",
        "semantic_meanings_validated", "heldout_used_to_fit_axes")}
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 1024**2:
        raise ValueError("native probe evidence metadata exceeds its byte bound")
    digest.update(encoded)
    for name in ("directions", "centers", "contrast_magnitudes"):
        value = bank[name].detach().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def native_probe_bank(root, *, tensor_name="final_hidden", max_axes=128):
    if isinstance(max_axes, bool) or not isinstance(max_axes, int) or not 0 < max_axes <= 2048:
        raise ValueError("native probe axis budget must be in [1, 2048]")
    if tensor_name != "final_hidden" and not re.fullmatch(r"layer_\d+_(hidden|expert_input)", tensor_name):
        raise ValueError("native probe contrasts require an explicit hidden-state site")
    manifest = load_archive_manifest(root)
    poles, pole_ids, source_hashes, site = defaultdict(dict), defaultdict(dict), set(), None
    for entry in manifest["entries"]:
        if entry["role"] != "capture":
            continue
        rows = entry["metadata"]["rows"]
        indices = [i for i, row in enumerate(rows) if row.get("kind") == "probe_framing"]
        if not indices:
            continue
        measured = entry["metadata"]["state_sites"].get(tensor_name)
        if not measured or site is not None and site != measured:
            raise ValueError("native probe measurement site differs between chunks")
        site = measured
        tensors = load_archive_entry(root, entry, identity=manifest["identity"])
        source_hashes.add(entry["sha256"])
        if tensor_name not in tensors:
            raise ValueError("native probe state was not captured at the requested site")
        for index in indices:
            row = rows[index]
            if row["split"] != "calibration" or row.get("polarity") not in (-1, 1) or not row.get("probe_id"):
                raise ValueError("native seed poles must be explicit calibration examples")
            probe, polarity = row["probe_id"], row["polarity"]
            if polarity in poles[probe]:
                raise ValueError("native seed bank requires one explicit measurement per pole")
            poles[probe][polarity] = tensors[tensor_name][index].detach().float().clone()
            pole_ids[probe][polarity] = row["id"]
            if len(poles) > max_axes:
                raise ValueError("native probe bank exceeds its axis budget")
    directions, centers, magnitudes, endpoints, ids, unpaired, zero = [], [], [], [], [], [], []
    for probe, pair in sorted(poles.items()):
        if set(pair) != {-1, 1}:
            unpaired.append(probe)
            continue
        contrast = pair[1] - pair[-1]
        magnitude = contrast.norm()
        if not torch.isfinite(magnitude) or magnitude <= 1e-6:
            zero.append(probe)
            continue
        directions.append(contrast / magnitude)
        centers.append((pair[1] + pair[-1]) / 2)
        magnitudes.append(magnitude)
        endpoints.append({"negative": pole_ids[probe][-1], "positive": pole_ids[probe][1]})
        ids.append(probe)
    if not directions:
        raise ValueError("no paired nonzero native probe contrasts were measured")
    scope = {"teacher": manifest["identity"], "tensor_name": tensor_name, "state_site": site}
    bank = {"format": "glm-native-probe-bank-v1", "scope": scope,
        "scope_sha256": hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest(),
        "origin": manifest["origin"], "probe_ids": ids,
        "directions": torch.stack(directions), "centers": torch.stack(centers),
        "contrast_magnitudes": torch.stack(magnitudes), "pole_record_ids": endpoints,
        "source_capture_sha256": sorted(source_hashes),
        "unpaired_probe_ids": unpaired, "zero_contrast_probe_ids": zero,
        "semantic_meanings_validated": False, "heldout_used_to_fit_axes": False}
    bank["evidence_sha256"] = native_probe_bank_fingerprint(bank)
    return bank


def gate_calibration_records(root, identity, *, split="calibration", max_records=8192,
                             max_pairs=4096, max_state_bytes=256 * 1024**2):
    if not isinstance(identity, ExpertGroupIdentity) or identity.projection not in ("gate_proj", "up_proj"):
        raise ValueError("shared gate/up captures cannot calibrate a down projection")
    if split not in ("calibration", "heldout"):
        raise ValueError("calibration and heldout records must be selected separately")
    for value in (max_records, max_pairs, max_state_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("calibration record, pair and state budgets must be positive integers")
    manifest = load_archive_manifest(root)
    teacher = manifest["identity"]
    if (teacher["model_id"] != identity.model_id or teacher["revision"] != identity.revision or
            teacher["hidden_size"] != identity.matrix_shape[1] or identity.layer >= teacher["num_layers"]):
        raise ValueError("capture teacher differs from the selected original expert group")
    name, site = f"layer_{identity.layer}_expert_input", "post_attention_norm_expert_gate_up_input"
    scope = InputStateScope.from_identity(identity, state_site=site)
    records, pairs, state_bytes = [], [], 0
    for entry in manifest["entries"]:
        if entry["role"] != "capture":
            continue
        rows = entry["metadata"]["rows"]
        chosen = {i for i, row in enumerate(rows) if row["split"] == split and row.get("kind") != "probe_framing"}
        if not chosen:
            continue
        if entry["metadata"]["state_sites"].get(name) != site:
            raise ValueError("expert calibration requires the exact normalized native gate/up site")
        expected_pairs = sum(a in chosen and b in chosen for a, b in entry["metadata"].get("pairs", []))
        # Production native activations are BF16. Use a conservative FP32
        # allocation bound before opening any tensor payload.
        state_bytes += len(chosen) * teacher["hidden_size"] * 4
        if len(records) + len(chosen) > max_records or len(pairs) + expected_pairs > max_pairs or state_bytes > max_state_bytes:
            raise MemoryError("native calibration exceeds its declared evidence budget")
        tensors = load_archive_entry(root, entry, identity=teacher)
        if name not in tensors:
            raise ValueError("selected expert input state is absent")
        for index in sorted(chosen):
            row = rows[index]
            records.append(CalibrationRecord(row["id"], row["environment_id"], row["trajectory_id"], scope,
                tensors[name][index].detach().clone(), row.get("provenance_weight", 1.0),
                row.get("stimulation_weight", 1.0), prediction_correct=None, origin=manifest["origin"]))
        for a, b in entry["metadata"].get("pairs", []):
            if a in chosen and b in chosen:
                pairs.append(TrajectoryPair(f"{rows[a]['id']}->{rows[b]['id']}", rows[a]["id"], rows[b]["id"]))
    return scope, records, pairs
