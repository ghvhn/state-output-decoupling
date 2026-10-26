"""Validate captured evidence and reconstruct the original teacher's readout locally.

Defaults to CPU inspection. Export requires native full-vocabulary spot checks
to pass first. This readout does not run the GLM decoder or validate a modified
model. Wrong next-token predictions remain measured evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tda.glm_capture_archive import (load_archive_manifest, load_archive_entry,
    load_output_head, reconstruct_logits, verify_spotchecks, _atomic_json)


def reconstruct_archive(root, *, out=None, batch_rows=16, max_export_bytes=4 * 1024**3,
                        max_working_bytes=2 * 1024**3, atol=0.03125, rtol=0.01):
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or not 1 <= batch_rows <= 1024:
        raise ValueError("readout batch rows must be in [1, 1024]")
    if isinstance(max_export_bytes, bool) or not isinstance(max_export_bytes, int) or max_export_bytes < 1:
        raise ValueError("export bytes must be a positive integer")
    root = Path(root).resolve()
    manifest = load_archive_manifest(root)
    entries = [entry for entry in manifest["entries"] if entry["role"] == "capture"]
    head = load_output_head(root, manifest)
    checks, positions, observed_positions = [], 0, 0
    for entry in entries:
        tensors = load_archive_entry(root, entry, identity=manifest["identity"])
        positions += len(entry["metadata"]["rows"])
        observed_positions += sum(row.get("next_token_observed", True) for row in entry["metadata"]["rows"])
        if "spotcheck_logits" in tensors:
            check = verify_spotchecks(tensors["final_hidden"][tensors["spotcheck_rows"]], head,
                tensors["spotcheck_logits"], atol=atol, rtol=rtol, max_working_bytes=max_working_bytes)
            checks.append(dict(check, artifact=entry["file"], positions=tensors["spotcheck_rows"].numel()))
    passed = bool(checks) and all(check["passed"] for check in checks)
    report = {"format": "glm-local-readout-v1", "identity": manifest["identity"],
        "source_archive_origin": manifest["origin"], "source_archive_status": manifest["status"],
        "positions": positions, "observed_next_tokens": observed_positions,
        "unobserved_probe_positions": positions - observed_positions,
        "spotchecks": checks, "readout_qualified": passed,
        "scope": "captured_original_teacher_final_readout_only",
        "modified_decoder_validated": False, "exported_bytes": 0, "entries": []}
    if out is None:
        return report
    if not passed:
        raise ValueError("readout export requires passing measured full-vocabulary spot checks")
    target = Path(out).resolve()
    if target.exists() or target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("readout export requires a new directory separate from its source archive")
    estimated = positions * manifest["identity"]["vocab_size"] * head.element_size()
    estimated += sum(math.ceil(len(entry["metadata"]["rows"]) / batch_rows) for entry in entries) * 128 * 1024
    if estimated > max_export_bytes:
        raise ValueError("full-vocabulary export exceeds its byte budget before writing")
    target.mkdir(parents=True)
    report["origin"] = "reconstructed_teacher_readout"
    report["native_dtype"] = str(head.dtype).removeprefix("torch.")
    _atomic_json(target / "report.json", report)
    nll_sum, top1_correct, measured = 0.0, 0, 0
    try:
        for entry_index, entry in enumerate(entries):
            tensors = load_archive_entry(root, entry, identity=manifest["identity"])
            rows = entry["metadata"]["rows"]
            for offset in range(0, len(rows), batch_rows):
                logits = reconstruct_logits(tensors["final_hidden"][offset:offset + batch_rows], head,
                    max_working_bytes=max_working_bytes, validate_output_head=False)
                labels = tensors["next_token_ids"][offset:offset + batch_rows].clone()
                observed = labels >= 0
                if bool(observed.any()):
                    selected, truth = logits[observed].float(), labels[observed]
                    nll_sum += float((torch.logsumexp(selected, dim=-1) - selected.gather(1, truth[:, None]).squeeze(1)).sum())
                    top1_correct += int((selected.argmax(-1) == truth).sum())
                    measured += int(observed.sum())
                filename = f"logits-{entry_index:06d}-{offset:06d}.pt"
                path, temporary = target / filename, target / ("." + filename + ".incomplete")
                try:
                    torch.save({"logits": logits.cpu().contiguous().clone(), "next_token_ids": labels}, temporary)
                    size = temporary.stat().st_size
                    if report["exported_bytes"] + size > max_export_bytes:
                        raise ValueError("serialized readout exceeds its export budget")
                    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                report["exported_bytes"] += size
                report["entries"].append({"file": filename, "sha256": digest, "bytes": size,
                    "source_chunk": entry["file"], "row_range": [offset, offset + len(labels)],
                    "rows": rows[offset:offset + len(labels)]})
                _atomic_json(target / "report.json", report, limit=32 * 1024**2)
        report["status"] = "complete"
        report["observed_target_metrics"] = {"positions": measured,
            "mean_nll": nll_sum / measured if measured else None,
            "top1_accuracy": top1_correct / measured if measured else None,
            "all_predictions_retained": True}
    except Exception as error:
        report["status"] = "partial"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _atomic_json(target / "report.json", report, limit=32 * 1024**2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, help="Optional new directory for native full-vocabulary readouts")
    parser.add_argument("--batch-rows", type=int, default=16)
    parser.add_argument("--max-export-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args(argv)
    try:
        report = reconstruct_archive(args.archive, out=args.out, batch_rows=args.batch_rows,
                                     max_export_bytes=args.max_export_bytes)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Readout stopped: {error}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "entries"}, indent=2))
    return 0 if report["readout_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
