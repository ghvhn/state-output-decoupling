"""Two-channel probe diagnostics, without generation or semantic authority.

The output-head projection is an algebraic fingerprint of each layer's probe
vector, not a semantic decoder. Measured reply language and input topics are
separate descriptive channels. Neither channel can validate a definition,
rename a namespace, or establish that the other channel is wrong.
"""

from __future__ import annotations

import collections
import hashlib
import math
import pickle
from pathlib import Path
import re
from typing import Mapping

import torch

GEOMETRIC_VERSION = "geometric-probe-diagnostic-v1"
_WORD_RE = re.compile(r"\b[^\W_]+(?:['’][^\W_]+)?\b", re.UNICODE)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _words(text):
    return [word.casefold() for word in _WORD_RE.findall(text)]


def _ngrams(text, n):
    words = _words(text)
    return [" ".join(words[index:index + n]) for index in range(len(words) - n + 1)]


def _special_ids(tokenizer):
    identifiers = getattr(tokenizer, "all_special_ids", []) or []
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in identifiers):
        raise ValueError("Tokenizer special IDs must be nonnegative integers")
    return sorted(set(identifiers))


def _validated_direction(value, width=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.is_floating_point():
        raise ValueError("Each probe direction must be a one-dimensional floating tensor")
    vector = value.detach().to(device="cpu", dtype=torch.float32)
    if width is not None and vector.numel() != width:
        raise ValueError(f"Probe width {vector.numel()} does not match output-head width {width}")
    if not vector.numel() or not bool(torch.isfinite(vector).all()):
        raise ValueError("Probe directions must be nonempty and finite")
    norm = float(vector.norm())
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Probe directions must have a finite nonzero norm")
    return vector / norm


def _probe_directions(probe, width=None):
    value = probe.get("direction") if isinstance(probe, Mapping) and "direction" in probe else probe
    if isinstance(value, torch.Tensor):
        return {"unspecified": _validated_direction(value, width)}
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Probe direction must be a nonempty layer-to-vector mapping")
    directions = {}
    for layer, vector in value.items():
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError("Probe layer identifiers must be nonnegative integers")
        directions[str(layer)] = _validated_direction(vector, width)
    return dict(sorted(directions.items(), key=lambda pair: int(pair[0])))


def _direction_from_probe_file(path):
    """Restricted loading, preserving the saved layer mapping or single vector."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        directions = _probe_directions(payload)
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError):
        return None
    if "unspecified" in directions:
        return directions["unspecified"]
    return {int(layer): direction for layer, direction in directions.items()}


def _output_head(model):
    native = getattr(model, "model", None)
    # HF bundles and native causal models both may expose .model. Prefer a
    # supported output head, without treating an inner decoder as the bundle.
    candidates = (model, native) if native is not None else (model,)
    for candidate in candidates:
        accessor = getattr(candidate, "get_output_embeddings", None)
        head = accessor() if callable(accessor) else getattr(candidate, "lm_head", None)
        weight = getattr(head, "weight", None)
        if isinstance(weight, torch.Tensor):
            if weight.ndim != 2 or not weight.is_floating_point() or min(weight.shape) <= 0:
                raise ValueError("Output-head weight must be a nonempty floating matrix")
            return candidate, weight
    raise ValueError("Model exposes no supported output embedding weight")


def _cpu_head_chunk(weight, start, stop):
    # Convert just one bounded slice, never the complete FP32 output head.
    return weight[start:stop].detach().to(device="cpu", dtype=torch.float32)


def _ranked_candidates(values, ids, count, descending):
    if not values.numel():
        return values, ids
    id_order = torch.argsort(ids, stable=True)
    values, ids = values[id_order], ids[id_order]
    order = torch.argsort(values, descending=descending, stable=True)[:count]
    return values[order], ids[order]


def vocab_fingerprint(direction, lm_head_weight, tokenizer, top_k=40, bottom_k=20,
                      *, chunk_size=4096):
    """Stream signed projections on CPU, preserving token IDs and controls.

    No forward pass is run. Earlier-layer fingerprints omit final normalization,
    the current state and later blocks, and therefore remain heuristic.
    """
    top_k = _positive_integer(top_k, "top_k")
    bottom_k = _positive_integer(bottom_k, "bottom_k")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    weight = lm_head_weight
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Output-head weight must be a floating matrix")
    if min(weight.shape) <= 0:
        raise ValueError("Output-head matrix must be nonempty")
    vector = _validated_direction(direction, weight.shape[1])
    count = int(weight.shape[0])
    top_k, bottom_k = min(top_k, count), min(bottom_k, count)
    positive_values = negative_values = torch.empty(0, dtype=torch.float32)
    positive_ids = negative_ids = torch.empty(0, dtype=torch.int64)
    with torch.no_grad():
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            chunk = _cpu_head_chunk(weight, start, stop)
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("Output-head matrix contains nonfinite values")
            values = chunk @ vector
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Vocabulary projection produced nonfinite scores")
            ids = torch.arange(start, stop, dtype=torch.int64)
            pos, neg = values > 0, values < 0
            positive_values, positive_ids = _ranked_candidates(
                torch.cat((positive_values, values[pos])), torch.cat((positive_ids, ids[pos])), top_k, True)
            negative_values, negative_ids = _ranked_candidates(
                torch.cat((negative_values, values[neg])), torch.cat((negative_ids, ids[neg])), bottom_k, False)

    special_ids = set(_special_ids(tokenizer))
    def decode(values, ids):
        entries = []
        for token_id, score in zip(ids.tolist(), values.tolist()):
            try:
                token = tokenizer.decode([token_id])
            except Exception:
                token = ""
            entries.append({"id": token_id, "token": token if isinstance(token, str) else "",
                            "score": float(score), "special": token_id in special_ids})
        return entries

    return {"positive": decode(positive_values, positive_ids),
            "negative": decode(negative_values, negative_ids),
            "width": int(weight.shape[1]), "vocab_size": count,
            "projection": "unit-direction-output-head-linear-cpu", "semantic_validated": False}


def _finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def behavioral_contrast(turn_records, probe_key=None, top_k=30, ngram_n=1,
                        threshold_pct=.33, *, field="reply", min_records=4):
    """Smoothed per-record document-frequency contrast with explicit support.

    Scoped rows use reading/reply/input. Legacy probe_key/text rows are marked
    as unseparated text. Input topics never enter reply counts. Readings label
    measured tails, not valence, confidence, causes, or intrinsic meanings.
    """
    top_k = _positive_integer(top_k, "top_k")
    ngram_n = _positive_integer(ngram_n, "ngram_n")
    min_records = _positive_integer(min_records, "min_records")
    if min_records < 4:
        raise ValueError("At least four records must be required")
    fraction = _finite_number(threshold_pct)
    if fraction is None or not 0 < fraction <= .5:
        raise ValueError("Tail fraction must be finite and between zero and one half")
    if field not in ("reply", "input"):
        raise ValueError("Behavior field must be reply or input")
    scored, skipped, legacy = [], 0, 0
    for index, row in enumerate(turn_records or []):
        explicit_id = index
        if isinstance(row, tuple) and len(row) == 2:
            explicit_id, row = row
        if not isinstance(row, Mapping):
            skipped += 1
            continue
        reading = _finite_number(row.get("reading", row.get(probe_key) if probe_key else None))
        text = row.get(field)
        used_legacy = field == "reply" and "reply" not in row and isinstance(row.get("text"), str)
        if used_legacy:
            text = row["text"]
        if reading is None or not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        example_id = row.get("example_id", explicit_id)
        if isinstance(example_id, bool) or not isinstance(example_id, (int, str)):
            example_id = index
        scored.append((reading, index, example_id, set(_ngrams(text, ngram_n))))
        legacy += int(used_legacy)
    scored.sort(key=lambda row: (row[0], row[1]))
    support = {"records": len(scored), "skipped_records": skipped,
               "legacy_unseparated_text_records": legacy, "tail_count": 0,
               "low_ids": [], "high_ids": [], "low_max": None, "high_min": None}
    result = {"status": "insufficient", "reason": "", "field": field,
              "statistic": "smoothed-document-log-odds", "ngram_n": ngram_n,
              "smoothing": .5, "support": support, "high": [], "low": [], "semantic_validated": False}
    if len(scored) < min_records:
        result["reason"] = f"Needs at least {min_records} measured records with {field} text"
        return result
    count = min(len(scored) // 2, max(2, int(len(scored) * fraction)))
    low_cut, high_cut = scored[count - 1][0], scored[-count][0]
    if low_cut >= high_cut:
        result["reason"] = "Measured tails overlap or readings are tied/constant"
        return result
    # Include complete boundary ties, avoiding arbitrary splitting of equally
    # measured observations into selected and discarded evidence.
    low = [row for row in scored if row[0] <= low_cut]
    high = [row for row in scored if row[0] >= high_cut]
    support.update(tail_count=count, low_ids=[row[2] for row in low], high_ids=[row[2] for row in high],
                   low_max=low_cut, high_min=high_cut, low_records=len(low), high_records=len(high))
    low_counts = collections.Counter(term for row in low for term in row[3])
    high_counts = collections.Counter(term for row in high for term in row[3])
    entries = []
    for term in sorted(set(low_counts) | set(high_counts)):
        high_count, low_count = high_counts[term], low_counts[term]
        high_odds = (high_count + .5) / (len(high) - high_count + .5)
        low_odds = (low_count + .5) / (len(low) - low_count + .5)
        entries.append({"term": term, "log_odds": math.log(high_odds / low_odds),
                        "high_count": high_count, "low_count": low_count})
    result.update(status="supported", reason="Descriptive association across distinct measured tails",
                  high=sorted((entry for entry in entries if entry["log_odds"] > 1e-12),
                              key=lambda entry: (-entry["log_odds"], entry["term"]))[:top_k],
                  low=sorted((entry for entry in entries if entry["log_odds"] < -1e-12),
                             key=lambda entry: (entry["log_odds"], entry["term"]))[:top_k])
    if not result["high"] and not result["low"]:
        result.update(status="insufficient", reason="Measured tails have no distinguishing lexical evidence")
    return result


def _lexical_entries(entries, field, top_n):
    words = set()
    for entry in entries[:top_n]:
        if isinstance(entry, Mapping) and entry.get("special"):
            continue
        text = entry.get(field, "") if isinstance(entry, Mapping) else entry[0]
        if isinstance(text, str):
            words.update(_words(text))
    return words


def _lexical_overlap(first, second):
    union = first | second
    return {"overlap_score": len(first & second) / len(union) if union else None,
            "shared": sorted(first & second), "vocab_only": sorted(first - second),
            "behavior_only": sorted(second - first), "status": "descriptive" if first and second else "unknown"}


def channel_divergence(fingerprint, contrast, top_n=10):
    """Same- and opposite-pole lexical overlap, without semantic verdicts."""
    top_n = _positive_integer(top_n, "top_n")
    positive = _lexical_entries(fingerprint.get("positive", []), "token", top_n)
    negative = _lexical_entries(fingerprint.get("negative", []), "token", top_n)
    high = _lexical_entries(contrast.get("high", []), "term", top_n)
    low = _lexical_entries(contrast.get("low", []), "term", top_n)
    supported = contrast.get("status") == "supported"
    return {"status": "descriptive" if supported and (positive or negative) else "unknown",
            "same_pole": {"positive_high": _lexical_overlap(positive, high),
                          "negative_low": _lexical_overlap(negative, low)},
            "opposite_pole": {"positive_low": _lexical_overlap(positive, low),
                              "negative_high": _lexical_overlap(negative, high)},
            "lexical_granularity": "casefolded-unigrams-from-decoded-token-fragments-and-terms",
            "subword_limit": "Decoded fragments need not identify whole words or concepts",
            "semantic_validated": False, "rename_authority": False}


def _model_identity(native, tokenizer, weight):
    config = getattr(native, "config", None)
    tokenizer_size = getattr(tokenizer, "vocab_size", None)
    specials = {name: value for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
                if isinstance((value := getattr(tokenizer, name, None)), int) and not isinstance(value, bool)}
    vocabulary_digest = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocabulary = get_vocab()
        if isinstance(vocabulary, Mapping) and vocabulary:
            digest = hashlib.sha256()
            for token, token_id in vocabulary.items():
                if not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int):
                    raise ValueError("Tokenizer vocabulary must map text to integer IDs")
            for token, token_id in sorted(vocabulary.items()):
                encoded = token.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little") + encoded)
                digest.update(str(token_id).encode("ascii") + b"\0")
            vocabulary_digest = digest.hexdigest()
    rows = sorted({0, int(weight.shape[0]) // 3, 2 * int(weight.shape[0]) // 3,
                   int(weight.shape[0]) - 1})
    sample_digest = hashlib.sha256()
    for row in rows:
        sample_digest.update(str(row).encode("ascii") + b"\0")
        sample_digest.update(_cpu_head_chunk(weight, row, row + 1).numpy().tobytes())
    return {"class": f"{type(native).__module__}.{type(native).__qualname__}",
            "name_or_path": str(getattr(config, "_name_or_path", "") or ""),
            "model_type": str(getattr(config, "model_type", "") or ""),
            "config_revision": str(getattr(config, "_commit_hash", "") or getattr(config, "revision", "") or ""),
            "output_head_shape": list(weight.shape), "output_head_dtype": str(weight.dtype),
            "output_head_sample_sha256": sample_digest.hexdigest(), "output_head_sample_rows": rows,
            "output_head_identity_limit": "Sample digest does not detect changes outside sampled rows",
            "tokenizer_identity": {"class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
                "vocab_size": tokenizer_size if isinstance(tokenizer_size, int) else int(weight.shape[0]),
                "vocabulary_sha256": vocabulary_digest,
                "vocabulary_identity_complete": vocabulary_digest is not None,
                "all_special_ids": _special_ids(tokenizer),
                "special_token_ids": specials}}


def analyze_probe_geometry(probe_name, probe, model, tokenizer=None, records=None, *,
                           top_k=20, bottom_k=20, chunk_size=4096, namespace=None):
    """Return JSON-safe scoped diagnostics, without forward/generate calls."""
    native, weight = _output_head(model)
    tokenizer = tokenizer if tokenizer is not None else getattr(model, "tok", None)
    if tokenizer is None or not callable(getattr(tokenizer, "decode", None)):
        raise ValueError("A tokenizer with a decode method is required")
    directions = _probe_directions(probe, weight.shape[1])
    records = list(records) if records is not None else []
    digest = hashlib.sha256()
    for layer, direction in directions.items():
        digest.update(layer.encode("utf-8") + b"\0")
        digest.update(str(tuple(direction.shape)).encode("ascii") + b"\0")
        digest.update(direction.numpy().tobytes())
    key = "probe_" + str(probe_name).removeprefix("probe_")
    reply = behavioral_contrast(records, key, field="reply")
    inputs = behavioral_contrast(records, key, field="input")
    fingerprints, alignments = {}, {}
    for layer, direction in directions.items():
        fingerprint = vocab_fingerprint(direction, weight, tokenizer, top_k, bottom_k, chunk_size=chunk_size)
        fingerprints[layer] = fingerprint
        alignments[layer] = channel_divergence(fingerprint, reply)
    return {"version": GEOMETRIC_VERSION, "probe_name": str(probe_name).removeprefix("probe_"),
            "namespace": namespace, "status": "candidate" if reply["status"] == "supported" else "insufficient",
            "semantic_validated": False, "rename_authority": False,
            "direction_sha256": digest.hexdigest(), "model_identity": _model_identity(native, tokenizer, weight),
            "vocabulary": {"status": "candidate", "layers": fingerprints},
            "behavior": {"reply": reply, "input": inputs},
            "alignment": {"status": "descriptive" if any(item["status"] == "descriptive" for item in alignments.values()) else "unknown",
                          "layers": alignments},
            "limitations": ["Earlier-layer direct unembedding is a heuristic, not a semantic decoder",
                "Probe readings are measured coordinates, not valence or causal effects",
                "Lexical overlap does not validate a definition or justify renaming"]}


def render_geometric_report(report, mode="run"):
    if mode not in ("run", "align"):
        raise ValueError("Diagnostic view must be run or align")
    lines = [f"[Define2: {report.get('probe_name', 'unknown')}] descriptive diagnostic; meaning remains unvalidated."]
    for channel in ("reply", "input"):
        evidence = report["behavior"][channel]
        support = evidence["support"]
        lines.append(f"  {channel}: {evidence['status']} | records={support['records']} | {evidence['reason']}")
        lines.append(f"    lower example IDs={support['low_ids']}; higher example IDs={support['high_ids']}")
        if mode == "run":
            for pole in ("high", "low"):
                terms = ", ".join(f"{entry['term']} ({entry['log_odds']:+.2f})" for entry in evidence[pole][:12])
                lines.append(f"    {pole} {channel} terms: {terms or 'insufficient lexical evidence'}")
    for layer, fingerprint in report["vocabulary"]["layers"].items():
        lines.append(f"  Layer {layer}: output-head projection heuristic")
        if mode == "run":
            for pole in ("positive", "negative"):
                tokens = ", ".join(f"{entry['token']!r} [id={entry['id']}, {entry['score']:+.3f}" +
                                   (", special]" if entry.get("special") else "]") for entry in fingerprint[pole][:12])
                lines.append(f"    {pole} projection: {tokens or 'no signed entries'}")
        alignment = report["alignment"]["layers"][layer]
        lines.append(f"    lexical alignment: {alignment['status']} (subword fragments may not be whole words)")
        for group in ("same_pole", "opposite_pole"):
            for pairing, comparison in alignment[group].items():
                score = comparison["overlap_score"]
                display = f"{score:.2f}" if score is not None else "unknown"
                lines.append(f"      {pairing}: overlap={display}; shared={comparison['shared']}")
    lines.append("  These channels do not authorize a definition, namespace rename, or vector change.")
    return "\n".join(lines)


def suggest_label_revision(probe_name, fingerprint, contrast, divergence):
    """Compatibility view: lexical diagnostics, deliberately no revision advice."""
    return (f"[Geometric diagnostic: {probe_name}] lexical alignment: {divergence['status']}. "
            "Vocabulary projection and measured language are candidate evidence; "
            "neither establishes meaning or authorizes label revision.")


def handle_define2_command(args, probes, model, tokenizer=None, turn_records=None,
                           probe_dir=None, *, behavior=None, return_report=False):
    """Compatibility adapter; watch scheduling belongs to the scoped caller."""
    if not args or len(args) > 2:
        raise ValueError(":probe define2 <probe_name> [align]")
    name = str(args[0]).removeprefix("probe_")
    mode = args[1].casefold() if len(args) > 1 else "run"
    if mode not in ("run", "align"):
        raise ValueError("Use run/align here; watch scheduling must use the shell's scoped scheduler")
    probe = probes.get(name)
    if probe is None:
        probe = probes.get("probe_" + name)
    if probe is None and probe_dir is not None:
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("Probe name cannot contain path components")
        direction = _direction_from_probe_file(Path(probe_dir) / (name + ".pt"))
        if direction is not None:
            probe = {"direction": direction}
    if probe is None:
        raise ValueError(f"No exact active or saved probe named '{name}'")
    namespace = None
    if behavior is not None:
        from invariants.probe_behavior import definition_key
        namespace = definition_key(name, probe)
        turn_records = behavior.records.get(namespace, [])
    report = analyze_probe_geometry(name, probe, model, tokenizer, turn_records, namespace=namespace)
    return report if return_report else render_geometric_report(report, mode=mode)
