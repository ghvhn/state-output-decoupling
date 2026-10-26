"""Bounded, state-conditioned attention over complete memory sources.

Provenance supplies explicit attention strengths, never a truth label or a
universal ranking of source types. Every recall outcome is one observation;
attention strength and evidence count are intentionally different quantities.
Raw source spans are encoded without a chat template or synthetic instruction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch


_AUXILIARY_ENCODING = ContextVar("memory_auxiliary_encoding", default=False)


@contextmanager
def auxiliary_encoding():
    """Mark source measurements so generation observers can ignore them."""
    token = _AUXILIARY_ENCODING.set(True)
    try:
        yield
    finally:
        _AUXILIARY_ENCODING.reset(token)


def is_auxiliary_encoding():
    return bool(_AUXILIARY_ENCODING.get())


def _bounded(value, default=1.0, upper=4.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return min(float(upper), max(0.0, value)) if math.isfinite(value) else float(default)


def _field(record, name, default=None):
    return record.get(name, default) if isinstance(record, dict) else getattr(record, name, default)


def _unit_vector(value):
    if not torch.is_tensor(value):
        try:
            value = torch.tensor(value, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError):
            return None
    if value.ndim != 1 or not value.numel():
        return None
    value = value.detach().float().cpu()
    if not torch.isfinite(value).all() or value.norm() <= 1e-12:
        return None
    return value / value.norm()


def provenance_weight(provenance=None, metrics=None):
    """Explicit finite attention strength, default one; no source trust rules.

    ``provenance_weight`` takes precedence over ``relevance_weight``. A caller
    may supply either in provenance or metrics. Role, source, confidence, and
    truth/correctness annotations do not silently change this strength.
    """
    for mapping in (provenance, metrics):
        if isinstance(mapping, dict):
            for key in ("provenance_weight", "relevance_weight"):
                if key in mapping:
                    return _bounded(mapping[key])
    return 1.0


def text_spans(text, span_chars=512):
    """Return contiguous source windows whose concatenation is the whole text.

    Prefer a nearby paragraph or sentence boundary, then whitespace; long
    unbroken text is split at the budget. Offsets always refer to the original
    source, including whitespace. ``span_chars`` bounds each window, not the
    amount of a memory that is eligible for attention.
    """
    text = str(text or "")
    span_chars = max(1, int(span_chars))
    spans = []
    start = 0
    while start < len(text):
        end = min(len(text), start + span_chars)
        if end < len(text):
            window = text[start:end]
            lower = max(1, len(window) // 2)
            paragraphs = list(re.finditer(r"\n\s*\n|\n", window))
            sentences = list(re.finditer(r"[.!?](?:[\"')\]]*)\s+", window))
            boundary = next((m.end() for m in reversed(paragraphs) if m.end() >= lower), None)
            if boundary is None:
                boundary = next((m.end() for m in reversed(sentences) if m.end() >= lower), None)
            if boundary is None:
                whitespace = list(re.finditer(r"\s+", window))
                boundary = next((m.end() for m in reversed(whitespace) if m.end() >= lower), None)
            if boundary is not None:
                end = start + boundary
        spans.append({"index": len(spans), "start": start, "end": end, "text": text[start:end]})
        start = end
    return spans


def _span_provenance(span, provenance):
    base = provenance_weight(provenance)
    annotations = provenance.get("spans", []) if isinstance(provenance, dict) else []
    strengths = []
    for item in annotations if isinstance(annotations, (list, tuple)) else []:
        if not isinstance(item, dict):
            continue
        try:
            overlap = max(0, min(span["end"], int(item.get("end", span["end"])))
                          - max(span["start"], int(item.get("start", 0))))
        except (TypeError, ValueError, OverflowError):
            continue
        if overlap:
            strengths.append((overlap, _bounded(item.get("weight", item.get("provenance_weight", 1.0)))))
    if strengths:
        base *= sum(length * strength for length, strength in strengths) / sum(length for length, _ in strengths)
    return _bounded(base)


def _stimulation(span, stimulation, provenance):
    value = None
    if callable(stimulation):
        value = stimulation(span)
    elif isinstance(stimulation, dict):
        value = stimulation.get(span["index"], stimulation.get(str(span["index"])))
    elif isinstance(stimulation, (list, tuple)) and span["index"] < len(stimulation):
        value = stimulation[span["index"]]
    if value is None and isinstance(provenance, dict):
        annotations = provenance.get("spans", [])
        for item in annotations if isinstance(annotations, (list, tuple)) else []:
            if not isinstance(item, dict) or "stimulation" not in item:
                continue
            try:
                if int(item.get("start", 0)) < span["end"] and int(item.get("end", span["end"])) > span["start"]:
                    value = item["stimulation"]
            except (TypeError, ValueError, OverflowError):
                continue
    return _bounded(value, default=1.0)


def _distributed_indices(count, budget, rotation=0):
    if count <= 0 or budget <= 0:
        return []
    budget = min(count, budget)
    offset = int(rotation) % count
    # A single sample starts at the tail. Multi-span initial samples include
    # both ends; rotating all positions makes every source span eligible.
    initial = [count - 1] if budget == 1 else [round(i * (count - 1) / (budget - 1)) for i in range(budget)]
    return [(index + offset) % count for index in initial]


def select_spans(spans, max_spans=4, stimulation=None, provenance=None, rotation=0):
    """Select bounded measured/annotated attention plus rotating coverage.

    Stimulation accepts an index-to-scalar mapping, scalar sequence, or
    ``callable(span)``. Raw hidden-state tensors are encoded/scored by
    ``encode_weighted_memory`` or ``SalientMemoryIndex``, not this pure helper.
    Even strongly weighted spans leave one slot for coverage of other content.
    The returned ``weight`` is attention strength, not an evidence count.
    """
    candidates = []
    for span in spans or []:
        candidate = dict(span)
        candidate["provenance_weight"] = _span_provenance(candidate, provenance)
        candidate["stimulation"] = _stimulation(candidate, stimulation, provenance)
        candidate["weight"] = _bounded(candidate["provenance_weight"] * candidate["stimulation"], upper=16.0)
        candidates.append(candidate)
    budget = min(len(candidates), max(0, int(max_spans)))
    if not budget:
        return []
    if len(candidates) <= budget:
        return candidates
    strengths = [item["weight"] for item in candidates]
    if max(strengths) - min(strengths) <= 1e-12:
        selected = _distributed_indices(len(candidates), budget, rotation)
    else:
        coverage = (len(candidates) - 1 + int(rotation)) % len(candidates)
        priority = sorted(range(len(candidates)), key=lambda i: (-strengths[i], i))
        selected = priority[:max(0, budget - 1)]
        if coverage not in selected:
            selected.append(coverage)
        for index in priority:
            if len(selected) >= budget:
                break
            if index not in selected:
                selected.append(index)
    return [candidates[index] for index in selected]


def _raw_encoder(model):
    language_model = getattr(model, "model", model)
    backbone = getattr(language_model, "model", None)
    if backbone is None:
        backbone = getattr(language_model, "transformer", None)

    def encode(text):
        if backbone is None or not text.strip():
            return None
        from invariants.engine import _inputs
        inputs = _inputs(model, text, pre_formatted=True)
        with torch.no_grad():
            output = backbone(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                              use_cache=False, output_hidden_states=False, return_dict=True)
            hidden = getattr(output, "last_hidden_state", None)
            if not torch.is_tensor(hidden) or hidden.ndim != 3 or not hidden.shape[1]:
                return None
            # Pool all tokens in a source window rather than treating its first
            # characters or last token as the entire memory.
            mask = inputs.get("attention_mask")
            values = hidden[0].detach().float()
            if torch.is_tensor(mask):
                mask = mask[0].to(values.device).float().unsqueeze(-1)
                values = (values * mask).sum(0) / mask.sum().clamp_min(1.0)
            else:
                values = values.mean(0)
            return _unit_vector(values)
    return encode


def encode_weighted_memory(model, text, *, state=None, provenance=None, metrics=None,
                           span_chars=512, max_spans=4, rotation=0, encoder=None):
    """Encode distributed source spans; return vectors, offsets, and weights.

    ``encoder(raw_span_text)`` can substitute a caller's activation measurement.
    All encoded vectors are detached and normalized on CPU. Current state
    changes the aggregation without rewriting provenance or source text.
    """
    spans = text_spans(text, span_chars)
    effective_provenance = dict(provenance) if isinstance(provenance, dict) else {}
    effective_provenance.setdefault("provenance_weight", provenance_weight(provenance, metrics))
    chosen = select_spans(spans, max_spans=max_spans, provenance=effective_provenance, rotation=rotation)
    encode = encoder or _raw_encoder(model)
    query = _unit_vector(state)
    encoded = []
    for span in chosen:
        try:
            with auxiliary_encoding():
                vector = _unit_vector(encode(span["text"]))
        except Exception:
            continue
        if vector is None:
            continue
        item = dict(span, vector=vector)
        if query is not None and query.shape == vector.shape:
            item["stimulation"] = max(0.0, float(torch.dot(query, vector).clamp(-1.0, 1.0)))
            item["weight"] = item["provenance_weight"] * item["stimulation"]
        encoded.append(item)
    compatible = [item for item in encoded if item["vector"].shape == encoded[0]["vector"].shape] if encoded else []
    combined = None
    if compatible:
        total = sum(item["weight"] for item in compatible)
        if total > 1e-12:
            combined = _unit_vector(sum(item["vector"] * item["weight"] for item in compatible) / total)
    return {"vector": combined, "spans": encoded, "span_count": len(spans),
            "selected_count": len(chosen), "newly_encoded": len(encoded)}


class SalientMemoryIndex:
    """Re-rank recallable records by actual state and explicit provenance.

    Encoding work and resident vectors are bounded. Unknown source regions
    rotate into the measured set; cached regions are rechecked when context
    changes. No record is marked permanently read or ineligible after recall.
    """

    OUTCOMES = {"helpful", "irrelevant", "contradicted", "unknown", "empty"}

    def __init__(self, model, *, max_cached_vectors=256, span_chars=512,
                 max_spans=4, max_new_vectors=16, max_evidence=256, encoder=None,
                 namespace="default"):
        self.model = model
        self.max_cached_vectors = max(1, min(4096, int(max_cached_vectors)))
        self.span_chars = max(1, int(span_chars))
        self.max_spans = max(1, int(max_spans))
        self.max_new_vectors = max(0, int(max_new_vectors))
        self.max_evidence = max(1, int(max_evidence))
        self.namespace = str(namespace)
        self._encode = encoder or _raw_encoder(model)
        self._cache = OrderedDict()
        self._prepared = OrderedDict()
        self._evidence = []
        self._rotation = 0
        self._record_rotation = {}
        self.last_stats = {"newly_encoded": 0, "candidates": 0, "cached_vectors": 0}

    @staticmethod
    def record_id(record):
        identifier = _field(record, "record_id")
        if identifier:
            return str(identifier)
        identity = (str(_field(record, "scope", "default")), str(_field(record, "text", "")))
        return hashlib.sha256("\0".join(identity).encode("utf-8")).hexdigest()

    def _evidence_weight(self, identifier, state, channel):
        numerator = denominator = 0.0
        for entry in self._evidence:
            if entry["record_id"] != identifier or entry["channel"] != str(channel or "speech"):
                continue
            previous = _unit_vector(entry["state"])
            if previous is None or previous.shape != state.shape:
                continue
            similarity = float(torch.dot(state, previous))
            if similarity < 0.8:
                continue
            # Contradiction describes content, not recall utility or truth.
            # Unresolved/empty attempts remain unit evidence with neutral vote.
            value = {"helpful": 1.0, "irrelevant": -1.0}.get(entry["outcome"], 0.0)
            numerator += similarity * value
            denominator += similarity
        return 1.0 if not denominator else 1.0 + 0.5 * numerator / denominator

    def prepare(self, records, scope=None, channel=None, scan=160):
        """Prewarm source vectors outside generation; expose measured coverage.

        Candidate records and their source spans rotate across the full allowed
        source. A exhausted encoding budget is partial coverage, not evidence
        that the unmeasured records are empty or irrelevant.
        """
        self.last_stats = {"newly_encoded": 0, "candidates": 0, "cached_vectors": len(self._cache)}
        eligible = [record for record in records or []
                    if (scope is None or _field(record, "scope", "default") == scope)
                    and str(_field(record, "text", "")).strip()]
        eligible_ids = {self.record_id(record) for record in eligible}
        for identifier in list(self._prepared):
            if identifier not in eligible_ids:
                del self._prepared[identifier]
        indices = _distributed_indices(len(eligible), min(len(eligible), max(0, int(scan))), self._rotation)
        self._rotation += 1
        budget = self.max_new_vectors
        for index in indices:
            record = eligible[index]
            identifier = self.record_id(record)
            text = str(_field(record, "text", ""))
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            raw_provenance = _field(record, "provenance", {})
            provenance = dict(raw_provenance) if isinstance(raw_provenance, dict) else {}
            provenance.setdefault("provenance_weight", provenance_weight(provenance, _field(record, "metrics", {})))
            source_spans = text_spans(text, self.span_chars)
            self._prepared[identifier] = {"record": record, "record_id": identifier, "digest": digest,
                                          "provenance": provenance, "spans": source_spans,
                                          "scope": _field(record, "scope", "default")}
            self._prepared.move_to_end(identifier)
            rotation = self._record_rotation.get(identifier, 0)
            self._record_rotation[identifier] = rotation + 1
            chosen = select_spans(source_spans, self.max_spans, provenance=provenance, rotation=rotation)
            for span in chosen:
                key = (identifier, digest, span["start"], span["end"])
                if key in self._cache:
                    self._cache.move_to_end(key)
                elif budget:
                    try:
                        with auxiliary_encoding():
                            vector = _unit_vector(self._encode(span["text"]))
                    except Exception:
                        vector = None
                    budget -= 1
                    if vector is not None:
                        self._cache[key] = vector
                        self.last_stats["newly_encoded"] += 1
                        while len(self._cache) > self.max_cached_vectors:
                            self._cache.popitem(last=False)
        self.last_stats["candidates"] = len(indices)
        self.last_stats["cached_vectors"] = len(self._cache)
        indexed_records = 0
        indexed_spans = 0
        total_spans = 0
        for item in self._prepared.values():
            item["measured_spans"] = [span for span in item["spans"]
                                      if (item["record_id"], item["digest"], span["start"], span["end"]) in self._cache]
            measured = len(item["measured_spans"])
            indexed_records += bool(measured)
            indexed_spans += measured
            total_spans += len(item["spans"])
        self.last_stats.update(total_records=len(eligible), prepared_records=len(self._prepared),
                               indexed_records=indexed_records, unindexed_records=len(eligible) - indexed_records,
                               indexed_spans=indexed_spans, prepared_spans=total_spans,
                               complete=indexed_records == len(eligible) and indexed_spans == total_spans)
        return dict(self.last_stats)

    def rank(self, records, state, scope=None, channel=None, scan=160, *, allow_encode=True):
        """Rank measured spans; ``allow_encode=False`` never forwards the model.

        Safe inside a generation seam after ``prepare`` has run. It changes
        neither sampling rotations nor source measurements. ``records=None``
        with encoding disabled queries all caller-prepared records.
        """
        query = _unit_vector(state)
        if query is None:
            return []
        if allow_encode:
            self.prepare(records, scope=scope, channel=channel, scan=scan)
        allowed_ids = None if records is None else {self.record_id(record) for record in records}
        ranked = []
        for item in self._prepared.values():
            if (allowed_ids is not None and item["record_id"] not in allowed_ids
                    or scope is not None and item["scope"] != scope):
                continue
            record, identifier, digest = item["record"], item["record_id"], item["digest"]
            provenance, source_spans = item["provenance"], item["spans"]
            evidence = self._evidence_weight(identifier, query, channel)
            measurements = []
            # The live seam visits at most the bounded resident cache, rather
            # than walking every unmeasured character window in long records.
            for span in item.get("measured_spans", []):
                key = (identifier, digest, span["start"], span["end"])
                vector = self._cache.get(key)
                if vector is None or vector.shape != query.shape:
                    continue
                stimulation = max(0.0, float(torch.dot(query, vector).clamp(-1.0, 1.0)))
                strength = _span_provenance(span, provenance)
                measurements.append(dict(span, vector=vector.clone(), stimulation=stimulation, provenance_weight=strength,
                                         weight=stimulation * strength, score=stimulation * strength * evidence))
            if measurements:
                best = max(measurements, key=lambda item: (item["score"], item["stimulation"], -item["index"]))
                total = sum(item["weight"] for item in measurements)
                vector = _unit_vector(sum(item["vector"] * item["weight"] for item in measurements) / total) if total > 1e-12 else None
                ranked.append({"record": record, "record_id": identifier, "score": best["score"],
                               "stimulation": best["stimulation"], "provenance_weight": best["provenance_weight"],
                               "evidence_weight": evidence, "span": best, "spans": measurements, "vector": vector,
                               "span_count": len(source_spans)})
        ranked.sort(key=lambda item: (-item["score"], -item["stimulation"], item["record_id"]))
        return ranked

    def query(self, state, *, scope=None, channel=None, max_records=5):
        """Pure cached-vector query suitable for a live generation observer."""
        return self.rank(None, state, scope=scope, channel=channel, allow_encode=False)[:max(0, int(max_records))]

    def observe(self, record_id, state, outcome="unknown", channel="speech", weight=1):
        """Keep one valid outcome observation, independent of attention weight.

        ``weight`` is accepted for callers carrying attention metadata; it is
        never used to multiply evidence count. Invalid/unmeasured state cannot
        invent a semantic neighborhood and therefore is not added.
        """
        if outcome not in self.OUTCOMES:
            raise ValueError("Unsupported recall outcome: " + str(outcome))
        vector = _unit_vector(state)
        if vector is None:
            return False
        self._evidence.append({"record_id": str(record_id), "state": vector.tolist(),
                               "outcome": outcome, "channel": str(channel or "speech"), "count": 1})
        self._evidence = self._evidence[-self.max_evidence:]
        return True

    def snapshot(self):
        """Compact learned evidence only; source records and base weights excluded."""
        return {"version": 1, "namespace": self.namespace,
                "evidence": [dict(item, state=list(item["state"])) for item in self._evidence]}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def load(self, path):
        path = Path(path)
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or payload.get("namespace") != self.namespace:
            return False
        evidence = []
        for item in payload.get("evidence", []):
            if not isinstance(item, dict) or item.get("outcome") not in self.OUTCOMES:
                continue
            state = _unit_vector(item.get("state"))
            if state is not None:
                evidence.append({"record_id": str(item.get("record_id", "")), "state": state.tolist(),
                                 "outcome": item["outcome"], "channel": str(item.get("channel") or "speech"), "count": 1})
        self._evidence = evidence[-self.max_evidence:]
        return True
