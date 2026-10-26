"""Document relevance conditioned on the reader's actual generation state.

This is observational selection, not a learned curiosity reward or a steer.
Source passages are encoded as raw text; no query or semantic instruction is
invented. Reading-pass coverage remains owned by document_engine.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import torch

from invariants.document_engine import document_layout_sha256, sha256_text


def _last_vector(output: Any):
    if not torch.is_tensor(output):
        return None
    if output.ndim == 3 and output.shape[0] == 1 and output.shape[1]:
        return output[0, -1, :].detach().clone()
    if output.ndim == 2 and output.shape[0]:
        return output[-1, :].detach().clone()
    if output.ndim == 1 and output.numel():
        return output.detach().clone()
    return None


def _unit_vector(value: Any):
    if not torch.is_tensor(value) or value.ndim != 1 or not value.numel():
        return None
    vector = value.detach().float().cpu()
    if not torch.isfinite(vector).all():
        return None
    magnitude = vector.norm()
    if magnitude <= 1e-12:
        return None
    return vector / magnitude


class IntrinsicDocumentReader:
    """Cache passage vectors, and re-rank them as the reader's state changes.

    ``latest_state`` is the last post-normalization state that produced a
    token in a successful generation. It is not a re-encoding of response
    text. One-token decoding naturally leaves the final emitted token not
    yet processed; this captures the state used to predict that token.
    """

    def __init__(self, model, *, max_cached_vectors: int = 256, max_chars: int = 600):
        self.model = model
        self.latest_state = None
        self.channel_states = {}
        self.current_channel = None
        self.max_cached_vectors = max(1, min(256, int(max_cached_vectors)))
        self.max_chars = max(1, int(max_chars))
        self._cache = OrderedDict()
        self._capturing = False
        self._language_model = getattr(model, "model", model)
        self._backbone = getattr(self._language_model, "model", None)
        if self._backbone is None:
            self._backbone = getattr(self._language_model, "transformer", None)
        self._norm = getattr(self._backbone, "norm", None)
        if self._norm is None:
            self._norm = getattr(self._backbone, "ln_f", None)

    @contextmanager
    def capture_generation(self, channel="speech"):
        """Observe generation without changing outputs; publish only on success.

        Candidate ranking does not run during capture. The hook is removed
        even when generation fails, and a failure retains the prior state.
        """
        if channel not in {"speech", "reader"}:
            raise ValueError("Capture channel must be 'speech' or 'reader'.")
        if self._capturing:
            raise RuntimeError("Document generation capture is already active.")
        self._capturing = True
        pending = None
        handle = None

        def observe(_module, _inputs, output):
            nonlocal pending
            # The normal outer forward overwrites any intermediate norm calls
            # made by routing. No graph or full-sequence tensor is retained.
            pending = _last_vector(output)

        try:
            if self._norm is not None:
                handle = self._norm.register_forward_hook(observe)
            yield self
            if pending is not None:
                unit = _unit_vector(pending)
                self.latest_state = pending.detach().float().cpu().clone() if unit is not None else None
                self.current_channel = channel if unit is not None else None
                if unit is not None:
                    self.channel_states[channel] = self.latest_state.clone()
        finally:
            if handle is not None:
                handle.remove()
            self._capturing = False

    def channel_difference(self):
        """Last reader state minus last speech state, or None if unavailable.

        This is a descriptive temporal comparison. The channels may have read
        different context; the difference does not isolate a causal mode effect
        and is never applied as a steering vector.
        """
        reader = self.channel_states.get("reader")
        speech = self.channel_states.get("speech")
        if _unit_vector(reader) is None or _unit_vector(speech) is None:
            return None
        if reader.shape != speech.shape:
            return None
        return (reader.detach().float().cpu() - speech.detach().float().cpu()).clone()

    def _encode_source(self, text: str):
        from invariants.engine import _inputs

        if self._backbone is None or not text.strip():
            return None
        # No chat template, assistant preamble, synthetic query, or instruction.
        inputs = _inputs(self.model, text[:self.max_chars], pre_formatted=True)
        with torch.no_grad():
            output = self._backbone(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = getattr(output, "last_hidden_state", None)
            return _unit_vector(_last_vector(hidden))

    def rank(self, sessions, scan: int = 160):
        """Return (cosine scores by (session,chunk), new vectors, candidates).

        The prefix limit bounds encoding work, so scores describe the first
        ``max_chars`` characters, not a guarantee of whole-passage relevance.
        Missing state/unsupported models return empty scores for an honest
        deterministic-order fallback. Candidate forwards never replace state.
        """
        state = _unit_vector(self.latest_state)
        if self._capturing or state is None or self._backbone is None:
            return {}, 0, 0
        sessions = list(sessions or [])
        identities = []
        candidates = []
        for si, session in enumerate(sessions):
            chunks = session.get("chunks") or []
            identity = (
                str(session.get("sha256") or session.get("source_path") or si),
                str(session.get("layout_sha256") or document_layout_sha256(chunks)),
                str(session.get("reading_pass_id") or ""),
            )
            identities.append(identity)
            for ci in range(min(len(chunks), int(session.get("chunk_count", len(chunks))))):
                if ci not in (session.get("read") or set()):
                    candidates.append((si, ci))
        active = set(identities)
        for key in list(self._cache):
            if key[1:4] not in active:
                del self._cache[key]
        candidates = candidates[:max(0, int(scan))]
        scores = {}
        newly_encoded = 0
        for si, ci in candidates:
            session = sessions[si]
            text = str(session["chunks"][ci])
            key = (id(self._language_model), *identities[si], ci, sha256_text(text))
            vector = self._cache.get(key)
            if vector is None:
                try:
                    vector = self._encode_source(text)
                except Exception:
                    # Missing/invalid candidate measurements are not scores.
                    continue
                if vector is None:
                    continue
                newly_encoded += 1
                self._cache[key] = vector
                while len(self._cache) > self.max_cached_vectors:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(key)
            if vector.shape != state.shape:
                continue
            scores[(si, ci)] = float(torch.dot(state, vector).clamp(-1.0, 1.0))
        return scores, newly_encoded, len(candidates)
