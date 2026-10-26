"""Read-only probe projections at the positions that actually predict output.

The logits processor commits a completed outer forward, so scratch forwards,
branch replays, and reprocessed prefix tokens never become extra observations.
Only scalar trajectories and the last state (plus optional running means) survive
a chunk. No model forward is added by this collector.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid

import torch


GENERATION_PROBE_SOURCE = "live_assistant_prediction"
GENERATION_PROBE_ENDPOINT = "decoder_block_outputs_with_final_model_norm"


class GenerationProbeCapture:
    def __init__(self, model, probes=None, replay_state=None, *, capture_states=False,
                 prompt_input_ids=None, generation_channel=None):
        self.model = model
        self.replay_state = replay_state if replay_state is not None else {}
        self.capture_states = bool(capture_states)
        self.generation_id = uuid.uuid4().hex
        self.generation_channel = generation_channel
        self.prompt_sha256 = None
        if prompt_input_ids is not None:
            encoded = json.dumps(prompt_input_ids.detach().cpu().tolist(), separators=(",", ":")).encode("utf-8")
            self.prompt_sha256 = hashlib.sha256(encoded).hexdigest()
        self.layers = model.model.model.layers
        self.n_layers = len(self.layers)
        self.probes = {}
        self.unavailable = {}
        self._layer_specs = {}
        for name, spec in (probes or {}).items():
            if not isinstance(spec, dict):
                self.unavailable[str(name)] = "Probe has no direction mapping."
                continue
            directions = spec.get("direction", spec)
            mode = str(spec.get("token_mode", "last")).lower()
            if mode not in {"last", "mean", "profile", "blend"}:
                mode = "last"
            valid = {}
            for key, value in directions.items() if isinstance(directions, dict) else ():
                try:
                    layer = int(key)
                    vector = torch.as_tensor(value).detach().float().reshape(-1)
                    norm = vector.norm()
                    if not 0 <= layer < self.n_layers or not vector.numel():
                        continue
                    if not bool(torch.isfinite(vector).all()) or not bool(norm > 0):
                        continue
                    valid[layer] = vector / norm
                except (TypeError, ValueError, RuntimeError):
                    continue
            if not valid:
                self.unavailable[str(name)] = "Probe has no compatible finite nonzero direction."
                continue
            self.probes[str(name)] = {"direction": valid, "token_mode": mode}
            for layer, vector in valid.items():
                self._layer_specs.setdefault(layer, []).append((str(name), vector))
        self._needs_means = any(p["token_mode"] != "last" for p in self.probes.values())
        self._handles = []
        self._matrices = {}
        self._active = False
        self._depth = 0
        self._pending = {}
        self._candidate = None
        self._last_block_seen = False
        self._chunk = []
        self._records = []
        self._sums = {}
        self._last_states = {}
        self._state_count = 0
        self._errors = {}

    def attach(self):
        """Attach after mutating hooks so observations see the returned states."""
        if self._handles:
            return self
        try:
            self._handles.append(self.model.model.register_forward_pre_hook(self._begin_forward))
            self._handles.append(self.model.model.register_forward_hook(
                self._end_forward, always_call=True))
            wanted = set(range(self.n_layers)) if self.capture_states else set(self._layer_specs)
            # The final raw-block marker distinguishes real terminal normalization
            # from norm calls made inside routing/synthesis hooks for scratch logits.
            for layer in sorted(wanted):
                self._handles.append(self.layers[layer].register_forward_hook(
                    self._layer_hook(layer)))
            if self.n_layers - 1 in wanted:
                self._handles.append(self.model.model.model.norm.register_forward_hook(self._norm_hook))
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        self._depth = 0
        self._pending.clear()
        self._candidate = None

    def start_chunk(self):
        self._chunk = []
        self._pending = {}
        self._candidate = None
        self._active = True

    def abort_chunk(self):
        self._chunk.clear()
        self._pending.clear()
        self._candidate = None
        self._active = False

    def _begin_forward(self, _module, _args):
        if self._active and self._depth == 0:
            self._pending = {}
            self._candidate = None
            self._last_block_seen = False
        self._depth += 1

    def _end_forward(self, _module, _args, output):
        if self._active and self._depth == 1 and output is not None:
            self._candidate = self._pending
            self._pending = {}
        self._depth = max(0, self._depth - 1)

    def _can_capture(self):
        return self._active and self._depth == 1 and not self.replay_state.get("replay_active", False)

    def _layer_hook(self, layer):
        def observe(_module, _args, output):
            if not self._can_capture():
                return
            if layer == self.n_layers - 1:
                self._last_block_seen = True
                return
            self._capture(layer, output)
        return observe

    def _norm_hook(self, _module, _args, output):
        if self._can_capture() and self._last_block_seen:
            self._capture(self.n_layers - 1, output)

    def _capture(self, layer, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
            self._errors[layer] = "Expected one outer generation sequence."
            return
        state = hidden[0, -1].detach().float()
        entries = self._layer_specs.get(layer, ())
        if entries:
            key = (layer, state.device, state.shape[-1])
            if key not in self._matrices:
                compatible = [(name, vector) for name, vector in entries
                              if vector.numel() == state.numel()]
                for name, vector in entries:
                    if vector.numel() != state.numel():
                        self.unavailable[name] = "Probe direction width differs from live hidden state."
                self._matrices[key] = (
                    [name for name, _ in compatible],
                    torch.stack([v.to(state.device) for _, v in compatible]) if compatible else None,
                )
            names, matrix = self._matrices[key]
            scores = (matrix @ state) / state.norm().clamp_min(1e-30) if matrix is not None else None
        else:
            names, scores = [], None
        # State snapshots are bounded to one chunk until it is accepted, then to
        # one last state and a running sum per layer. No full hidden trajectory.
        keep_state = self.capture_states or self._needs_means
        self._pending[layer] = {
            "names": names,
            "scores": scores.detach() if scores is not None else None,
            "state": state.clone() if keep_state else None,
        }

    def commit_prediction(self, input_ids):
        """Called only by the outer generate's logits processor, once per token."""
        candidate, self._candidate = self._candidate, None
        if not self._active or not candidate:
            return
        position = int(input_ids.shape[-1])
        previous = self._chunk[-1]["position"] if self._chunk else (
            self._records[-1]["position"] if self._records else -1)
        if position <= previous:
            return
        self._chunk.append({"position": position, "layers": candidate})

    def commit_chunk(self, emitted_count):
        """Keep only predictions whose tokens survive the generation call."""
        accepted = self._chunk[:max(0, int(emitted_count))]
        # Transfer each layer's scalar matrix once per chunk, rather than forcing
        # a CPU/GPU synchronization for every probe on every prediction.
        scalar_rows = {}
        for layer in {layer for record in accepted for layer in record["layers"]}:
            indexed = [(index, record["layers"][layer]) for index, record in enumerate(accepted)
                       if layer in record["layers"] and record["layers"][layer]["scores"] is not None]
            if indexed:
                values = torch.stack([row["scores"] for _, row in indexed]).cpu().tolist()
                scalar_rows[layer] = {
                    index: dict(zip(row["names"], scores))
                    for (index, row), scores in zip(indexed, values)
                }
        for index, record in enumerate(accepted):
            scalar_layers = {}
            complete_states = {}
            for layer, row in record["layers"].items():
                if index in scalar_rows.get(layer, {}):
                    scalar_layers[layer] = scalar_rows[layer][index]
                state = row["state"]
                if state is not None:
                    complete_states[layer] = state
                    if self._needs_means:
                        if layer not in self._sums:
                            self._sums[layer] = state.clone()
                        else:
                            self._sums[layer].add_(state)
            if complete_states:
                self._last_states = complete_states
                self._state_count += 1
            self._records.append({"position": record["position"], "layers": scalar_layers})
        self.abort_chunk()

    def last_hidden_states(self):
        """Last committed prediction state, matching _hidden_states layer order."""
        if not self.capture_states or len(self._last_states) != self.n_layers:
            return None
        return torch.stack([self._last_states[l].detach().cpu() for l in range(self.n_layers)]).unsqueeze(1)

    def summary(self):
        result = {}
        for name, probe in self.probes.items():
            trajectory, positions, layer_trajectories = [], [], {}
            required = set(probe["direction"])
            for record in self._records:
                values = {layer: row[name] for layer, row in record["layers"].items() if name in row}
                if set(values) != required or not all(math.isfinite(v) for v in values.values()):
                    continue
                positions.append(record["position"])
                trajectory.append(sum(values.values()) / len(values))
                for layer, value in values.items():
                    layer_trajectories.setdefault(layer, []).append(value)
            if not trajectory:
                continue
            by_layer = {layer: values[-1] for layer, values in layer_trajectories.items()}
            mode = probe["token_mode"]
            if mode != "last" and self._state_count:
                for layer, vector in probe["direction"].items():
                    if layer not in self._sums or layer not in self._last_states:
                        continue
                    representative = self._sums[layer] / self._state_count
                    if mode in {"profile", "blend"}:
                        representative = .5 * representative + .5 * self._last_states[layer]
                    value = (representative @ vector.to(representative.device)) / representative.norm().clamp_min(1e-30)
                    by_layer[layer] = float(value.cpu())
            raw = sum(by_layer.values()) / len(by_layer)
            result[name] = {
                "raw_score": raw,
                "score": raw,
                "last_token_score": trajectory[-1],
                "mean_token_score": sum(trajectory) / len(trajectory),
                "token_profile_score": .5 * trajectory[-1] + .5 * sum(trajectory) / len(trajectory),
                "by_token": trajectory,
                "by_layer": by_layer,
                "prediction_positions": positions,
                "token_count": len(trajectory),
                "token_mode": mode,
                "measurement": GENERATION_PROBE_SOURCE,
                "endpoint": GENERATION_PROBE_ENDPOINT,
            }
        return result

    def metadata(self):
        return {
            "source": GENERATION_PROBE_SOURCE,
            "endpoint": GENERATION_PROBE_ENDPOINT,
            "generation_id": self.generation_id,
            "prompt_sha256": self.prompt_sha256,
            "prompt_hash_encoding": "json_token_ids",
            "generation_channel": self.generation_channel,
            "decision_position": self._records[-1]["position"] - 1 if self._records else None,
            "token_count": len(self._records),
            "position_semantics": "absolute_position_of_token_predicted_by_captured_state",
            "prefix_reprocessing": "last_prediction_only",
            "scratch_replays": "excluded",
            "unavailable_probes": dict(self.unavailable),
            "layer_errors": {str(layer): message for layer, message in self._errors.items()},
        }
