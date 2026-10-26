"""
Context-local future waypoint prediction and post-maturity adjustment.

Foundational Architecture:
1. Future Token & Waypoint Prediction:
   Predicts future tokens and latent states at arbitrary horizons (k) in any order (non-sequential, out-of-order waypoints).
2. Dual-Stream Operation:
   - MODE_EXTERNAL: Ingesting external text (prompts, :doc read chunks, tool output).
   - MODE_SELF: Autoregressive token generation during model replies.
3. Waypoint Flight Ledger:
   Stores flight tickets indexed by scheduled arrival step (t + k), waiting for timeline arrival.
4. Arrival Inspection & ClaimMap Discrepancy:
   When arrival step lands, measures prediction error (token hit/miss, latent error vector)
   and projects error onto the active ClaimMap tension axis (h_A - h_B).
5. Contextual Learning:
   Every hit and miss is unit-weight evidence in its source-state neighborhood.
   Empirical transitions and token distributions are learned separately by horizon
   and stream; no shared model weights spread an observation to other contexts.
6. Post-Mature Layer Adjustment:
   Only supported trajectory dimensions are steered in a matching environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from invariants.trigger_tuner import TriggerTuner


@dataclass
class WaypointTicket:
    """A scheduled future waypoint prediction ticket waiting for timeline arrival."""
    mode: str                           # "self" or "external"
    created_step: int                   # step index when ticket was created (t)
    arrival_step: int                   # step index when ticket matures (t + k)
    horizon_k: int                      # distance ahead (k)
    predicted_token_id: Optional[int]   # predicted token ID if vocabulary available
    predicted_state: torch.Tensor       # [d_model] predicted latent vector at arrival
    claimmap_axis: Optional[torch.Tensor] = None  # [d_model] active tension axis at prediction time
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_state: Optional[torch.Tensor] = None
    environment_id: Optional[int] = None
    namespace: str = "default"
    dimension_certainty: Optional[torch.Tensor] = None


@dataclass
class ArrivalResult:
    """Inspection report when a scheduled waypoint lands on the timeline."""
    ticket: WaypointTicket
    actual_step: int
    actual_token_id: Optional[int]
    actual_state: Optional[torch.Tensor]
    token_matched: bool
    error_vector: torch.Tensor          # [d_model] actual_state - predicted_state
    error_norm: float
    claimmap_drift: float               # projection of error onto ClaimMap tension axis
    post_mature_delta: torch.Tensor     # [d_model] corrective vector to apply to intermediate layers


class FutureWaypointPredictor(nn.Module):
    """
    Neural network that predicts future waypoint states down-the-line.
    Accepts arbitrary horizon offsets k in ANY order (e.g. k=12, then k=5, then k=2).
    """

    def __init__(self, d_model: int = 4096, hidden_dim: int = 1024, max_horizon: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_horizon = max_horizon

        self.horizon_dim = max(8, hidden_dim // 4)
        # Continuous horizon embedding (sinusoidal + projection)
        self.horizon_embed = nn.Sequential(
            nn.Linear(self.horizon_dim, self.horizon_dim),
            nn.GELU(),
            nn.Linear(self.horizon_dim, d_model),
        )

        # Predictor backbone: (h_t + horizon_emb + optional_claimmap) -> h_{t+k}
        self.backbone = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        # Cold start is persistence, not a random force in the residual stream.
        nn.init.zeros_(self.backbone[-1].weight)
        nn.init.zeros_(self.backbone[-1].bias)

    def _sinusoidal_horizon(self, k: int, dim: int, device: torch.device) -> torch.Tensor:
        """Create a continuous sinusoidal representation of horizon k."""
        half_dim = max(1, dim // 2)
        emb_scale = math.log(10000) / max(1, half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb_scale)
        val = torch.tensor([float(k)], dtype=torch.float32, device=device)
        args = val * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        elif emb.shape[-1] > dim:
            emb = emb[..., :dim]
        return emb

    def forward(
        self,
        hidden_state: torch.Tensor,
        horizon_k: int,
        condition_axis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict latent state h_{t+k} for any arbitrary horizon k.
        """
        device = hidden_state.device
        h_flat = hidden_state.view(-1, self.d_model)

        # 1. Embed horizon k
        raw_sin = self._sinusoidal_horizon(horizon_k, self.horizon_dim, device)
        h_k = self.horizon_embed(raw_sin).expand_as(h_flat)

        # 2. Combine with condition axis (e.g. ClaimMap tension) if present
        if condition_axis is not None:
            c_axis = condition_axis.view(-1, self.d_model).to(device).expand_as(h_flat)
            context = h_k + c_axis
        else:
            context = h_k

        # 3. Predict future state h_{t+k}
        combined = torch.cat([h_flat, context], dim=-1)
        delta_future = self.backbone(combined)
        predicted_state = h_flat + delta_future
        return predicted_state.view_as(hidden_state)


class LookaheadApparatus:
    """Forecast, verify, and learn locally without changing shared model weights.

    Every matured ticket contributes one full observation, including misses.
    Forecasts use conditional transition means/token frequencies in a matching
    semantic neighborhood. Residual dispersion is tracked independently per
    dimension; it never changes a sample's evidence weight.
    """
    MODE_SELF = "self"
    MODE_EXTERNAL = "external"

    def __init__(self, predictor=None, tuner=None, embedding_matrix=None,
                 d_model=4096, hidden_dim=512, device="cpu"):
        from invariants.lookahead_evidence import SemanticEvidence
        self.device = torch.device(device)
        self.d_model = int(d_model)
        self.hidden_dim = hidden_dim
        self._custom_prior = predictor is not None
        self.predictor = (predictor or FutureWaypointPredictor(d_model, hidden_dim)).to(self.device).eval()
        self.predictor.requires_grad_(False)
        self.tuner = tuner or TriggerTuner()
        # Keep the frozen vocabulary in its existing dtype; no per-token fp32 copy.
        self.embedding_matrix = embedding_matrix.detach() if embedding_matrix is not None else None
        if self.embedding_matrix is not None and self.embedding_matrix.shape[1] != self.d_model:
            raise ValueError("embedding width does not match predictor width")
        self.register_tuner_knobs()
        self.evidence = SemanticEvidence(self.d_model, self._knob("lookahead_environment_similarity", 0.90))
        self.environment_namespace = "default"
        self.composition = "linear"
        self._streams = {mode: {"step": 0, "ledger": {}, "delta": None, "context": None,
                                "active_environment": None, "expected_direction": None,
                                "previous_state": None, "last_token": None}
                         for mode in (self.MODE_SELF, self.MODE_EXTERNAL)}
        self.current_mode = self.MODE_SELF
        self.arrivals = []
        self.claimmap_context = None
        self.last_observed_state = None
        self.latest_telemetry = {}
        self._target_layers = set(range(14, 25))
        self._capture_enabled = True
        self._last_saved_updates = -1

    @property
    def learning_updates(self):
        return self.evidence.updates

    @property
    def current_step(self):
        return self._streams[self.current_mode]["step"]

    @current_step.setter
    def current_step(self, value):
        self._streams[self.current_mode]["step"] = int(value)

    @property
    def ledger(self):
        return self._streams[self.current_mode]["ledger"]

    @property
    def active_post_mature_delta(self):
        return self._streams[self.current_mode]["delta"]

    @active_post_mature_delta.setter
    def active_post_mature_delta(self, value):
        self._streams[self.current_mode]["delta"] = value

    def _knob(self, name, default):
        value = float(self.tuner.get(name, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    def _vector(self, value):
        value = value.detach().float().to(self.device).reshape(-1, self.d_model)[-1].clone()
        if not torch.isfinite(value).all():
            raise ValueError("lookahead state must be finite")
        return value

    def register_tuner_knobs(self):
        defaults = {"lookahead_alpha_self": 0.15, "lookahead_alpha_ext": 0.10,
                    "lookahead_claimmap_weight": 0.50, "lookahead_default_k": 3,
                    "lookahead_max_ledger": 512, "lookahead_max_arrivals": 256,
                    "lookahead_environment_similarity": 0.90, "lookahead_phase_scale": 0.25}
        defaults.update({f"degree_scale_{key}": 1.0 for key in
                         ("diff", "horizon", "direction", "distance", "logits", "factor")})
        for name, value in defaults.items():
            self.tuner.register(name, value, kind="coefficient")

    def attach_claimmap(self, unified_axis, tension_score=0.0, mean_sim=1.0, steer_delta=None):
        axis = None if unified_axis is None else self._vector(unified_axis)
        if axis is not None:
            axis = axis / axis.norm().clamp_min(1e-6)
        self.claimmap_context = {"unified_axis": axis, "tension_score": float(tension_score),
                                 "mean_sim": float(mean_sim), "steer_delta": steer_delta}

    def detach_claimmap(self):
        self.claimmap_context = None

    def set_mode(self, mode):
        if mode not in self._streams:
            raise ValueError(f"unknown lookahead stream {mode!r}")
        self.current_mode = mode

    def set_environment(self, namespace):
        """Optional named boundary in addition to latent similarity."""
        self.environment_namespace = str(namespace)
        for stream in self._streams.values():
            stream["delta"] = None

    def reset_sequence(self, start_step=0):
        stream = self._streams[self.current_mode]
        stream.update(step=int(start_step), ledger={}, delta=None, context=None, active_environment=None,
                      expected_direction=None, previous_state=None, last_token=None)
        self.last_observed_state = None

    def _horizons(self):
        k = max(1, round(self._knob("lookahead_default_k", 3) * self._knob("degree_scale_horizon", 1)))
        cap = self.predictor.max_horizon
        return list(dict.fromkeys(min(cap, offset) for offset in (2*k, k, max(1, k//2))))

    def _token_prior(self, h):
        if self.embedding_matrix is None:
            return None
        matrix = self.embedding_matrix
        # A cold-start persistence prior. Learned tokens come from actual local
        # arrivals, not from treating an intermediate residual as next-token logits.
        query = h.to(matrix.device, matrix.dtype)
        with torch.no_grad():
            return int(torch.mv(matrix, query).argmax())

    def predict_waypoint(self, hidden_state, horizon_k, metadata=None):
        if int(horizon_k) != horizon_k or not 1 <= int(horizon_k) <= self.predictor.max_horizon:
            raise ValueError(f"horizon must be an integer in [1, {self.predictor.max_horizon}]")
        k = int(horizon_k)
        maximum = max(1, int(self._knob("lookahead_max_ledger", 512)))
        if sum(map(len, self.ledger.values())) >= maximum:
            raise ValueError("pending forecast capacity reached; wait for arrivals or increase lookahead_max_ledger")
        h = self._vector(hidden_state)
        stream = self._streams[self.current_mode]
        context = stream["context"] if stream["context"] is not None else h
        self.evidence.similarity = self._knob("lookahead_environment_similarity", 0.90)
        env = self.evidence.locate(context, self.current_mode, self.environment_namespace, create=True)
        source_token = stream["last_token"]
        row = self.evidence.stats(env, k, source_token)
        # Local transition regression is sufficient-statistic online learning:
        # all observed actual-minus-source transitions have equal unit weight.
        axis = (self.claimmap_context or {}).get("unified_axis")
        with torch.no_grad():
            pred_h = self.predictor(h, k, condition_axis=axis).clone() if self._custom_prior else h.clone()
        if row is not None and row["state_count"]:
            pred_h = h + row["mean_delta"].to(self.device) * self._knob("degree_scale_diff", 1)
        token = None
        probability = 0.0
        if row and row["token_counts"]:
            counts = row["token_counts"]
            token = max(counts, key=lambda key: (counts[key], -int(key)))
            # Temperature changes reported forecast sharpness, never sample validity.
            scale = self._knob("degree_scale_logits", 1)
            values = torch.tensor([math.log(count) * scale for count in counts.values()])
            probs = torch.softmax(values, 0)
            probability = float(probs[list(counts).index(token)])
        else:
            token = source_token if source_token is not None else self._token_prior(h)
        ticket = WaypointTicket(self.current_mode, self.current_step, self.current_step+k, k,
                                token, pred_h.cpu(), None if axis is None else axis.cpu().clone(),
                                dict(metadata or {}))
        ticket.source_state = h.cpu()
        ticket.environment_id = env
        ticket.namespace = self.environment_namespace
        ticket.dimension_certainty = self.evidence.certainty(env, k, source_token)
        ticket.metadata.update(evidence_weight=1.0, environment_id=env,
                               local_samples=0 if row is None else row["count"],
                               token_probability=probability, source_token_id=source_token)
        self.ledger.setdefault(ticket.arrival_step, []).append(ticket)
        return ticket

    def map_future_waypoints(self, hidden_state, horizons):
        tickets = []
        maximum = max(1, int(self._knob("lookahead_max_ledger", 512)))
        for horizon in dict.fromkeys(horizons):
            if sum(map(len, self.ledger.values())) >= maximum:
                break  # do not issue an unrecorded forecast or discard an existing one
            tickets.append(self.predict_waypoint(hidden_state, horizon))
        return tickets

    def step_arrival(self, actual_token_id, actual_hidden_state=None):
        # A final generated token has no residual yet. It still supplies a full
        # token observation; no fabricated latent target is used for it.
        actual = None if actual_hidden_state is None else self._vector(actual_hidden_state)
        if actual_token_id is not None:
            actual_token_id = int(actual_token_id)
            if self.embedding_matrix is not None and not 0 <= actual_token_id < len(self.embedding_matrix):
                raise ValueError("arrival token outside vocabulary")
        self.current_step += 1
        stream = self._streams[self.current_mode]
        context = stream["context"] if stream["context"] is not None else actual
        tickets = self.ledger.pop(self.current_step, [])
        results, deltas, directions = [], [], []
        current_env = None if context is None else self.evidence.locate(
            context, self.current_mode, self.environment_namespace)
        for ticket in tickets:
            matched = ticket.predicted_token_id is not None and ticket.predicted_token_id == actual_token_id
            error = torch.zeros(self.d_model, device=self.device) if actual is None else actual-ticket.predicted_state.to(self.device)
            axis = None if ticket.claimmap_axis is None else ticket.claimmap_axis.to(self.device)
            drift = 0.0 if axis is None else float(torch.dot(error, axis))
            # Record EVERY observation before considering where an intervention
            # applies. Data remains in the originating environment after a shift.
            self.evidence.observe(ticket.environment_id, ticket.horizon_k, ticket.source_state,
                                  actual, actual_token_id, matched, ticket.metadata.get("source_token_id"))
            delta = torch.zeros_like(error)
            applicable = actual is not None and self.evidence.applies(
                ticket.environment_id, context, self.current_mode, self.environment_namespace)
            if applicable:
                alpha = self._knob("lookahead_alpha_self" if ticket.mode == self.MODE_SELF else "lookahead_alpha_ext", 0)
                certainty = ticket.dimension_certainty.to(self.device)
                # Anisotropy scales the intervention only. Evidence remains unit
                # weight even for dimensions with low certainty.
                direction = self._knob("degree_scale_direction", 1)
                anisotropy = (1-direction) + direction*certainty if direction <= 1 else certainty.pow(direction)
                corrective = -error * anisotropy
                if axis is not None:
                    corrective -= self._knob("lookahead_claimmap_weight", .5)*drift*axis
                magnitude = self._knob("degree_scale_distance", 1)*self._knob("degree_scale_factor", 1)
                expected = ticket.predicted_state.to(self.device) - ticket.source_state.to(self.device)
                travelled = actual - ticket.source_state.to(self.device)
                delta = alpha*magnitude*corrective*self._direction_support(expected, travelled)
                deltas.append(delta)
                directions.append(expected)
            report = ArrivalResult(ticket, self.current_step, actual_token_id,
                                   None if actual is None else actual.cpu(), matched, error.cpu(),
                                   float(error.norm()), drift, delta.cpu())
            results.append(report)
            self.arrivals.append(report)
        # No intervention is carried into a different semantic environment.
        if deltas:
            amplitude = max(float(actual.norm()) / math.sqrt(self.d_model), 1e-6) * self._knob("lookahead_phase_scale", .25)
            stream["delta"] = self.combine_corrections(deltas, amplitude=amplitude)
            stream["active_environment"] = current_env
            stream["expected_direction"] = torch.stack(directions).mean(0)
        elif not self.evidence.applies(stream["active_environment"], context, self.current_mode, self.environment_namespace):
            stream["delta"] = None
        elif stream["delta"] is not None:
            stream["delta"] *= .85
        if actual is not None:
            stream["previous_state"] = actual
        stream["last_token"] = actual_token_id
        maximum = max(1, int(self._knob("lookahead_max_arrivals", 256)))
        self.arrivals = self.arrivals[-maximum:]
        if results:
            judged = [r for r in results if r.ticket.predicted_token_id is not None and actual_token_id is not None]
            self.latest_telemetry = {
                "step": self.current_step, "mode": self.current_mode, "matured_count": len(results),
                "match_rate": sum(r.token_matched for r in judged)/len(judged) if judged else None,
                "mean_error_norm": sum(r.error_norm for r in results)/len(results) if actual is not None else None,
                "post_mature_norm": float(stream["delta"].norm()) if stream["delta"] is not None else 0,
                "claimmap_drift": results[-1].claimmap_drift, "pending_tickets": sum(map(len, self.ledger.values())),
                "learning_updates": self.learning_updates, "evidence_weight": 1.0,
                "semantic_environments": len(self.evidence.environments), "environment_id": current_env,
                "composition": self.composition,
            }
        return results

    def set_composition(self, mode):
        mode = str(mode).lower()
        if mode not in ("linear", "sinusoidal"):
            raise ValueError("composition must be linear or sinusoidal")
        self.composition = mode
        for stream in self._streams.values():
            stream["delta"] = None
        self._last_saved_updates = -1

    def combine_corrections(self, corrections, amplitude=1.0):
        """Compose only eligible corrections; internal states are not sin-transformed.

        Linear: sum(d_i). Sinusoidal: A*sin(sum(asin(clamp(d_i/A,-1,1)))).
        A is the residual RMS times lookahead_phase_scale. A single in-range
        contribution is unchanged; opposing phases cancel; reinforcement and
        interference remain bounded per dimension by A.
        """
        stacked = torch.stack(corrections)
        if self.composition == "linear":
            return stacked.sum(0)
        if not math.isfinite(amplitude) or amplitude < 0:
            raise ValueError("phase amplitude must be finite and nonnegative")
        if amplitude == 0:
            return torch.zeros_like(stacked[0])
        phases = torch.asin((stacked/amplitude).clamp(-1, 1))
        return amplitude*torch.sin(phases.sum(0))

    @staticmethod
    def _direction_support(expected, travelled):
        """A directional intervention is applicable only along a supported path."""
        denom = expected.norm()*travelled.norm()
        if float(denom) < 1e-8:
            return torch.zeros_like(expected)
        alignment = (torch.dot(expected, travelled)/denom).clamp(0, 1)
        return alignment * (expected*travelled > 0).to(expected.dtype)

    def get_post_mature_adjustment(self, layer_idx=None, context=None):
        if layer_idx is not None and layer_idx not in self._target_layers:
            return None
        stream = self._streams[self.current_mode]
        if context is not None and not self.evidence.applies(stream["active_environment"], context,
                                                           self.current_mode, self.environment_namespace):
            return None
        alpha = self._knob("lookahead_alpha_self" if self.current_mode == self.MODE_SELF else "lookahead_alpha_ext", 0)
        delta = stream["delta"] if alpha > 0 else None
        if delta is not None and context is not None and stream["previous_state"] is not None:
            expected = stream["expected_direction"]
            if expected is not None:
                delta = delta*self._direction_support(expected, context-stream["previous_state"])
        return delta

    def ingest_external_sequence(self, token_ids, hidden_states=None, horizons=None):
        self.set_mode(self.MODE_EXTERNAL)
        tokens = list(token_ids)
        if hidden_states is not None and len(hidden_states) != len(tokens):
            raise ValueError("one external hidden state is required per token")
        horizons = self._horizons() if horizons is None else list(horizons)
        arrivals = []
        stream = self._streams[self.MODE_EXTERNAL]
        for index, token in enumerate(tokens):
            token = int(token)
            if hidden_states is not None:
                h = self._vector(hidden_states[index])
            elif self.embedding_matrix is not None and 0 <= token < len(self.embedding_matrix):
                h = self._vector(self.embedding_matrix[token])
            else:
                raise ValueError("external ingestion requires valid token embeddings or hidden states")
            previous = stream["context"]
            stream["context"] = h if previous is None else .75*previous + .25*h
            arrivals.extend(self.step_arrival(token, h))
            self.map_future_waypoints(h, horizons)
        return arrivals

    def make_steer_handles(self, model, target_layers=tuple(range(14, 25)), cap_fraction=None):
        from invariants.engine import _cap_steer
        wrapped = getattr(model, "model", model)
        core = getattr(wrapped, "model", wrapped)
        layers = getattr(core, "layers", getattr(model, "layers", None))
        if layers is None:
            return []
        valid = [index for index in target_layers if 0 <= index < len(layers)]
        if not valid:
            valid = [len(layers)//2] if len(layers) else []
        self._target_layers = set(valid)
        # One consistent sensor coordinate system for every self forecast.
        sensor = valid[0] if valid else None
        handles = []
        for index in valid:
            def hook(module, inputs, output, layer_idx=index):
                if not self._capture_enabled:
                    return output
                h = output[0] if isinstance(output, tuple) else output
                if h.ndim != 3 or h.shape[0] != 1:
                    return output
                if layer_idx == sensor:
                    self.last_observed_state = self._vector(h[0, -1])
                    self._streams[self.MODE_SELF]["context"] = self.last_observed_state
                    self._token_adjustment = self.get_post_mature_adjustment(context=self.last_observed_state)
                delta = getattr(self, "_token_adjustment", None)
                if delta is None or not bool(torch.count_nonzero(delta)):
                    return output
                # Only the current token changes. Historical prefill positions
                # never receive a correction learned from their own future.
                updated = h.clone()
                current = h[:, -1:, :]
                add = delta.to(h.device, h.dtype).view(1, 1, -1)
                updated[:, -1:, :] = current + _cap_steer(add, current, cap_fraction=cap_fraction)
                return (updated,) + tuple(output[1:]) if isinstance(output, tuple) else updated
            handles.append(layers[index].register_forward_hook(hook))
        return handles

    def create_logits_processor(self):
        from transformers import LogitsProcessor
        apparatus = self

        class LookaheadArrivalProcessor(LogitsProcessor):
            def __init__(self):
                self.seen = None
                self.active = False

            def begin_generation(self, input_ids):
                apparatus.set_mode(apparatus.MODE_SELF)
                apparatus.reset_sequence()
                self.seen = int(input_ids.shape[1])
                self.last_forecast = None
                self.active = True
                apparatus._capture_enabled = True

            def begin_chunk(self, input_ids):
                if not self.active:
                    self.begin_generation(input_ids)
                length = int(input_ids.shape[1])
                if length != self.seen:
                    raise ValueError("lookahead chunk boundary must report external or generated arrivals first")
                apparatus.set_mode(apparatus.MODE_SELF)
                apparatus.last_observed_state = None

            def ingest_external(self, input_ids):
                tokens = input_ids.reshape(-1).tolist() if torch.is_tensor(input_ids) else list(input_ids)
                apparatus.ingest_external_sequence(tokens)
                self.seen += len(tokens)
                apparatus.set_mode(apparatus.MODE_SELF)
                # External text pauses the self timeline. Outstanding self
                # forecasts still mature on future self tokens, at full weight
                # in their original context, even if the tool changed direction.
                apparatus._streams[apparatus.MODE_SELF].update(
                    delta=None, context=None, previous_state=None,
                    expected_direction=None, active_environment=None, last_token=None)
                apparatus.last_observed_state = None

            def finish_chunk(self, output_ids):
                for token in output_ids[0, self.seen:].tolist():
                    apparatus.set_mode(apparatus.MODE_SELF)
                    apparatus.step_arrival(token, None)
                    self.seen += 1

            def finish_generation(self):
                self.active = False
                apparatus._capture_enabled = False
                apparatus.set_mode(apparatus.MODE_SELF)
                apparatus.ledger.clear()  # beyond end-of-sequence is unobserved, not false
                apparatus.active_post_mature_delta = None
                apparatus.last_observed_state = None
                apparatus._token_adjustment = None
                for mode in apparatus._streams:
                    apparatus._streams[mode].update(
                        delta=None, context=None, active_environment=None,
                        expected_direction=None, previous_state=None, last_token=None, ledger={},
                    )
                if hasattr(apparatus.evidence, "clear_cache"):
                    apparatus.evidence.clear_cache()
                elif hasattr(apparatus.evidence, "_center_cache"):
                    apparatus.evidence._center_cache.clear()

            def __call__(self, input_ids, scores):
                if input_ids.shape[0] != 1:
                    raise ValueError("lookahead currently supports one generation sequence at a time")
                if not self.active:
                    # Direct callers retain the already captured initial state.
                    captured = apparatus.last_observed_state
                    self.begin_generation(input_ids)
                    apparatus.last_observed_state = captured
                apparatus.set_mode(apparatus.MODE_SELF)
                length = int(input_ids.shape[1])
                if length < self.seen or length > self.seen+1:
                    raise ValueError("lookahead observed a noncausal token boundary")
                if length > self.seen:
                    apparatus.step_arrival(int(input_ids[0, -1]), apparatus.last_observed_state)
                    self.seen = length
                apparatus._streams[apparatus.MODE_SELF]["last_token"] = int(input_ids[0, -1])
                # There is exactly one forecast set per observed position.
                key = (apparatus.current_step, self.seen)
                if apparatus.last_observed_state is not None and getattr(self, "last_forecast", None) != key:
                    apparatus.map_future_waypoints(apparatus.last_observed_state, apparatus._horizons())
                    self.last_forecast = key
                return scores

        return LookaheadArrivalProcessor()

    def save_checkpoint(self, path):
        from pathlib import Path
        path = Path(path)
        if self.learning_updates == self._last_saved_updates and path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "d_model": self.d_model, "evidence": self.evidence.state_dict(),
                   "composition": self.composition}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        self._last_saved_updates = self.learning_updates

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("version") != 1 or state.get("d_model") != self.d_model:
            raise ValueError("incompatible lookahead checkpoint")
        self.evidence.load_state_dict(state["evidence"])
        self.set_composition(state.get("composition", "linear"))
        self._last_saved_updates = self.learning_updates
