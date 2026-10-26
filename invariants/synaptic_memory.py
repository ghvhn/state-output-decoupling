"""Synaptic Memory & Weight-Integrated Plasticity Engine.

Integrates episodic memories directly into transformer layer weights (e.g. MLP down_proj
and Attention o_proj) via low-rank associative outer-products and bounded weight updates.
This allows prior experiences and consolidated insights to alter internal representations
natively during forward passes, rather than relying on prompt-stuffed active retrieval.
"""

from __future__ import annotations

import os
import math
import hashlib
import threading
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def _unit_features(value: torch.Tensor) -> torch.Tensor:
    """Normalize along features without overflowing for large activations."""
    value = value.float()
    scale = value.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scaled = value / scale
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# A generation selects its own learned overlay without changing shared weights
# or another thread's selection. Unselected/background contexts retain main.
_ACTIVE_ENGINE: ContextVar[Optional["SynapticMemoryEngine"]] = ContextVar(
    "active_synaptic_engine", default=None,
)
_ACTIVATION_SESSION = ContextVar("synaptic_activation_session", default=None)


@dataclass
class SynapticTrace:
    """An individual associative trace compiled into layer weights."""
    trace_id: str
    layer_idx: int
    target_module: str  # "mlp.down_proj" or "self_attn.o_proj"
    conductance: float  # Salience / reinforcement weight
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlasticWeightHook:
    """Apply compact associative factors gated by current-state stimulation.

    Positive cosine matches receive a quadratic gate. The combined additive
    effect is bounded per position by both the unmodified output norm and an
    absolute budget, independently of the number or magnitude of input traces.
    """
    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        module_name: str = "mlp.down_proj",
        max_norm_ratio: float = 0.05,  # Max additive change / unmodified output norm
        should_apply: Optional[Callable[[], bool]] = None,
        activation_session: Optional[Callable[[], Any]] = None,
    ):
        self.module = module
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.max_norm_ratio = float(max_norm_ratio)
        if not math.isfinite(self.max_norm_ratio) or self.max_norm_ratio < 0:
            raise ValueError("max_norm_ratio must be finite and nonnegative")
        self.should_apply = should_apply
        from invariants.activation_memory import ActivationMemory
        self.activation_memory = ActivationMemory()
        self.activation_session = activation_session
        self.traces: Dict[str, SynapticTrace] = {}
        
        # Associative memory buffers: keys (in_features) and values (out_features)
        self.keys: List[torch.Tensor] = []    # [d_in]
        self.values: List[torch.Tensor] = []  # [d_out]
        self.alphas: List[float] = []
        self.provenance_weights: List[float] = []
        self._lock = threading.Lock()
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register()

    def _register(self) -> None:
        def forward_hook(mod, inp, out):
            if self.should_apply is not None and not self.should_apply():
                return out
            session = self.activation_session() if self.activation_session else None
            raw = out
            if session is not None:
                self.activation_memory.set_capacity(session.capacity)
                out, stats = self.activation_memory.read(
                    out, hops=session.hops, threshold=session.threshold,
                    navigation_gain=session.navigation_gain,
                    max_norm_ratio=session.max_norm_ratio, absolute_budget=session.absolute_budget,
                    episode_binding=session.episode_binding, link_gain=session.link_gain, top_k=session.top_k,
                )
                session.stats['linked_matches'] += stats['linked_matches']
                session.stats['episode_locked_positions'] += stats['episode_locked_positions']
                session.stats['matched_memories'] += stats['matched_memories']
                session.stats['max_delta_norm'] = max(session.stats['max_delta_norm'], stats['max_delta_norm'])
                session.stats['max_hops'] = max(session.stats['max_hops'], stats['hops'])
                if session.write_enabled:
                    # Record experienced states before this bank's recalled correction.
                    session.observe(self.layer_idx, self.activation_memory, raw)
            with self._lock:
                if not self.keys:
                    return out
                
                # Input x can be [batch, seq_len, d_in] or [seq_len, d_in]
                x = inp[0] if isinstance(inp, tuple) else inp
                if not isinstance(x, torch.Tensor):
                    return out

                orig_dtype = out.dtype
                x_unit = _unit_features(x)

                delta_out = torch.zeros_like(out, dtype=torch.float32)
                for index, (key, val, alpha) in enumerate(zip(self.keys, self.values, self.alphas)):
                    k_dev = key.to(x.device)
                    v_dev = val.to(x.device)

                    if k_dev.shape[-1] != x_unit.shape[-1]:
                        if k_dev.shape[-1] < x_unit.shape[-1]:
                            k_dev = torch.nn.functional.pad(k_dev, (0, x_unit.shape[-1] - k_dev.shape[-1]))
                        else:
                            k_dev = k_dev[:x_unit.shape[-1]]

                    if v_dev.shape[-1] != out.shape[-1]:
                        if v_dev.shape[-1] < out.shape[-1]:
                            v_dev = torch.nn.functional.pad(v_dev, (0, out.shape[-1] - v_dev.shape[-1]))
                        else:
                            v_dev = v_dev[:out.shape[-1]]

                    # Magnitude alone never activates a memory; opposing keys
                    # do not turn its content into a reversed correction.
                    match = torch.matmul(x_unit, _unit_features(k_dev)).clamp(0, 1)
                    gate = match.square()
                    strength = self.provenance_weights[index] if index < len(self.provenance_weights) else 1.0
                    delta_out += (float(alpha) * strength * gate).unsqueeze(-1) * _unit_features(v_dev)

                original_norm = out.float().norm(dim=-1, keepdim=True)
                budget = (self.max_norm_ratio * original_norm).clamp(max=0.25)
                delta_norm = delta_out.norm(dim=-1, keepdim=True)
                delta_out *= (budget / delta_norm.clamp_min(1e-30)).clamp(max=1)

                return (out + delta_out.to(orig_dtype))

        self.hook_handle = self.module.register_forward_hook(forward_hook)

    def add_associative_trace(
        self,
        trace_id: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        alpha: float = 0.05,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an associative memory trace: when input aligns with key, output is biased by value."""
        from invariants.memory_attention import provenance_weight
        with self._lock:
            k = key_vector.detach().cpu().float().reshape(-1)
            v = value_vector.detach().cpu().float().reshape(-1)

            if not torch.isfinite(k).all() or not torch.isfinite(v).all() or not math.isfinite(float(alpha)):
                raise ValueError("Synaptic factors and strength must be finite")
            k, v = _unit_features(k), _unit_features(v)

            # Clamp alpha to prevent unbounded distortion
            bounded_alpha = max(-0.25, min(0.25, float(alpha)))

            self.keys.append(k)
            self.values.append(v)
            self.alphas.append(bounded_alpha)
            self.provenance_weights.append(provenance_weight(metadata or {}))

            self.traces[trace_id] = SynapticTrace(
                trace_id=trace_id,
                layer_idx=self.layer_idx,
                target_module=self.module_name,
                conductance=abs(bounded_alpha),
                created_at=float(torch.empty(1).uniform_().item()),
                metadata=deepcopy(metadata or {}),
            )

    def decay(self, rate: float = 0.05) -> None:
        """Passive synaptic decay: gradually relaxes memory deltas toward base state."""
        with self._lock:
            retained_keys, retained_vals, retained_alphas, retained_weights = [], [], [], []
            for index, (k, v, a) in enumerate(zip(self.keys, self.values, self.alphas)):
                new_a = a * (1.0 - rate)
                if abs(new_a) >= 1e-4:
                    retained_keys.append(k)
                    retained_vals.append(v)
                    retained_alphas.append(new_a)
                    retained_weights.append(self.provenance_weights[index] if index < len(self.provenance_weights) else 1.0)
            self.keys = retained_keys
            self.values = retained_vals
            self.alphas = retained_alphas
            self.provenance_weights = retained_weights

    def clear(self) -> None:
        with self._lock:
            self.keys.clear()
            self.values.clear()
            self.alphas.clear()
            self.provenance_weights.clear()
            self.traces.clear()

    def remove(self) -> None:
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class SynapticMemoryEngine:
    """Master orchestrator for integrating episodic memory into model weights.
    
    Acts as the neocortical target of the Default Mode Network (DMN), converting
    consolidated activation trajectories and epistemic friction boundaries into
    direct synaptic weight updates across transformer layers.
    """
    _INSTANCE: Optional[SynapticMemoryEngine] = None

    @classmethod
    def get_instance(cls, model: Any = None) -> SynapticMemoryEngine:
        selected = _ACTIVE_ENGINE.get()
        if selected is not None:
            if model is not None and selected.model is None:
                selected.attach_model(model)
            return selected
        if cls._INSTANCE is None:
            cls._INSTANCE = SynapticMemoryEngine(model=model)
        elif model is not None and cls._INSTANCE.model is None:
            cls._INSTANCE.attach_model(model)
        return cls._INSTANCE

    def __init__(
        self,
        model: Any = None,
        target_layers: Optional[List[int]] = None,
        target_module: str = "mlp.down_proj",
        persistence_path: Optional[Path] = None,
        learnable: bool = True,
    ):
        self.model = model
        self.learnable = bool(learnable)
        # Target central reasoning band by default (Layer 14-22 for 32-layer LLMs)
        self.target_layers = target_layers or [14, 16, 18, 20]
        self.target_module_name = target_module
        self.hooks: Dict[int, PlasticWeightHook] = {}
        self.persistence_path = Path(persistence_path or "invariants/out/synaptic_weights.pt")
        self.consolidated_trace_count = 0
        self.last_activation_stats = {}
        self._span_rotations = {}
        self._lock = threading.Lock()

        if model is not None:
            self.attach_model(model)

    @property
    def enabled(self) -> bool:
        """Whether this context applies this engine's learned delta buffers."""
        if not self.learnable:
            return False
        selected = _ACTIVE_ENGINE.get()
        return selected is self if selected is not None else type(self)._INSTANCE is self

    @contextmanager
    def activate(self):
        """Select this overlay temporarily, including for singleton callers."""
        token = _ACTIVE_ENGINE.set(self)
        try:
            yield self
        finally:
            _ACTIVE_ENGINE.reset(token)

    @classmethod
    def current_for_model(cls, model):
        """Find an existing owner; do not create hooks during generation setup."""
        owner = _ACTIVE_ENGINE.get() or cls._INSTANCE
        if owner is None:
            return None
        same_model = owner.model is model or getattr(owner.model, 'model', owner.model) is getattr(model, 'model', model)
        return owner if same_model and owner.enabled else None

    @contextmanager
    def activation_generation(self, **settings):
        from invariants.activation_memory import ActivationSession
        from invariants.memory_attention import is_auxiliary_encoding
        if not self.enabled or is_auxiliary_encoding():
            yield None
            return
        session = ActivationSession(**settings)
        token = _ACTIVATION_SESSION.set((self, session))
        try:
            yield session
        finally:
            self.last_activation_stats = dict(session.stats)
            _ACTIVATION_SESSION.reset(token)

    def _activation_session(self):
        from invariants.memory_attention import is_auxiliary_encoding
        active = _ACTIVATION_SESSION.get()
        return active[1] if active and active[0] is self and active[1].running and not is_auxiliary_encoding() else None

    def set_memory_strength(self, memory_id, strength):
        """Change one activation association, including setting it to zero."""
        if not self.learnable:
            raise ValueError("Base-only owner cannot change memories")
        for hook in self.hooks.values():
            if memory_id in hook.activation_memory.records:
                hook.activation_memory.set_strength(memory_id, strength)
                return
        raise KeyError(memory_id)

    def attach_model(self, model: Any) -> None:
        """Attach synaptic hooks to transformer layers of the given model."""
        self.detach()
        self.model = model
        raw_model = getattr(model, "model", model)
        layers = None
        if hasattr(raw_model, "layers"):
            layers = raw_model.layers
        elif hasattr(raw_model, "model") and hasattr(raw_model.model, "layers"):
            layers = raw_model.model.layers

        if layers is None:
            return

        num_layers = len(layers)
        for l_idx in self.target_layers:
            if l_idx >= num_layers:
                continue
            layer = layers[l_idx]
            # Locate target sub-module (e.g. mlp.down_proj)
            submod = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                submod = layer.mlp.down_proj
            elif hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                submod = layer.self_attn.o_proj

            if submod is not None:
                self.hooks[l_idx] = PlasticWeightHook(
                    module=submod,
                    layer_idx=l_idx,
                    module_name=self.target_module_name,
                    should_apply=lambda owner=self: owner.enabled,
                    activation_session=self._activation_session,
                )

    def detach(self) -> None:
        """Remove all active synaptic hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

    def consolidate_associative_memory(
        self,
        key_activation: torch.Tensor,
        target_shift: torch.Tensor,
        layer_idx: Optional[int] = None,
        learning_rate: float = 0.05,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Directly write an associative memory into the layer's synaptic weights.
        
        When input hidden state aligns with key_activation, the layer weight output
        inherently produces target_shift.
        """
        if not self.learnable:
            return ""
        with self._lock:
            self.consolidated_trace_count += 1
            t_id = trace_id or f"SYN_{self.consolidated_trace_count:05d}"
            target_l = layer_idx if (layer_idx is not None and layer_idx in self.hooks) else (self.target_layers[0] if self.target_layers else 16)

            if target_l in self.hooks:
                self.hooks[target_l].add_associative_trace(
                    trace_id=t_id,
                    key_vector=key_activation,
                    value_vector=target_shift,
                    alpha=learning_rate,
                    metadata=metadata,
                )
            return t_id

    def consolidate_from_dmn_episode(self, episode: Dict[str, Any], reference_state: Optional[torch.Tensor] = None) -> List[str]:
        """Convert a Default Mode Network consolidated episode into synaptic weight updates.
        
        Uses trajectory direction and mean friction to determine synaptic plasticity.
        """
        if not self.learnable:
            return []
        created_traces = []
        friction = float(episode.get("mean_friction", 0.01))
        # Plasticity is proportional to epistemic friction (higher surprise/friction = stronger imprint)
        learning_rate = max(0.01, min(0.12, friction * 2.5))
        
        tokens = episode.get("tokens", [])
        tok_label = "_".join(tokens[:2]) if tokens else "episode"

        # If reference hidden state is available, associate it with trajectory direction
        if reference_state is not None:
            k = reference_state.squeeze().float()
            # If j_space or direction is available, derive target shift
            v = torch.randn_like(k)  # Default fallback orthogonal basis
            for l_idx in self.hooks:
                t_id = self.consolidate_associative_memory(
                    key_activation=k,
                    target_shift=v,
                    layer_idx=l_idx,
                    learning_rate=learning_rate,
                    trace_id=f"EP_{episode.get('episode_id', '0')}_L{l_idx}",
                    metadata={"source": "DMN_consolidation", "tokens": tok_label, "friction": friction},
                )
                created_traces.append(t_id)

        return created_traces

    def integrate_text_memory(
        self,
        text: str,
        learning_rate: float = 0.04,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        provenance: Optional[Dict[str, Any]] = None,
        stimulation: Any = None,
        max_spans: int = 4,
    ) -> List[str]:
        """Associate bounded spans across the source with the owner's state.

        Explicit provenance strengths influence selection and trace priority;
        their source labels never imply truth. A tensor stimulation supplies an
        actual state for measuring semantic match. Span-score maps/callbacks can
        provide already measured stimulation without encoding the whole source.
        Source offsets and full origin metadata accompany every compact trace.
        """
        if not self.learnable or not text or not self.hooks or self.model is None:
            return []
        try:
            from invariants.engine import _inputs, _hidden_states
            from invariants.memory_attention import text_spans, select_spans, auxiliary_encoding
            origin = deepcopy(provenance if provenance is not None else (metadata or {}))
            source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rotation = self._span_rotations.get(source_hash, 0)
            spans = text_spans(text, span_chars=512)
            state = stimulation if torch.is_tensor(stimulation) else None
            span_scores = None if state is not None else stimulation
            if isinstance(stimulation, dict) and "state" in stimulation:
                state = stimulation.get("state")
                span_scores = stimulation.get("scores")
            selected = select_spans(spans, max_spans=max_spans, stimulation=span_scores,
                                    provenance=origin, rotation=rotation)
            if not selected:
                return []
            self._span_rotations[source_hash] = rotation + 1
            if len(self._span_rotations) > 128:
                self._span_rotations.pop(next(iter(self._span_rotations)))
            measured = []
            with torch.no_grad():
                for span in selected:
                    try:
                        observed = {}
                        handles = []

                        def capture(_module, inputs, output, index):
                            value = inputs[0] if isinstance(inputs, tuple) and inputs else None
                            if not torch.is_tensor(value) or not torch.is_tensor(output):
                                return
                            key = value.reshape(-1, value.shape[-1])[-1].detach().float().cpu().clone()
                            states = output.reshape(-1, output.shape[-1])
                            shift = states[-1] - states[0] if states.shape[0] > 1 else states[-1]
                            observed[index] = (key, shift.detach().float().cpu().clone())

                        try:
                            for layer, hook in self.hooks.items():
                                handles.append(hook.module.register_forward_hook(
                                    lambda module, inputs, output, index=layer: capture(module, inputs, output, index),
                                ))
                            with auxiliary_encoding():
                                ids = _inputs(self.model, span["text"], pre_formatted=True)
                                hs = _hidden_states(self.model, ids["input_ids"], ids.get("attention_mask"))
                        finally:
                            for handle in handles:
                                handle.remove()
                        if hs.ndim != 3 or not torch.isfinite(hs).all():
                            continue
                        state_match = 1.0
                        if state is not None:
                            if not torch.is_tensor(state) or not state.numel():
                                continue
                            actual = state.detach().float().reshape(-1, state.shape[-1])[-1]
                            represented = hs[-1, -1, :].detach().float()
                            if actual.shape != represented.shape or not torch.isfinite(actual).all():
                                continue
                            state_match = max(0.0, float(torch.dot(
                                _unit_features(actual.cpu()), _unit_features(represented.cpu()),
                            ))) ** 2
                        weight = max(0.0, float(span.get("weight", 1.0))) * state_match
                        if math.isfinite(weight) and weight > 0:
                            measured.append((span, hs, weight, state_match, observed))
                    except Exception:
                        continue
                total_weight = sum(item[2] for item in measured)
                if not total_weight:
                    return []
                created = []
                for span, hs, weight, state_match, observed in measured:
                    trace_metadata = deepcopy(metadata or {})
                    trace_metadata.update({
                        "provenance": deepcopy(origin), "source_sha256": source_hash,
                        "source_chars": len(text), "source_span_count": len(spans),
                        "span_index": span["index"], "span_start": span["start"], "span_end": span["end"],
                        "span_weight": weight, "provenance_weight": span.get("provenance_weight", 1.0),
                        "stimulation": span.get("stimulation", 1.0), "state_match": state_match,
                        "source_pass": rotation, "text_sample": span["text"][:100],
                    })
                    for l_idx in self.hooks:
                        if l_idx < hs.shape[0]:
                            if l_idx in observed:
                                k, v = observed[l_idx]
                            else:
                                k = hs[l_idx, -1, :].squeeze().float()
                                v = (hs[l_idx, -1, :] - hs[l_idx, 0, :]).squeeze().float() if hs.shape[1] > 1 else k
                            layer_metadata = deepcopy(trace_metadata)
                            layer_metadata["key_source"] = "module_input" if l_idx in observed else "layer_hidden"
                            t_id = self.consolidate_associative_memory(
                                key_activation=k, target_shift=v, layer_idx=l_idx,
                                learning_rate=float(learning_rate) * weight / total_weight,
                                metadata=layer_metadata,
                            )
                            created.append(t_id)
                return created
        except Exception:
            return []

    def apply_synaptic_decay(self, decay_rate: float = 0.05) -> None:
        """Simulate forgetting / homeostatic synaptic scaling."""
        if not self.learnable:
            return
        for hook in self.hooks.values():
            hook.decay(rate=decay_rate)

    def state_dict(self) -> Dict[str, Any]:
        """Copy the compact overlay only; checkpoint parameters stay shared."""
        layers = {}
        with self._lock:
            for l_idx, hook in self.hooks.items():
                with hook._lock:
                    layers[l_idx] = {
                        "keys": [k.detach().cpu().clone() for k in hook.keys],
                        "values": [v.detach().cpu().clone() for v in hook.values],
                        "alphas": list(hook.alphas),
                        "provenance_weights": list(hook.provenance_weights),
                        "traces": {tid: deepcopy(t.__dict__) for tid, t in hook.traces.items()},
                        "activation_memory": hook.activation_memory.state_dict(),
                    }
            result = {
                "version": 2,
                "consolidated_trace_count": self.consolidated_trace_count,
                "layers": layers,
            }
            if self._span_rotations:
                result["span_rotations"] = dict(self._span_rotations)
            return result

    def load_state_dict(self, data: Dict[str, Any]) -> bool:
        """Replace this overlay, accepting both versioned and legacy dumps."""
        if not self.learnable or not isinstance(data, dict):
            return False
        layers = data.get("layers") if "version" in data else data
        if not isinstance(layers, dict):
            return False
        prepared = {}
        try:
            for raw_idx, hook_data in layers.items():
                l_idx = int(raw_idx)
                if l_idx not in self.hooks:
                    continue
                keys = [k.detach().cpu().float().reshape(-1).clone() for k in hook_data.get("keys", [])]
                values = [v.detach().cpu().float().reshape(-1).clone() for v in hook_data.get("values", [])]
                alphas = [float(a) for a in hook_data.get("alphas", [])]
                if not (len(keys) == len(values) == len(alphas)):
                    return False
                if any(not torch.isfinite(value).all() for value in keys + values) or any(not math.isfinite(value) for value in alphas):
                    return False
                alphas = [max(-0.25, min(0.25, value)) for value in alphas]
                traces = {
                    tid: SynapticTrace(**deepcopy(trace))
                    for tid, trace in hook_data.get("traces", {}).items()
                }
                weights = [float(value) for value in hook_data.get("provenance_weights", [1.0] * len(keys))]
                if len(weights) != len(keys) or any(not math.isfinite(value) or value < 0 for value in weights):
                    return False
                weights = [min(4.0, value) for value in weights]
                from invariants.activation_memory import ActivationMemory
                activation = ActivationMemory.from_state_dict(hook_data['activation_memory']) if 'activation_memory' in hook_data else ActivationMemory()
                expected_width = getattr(self.hooks[l_idx].module, 'out_features', None)
                if expected_width is not None and any(record['key'].numel() != expected_width for record in activation.records.values()):
                    return False
                prepared[l_idx] = (keys, values, alphas, traces, weights, activation)
            count = int(data.get("consolidated_trace_count", 0)) if "version" in data else 0
            if count < 0:
                return False
            if "version" not in data:
                count = max((len(item[0]) for item in prepared.values()), default=0)
            rotations = {str(key): max(0, int(value)) for key, value in data.get("span_rotations", {}).items()}
        except (AttributeError, TypeError, ValueError, KeyError, RuntimeError):
            return False
        with self._lock:
            for l_idx, hook in self.hooks.items():
                from invariants.activation_memory import ActivationMemory
                keys, values, alphas, traces, weights, activation = prepared.get(l_idx, ([], [], [], {}, [], ActivationMemory()))
                with hook._lock:
                    hook.keys, hook.values, hook.alphas, hook.traces = keys, values, alphas, traces
                    hook.provenance_weights = weights
                    hook.activation_memory = activation
            self.consolidated_trace_count = count
            self._span_rotations = rotations
        return True

    def save_synaptic_weights(self, path: Optional[Path] = None) -> None:
        """Atomically persist this owner's compact deltas and trace metadata."""
        if not self.learnable:
            return
        target = Path(path or self.persistence_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False) as fh:
            temporary = Path(fh.name)
        try:
            torch.save(self.state_dict(), temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_synaptic_weights(self, path: Optional[Path] = None) -> bool:
        """Load persisted synaptic weight deltas into active hooks."""
        target = Path(path or self.persistence_path)
        if not self.learnable or not target.is_file():
            return False
        try:
            data = torch.load(target, map_location="cpu", weights_only=True)
            return self.load_state_dict(data)
        except Exception:
            return False

    def telemetry(self) -> Dict[str, Any]:
        """Return diagnostic report of active synaptic modifications."""
        total_traces = sum(len(h.traces) for h in self.hooks.values())
        active_layers = list(self.hooks.keys())
        return {
            "enabled": self.enabled,
            "learnable": self.learnable,
            "active_plastic_layers": active_layers,
            "total_synaptic_traces": total_traces,
            "consolidated_trace_count": self.consolidated_trace_count,
            "per_layer_traces": {l: len(h.traces) for l, h in self.hooks.items()},
            "activation_memories": {l: len(h.activation_memory.records) for l, h in self.hooks.items()},
        }
