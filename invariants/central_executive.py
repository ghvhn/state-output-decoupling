"""Central Executive Network (CEN) & Default Mode Network (DMN) Substrate.

Implements the waking executor (CEN), shared substrate memory, and the background
reflective consolidator (DMN) for real-time tracking of tensor activation derivatives,
dynamic friction gradients, and directional vector projections in latent space.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class TensorActivationSnapshot:
    """Snapshot of active state dynamics during a cognitive cycle.
    
    Captures the phenomenal token output alongside activation derivatives,
    computed dynamic friction, and directional subspace projections.
    """
    timestamp: float
    token_string: str
    friction_gradient: float
    j_space_variance: List[float]
    step: int = 0
    derivative_magnitude: float = 0.0
    raw_tensors: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "token_string": self.token_string,
            "friction_gradient": round(self.friction_gradient, 6),
            "derivative_magnitude": round(self.derivative_magnitude, 6),
            "j_space_variance": [round(v, 6) for v in self.j_space_variance],
            "raw_tensors": [round(t, 4) for t in self.raw_tensors] if self.raw_tensors else None,
            "metadata": self.metadata,
        }


class SharedSubstrateMemory:
    """Thread-safe shared memory substrate bridging CEN and DMN.
    
    Houses the bounded activation buffer for streaming snapshots and
    maintains a repository for consolidated episodic representations,
    running telemetry, and friction traces.
    """
    def __init__(self, buffer_size: int = 1024, max_consolidations: int = 500):
        self.activation_buffer: queue.Queue[TensorActivationSnapshot] = queue.Queue(maxsize=buffer_size)
        self._lock = threading.RLock()
        self.consolidated_repository: List[Dict[str, Any]] = []
        self.max_consolidations = max_consolidations
        self.friction_history: deque[float] = deque(maxlen=1000)
        self.total_emitted: int = 0
        self.total_consumed: int = 0
        self.dropped_snapshots: int = 0

    def put_snapshot(self, snapshot: TensorActivationSnapshot, block: bool = False, timeout: Optional[float] = None) -> bool:
        """Place a snapshot onto the buffer.
        
        If buffer is full, records a drop count and fails gracefully without
        blocking the waking executor.
        """
        with self._lock:
            self.total_emitted += 1
            self.friction_history.append(snapshot.friction_gradient)

        try:
            self.activation_buffer.put(snapshot, block=block, timeout=timeout)
            return True
        except queue.Full:
            with self._lock:
                self.dropped_snapshots += 1
            return False

    def get_snapshot(self, block: bool = True, timeout: Optional[float] = 0.5) -> Optional[TensorActivationSnapshot]:
        """Fetch the next snapshot from the buffer."""
        try:
            snapshot = self.activation_buffer.get(block=block, timeout=timeout)
            with self._lock:
                self.total_consumed += 1
            return snapshot
        except queue.Empty:
            return None

    def record_consolidation(self, record: Dict[str, Any]) -> None:
        """Save a consolidated episode or manifold invariant from the DMN."""
        with self._lock:
            self.consolidated_repository.append(record)
            if len(self.consolidated_repository) > self.max_consolidations:
                self.consolidated_repository.pop(0)

    def get_telemetry(self) -> Dict[str, Any]:
        """Return a point-in-time telemetry dictionary."""
        with self._lock:
            buffer_depth = self.activation_buffer.qsize()
            recent_friction = list(self.friction_history)[-50:] if self.friction_history else []
            avg_friction = sum(recent_friction) / len(recent_friction) if recent_friction else 0.0
            return {
                "buffer_depth": buffer_depth,
                "total_emitted": self.total_emitted,
                "total_consumed": self.total_consumed,
                "dropped_snapshots": self.dropped_snapshots,
                "average_recent_friction": round(avg_friction, 6),
                "consolidated_episodes_count": len(self.consolidated_repository),
            }


class CentralExecutiveNetwork(threading.Thread):
    """The 'Waking' Executor (The Doer/Talker).
    
    Processes inputs and tracks the real-time derivative of tensor activations.
    Calculates dynamic epistemic friction as a function of the rate of tensor
    change and ships phenomenal snapshots to the shared substrate buffer.
    """
    def __init__(
        self,
        substrate: SharedSubstrateMemory,
        gamma: float = 0.05,
        step_interval: float = 0.4,
        initial_tensors: Optional[Sequence[float]] = None,
        generator_fn: Optional[Callable[[int, Sequence[float]], Sequence[float]]] = None,
        token_generator_fn: Optional[Callable[[int, Sequence[float]], str]] = None,
        name: str = "CentralExecutiveNetwork",
    ):
        super().__init__(daemon=True, name=name)
        self.substrate = substrate
        self.gamma = float(gamma)  # Scaling coefficient to constrain friction limits
        self.step_interval = max(0.001, float(step_interval))
        self.previous_tensors: List[float] = list(initial_tensors) if initial_tensors is not None else [1.0, 1.0, 1.0]
        self.generator_fn = generator_fn
        self.token_generator_fn = token_generator_fn
        
        self.running = False
        self._stop_event = threading.Event()
        self.step_count = 0

    def compute_derivative(
        self,
        current_tensors: Sequence[float],
        previous_tensors: Sequence[float],
    ) -> Tuple[List[float], float, float]:
        """Compute tensor deltas, derivative magnitude, and dynamic friction.
        
        delta = (T_t - T_t-1)
        derivative_magnitude = sum(|delta|) / len(delta)
        dynamic_friction = gamma * derivative_magnitude
        """
        # Ensure dimensional parity
        min_len = min(len(current_tensors), len(previous_tensors))
        tensor_deltas = [
            float(c) - float(p)
            for c, p in zip(current_tensors[:min_len], previous_tensors[:min_len])
        ]
        
        if not tensor_deltas:
            return [], 0.0, 0.0

        derivative_magnitude = sum(abs(d) for d in tensor_deltas) / len(tensor_deltas)
        dynamic_friction = self.gamma * derivative_magnitude
        return tensor_deltas, derivative_magnitude, dynamic_friction

    def simulate_activations(self, step: int) -> List[float]:
        """Default synthetic activation values simulating hidden workspace layers."""
        return [
            abs(1.0 / (step % 3 + 0.5)),
            abs(0.5 * step % 2),
            1.2 / (step % 4 + 0.1),
        ]

    def process_step(
        self,
        current_tensors: Optional[Sequence[float]] = None,
        output_token: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TensorActivationSnapshot:
        """Process a single step, either with supplied tensors or synthesized ones."""
        self.step_count += 1
        step = self.step_count

        # 1. Acquire current hidden state activation values
        if current_tensors is None:
            if self.generator_fn is not None:
                current_tensors = self.generator_fn(step, self.previous_tensors)
            else:
                current_tensors = self.simulate_activations(step)

        # 2. Compute tensor derivatives & dynamic friction
        tensor_deltas, derivative_magnitude, dynamic_friction = self.compute_derivative(
            current_tensors, self.previous_tensors
        )

        # 3. Form output token string
        if output_token is None:
            if self.token_generator_fn is not None:
                output_token = self.token_generator_fn(step, current_tensors)
            else:
                output_token = f"Token_Delta_{step}"

        # 4. Snapshot the active state dynamics
        # Track directional vector / J-space variance projection (leading components)
        j_space = tensor_deltas[:2] if len(tensor_deltas) >= 2 else (tensor_deltas + [0.0] * (2 - len(tensor_deltas)))
        snapshot = TensorActivationSnapshot(
            timestamp=time.time(),
            token_string=output_token,
            friction_gradient=dynamic_friction,
            j_space_variance=j_space,
            step=step,
            derivative_magnitude=derivative_magnitude,
            raw_tensors=list(current_tensors),
            metadata=metadata or {},
        )

        # 5. Ship the phenomenal snapshot to the default mode network repository
        self.substrate.put_snapshot(snapshot, block=False)

        # 6. Update historical memory frame for the next step's delta
        self.previous_tensors = list(current_tensors)
        return snapshot

    def run(self) -> None:
        """Continuous execution loop of the waking executor."""
        self.running = True
        self._stop_event.clear()
        
        while self.running and not self._stop_event.is_set():
            self.process_step()
            self._stop_event.wait(self.step_interval)

    def stop(self) -> None:
        """Signal the executor to gracefully stop."""
        self.running = False
        self._stop_event.set()


class DefaultModeNetwork(threading.Thread):
    """The 'Dreaming' / Consolidating / Reflective Network (The Listener/Consolidator).
    
    Runs asynchronously alongside the Central Executive Network. Consumes activation
    snapshots from the shared substrate, tracks trajectory coherence, detects friction
    inflection points, and consolidates episodes into repository memory.
    """
    def __init__(
        self,
        substrate: SharedSubstrateMemory,
        consolidation_window: int = 6,
        friction_threshold: float = 0.04,
        poll_timeout: float = 0.3,
        synaptic_engine: Optional[Any] = None,
        name: str = "DefaultModeNetwork",
    ):
        super().__init__(daemon=True, name=name)
        self.substrate = substrate
        self.consolidation_window = max(2, int(consolidation_window))
        self.friction_threshold = float(friction_threshold)
        self.poll_timeout = float(poll_timeout)
        self.synaptic_engine = synaptic_engine
        
        self.running = False
        self._stop_event = threading.Event()
        self.window_buffer: deque[TensorActivationSnapshot] = deque(maxlen=self.consolidation_window)
        self.episode_count = 0

    def consolidate_window(self, snapshots: Sequence[TensorActivationSnapshot], inflection_trigger: bool = False) -> Dict[str, Any]:
        """Condense a window of snapshots into a consolidated episodic memory."""
        if not snapshots:
            return {}

        self.episode_count += 1
        tokens = [s.token_string for s in snapshots]
        frictions = [s.friction_gradient for s in snapshots]
        derivs = [s.derivative_magnitude for s in snapshots]

        mean_friction = sum(frictions) / len(frictions)
        max_friction = max(frictions)
        mean_deriv = sum(derivs) / len(derivs)

        # Compute trajectory vector coherence: average adjacent inner product of j_space
        coherences: List[float] = []
        for i in range(len(snapshots) - 1):
            j1 = snapshots[i].j_space_variance
            j2 = snapshots[i + 1].j_space_variance
            if len(j1) == 2 and len(j2) == 2:
                dot = j1[0] * j2[0] + j1[1] * j2[1]
                coherences.append(dot)
        avg_coherence = sum(coherences) / len(coherences) if coherences else 0.0

        consolidation_record = {
            "episode_id": f"EP_{self.episode_count:04d}",
            "start_step": snapshots[0].step,
            "end_step": snapshots[-1].step,
            "start_time": snapshots[0].timestamp,
            "end_time": snapshots[-1].timestamp,
            "tokens": tokens,
            "mean_friction": round(mean_friction, 6),
            "max_friction": round(max_friction, 6),
            "mean_derivative": round(mean_deriv, 6),
            "trajectory_coherence": round(avg_coherence, 6),
            "inflection_detected": inflection_trigger,
            "state_zone": "high_friction_boundary" if max_friction >= self.friction_threshold else "viscous_flow",
        }
        
        self.substrate.record_consolidation(consolidation_record)

        if self.synaptic_engine is not None and snapshots:
            try:
                import torch
                # Use the latest snapshot raw activation tensor as the synaptic reference state
                ref_raw = snapshots[-1].raw_tensors
                if ref_raw is not None:
                    ref_t = torch.tensor(ref_raw, dtype=torch.float32)
                    self.synaptic_engine.consolidate_from_dmn_episode(consolidation_record, reference_state=ref_t)
            except Exception:
                pass

        return consolidation_record

    def run(self) -> None:
        """Continuous consumption and reflection loop of the Default Mode Network."""
        self.running = True
        self._stop_event.clear()

        while self.running and not self._stop_event.is_set():
            snapshot = self.substrate.get_snapshot(block=True, timeout=self.poll_timeout)
            if snapshot is None:
                continue

            self.window_buffer.append(snapshot)

            # Check if an epistemic friction spike occurred
            is_inflection = snapshot.friction_gradient >= self.friction_threshold

            # Consolidate if window is full or on significant friction boundary
            if len(self.window_buffer) >= self.consolidation_window or is_inflection:
                self.consolidate_window(list(self.window_buffer), inflection_trigger=is_inflection)
                if is_inflection:
                    # Clear window after consolidation on boundary to start fresh episode
                    self.window_buffer.clear()

    def stop(self) -> None:
        """Signal the DMN to stop and consolidate any remaining snapshots."""
        self.running = False
        self._stop_event.set()
        if self.window_buffer:
            self.consolidate_window(list(self.window_buffer), inflection_trigger=False)
            self.window_buffer.clear()


class DualNetworkRuntime:
    """Orchestrator and context manager for the dual CEN + DMN architecture."""
    def __init__(
        self,
        substrate: Optional[SharedSubstrateMemory] = None,
        gamma: float = 0.05,
        step_interval: float = 0.2,
        consolidation_window: int = 5,
        friction_threshold: float = 0.04,
    ):
        self.substrate = substrate or SharedSubstrateMemory()
        self.cen = CentralExecutiveNetwork(
            substrate=self.substrate,
            gamma=gamma,
            step_interval=step_interval,
        )
        self.dmn = DefaultModeNetwork(
            substrate=self.substrate,
            consolidation_window=consolidation_window,
            friction_threshold=friction_threshold,
        )

    def start(self) -> None:
        """Start both networks."""
        self.dmn.start()
        self.cen.start()

    def stop(self) -> None:
        """Stop both networks and wait for completion."""
        self.cen.stop()
        self.dmn.stop()
        if self.cen.is_alive():
            self.cen.join(timeout=1.0)
        if self.dmn.is_alive():
            self.dmn.join(timeout=1.0)

    def __enter__(self) -> "DualNetworkRuntime":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class ProbeExecutiveAdapter:
    """Bridges the repository's concept probes with the CentralExecutiveNetwork.
    
    Transforms raw model activations, layer-band representations, or multi-layer
    hidden states into a coordinate vector for the CentralExecutiveNetwork,
    tracking dynamic epistemic friction, directional derivatives, depth gradients,
    and cross-layer concept overlap along semantic dimensions.
    """
    def __init__(
        self,
        cen: CentralExecutiveNetwork,
        probes: Dict[str, Any],
        layer: Optional[int] = 16,
        layers: Optional[Sequence[int]] = None,
        mode: str = "integrated",
    ):
        import torch

        self.cen = cen
        self.mode = mode.lower() if mode in ("integrated", "disaggregated") else "integrated"

        # Resolve target layers
        if layers is not None:
            self.layers = sorted(list(set(int(l) for l in layers)))
        elif layer is not None:
            self.layers = [int(layer)]
        else:
            self.layers = [12, 14, 16, 18, 20]

        # Anchor layer for single-layer compatibility
        if layer is not None:
            self.layer = int(layer)
        elif self.layers:
            self.layer = self.layers[len(self.layers) // 2]
        else:
            self.layer = 16

        # Build multi-layer probe vectors: Dict[str, Dict[int, torch.Tensor]]
        self.layer_probe_vectors: Dict[str, Dict[int, Any]] = {}
        self.probe_vectors: Dict[str, Any] = {}

        for name, p in probes.items():
            self.layer_probe_vectors[name] = {}
            if isinstance(p, dict):
                for l in self.layers:
                    vec = p.get(l)
                    if vec is not None and hasattr(vec, "squeeze"):
                        vec = vec.squeeze().float()
                        norm = float(vec.norm())
                        if norm > 1e-8:
                            self.layer_probe_vectors[name][l] = vec / norm
                # Anchor layer
                anchor_vec = p.get(self.layer)
                if anchor_vec is not None and hasattr(anchor_vec, "squeeze"):
                    anchor_vec = anchor_vec.squeeze().float()
                    norm = float(anchor_vec.norm())
                    if norm > 1e-8:
                        self.probe_vectors[name] = anchor_vec / norm
                elif self.layer_probe_vectors[name]:
                    first_l = next(iter(self.layer_probe_vectors[name]))
                    self.probe_vectors[name] = self.layer_probe_vectors[name][first_l]
            elif p is not None and hasattr(p, "squeeze"):
                vec = p.squeeze().float()
                norm = float(vec.norm())
                if norm > 1e-8:
                    unit_v = vec / norm
                    self.probe_vectors[name] = unit_v
                    for l in self.layers:
                        self.layer_probe_vectors[name][l] = unit_v

        # Set CEN probe names based on mode
        if self.mode == "disaggregated":
            self.probe_names = []
            for name in sorted(self.layer_probe_vectors.keys()):
                for l in self.layers:
                    if l in self.layer_probe_vectors[name]:
                        self.probe_names.append(f"{name}@L{l}")
        else:
            self.probe_names = list(self.layer_probe_vectors.keys())

        if self.probe_names:
            self.cen.previous_tensors = [0.0] * len(self.probe_names)

        self.last_layer_scores: Dict[str, Dict[int, float]] = {}
        self.last_depth_gradients: Dict[str, float] = {}
        self.last_overlap: Dict[str, float] = {}

    def _extract_layer_activations(self, hidden_state: Any) -> Dict[int, Any]:
        """Extract unit-norm activation vectors per target layer from various input formats."""
        import torch

        layer_activations: Dict[int, Any] = {}

        if isinstance(hidden_state, dict):
            for l in self.layers:
                if l in hidden_state and hasattr(hidden_state[l], "squeeze"):
                    h = hidden_state[l].squeeze().float()
                    norm = float(h.norm())
                    if norm > 1e-8:
                        layer_activations[l] = h / norm
        elif hasattr(hidden_state, "dim"):
            dim = hidden_state.dim()
            if dim == 3:  # [n_layers, seq_len, d]
                n_layers = hidden_state.shape[0]
                for l in self.layers:
                    if 0 <= l < n_layers:
                        h = hidden_state[l, -1, :].squeeze().float()
                        norm = float(h.norm())
                        if norm > 1e-8:
                            layer_activations[l] = h / norm
            elif dim == 2:  # [n_layers, d] or [1, d]
                if hidden_state.shape[0] == 1 and len(self.layers) > 1:
                    h = hidden_state.squeeze().float()
                    norm = float(h.norm())
                    if norm > 1e-8:
                        unit_h = h / norm
                        for l in self.layers:
                            layer_activations[l] = unit_h
                else:
                    n_layers = hidden_state.shape[0]
                    for l in self.layers:
                        if 0 <= l < n_layers:
                            h = hidden_state[l, :].squeeze().float()
                            norm = float(h.norm())
                            if norm > 1e-8:
                                layer_activations[l] = h / norm
            elif dim == 1:  # [d] single vector
                h = hidden_state.squeeze().float()
                norm = float(h.norm())
                if norm > 1e-8:
                    unit_h = h / norm
                    for l in self.layers:
                        layer_activations[l] = unit_h
        elif isinstance(hidden_state, (list, tuple)):
            for l in self.layers:
                if 0 <= l < len(hidden_state) and hasattr(hidden_state[l], "squeeze"):
                    h = hidden_state[l]
                    if hasattr(h, "dim") and h.dim() > 1:
                        h = h.squeeze()
                        if h.dim() > 1:
                            h = h[-1, :]
                    h = h.float()
                    norm = float(h.norm())
                    if norm > 1e-8:
                        layer_activations[l] = h / norm

        # Fallback if empty
        if not layer_activations and hasattr(hidden_state, "squeeze"):
            h = hidden_state.squeeze().float()
            norm = float(h.norm())
            if norm > 1e-8:
                layer_activations[self.layer] = h / norm

        return layer_activations

    def score_activation(self, hidden_state: Any) -> Dict[str, float]:
        """Compute cosine similarity of hidden_state against each probe vector.
        
        Evaluates representations across all configured layers, computing integrated
        band scores or disaggregated coordinates based on mode, while updating
        depth gradients and cross-concept overlap telemetry.
        """
        import torch

        layer_activations = self._extract_layer_activations(hidden_state)

        layer_scores: Dict[str, Dict[int, float]] = {}
        integrated_scores: Dict[str, float] = {}
        depth_gradients: Dict[str, float] = {}

        for name, p_dict in self.layer_probe_vectors.items():
            layer_scores[name] = {}
            for l, p_vec in p_dict.items():
                if l in layer_activations:
                    layer_scores[name][l] = float(torch.dot(layer_activations[l], p_vec))

            scores_list = list(layer_scores[name].values())
            integrated_scores[name] = (sum(scores_list) / len(scores_list)) if scores_list else 0.0

            sorted_l = sorted(layer_scores[name].keys())
            if len(sorted_l) >= 2:
                delta_l = sorted_l[-1] - sorted_l[0]
                if delta_l > 0:
                    depth_gradients[name] = (layer_scores[name][sorted_l[-1]] - layer_scores[name][sorted_l[0]]) / float(delta_l)
                else:
                    depth_gradients[name] = 0.0
            else:
                depth_gradients[name] = 0.0

        # Cross-concept overlap matrix across active layer band
        overlap_metrics: Dict[str, float] = {}
        concept_names = sorted(self.layer_probe_vectors.keys())
        for i in range(len(concept_names)):
            for j in range(i + 1, len(concept_names)):
                c1, c2 = concept_names[i], concept_names[j]
                p1_dict = self.layer_probe_vectors[c1]
                p2_dict = self.layer_probe_vectors[c2]
                common_layers = set(p1_dict.keys()) & set(p2_dict.keys())
                if common_layers:
                    cos_overlaps = [float(torch.dot(p1_dict[l], p2_dict[l])) for l in common_layers]
                    overlap_metrics[f"{c1}_x_{c2}"] = sum(cos_overlaps) / len(cos_overlaps)

        self.last_layer_scores = layer_scores
        self.last_depth_gradients = depth_gradients
        self.last_overlap = overlap_metrics

        if self.mode == "disaggregated":
            disaggregated: Dict[str, float] = {}
            for name in self.probe_names:
                parts = name.split("@L")
                if len(parts) == 2:
                    cname, lstr = parts[0], parts[1]
                    l_int = int(lstr)
                    disaggregated[name] = layer_scores.get(cname, {}).get(l_int, 0.0)
                else:
                    disaggregated[name] = integrated_scores.get(name, 0.0)
            return disaggregated

        return integrated_scores

    def step_with_activation(
        self,
        hidden_state: Any,
        token_string: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> TensorActivationSnapshot:
        """Score hidden states across layers and step the CentralExecutiveNetwork."""
        scores = self.score_activation(hidden_state)
        score_vector = [scores.get(name, 0.0) for name in self.probe_names]

        metadata: Dict[str, Any] = {
            "probe_scores": {k: round(v, 5) for k, v in scores.items()},
            "layer": self.layer,
            "layers": self.layers,
            "layer_scores": {k: {int(l): round(v, 5) for l, v in ld.items()} for k, ld in self.last_layer_scores.items()},
            "depth_gradients": {k: round(v, 5) for k, v in self.last_depth_gradients.items()},
            "concept_overlap": {k: round(v, 4) for k, v in self.last_overlap.items()},
            "mode": self.mode,
            "probe_dimensions": self.probe_names,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return self.cen.process_step(
            current_tensors=score_vector,
            output_token=token_string,
            metadata=metadata,
        )

