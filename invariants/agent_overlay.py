"""Small, independently persisted agent changes on one shared checkpoint."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import torch

from invariants.config import AgenticConfig
from invariants.cognitive_cache import CognitiveCache
from invariants.lookahead_predictor import LookaheadApparatus
from invariants.memory_engine import MemoryEngine
from invariants.self_concept_controller import SelfConceptController
from invariants.steer_map_store import SteerMapStore
from invariants.synaptic_memory import SynapticMemoryEngine
from invariants.vector_travel import VectorTravelEngine


def agent_directory(root, model_name, d_model, name):
    """Model and exact-name keys prevent residual-width and slug collisions."""
    model_key = hashlib.sha256(f"{model_name}:{d_model}".encode()).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9_-]", "_", str(name).lower()).strip("_")[:40] or "agent"
    name_key = hashlib.sha256(str(name).encode()).hexdigest()[:12]
    return Path(root) / "invariants" / "out" / "agents" / model_key / f"{slug}_{name_key}"


class AgentOverlay:
    """Own learned sidecars; never clone or serialize model parameters."""
    def __init__(self, model, directory, *, tuner, lookahead_template=None,
                 target_layers=None, base_model_only=False):
        self.directory = Path(directory)
        self.base_model_only = bool(base_model_only)
        self.config = AgenticConfig()
        self.config.base_model_only = self.base_model_only
        self.config.use_expert_vectors = False
        self.config.allow_organic_fallback = False
        self.config.interactive_disambiguation = False
        self.config.cache_write_enabled = not self.base_model_only
        self.config.cache_verified_only = False
        self.config.cache_write_scope = "agent:" + self.directory.name
        self.config.expert_persistence_dir = self.directory / "experts"
        width = getattr(model, "d_model", getattr(getattr(model, "config", None), "hidden_size", 4096))
        self.travel = VectorTravelEngine(persistence_dir=self.directory, autoload=not self.base_model_only,
                                         d_model=width)
        self.cache = CognitiveCache(file=self.directory / "cognitive_cache.pt",
                                    autoload=not self.base_model_only, vector_travel=self.travel)
        self.config.cognitive_cache = self.cache
        self.synaptic = SynapticMemoryEngine(
            model, target_layers=target_layers,
            persistence_path=self.directory / "synaptic_weights.pt",
            learnable=not self.base_model_only,
        )
        self.lookahead = None
        self.lookahead_path = self.directory / "lookahead.pt"
        if not self.base_model_only:
            self.synaptic.load_synaptic_weights()
            if lookahead_template is not None:
                # The frozen predictor/vocabulary remain shared. Only evidence is new.
                self.lookahead = LookaheadApparatus(
                    predictor=lookahead_template.predictor, tuner=tuner,
                    embedding_matrix=lookahead_template.embedding_matrix,
                    d_model=lookahead_template.d_model, hidden_dim=lookahead_template.hidden_dim,
                    device=str(lookahead_template.device),
                )
                self.lookahead._custom_prior = lookahead_template._custom_prior
                if self.lookahead_path.exists():
                    self.lookahead.load_checkpoint(self.lookahead_path)
        self.memory = MemoryEngine(path=self.directory / "memory.jsonl", scope="agent:" + self.directory.name,
                                   load_existing=not self.base_model_only)
        self.steer_map = SteerMapStore(events_path=self.directory / "steer_map_events.jsonl",
                                      summary_path=self.directory / "steer_map_summary.json")
        self.self_concept = SelfConceptController(out_dir=self.directory)
        self.executive_owner = SimpleNamespace()
        self.shell_state = {}  # Shell-owned tool/clock/queue state, restored at each boundary.
        if not self.base_model_only:
            self._load_settings()

    @contextmanager
    def activate(self):
        with ExitStack() as stack:
            stack.enter_context(self.synaptic.activate())
            stack.enter_context(self.travel.activate())
            yield self

    def _load_settings(self):
        path = self.directory / "settings.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, value in payload.items():
                current = getattr(self.config, key, None)
                if isinstance(current, (bool, str, int, float)) and isinstance(value, type(current)):
                    setattr(self.config, key, value)
        path = self.directory / "corrections.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.config.organic_correction_vector = payload.get("organic_correction_vector")
            self.config.recent_corrections = payload.get("recent_corrections", [])
        # Runtime ownership flags cannot be enabled by an old settings file.
        self.config.base_model_only = False
        self.config.use_expert_vectors = False
        self.config.allow_organic_fallback = False

    def save(self):
        if self.base_model_only:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self.synaptic.save_synaptic_weights()
        self.cache.save()
        self.travel.chemicals.save()
        self.travel.pathways.save()
        if self.lookahead is not None and (self.lookahead.learning_updates
                                          or self.lookahead.composition != "linear"
                                          or self.lookahead_path.exists()):
            self.lookahead.save_checkpoint(self.lookahead_path)
        settings = {f.name: getattr(self.config, f.name) for f in fields(self.config)
                    if isinstance(getattr(self.config, f.name), (bool, str, int, float))}
        target = self.directory / "settings.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        temporary.replace(target)
        payload = {"organic_correction_vector": getattr(self.config, "organic_correction_vector", None),
                   "recent_corrections": self.config.recent_corrections}
        target = self.directory / "corrections.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    def close(self):
        try:
            networks = [getattr(self.executive_owner, name, None)
                        for name in ("_cen_network", "_dmn_network")]
            for network in networks:
                if network is not None:
                    network.stop()
            for network in networks:
                if isinstance(network, threading.Thread) and network is not threading.current_thread() and network.is_alive():
                    network.join(timeout=1.0)
            self.save()
        finally:
            self.synaptic.detach()
