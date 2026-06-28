import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import os

@dataclass
class AgenticConfig:
    """Global configuration for the Agentic Engine."""
    
    # Feature Flags
    synthesis_enabled: bool = True
    use_expert_vectors: bool = True
    cache_enabled: bool = True
    cache_write_enabled: bool = False
    cache_verified_only: bool = True
    ignore_oracle_cache: bool = False
    exclude_same_question_oracle_cache: bool = False
    benchmark_question_key: Optional[str] = None
    interactive_disambiguation: bool = False
    defer_disambiguation: bool = False
    clarification_fallback: Optional[str] = (
        "Resolve the uncertainty internally: list the plausible interpretations, "
        "choose the one best supported by the original wording, and continue without external information."
    )
    clarifying_questions: list[Dict[str, Any]] = field(default_factory=list)
    chatty_log: bool = False
    force_synthesis: bool = False
    provide_time_context: bool = True
    time_awareness_gated_urgency: bool = True
    time_awareness_threshold: float = 0.45
    urgency_max_coefficient: float = 0.8
    continuous_urgency_injection: bool = False
    
    # Hyperparameters
    max_loops: int = 3
    entropy_threshold: float = 2.0
    alpha: float = 15.0  # Steer strength
    epsilon: float = 0.05
    max_synthesis_events: int = 1
    max_routing_events: int = 4
    
    # High-Level Solver Logic
    max_rounds: int = 5
    required_agreement: int = 3
    max_elapsed_sec: Optional[float] = None
    stop_on_critical_urgency: bool = True
    
    # Attempt Budgets
    max_new_tokens: int = 220
    repair_token_multiplier: float = 2.0
    max_attempt_tokens: Optional[int] = None
    
    @classmethod
    def from_preset(cls, preset_name: str) -> "AgenticConfig":
        if preset_name == "fast":
            return cls(
                max_loops=1,
                synthesis_enabled=False,
                cache_enabled=True,
                max_routing_events=2,
            )
        elif preset_name == "thorough":
            return cls(
                max_loops=3,
                synthesis_enabled=True,
                cache_enabled=True,
                max_routing_events=5,
                max_synthesis_events=3,
                entropy_threshold=1.5,
            )
        elif preset_name == "default":
            return cls()
        else:
            raise ValueError(f"Unknown preset: {preset_name}")


class VectorRegistry:
    """Auto-discovers and caches steering vectors."""
    def __init__(self):
        self._vecs: Dict[str, torch.Tensor] = {}
        self._special_vecs: Dict[str, torch.Tensor] = {}
        
    def get_vecs(self, M) -> Dict[str, torch.Tensor]:
        if self._vecs:
            return self._vecs
            
        try:
            from invariants.multi_domain_benchmark import DOMAINS
            from invariants.social_hunt import get_steer_vector
            for name, spec in DOMAINS.items():
                self._vecs[name] = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        except ImportError:
            pass
            
        model_dir = Path(__file__).parent / "models"
        if model_dir.exists():
            for f in model_dir.glob("*.pt"):
                name = f.stem
                if name not in self._vecs:
                    try:
                        self._vecs[name] = torch.load(f, map_location="cpu")
                        print(f"  [VectorRegistry] Discovered local expert vector: {name}")
                    except Exception as e:
                        print(f"  [VectorRegistry] Failed to load {f.name}: {e}")
                    
        return self._vecs

    def get_special_vec(self, name: str) -> Optional[torch.Tensor]:
        if name in self._special_vecs:
            return self._special_vecs[name]
        model_dir = Path(__file__).parent / "models"
        f = model_dir / f"{name}.pt"
        if f.exists():
            try:
                vec = torch.load(f, map_location="cpu")
                self._special_vecs[name] = vec
                print(f"  [VectorRegistry] Discovered special vector: {name}")
                return vec
            except Exception as e:
                print(f"  [VectorRegistry] Failed to load special vector {f.name}: {e}")
        return None

# Global singleton
_global_registry = VectorRegistry()
