import torch
from pathlib import Path
import os
import re
import torch.nn.functional as F

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

CACHE_FILE = Path(
    os.environ.get(
        "COGNITIVE_CACHE_FILE",
        str(Path(__file__).parent / "data" / "cognitive_cache.pt"),
    )
)


def model_cache_file(model_name, d_model=None):
    """Per-model cache path.

    The shipped cache is calibrated for DEFAULT_MODEL. A swapped-in model has a
    different residual geometry, so it must build and read its OWN cache rather
    than be handed the (dimensionally inert) default file. The default model
    keeps the original path for backward compatibility.
    """
    base = CACHE_FILE
    if not model_name or model_name == DEFAULT_MODEL:
        return base
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name)).strip("_")
    suffix = f"__{slug}"
    if d_model:
        suffix += f"_d{d_model}"
    return base.with_name(base.stem + suffix + base.suffix)


class CognitiveCache:
    def __init__(self, threshold=0.995, max_memories=512):
        self.threshold = threshold
        self.max_memories = max_memories
        self.memory = [] # List of dicts: {trigger, delta, metadata}
        self.file = CACHE_FILE
        self.load()

    def use_file(self, path):
        """Re-point this cache at a model-specific file and reload its memories."""
        self.file = Path(path)
        self.load()
        return self.file

    @staticmethod
    def _last_token_vector(tensor):
        t = tensor.detach().cpu().float()
        if t.ndim == 0:
            raise ValueError("Cannot cache a scalar state.")
        if t.ndim == 1:
            return t.contiguous()
        return t.reshape(-1, t.shape[-1])[-1].contiguous()

    def _coerce_entry(self, item):
        if isinstance(item, dict):
            trigger = item.get("trigger")
            delta = item.get("delta", item.get("learned_vector"))
            metadata = item.get("metadata", {})
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            trigger, delta = item[:2]
            metadata = {"format": "legacy_tuple"}
        else:
            return None

        if trigger is None or delta is None:
            return None

        try:
            return {
                "trigger": self._last_token_vector(trigger),
                "delta": self._last_token_vector(delta),
                "metadata": metadata,
            }
        except Exception:
            return None
        
    def load(self):
        if self.file.exists():
            try:
                raw = torch.load(self.file, map_location="cpu")
                if not isinstance(raw, list):
                    raw = []
                self.memory = []
                for item in raw:
                    entry = self._coerce_entry(item)
                    if entry is not None:
                        self.memory.append(entry)
                print(f"[Cognitive Cache] Loaded {len(self.memory)} episodic memories from {self.file.name}.")
            except Exception as e:
                print(f"[Cognitive Cache] Error loading cache: {e}")
                self.memory = []
        else:
            # New (e.g. model-specific) cache; ensure data dir exists.
            self.memory = []
            os.makedirs(self.file.parent, exist_ok=True)

    def save(self):
        torch.save(self.memory, self.file)
        
    def store(self, trigger_state, learned_vector, metadata=None):
        """
        Stores the newly minted layer to memory.
        """
        trigger = self._last_token_vector(trigger_state)
        vec = self._last_token_vector(learned_vector)

        if trigger.numel() != vec.numel():
            print("    [Cognitive Cache] Skipping store: trigger/vector shape mismatch.")
            return

        self.memory.append({
            "trigger": trigger,
            "delta": vec,
            "metadata": metadata or {},
        })
        if len(self.memory) > self.max_memories:
            self.memory = self.memory[-self.max_memories:]
        self.save()
        scope = (metadata or {}).get("cache_write_scope") if isinstance(metadata, dict) else None
        if scope and scope != "default":
            label = f"{scope} memory"
        elif isinstance(metadata, dict) and metadata.get("promoted_by") == "humble_verifier":
            label = "Verified lesson"
        else:
            label = "Epiphany"
        print(f"    [Cognitive Cache] {label} stored! Total memories: {len(self.memory)}")
        
    def retrieve(
        self,
        current_state,
        verified_only=False,
        ignore_oracle_cache=False,
        excluded_question_key=None,
        excluded_oracle_question_key=None,
    ):
        """
        Computes cosine similarity between current state and all saved triggers.
        Returns the learned vector if similarity > threshold.
        """
        if len(self.memory) == 0:
            return None
            
        curr = self._last_token_vector(current_state)
        
        best_sim = -1.0
        best_vec = None
        best_meta = None
        
        for item in self.memory:
            entry = self._coerce_entry(item)
            if entry is None:
                continue
            metadata = entry.get("metadata", {})
            if verified_only and (
                not isinstance(metadata, dict)
                or metadata.get("promoted_by") != "humble_verifier"
            ):
                continue
            if ignore_oracle_cache and isinstance(metadata, dict) and metadata.get("tag") == "oracle_repair":
                continue
            if (
                excluded_question_key is not None
                and isinstance(metadata, dict)
                and metadata.get("question_key") == excluded_question_key
            ):
                continue
            if (
                excluded_oracle_question_key is not None
                and isinstance(metadata, dict)
                and metadata.get("tag") == "oracle_repair"
                and metadata.get("question_key") == excluded_oracle_question_key
            ):
                continue
            trigger = entry["trigger"]
            vec = entry["delta"]
            if trigger.numel() != curr.numel() or vec.numel() != curr.numel():
                continue
            sim = F.cosine_similarity(curr.view(-1), trigger.view(-1), dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_vec = vec
                best_meta = metadata
                
        if best_sim > self.threshold:
            reason = best_meta.get("reason", "cached_delta") if isinstance(best_meta, dict) else "cached_delta"
            print(f"    [Cognitive Cache] HIT! Found matching cognitive state (Sim: {best_sim:.4f}, reason: {reason})")
            return best_vec.to(current_state.device).to(current_state.dtype)
            
        return None
