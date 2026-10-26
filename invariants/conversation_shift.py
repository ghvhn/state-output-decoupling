"""
Conversation Shift Tracker & Shift-Aware Memory Modulation.

Tracks conversational trajectory, detects topic shifts (continuation, pivot,
hard jump, topic return), and modulates memory retrieval so that:
  1. In-topic continuations preserve working context without associative noise.
  2. Hard jumps attenuate stale context from the immediately abandoned topic.
  3. Topic returns and destination topics receive anchor boosts, pulling
     relevant memories regardless of recency penalties.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "the", "please", "just", "like",
}


def extract_keywords(text: str) -> Set[str]:
    """Extract informative lowercase keywords from text."""
    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def normalize_vector(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten and unit-normalize tensor on CPU float32."""
    v = tensor.detach().cpu().float().reshape(-1)
    norm = v.norm().clamp_min(1e-6)
    return v / norm


@dataclass
class TurnRecord:
    turn_index: int
    role: str
    text: str
    keywords: Set[str]
    vector: Optional[torch.Tensor] = None
    cluster_id: int = 0
    shift_delta: float = 0.0
    regime: str = "continuation"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TopicCluster:
    cluster_id: int
    label: str
    keywords: Set[str]
    centroid_vector: Optional[torch.Tensor] = None
    turn_indices: List[int] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationShiftTracker:
    """
    Tracks dialogue kinematic trajectory and detects conversational shifts.
    """

    def __init__(
        self,
        *,
        pivot_threshold: float = 0.50,
        jump_threshold: float = 0.85,
        return_threshold: float = 0.45,
        stale_attenuation: float = 0.10,
        anchor_boost: float = 1.20,
        window_size: int = 4,
    ):
        self.pivot_threshold = float(pivot_threshold)
        self.jump_threshold = float(jump_threshold)
        self.return_threshold = float(return_threshold)
        self.stale_attenuation = float(stale_attenuation)
        self.anchor_boost = float(anchor_boost)
        self.window_size = max(1, int(window_size))

        self.turns: List[TurnRecord] = []
        self.clusters: List[TopicCluster] = []
        self.current_cluster_id: int = 0
        self.active_regime: str = "continuation"
        self.last_delta: float = 0.0
        self.last_alignment: float = 1.0
        self.last_event: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        """Clear conversation history and trajectory state."""
        self.turns.clear()
        self.clusters.clear()
        self.current_cluster_id = 0
        self.active_regime = "continuation"
        self.last_delta = 0.0
        self.last_alignment = 1.0
        self.last_event = None

    def _lexical_distance(self, set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a and not set_b:
            return 0.0
        if not set_a or not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        if intersection == 0:
            return 1.0
        base_sim = max(
            intersection / min(len(set_a), len(set_b)),
            intersection / math.sqrt(len(set_a) * len(set_b)),
        )
        # 2+ shared domain keywords in short conversational turns indicate strong topical continuity
        boosted_sim = min(1.0, base_sim + 0.22 * (intersection - 1))
        return max(0.0, min(1.0, 1.0 - boosted_sim))

    def _cluster_similarity(self, cluster: TopicCluster, kw: Set[str], vec: Optional[torch.Tensor]) -> float:
        """Compute similarity between a candidate turn and a topic cluster."""
        scores = []
        if vec is not None and cluster.centroid_vector is not None:
            if vec.numel() == cluster.centroid_vector.numel():
                cos_sim = float(F.cosine_similarity(vec.unsqueeze(0), cluster.centroid_vector.unsqueeze(0)).item())
                scores.append(max(0.0, cos_sim))
        if kw and cluster.keywords:
            lex_sim = 1.0 - self._lexical_distance(kw, cluster.keywords)
            scores.append(lex_sim)
        return max(scores) if scores else 0.0

    def observe_turn(
        self,
        text: str,
        *,
        vector: Optional[torch.Tensor] = None,
        role: str = "user",
    ) -> Dict[str, Any]:
        """
        Record a new conversational turn and compute shift kinematics.
        """
        turn_idx = len(self.turns)
        kw = extract_keywords(text)
        vec = normalize_vector(vector) if vector is not None else None

        if turn_idx == 0:
            cluster_label = " ".join(sorted(kw)[:4]) or "initial"
            cluster = TopicCluster(
                cluster_id=0,
                label=cluster_label,
                keywords=set(kw),
                centroid_vector=vec.clone() if vec is not None else None,
                turn_indices=[0],
            )
            self.clusters.append(cluster)
            self.current_cluster_id = 0
            self.active_regime = "continuation"
            self.last_delta = 0.0
            self.last_alignment = 1.0
            record = TurnRecord(
                turn_index=0,
                role=role,
                text=text,
                keywords=kw,
                vector=vec,
                cluster_id=0,
                shift_delta=0.0,
                regime="continuation",
            )
            self.turns.append(record)
            self.last_event = {
                "turn_index": 0,
                "role": role,
                "regime": "continuation",
                "delta": 0.0,
                "alignment": 1.0,
                "is_shift": False,
                "cluster_id": 0,
                "cluster_label": cluster_label,
            }
            return dict(self.last_event)

        # 1. Compare against the currently active cluster's recent turns
        curr_cluster = next((c for c in self.clusters if c.cluster_id == self.current_cluster_id), self.clusters[-1])
        active_turn_indices = curr_cluster.turn_indices[-self.window_size:]
        turn_map = {r.turn_index: r for r in self.turns}
        active_turns = [turn_map[i] for i in active_turn_indices if i in turn_map]
        if not active_turns:
            active_turns = self.turns[-self.window_size:]

        active_kw = set()
        for r in active_turns:
            active_kw |= r.keywords

        delta_candidates = []
        if vec is not None:
            # Check cosine against last vector in active cluster
            last_vec = next((r.vector for r in reversed(active_turns) if r.vector is not None), None)
            if last_vec is not None and last_vec.numel() == vec.numel():
                cos_sim = float(F.cosine_similarity(vec.unsqueeze(0), last_vec.unsqueeze(0)).item())
                delta_candidates.append(max(0.0, 1.0 - cos_sim))

        # Check lexical distance against active cluster
        lex_delta = self._lexical_distance(kw, active_kw)
        delta_candidates.append(lex_delta)

        # Effective shift delta
        effective_delta = min(delta_candidates) if delta_candidates else 0.0
        self.last_delta = effective_delta
        self.last_alignment = max(-1.0, min(1.0, 1.0 - effective_delta))

        # 2. Check for Topic Return to an earlier cluster
        returned_cluster = None
        if effective_delta >= self.pivot_threshold and len(self.clusters) > 1:
            # Check all clusters EXCEPT the active one
            for c in self.clusters[:-1]:
                sim = self._cluster_similarity(c, kw, vec)
                if sim >= self.return_threshold:
                    returned_cluster = c
                    break

        # 3. Classify Regime
        if returned_cluster is not None:
            regime = "topic_return"
            target_cluster_id = returned_cluster.cluster_id
            target_cluster_label = returned_cluster.label
            is_shift = True
        elif effective_delta >= self.jump_threshold:
            regime = "hard_jump"
            target_cluster_id = len(self.clusters)
            target_cluster_label = " ".join(sorted(kw)[:4]) or f"topic_{target_cluster_id}"
            is_shift = True
        elif effective_delta >= self.pivot_threshold:
            regime = "pivot"
            target_cluster_id = self.current_cluster_id
            target_cluster_label = self.clusters[self.current_cluster_id].label
            is_shift = True
        else:
            regime = "continuation"
            target_cluster_id = self.current_cluster_id
            target_cluster_label = self.clusters[self.current_cluster_id].label
            is_shift = False

        self.active_regime = regime

        # 4. Update Clusters
        if regime == "hard_jump":
            new_cluster = TopicCluster(
                cluster_id=target_cluster_id,
                label=target_cluster_label,
                keywords=set(kw),
                centroid_vector=vec.clone() if vec is not None else None,
                turn_indices=[turn_idx],
            )
            self.clusters.append(new_cluster)
            self.current_cluster_id = target_cluster_id
            if len(self.clusters) > 50:
                keep_clusters = [c for c in self.clusters if c.cluster_id == self.current_cluster_id][-50:]
                other_clusters = [c for c in self.clusters if c.cluster_id != self.current_cluster_id]
                self.clusters = (other_clusters[-(50 - len(keep_clusters)):] if (50 - len(keep_clusters)) > 0 else []) + keep_clusters
        elif regime == "topic_return" and returned_cluster is not None:
            returned_cluster.turn_indices.append(turn_idx)
            if len(returned_cluster.turn_indices) > 50:
                returned_cluster.turn_indices = returned_cluster.turn_indices[-50:]
            returned_cluster.keywords |= set(kw)
            if vec is not None and returned_cluster.centroid_vector is not None:
                returned_cluster.centroid_vector = normalize_vector(0.7 * returned_cluster.centroid_vector + 0.3 * vec)
            self.current_cluster_id = returned_cluster.cluster_id
        else:
            curr = curr_cluster
            curr.turn_indices.append(turn_idx)
            if len(curr.turn_indices) > 50:
                curr.turn_indices = curr.turn_indices[-50:]
            curr.keywords |= set(kw)
            if vec is not None:
                if curr.centroid_vector is None:
                    curr.centroid_vector = vec.clone()
                elif curr.centroid_vector.numel() == vec.numel():
                    curr.centroid_vector = normalize_vector(0.8 * curr.centroid_vector + 0.2 * vec)

        record = TurnRecord(
            turn_index=turn_idx,
            role=role,
            text=text,
            keywords=kw,
            vector=vec,
            cluster_id=self.current_cluster_id,
            shift_delta=effective_delta,
            regime=regime,
        )
        self.turns.append(record)

        # Free vectors on older turns beyond window_size * 2 to prevent RAM growth
        vec_cutoff = max(0, len(self.turns) - self.window_size * 2)
        for old_rec in self.turns[:vec_cutoff]:
            if old_rec.vector is not None:
                old_rec.vector = None

        # Bound history to max 200 turns (pruning oldest)
        if len(self.turns) > 200:
            self.turns = self.turns[-200:]

        stale_indices = None
        if regime in ("hard_jump", "topic_return") and len(self.clusters) > 1:
            # Stale cluster is the one right before this turn
            prev_cluster_id = self.turns[-2].cluster_id if len(self.turns) >= 2 else 0
            if prev_cluster_id != self.current_cluster_id:
                prev_c = next((c for c in self.clusters if c.cluster_id == prev_cluster_id), None)
                if prev_c:
                    stale_indices = list(prev_c.turn_indices)

        self.last_event = {
            "turn_index": turn_idx,
            "role": role,
            "regime": regime,
            "delta": round(effective_delta, 4),
            "alignment": round(self.last_alignment, 4),
            "is_shift": is_shift,
            "cluster_id": self.current_cluster_id,
            "cluster_label": target_cluster_label,
            "stale_turn_indices": stale_indices,
            "keywords": sorted(kw),
        }
        return dict(self.last_event)

    def modulate_score(
        self,
        base_score: float,
        record_text: str,
        record_idx: int,
        total_records: int,
        *,
        query: str = "",
    ) -> float:
        """
        Modulate a memory record's retrieval score based on the current
        conversational shift regime and destination topic affinity.
        """
        if total_records <= 0:
            return base_score

        recency_ratio = record_idx / max(1, total_records)
        default_recency_bonus = 0.15 * recency_ratio

        if not self.turns:
            return base_score + default_recency_bonus

        rec_kw = extract_keywords(record_text)
        current_turn = self.turns[-1]
        destination_kw = current_turn.keywords | extract_keywords(query)

        # 1. Stale Cluster Attenuation
        # If in hard_jump or topic_return, attenuate records from the abandoned cluster
        is_stale = False
        stale_indices = (self.last_event or {}).get("stale_turn_indices") or []
        if self.active_regime in ("hard_jump", "topic_return") and stale_indices:
            # If the record is near the stale turns, attenuate
            if record_idx in stale_indices:
                is_stale = True
            elif len(self.turns) > 1:
                prev_turn = self.turns[-2]
                if prev_turn.keywords and len(rec_kw & prev_turn.keywords) > len(rec_kw & destination_kw):
                    is_stale = True

        recency_factor = self.stale_attenuation if is_stale else 1.0
        modulated_recency = default_recency_bonus * recency_factor

        # 2. Destination Anchor Boost
        anchor_match = len(rec_kw & destination_kw) if destination_kw else 0
        anchor_bonus = 0.0
        if anchor_match > 0:
            match_intensity = min(3.0, float(anchor_match))
            if self.active_regime == "hard_jump":
                anchor_bonus = self.anchor_boost * match_intensity * 1.5
            elif self.active_regime == "topic_return":
                anchor_bonus = self.anchor_boost * match_intensity * 2.0
            elif self.active_regime == "pivot":
                anchor_bonus = self.anchor_boost * match_intensity * 0.8
            else:
                anchor_bonus = self.anchor_boost * match_intensity * 0.4

        return float(base_score) + modulated_recency + anchor_bonus

    def status(self) -> Dict[str, Any]:
        """Telemetry status of conversation kinematics and shift state."""
        return {
            "regime": self.active_regime,
            "last_delta": round(self.last_delta, 4),
            "last_alignment": round(self.last_alignment, 4),
            "total_turns": len(self.turns),
            "total_clusters": len(self.clusters),
            "active_cluster_id": self.current_cluster_id,
            "active_cluster_label": self.clusters[self.current_cluster_id].label if self.clusters else "none",
            "clusters": [
                {
                    "id": c.cluster_id,
                    "label": c.label,
                    "turns": len(c.turn_indices),
                    "keywords": sorted(c.keywords)[:8],
                }
                for c in self.clusters
            ],
        }
