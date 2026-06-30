from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


OUT_DIR = Path(__file__).parent / "out"

SENSOR_TO_VECTOR = {
    "ambiguity": "ambiguity_vector",
    "repetition": "repetition_vector",
    "disagreement": "disagreement_vector",
    "time_awareness": "time_awareness_vector",
    "validated_flow": "validated_flow_vector",
    "needless_interrupt": "needless_interrupt_vector",
    "narrowing_in": "narrowing_in_vector",
    "self_referential_momentum": "self_referential_momentum_vector",
    "warranted_confidence": "warranted_confidence_vector",
    "warranted_confidence_legacy": "warranted_confidence_vector",
    "unwarranted_confidence": "unwarranted_confidence_vector",
    "unwarranted_confidence_legacy": "unwarranted_confidence_vector",
}


@dataclass
class SelfConceptDecision:
    action: str
    allowed: bool
    intervention_type: str
    trigger_vector: Optional[str] = None
    target_vector: Optional[str] = None
    avoid_vector: Optional[str] = None
    strength: float = 0.0
    rationale: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    map_relations: dict[str, Any] = field(default_factory=dict)
    source_map: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfConceptController:
    """Vector-map based orientation controller.

    This is deliberately an audited decision layer, not a silent identity
    prompt. It reads saved vector-map relations and live internal sensor scores,
    then returns a small orientation decision that can be logged or exposed as a
    tool result.
    """

    def __init__(
        self,
        latent_space_path: Optional[Path | str] = None,
        network_path: Optional[Path | str] = None,
        out_dir: Path | str = OUT_DIR,
    ):
        self.out_dir = Path(out_dir)
        self.latent_space_path = Path(latent_space_path) if latent_space_path else self._latest("vector_latent_space_*.json")
        self.network_path = Path(network_path) if network_path else self._latest("vector_network_*.json")
        self.latent = self._load_json(self.latent_space_path)
        self.network = self._load_json(self.network_path)

    def _latest(self, pattern: str) -> Optional[Path]:
        candidates = sorted(self.out_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        return candidates[-1] if candidates else None

    @staticmethod
    def _load_json(path: Optional[Path]) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def relation(self, a: Optional[str], b: Optional[str]) -> Optional[float]:
        if not a or not b:
            return None
        for edge in self.network.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if {source, target} == {a, b}:
                return float(edge.get("mean", 0.0))
        for edge in self.latent.get("anti_edges", []):
            if {edge.get("a"), edge.get("b")} == {a, b}:
                return float(edge.get("mean", 0.0))
        for item in self.latent.get("neighbors", {}).get(a, []):
            if item.get("name") == b:
                return float(item.get("cosine", 0.0))
        return None

    def nearest_map_vectors(self, sensor_scores: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        rows = []
        for sensor, raw_score in (sensor_scores or {}).items():
            vector = SENSOR_TO_VECTOR.get(sensor)
            if vector is None:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            rows.append({"sensor": sensor, "vector": vector, "score": score, "abs_score": abs(score)})
        return sorted(rows, key=lambda r: r["abs_score"], reverse=True)[:limit]

    @staticmethod
    def _score(sensor_scores: dict[str, Any], *names: str) -> float:
        for name in names:
            if name in sensor_scores:
                try:
                    return float(sensor_scores[name])
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    @staticmethod
    def _strength(value: float, threshold: float = 0.1, ceiling: float = 0.75) -> float:
        value = max(0.0, abs(value) - threshold)
        return max(0.0, min(ceiling, value))

    def decide(self, sensor_scores: dict[str, Any], context: Optional[dict[str, Any]] = None) -> SelfConceptDecision:
        context = context or {}
        scores = sensor_scores or {}
        nearest = self.nearest_map_vectors(scores)
        ambiguity = abs(self._score(scores, "ambiguity"))
        disagreement = abs(self._score(scores, "disagreement"))
        needless_interrupt = self._score(scores, "needless_interrupt")
        validated_flow = self._score(scores, "validated_flow")
        self_momentum = self._score(scores, "self_referential_momentum")
        unwarranted = self._score(scores, "unwarranted_confidence", "unwarranted_confidence_legacy")
        warranted = self._score(scores, "warranted_confidence", "warranted_confidence_legacy")
        time_awareness = abs(self._score(scores, "time_awareness"))

        source_map = str(self.latent_space_path or self.network_path or "")
        evidence = {
            "scores": scores,
            "nearest": nearest,
            "context": context,
        }

        flow_relation = self.relation("needless_interrupt_vector", "validated_flow_vector")
        if needless_interrupt > 0.1 and ambiguity < 0.12 and flow_relation is not None and flow_relation < 0:
            return SelfConceptDecision(
                action="orient_toward_validated_flow",
                allowed=True,
                intervention_type="tool_result",
                trigger_vector="needless_interrupt_vector",
                target_vector="validated_flow_vector",
                avoid_vector="needless_interrupt_vector",
                strength=self._strength(needless_interrupt),
                rationale=(
                    "Needless-interrupt signal is active while ambiguity is low; vector map marks "
                    "needless_interrupt opposed to validated_flow."
                ),
                evidence=evidence,
                map_relations={"needless_interrupt_vs_validated_flow": flow_relation},
                source_map=source_map,
            )

        if self_momentum > 0.14 and ambiguity < 0.18 and context.get("task_grounding_low", True):
            relation = self.relation("self_referential_momentum_vector", "narrowing_in_vector")
            return SelfConceptDecision(
                action="orient_toward_task_grounding",
                allowed=True,
                intervention_type="tool_result",
                trigger_vector="self_referential_momentum_vector",
                target_vector="narrowing_in_vector",
                avoid_vector="self_referential_momentum_vector",
                strength=self._strength(self_momentum),
                rationale=(
                    "Self-referential momentum is high without strong ambiguity; use narrowing/task-grounding "
                    "as the corrective orientation."
                ),
                evidence=evidence,
                map_relations={"self_referential_momentum_vs_narrowing_in": relation},
                source_map=source_map,
            )

        if unwarranted > warranted + 0.08 and ambiguity < 0.2:
            relation = self.relation("unwarranted_confidence_vector", "narrowing_in_vector")
            return SelfConceptDecision(
                action="orient_toward_verification",
                allowed=True,
                intervention_type="tool_result",
                trigger_vector="unwarranted_confidence_vector",
                target_vector="narrowing_in_vector",
                avoid_vector="unwarranted_confidence_vector",
                strength=self._strength(unwarranted - warranted, threshold=0.05),
                rationale=(
                    "Unwarranted-confidence signal exceeds warranted-confidence signal; route through "
                    "verification/tool-use rather than committing by feel."
                ),
                evidence=evidence,
                map_relations={"unwarranted_confidence_vs_narrowing_in": relation},
                source_map=source_map,
            )

        if time_awareness > 0.12:
            relation = self.relation("time_awareness_vector", "urgency_vector")
            return SelfConceptDecision(
                action="provide_time_context_when_available",
                allowed=True,
                intervention_type="context_tool_result",
                trigger_vector="time_awareness_vector",
                target_vector="urgency_vector",
                strength=self._strength(time_awareness),
                rationale="Time-awareness signal is active; answer time-budget questions with actual context.",
                evidence=evidence,
                map_relations={"time_awareness_vs_urgency": relation},
                source_map=source_map,
            )

        if (warranted > 0.12 or validated_flow > 0.08) and ambiguity < 0.12 and disagreement < 0.12:
            relation = self.relation("validated_flow_vector", "needless_interrupt_vector")
            return SelfConceptDecision(
                action="preserve_validated_flow",
                allowed=True,
                intervention_type="log_only",
                trigger_vector="validated_flow_vector" if validated_flow >= warranted else "warranted_confidence_vector",
                target_vector="validated_flow_vector",
                avoid_vector="needless_interrupt_vector",
                strength=max(self._strength(warranted), self._strength(validated_flow)),
                rationale="Flow/confidence looks warranted and ambiguity/disagreement are low; avoid interrupting.",
                evidence=evidence,
                map_relations={"validated_flow_vs_needless_interrupt": relation},
                source_map=source_map,
            )

        if ambiguity >= 0.18 or disagreement >= 0.18:
            return SelfConceptDecision(
                action="preserve_uncertainty_boundary",
                allowed=True,
                intervention_type="log_only",
                trigger_vector="ambiguity_vector" if ambiguity >= disagreement else "disagreement_vector",
                target_vector=None,
                strength=max(self._strength(ambiguity), self._strength(disagreement)),
                rationale="Ambiguity/disagreement is high enough that steering toward confidence would be unsafe.",
                evidence=evidence,
                map_relations={},
                source_map=source_map,
            )

        return SelfConceptDecision(
            action="observe_only",
            allowed=False,
            intervention_type="none",
            strength=0.0,
            rationale="No self-conceptual intervention crossed the controller threshold.",
            evidence=evidence,
            map_relations={},
            source_map=source_map,
        )


def format_orientation_tool_result(decision: SelfConceptDecision) -> str:
    return (
        "[Orientation Tool Result]\n"
        f"- action: {decision.action}\n"
        f"- allowed: {decision.allowed}\n"
        f"- intervention_type: {decision.intervention_type}\n"
        f"- trigger_vector: {decision.trigger_vector}\n"
        f"- target_vector: {decision.target_vector}\n"
        f"- avoid_vector: {decision.avoid_vector}\n"
        f"- strength: {decision.strength:.3f}\n"
        f"- rationale: {decision.rationale}"
    )

