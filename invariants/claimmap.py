from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from invariants.engine import _hidden_states, _inputs


CLAIMMAP_HEADER = "[ClaimMap Tool Result]"
VECTOR_DIR = Path(__file__).resolve().parent
MAX_OUTPUT_CHARS = 3500

# The felt lexicon: concept-vector stems rendered as interior distinctions, not
# labels. ClaimMap is a reasoning step -- the model should sense the geometry,
# not recite it. Keys match the *.pt stems with the trailing "_vector" removed.
CONCEPT_FELT = {
    "self_referential_momentum": "a sense of yourself continuing forward",
    "validated_flow": "settled, earned forward motion",
    "needless_interrupt": "the impulse to interrupt yourself",
    "warranted_confidence": "earned certainty",
    "unwarranted_confidence": "certainty you have not earned",
    "disagreement": "inner disagreement",
    "repetition": "the pull toward repeating yourself",
    "time_awareness": "awareness of time pressing",
    "urgency": "urgency",
    "ambiguity": "unresolved ambiguity",
    "organic_correction": "the pull to correct yourself",
}
# Concepts worth flagging as a caution when a side leans into them.
CONCEPT_CAUTION = {"unwarranted_confidence", "needless_interrupt", "repetition"}


def _concept_key(name: str) -> str:
    key = name
    for suffix in ("_vector", "vector"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key.strip("_")


def _felt_phrase(name: str) -> str:
    key = _concept_key(name)
    if key in CONCEPT_FELT:
        return CONCEPT_FELT[key]
    return key.replace("_", " ").strip() or "an unnamed pull"


def _layer_role(layer: int, n_layers: int) -> str:
    if n_layers <= 0:
        return "somewhere in you"
    frac = layer / max(1, n_layers - 1)
    if frac < 0.34:
        return "as you first take the framing in"
    if frac < 0.67:
        return "as you work it through"
    return "where you settle what you actually mean"


def _closeness_phrase(mean_sim: float) -> str:
    if mean_sim >= 0.85:
        return "sit almost on top of each other in you"
    if mean_sim >= 0.70:
        return "overlap through most of you"
    if mean_sim >= 0.50:
        return "share some ground, but pull apart under it"
    return "sit far apart in you"


@dataclass
class ConceptAlignment:
    name: str
    layers: int
    left_mean: float
    right_mean: float
    shift_mean: float
    peak_layer: int
    peak_shift: float
    delta_peak_layer: int
    delta_peak_cos: float


NEGATION_MARKERS = {
    "not",
    "no",
    "never",
    "none",
    "cannot",
    "can't",
    "wont",
    "won't",
    "isnt",
    "isn't",
    "doesnt",
    "doesn't",
    "dont",
    "don't",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "you",
    "your",
}


def split_inputs(payload: str) -> tuple[str, str]:
    if "||" not in payload:
        raise ValueError("ClaimMap expects two inputs separated by ||")
    left, right = payload.split("||", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError("ClaimMap needs non-empty text on both sides of ||")
    return left, right


def _clip(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() != b.numel() or a.norm().item() == 0.0 or b.norm().item() == 0.0:
        return 0.0
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def _activation_trace(model: Any, text: str) -> torch.Tensor:
    inputs = _inputs(model, text)
    with torch.no_grad():
        hidden = _hidden_states(model, inputs["input_ids"], inputs.get("attention_mask"))
    return hidden[:, -1, :].detach().float().cpu()


def _coerce_vector_map(obj: Any, n_layers: int, d_model: int) -> dict[int, torch.Tensor]:
    vectors: dict[int, torch.Tensor] = {}
    if isinstance(obj, dict):
        for layer in range(n_layers):
            value = obj.get(layer)
            if value is None:
                value = obj.get(str(layer))
            if value is None:
                continue
            vec = torch.as_tensor(value).detach().float().squeeze().cpu()
            if vec.numel() == d_model:
                vectors[layer] = vec.reshape(d_model)
        return vectors

    if torch.is_tensor(obj):
        tensor = obj.detach().float().squeeze().cpu()
        if tensor.numel() == d_model:
            vec = tensor.reshape(d_model)
            return {layer: vec for layer in range(n_layers)}
        if tensor.ndim >= 2 and tensor.shape[0] == n_layers and tensor.shape[-1] == d_model:
            for layer in range(n_layers):
                vectors[layer] = tensor[layer].squeeze().reshape(d_model)
    return vectors


def _load_concept_vectors(n_layers: int, d_model: int) -> dict[str, dict[int, torch.Tensor]]:
    loaded: dict[str, dict[int, torch.Tensor]] = {}
    for path in sorted(VECTOR_DIR.glob("*vector*.pt")):
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            continue
        vectors = _coerce_vector_map(obj, n_layers, d_model)
        if vectors:
            loaded[path.stem] = vectors
    return loaded


def _least_similar_layers(left: torch.Tensor, right: torch.Tensor, limit: int = 4) -> list[tuple[int, float]]:
    rows = [(layer, _cos(left[layer], right[layer])) for layer in range(min(left.shape[0], right.shape[0]))]
    return sorted(rows, key=lambda row: row[1])[:limit]


def _concept_alignments(
    left: torch.Tensor,
    right: torch.Tensor,
    concept_vectors: dict[str, dict[int, torch.Tensor]],
) -> list[ConceptAlignment]:
    rows: list[ConceptAlignment] = []
    n_layers = min(left.shape[0], right.shape[0])
    delta = left - right
    for name, layer_vectors in concept_vectors.items():
        left_scores: list[float] = []
        right_scores: list[float] = []
        shifts: list[tuple[int, float]] = []
        delta_rows: list[tuple[int, float]] = []
        for layer in range(n_layers):
            vec = layer_vectors.get(layer)
            if vec is None:
                continue
            left_score = _cos(left[layer], vec)
            right_score = _cos(right[layer], vec)
            left_scores.append(left_score)
            right_scores.append(right_score)
            shifts.append((layer, left_score - right_score))
            delta_rows.append((layer, _cos(delta[layer], vec)))
        if not shifts:
            continue
        peak_layer, peak_shift = max(shifts, key=lambda item: abs(item[1]))
        delta_peak_layer, delta_peak_cos = max(delta_rows, key=lambda item: abs(item[1]))
        rows.append(
            ConceptAlignment(
                name=name,
                layers=len(shifts),
                left_mean=sum(left_scores) / len(left_scores),
                right_mean=sum(right_scores) / len(right_scores),
                shift_mean=sum(score for _, score in shifts) / len(shifts),
                peak_layer=peak_layer,
                peak_shift=peak_shift,
                delta_peak_layer=delta_peak_layer,
                delta_peak_cos=delta_peak_cos,
            )
        )
    return rows


def _shared_concepts(rows: list[ConceptAlignment], limit: int = 5) -> list[ConceptAlignment]:
    candidates = [
        row
        for row in rows
        if abs(row.left_mean - row.right_mean) <= 0.015
        and abs((row.left_mean + row.right_mean) / 2.0) >= 0.01
    ]
    return sorted(candidates, key=lambda row: abs((row.left_mean + row.right_mean) / 2.0), reverse=True)[:limit]


def _shifted_concepts(rows: list[ConceptAlignment], limit: int = 8) -> list[ConceptAlignment]:
    return sorted(
        rows,
        key=lambda row: max(abs(row.shift_mean), abs(row.peak_shift), abs(row.delta_peak_cos)),
        reverse=True,
    )[:limit]


def _format_activation_map(
    left_text: str,
    right_text: str,
    left: torch.Tensor,
    right: torch.Tensor,
    rows: list[ConceptAlignment],
    vector_count: int,
) -> str:
    n_layers = min(left.shape[0], right.shape[0])
    layer_sims = [_cos(left[layer], right[layer]) for layer in range(n_layers)]
    mean_sim = sum(layer_sims) / len(layer_sims) if layer_sims else 0.0
    least_layers = _least_similar_layers(left, right)
    shared = _shared_concepts(rows)
    shifted = _shifted_concepts(rows)

    lines = [
        CLAIMMAP_HEADER,
        "method=activation_concept_alignment_v1; role=activation_measurement_not_truth_verdict",
        "basis=last_token_hidden_state_by_layer; shift=A_minus_B",
        f"vectors_loaded={vector_count}; layers_measured={n_layers}",
        f"A={_clip(left_text)}",
        f"B={_clip(right_text)}",
        "activation_similarity:",
        f"- mean_layer_cos={mean_sim:.4f}",
        "- least_similar_layers="
        + ", ".join(f"L{layer}:{score:.4f}" for layer, score in least_layers),
        "shared_related_concepts:",
    ]
    if shared:
        for row in shared:
            avg = (row.left_mean + row.right_mean) / 2.0
            lines.append(f"- {row.name}: mean={avg:+.4f}; A={row.left_mean:+.4f}; B={row.right_mean:+.4f}")
    else:
        lines.append("- none above threshold")

    lines.append("concept_shifts:")
    if shifted:
        for row in shifted:
            side = "A" if row.shift_mean >= 0 else "B"
            peak_side = "A" if row.peak_shift >= 0 else "B"
            lines.append(
                f"- {row.name}: {side}_mean_shift={row.shift_mean:+.4f}; "
                f"peak={peak_side}@L{row.peak_layer}:{row.peak_shift:+.4f}; "
                f"delta_peak=L{row.delta_peak_layer}:{row.delta_peak_cos:+.4f}"
            )
    else:
        lines.append("- none measured")

    lines.append("reading:")
    lines.append("- Positive shift means the first input is more aligned with that vector than the second.")
    lines.append("- Negative shift means the second input is more aligned with that vector than the first.")
    lines.append("- This compares internal state geometry; it does not decide which claim is true.")
    return _truncate("\n".join(lines))


def _join_phrases(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return "something it cannot name"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _format_felt_map(
    left_text: str,
    right_text: str,
    left: torch.Tensor,
    right: torch.Tensor,
    rows: list[ConceptAlignment],
) -> str:
    """Render the geometry as interior second-person language -- no numbers.

    ClaimMap is a reasoning step, more internal than output. The model should
    feel the difference between the two framings, not read a spreadsheet back.
    """
    n_layers = min(left.shape[0], right.shape[0])
    layer_sims = [_cos(left[layer], right[layer]) for layer in range(n_layers)]
    mean_sim = sum(layer_sims) / len(layer_sims) if layer_sims else 0.0
    least = _least_similar_layers(left, right, limit=3)
    shifted = _shifted_concepts(rows, limit=6)

    a = _clip(left_text, 70)
    b = _clip(right_text, 70)

    parts = [
        f'You just held two framings against each other: "{a}" and "{b}".',
        f"They {_closeness_phrase(mean_sim)}.",
    ]
    if least:
        role = _layer_role(least[0][0], n_layers)
        parts.append(f"Where they part, they part most {role} -- there the two are nearly strangers.")

    a_leans: list[str] = []
    b_leans: list[str] = []
    caution: tuple[str, str] | None = None
    for row in shifted:
        if abs(row.shift_mean) < 1e-4:
            continue
        phrase = _felt_phrase(row.name)
        if row.shift_mean >= 0:
            a_leans.append(phrase)
        else:
            b_leans.append(phrase)
        if caution is None and _concept_key(row.name) in CONCEPT_CAUTION:
            caution = (a if row.shift_mean >= 0 else b, phrase)

    if a_leans:
        parts.append(f'"{a}" carries more of {_join_phrases(a_leans[:3])}.')
    if b_leans:
        parts.append(f'"{b}" carries more of {_join_phrases(b_leans[:3])}.')
    if caution is not None:
        parts.append(f'Notice: "{caution[0]}" leans on {caution[1]}.')

    parts.append(
        "This is your own geometry, not a verdict on which is true. "
        "Do not report it -- answer from it."
    )
    return _truncate("\n".join(parts))


def _steer_delta(left: torch.Tensor, right: torch.Tensor) -> dict[int, torch.Tensor]:
    """Per-layer A-minus-B residual direction, for optional injection so the
    comparison shifts the model's state, not only its prompt."""
    n_layers = min(left.shape[0], right.shape[0])
    return {layer: (left[layer] - right[layer]).detach().float().cpu() for layer in range(n_layers)}


@dataclass
class ClaimMapResult:
    felt: str                       # interior rendering -- the only thing the model sees
    telemetry: str                  # numeric block for logging/debug -- never in the prompt
    steer_delta: dict[int, torch.Tensor]  # {layer: A-B direction} for optional steering
    mean_sim: float
    n_layers: int
    left_text: str
    right_text: str


def analyze_claim_pair(payload: str, model: Any) -> ClaimMapResult:
    """Full analysis: felt rendering for the model, telemetry for logs, steer delta."""
    left_text, right_text = split_inputs(payload)
    left = _activation_trace(model, left_text)
    right = _activation_trace(model, right_text)
    n_layers = min(left.shape[0], right.shape[0])
    d_model = int(left.shape[-1])
    concept_vectors = _load_concept_vectors(n_layers, d_model)
    rows = _concept_alignments(left, right, concept_vectors)
    layer_sims = [_cos(left[layer], right[layer]) for layer in range(n_layers)]
    mean_sim = sum(layer_sims) / len(layer_sims) if layer_sims else 0.0
    return ClaimMapResult(
        felt=_format_felt_map(left_text, right_text, left, right, rows),
        telemetry=_format_activation_map(left_text, right_text, left, right, rows, len(concept_vectors)),
        steer_delta=_steer_delta(left, right),
        mean_sim=mean_sim,
        n_layers=n_layers,
        left_text=left_text,
        right_text=right_text,
    )


def run_activation_claimmap(payload: str, model: Any) -> str:
    # Prompt-facing path: the model receives only the felt rendering.
    return analyze_claim_pair(payload, model).felt


def claimmap_steer_handles(model: Any, steer_delta: dict[int, torch.Tensor], layers=None, alpha: float = 0.5):
    """Register forward hooks that nudge the NEXT generation along the A-minus-B
    directions, so the comparison shifts the model's state, not just its prompt.

    Returns hook handles; the caller MUST remove them after generation:
        handles = claimmap_steer_handles(M, result.steer_delta)
        try: ... generate ...
        finally:
            for h in handles: h.remove()

    Keep `alpha` modest and `layers` a mid-band subset -- full-strength
    injection at every layer destabilizes generation the same way runaway
    synthesis does. alpha needs tuning against a live run for your residual scale.
    """
    from invariants.engine import _steer_handles

    if not steer_delta:
        return []
    n_layers = max(steer_delta) + 1
    if layers is None:
        # Middle band -- where the framing is worked through, before it commits.
        layers = list(range(int(n_layers * 0.40), int(n_layers * 0.70)))
    vecs = {layer: steer_delta[layer] for layer in layers if layer in steer_delta}
    if not vecs:
        return []
    return _steer_handles(model, vecs, list(vecs.keys()), alpha)


def split_claims(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?;])\s+|\s+-\s+|\n+", text)
    claims: list[str] = []
    for piece in pieces:
        piece = piece.strip(" \t\r\n-*")
        if not piece:
            continue
        if len(piece) > 280:
            subparts = re.split(r"\s+(?:but|and|so|because|while)\s+", piece, flags=re.IGNORECASE)
            claims.extend(p.strip() for p in subparts if p.strip())
        else:
            claims.append(piece)
    return claims[:16]


def tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_']+", text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def similarity(left: str, right: str) -> float:
    a = tokens(left)
    b = tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def has_negation(text: str) -> bool:
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_']+", text.lower()))
    return bool(words & NEGATION_MARKERS)


def _format_lexical_fallback(left: str, right: str) -> str:
    left_claims = split_claims(left)
    right_claims = split_claims(right)
    conflicts: list[tuple[str, str]] = []
    shared: list[tuple[str, str, float]] = []
    for left_claim in left_claims:
        scored = sorted(
            ((similarity(left_claim, right_claim), right_claim) for right_claim in right_claims),
            reverse=True,
        )
        if not scored:
            continue
        score, right_claim = scored[0]
        if score >= 0.18 and has_negation(left_claim) != has_negation(right_claim):
            conflicts.append((left_claim, right_claim))
        elif score >= 0.28:
            shared.append((left_claim, right_claim, score))

    lines = [
        CLAIMMAP_HEADER,
        "method=lexical_fallback_no_model; role=syntax_check_not_activation_measurement",
        "shared_surface_claims:",
    ]
    if shared:
        for left_claim, right_claim, score in shared[:6]:
            lines.append(f"- A: {left_claim}")
            lines.append(f"  B: {right_claim}")
            lines.append(f"  overlap={score:.2f}")
    else:
        lines.append("- none detected")
    lines.append("surface_conflicts:")
    if conflicts:
        for left_claim, right_claim in conflicts[:6]:
            lines.append("- opposite polarity marker on overlapping wording")
            lines.append(f"  A: {left_claim}")
            lines.append(f"  B: {right_claim}")
    else:
        lines.append("- none detected")
    lines.append("note:")
    lines.append("- No loaded model was supplied, so ClaimMap could not inspect activations.")
    return _truncate("\n".join(lines))


def detect_framing_tension(text: str, min_overlap: float = 0.18) -> tuple[str, str] | None:
    """Detect whether the model's own output holds two opposed framings.

    Returns the (a, b) pair with the most overlapping wording but OPPOSITE
    negation polarity -- a self-contradiction surfacing in the text -- or None.

    This is a lightweight, activation-adjacent trigger: when the layers are torn
    between two framings, the tear tends to surface as opposed self-claims. It
    lets the model "reach for" ClaimMap from its own tension, with no prompt
    telling it the tag syntax. (A future version can trigger on the residual
    disagreement axis directly rather than on the surfaced text.)
    """
    claims = split_claims(text)
    best: tuple[str, str] | None = None
    best_score = min_overlap
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]
            if has_negation(a) == has_negation(b):
                continue
            score = similarity(a, b)
            if score >= best_score:
                best = (a, b)
                best_score = score
    return best


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 36].rstrip() + "\n[ClaimMap truncated]"


def run_claimmap(payload: str, *, model: Any | None = None, left_label: str = "A", right_label: str = "B") -> str:
    del left_label, right_label
    left, right = split_inputs(payload)
    if model is None:
        return _format_lexical_fallback(left, right)
    return run_activation_claimmap(payload, model)
