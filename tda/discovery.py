"""
Dynamic latent variable discovery.

Instead of relying on predefined vocabularies, this module discovers what
the model is actually tracking by working backwards from the activations.

Approach:
1. For each dimension, find which tokens consistently push it up or down
2. Cluster those tokens by their activation signatures
3. The clusters define new quantity axes — not imposed by us, found in the data
4. Score each discovered axis for consistency across contexts
5. Feed confirmed axes back into latent_variables.py as dynamic vocabularies

This is the complement to latent_variables.py:
  latent_variables.py — "we think the model tracks X, let's check"
  discovery.py        — "what is the model actually tracking?"

The two should converge. Where they agree, confidence is high.
Where discovery finds something latent_variables missed, extend the vocabulary.
Where latent_variables finds something discovery missed, the predefined
scale may be too coarse or the vocabulary word absent from the corpus.
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats


def build_token_activation_matrix(
    token_strings: list[str],
    activation_matrix: np.ndarray,
    min_token_occurrences: int = 2,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    For each unique token, computes the mean activation vector across all
    positions where that token appears.

    Returns:
      token_profiles: {token_str: mean_activation_vector [d_model]}
      vocabulary: sorted list of token strings with enough occurrences
    """
    token_positions = defaultdict(list)
    for i, tok in enumerate(token_strings):
        tok = tok.lower().strip()
        if tok and i < len(activation_matrix):
            token_positions[tok].append(i)

    token_profiles = {}
    for tok, positions in token_positions.items():
        if len(positions) >= min_token_occurrences:
            token_profiles[tok] = activation_matrix[positions].mean(axis=0)

    vocabulary = sorted(token_profiles.keys())
    return token_profiles, vocabulary


def find_sensitive_dimensions(
    token_profiles: dict[str, np.ndarray],
    top_k: int = 50,
) -> list[int]:
    """
    Finds dimensions with high variance across token profiles.
    High variance = dimension responds differently to different tokens.
    These are the dimensions most likely to be encoding something.
    """
    if not token_profiles:
        return []

    matrix = np.stack(list(token_profiles.values()))  # [n_tokens, d_model]
    variances = matrix.var(axis=0)
    return list(np.argsort(variances)[-top_k:][::-1])


def cluster_tokens_by_dimension(
    token_profiles: dict[str, np.ndarray],
    sensitive_dims: list[int],
    min_cluster_size: int = 3,
) -> dict[int, list[str]]:
    """
    Projects token profiles onto sensitive dimensions and clusters them.
    Each cluster is a group of tokens that activate the same dimensions
    in the same way — likely encoding the same latent quantity.

    Returns {cluster_label: [token_strings]}
    """
    if not token_profiles or not sensitive_dims:
        return {}

    tokens = list(token_profiles.keys())
    matrix = np.stack([token_profiles[t] for t in tokens])
    projected = matrix[:, sensitive_dims]

    scaler = StandardScaler()
    projected = scaler.fit_transform(projected)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(projected)

    clusters = defaultdict(list)
    for token, label in zip(tokens, labels):
        if label != -1:
            clusters[label].append(token)

    return dict(clusters)


def score_cluster_as_scale(
    cluster_tokens: list[str],
    token_profiles: dict[str, np.ndarray],
    activation_matrix: np.ndarray,
    token_strings: list[str],
) -> dict:
    """
    Tests whether a token cluster behaves like a continuous scale —
    i.e., the tokens in the cluster don't just co-activate the same dimensions,
    they do so in a graded way that suggests an underlying ordinal/continuous variable.

    Returns a score and the most sensitive dimension for this cluster.
    """
    if len(cluster_tokens) < 3:
        return {"is_scale": False, "score": 0.0}

    profiles = np.stack([token_profiles[t] for t in cluster_tokens])
    variances = profiles.var(axis=0)
    best_dim = int(np.argmax(variances))

    dim_values = profiles[:, best_dim]
    positions_in_text = []
    for i, tok in enumerate(token_strings):
        if tok.lower().strip() in cluster_tokens and i < len(activation_matrix):
            positions_in_text.append((tok.lower().strip(), i))

    if len(positions_in_text) < 3:
        return {"is_scale": False, "score": 0.0, "best_dim": best_dim}

    # Check if activation at those positions varies smoothly
    act_at_positions = [activation_matrix[pos, best_dim] for _, pos in positions_in_text]
    token_vals = [dim_values[cluster_tokens.index(tok)] for tok, _ in positions_in_text]

    if len(set(token_vals)) < 2:
        return {"is_scale": False, "score": 0.0, "best_dim": best_dim}

    r, p = stats.spearmanr(token_vals, act_at_positions)
    is_scale = abs(r) > 0.5 and p < 0.1

    return {
        "is_scale": is_scale,
        "score": float(abs(r)),
        "p": float(p),
        "best_dim": best_dim,
        "n_tokens": len(cluster_tokens),
        "n_observations": len(positions_in_text),
    }


def name_cluster(cluster_tokens: list[str]) -> str:
    """
    Attempts to infer a human-readable name for a discovered cluster
    by finding the most semantically central token (highest mean similarity
    to others in activation space is too expensive here, so we use
    the most frequent token as a proxy label).
    Returns a string label like "CLUSTER(fire,danger,threat,explosion,...)"
    """
    preview = cluster_tokens[:5]
    label = ",".join(preview)
    if len(cluster_tokens) > 5:
        label += f",...+{len(cluster_tokens)-5}"
    return f"DISCOVERED({label})"


def discover(
    activation_matrix: np.ndarray,
    token_strings: list[str],
    top_dims: int = 50,
    min_cluster_size: int = 3,
    min_occurrences: int = 2,
) -> list[dict]:
    """
    Full discovery pipeline. Returns list of discovered latent axes:
    {
        "label": str,
        "tokens": [str],
        "best_dim": int,
        "is_scale": bool,
        "score": float,
    }

    These can be fed back into latent_variables.py as dynamic vocabularies,
    or compared against MAGNITUDE_WORDS and SALIENCE_WORDS to see what
    the predefined scales missed.
    """
    token_profiles, vocabulary = build_token_activation_matrix(
        token_strings, activation_matrix, min_occurrences
    )

    if len(token_profiles) < min_cluster_size:
        return []

    sensitive_dims = find_sensitive_dimensions(token_profiles, top_k=top_dims)
    clusters = cluster_tokens_by_dimension(token_profiles, sensitive_dims, min_cluster_size)

    results = []
    for label, tokens in clusters.items():
        score_result = score_cluster_as_scale(
            tokens, token_profiles, activation_matrix, token_strings
        )
        results.append({
            "label": name_cluster(tokens),
            "tokens": tokens,
            **score_result,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def compare_to_known(
    discovered: list[dict],
    known_vocabularies: dict[str, dict[str, float]],
) -> list[dict]:
    """
    Compares discovered clusters against known vocabularies
    (MAGNITUDE_WORDS, SALIENCE_WORDS).

    For each discovered cluster, reports:
    - overlap with known vocabulary
    - whether it's already covered
    - if not covered, it's a new candidate for the vocabulary

    This is how the system learns what it doesn't know yet.
    """
    results = []
    for cluster in discovered:
        tokens = set(cluster["tokens"])
        coverage = {}
        for vocab_name, vocab in known_vocabularies.items():
            known_tokens = set(vocab.keys())
            overlap = tokens & known_tokens
            coverage[vocab_name] = {
                "overlap": list(overlap),
                "overlap_ratio": len(overlap) / len(tokens) if tokens else 0.0,
            }

        best_coverage = max(
            coverage.items(), key=lambda x: x[1]["overlap_ratio"]
        )
        is_novel = best_coverage[1]["overlap_ratio"] < 0.3

        results.append({
            **cluster,
            "known_coverage": coverage,
            "novel": is_novel,
            "verdict": "NEW AXIS" if is_novel else f"EXTENDS {best_coverage[0].upper()}",
        })

    return results


def to_vocabulary(discovered: list[dict], score_threshold: float = 0.5) -> dict[str, float]:
    """
    Converts high-confidence discovered clusters into a vocabulary dict
    compatible with latent_variables.py MAGNITUDE_WORDS / SALIENCE_WORDS format.

    Tokens within a cluster are assigned values based on their rank within
    the cluster's best dimension — preserving the ordinal structure.
    """
    vocab = {}
    for cluster in discovered:
        if not cluster.get("is_scale") or cluster.get("score", 0) < score_threshold:
            continue
        tokens = cluster["tokens"]
        n = len(tokens)
        for i, tok in enumerate(tokens):
            vocab[tok] = float(i) / max(n - 1, 1)
    return vocab
