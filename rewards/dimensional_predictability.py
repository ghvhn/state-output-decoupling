"""
Reward model for prediction-encouraging compression.

Core idea: a representation is good if its dimensions are:
  1. Persistent   — topological features survive across scales (signal, not noise)
  2. Predictable  — knowing the current value forecasts the next (low surprise)
  3. Compressed   — few dimensions carry most variance (efficient encoding)
  4. Aligned      — same dimension encodes the same thing across languages (bilingual case)

These four signals combine into a scalar reward that can be used to shape
pre-training via a KL penalty, auxiliary loss, or PPO reward signal.

Intended for: bilingual pre-training experiment where the model is rewarded
for representations whose dimensions are consistent and predictable across
Spanish and English parallel corpora.

TDA connection: persistence_reward uses the same diagrams produced by
tda/homology.py — meaning the domain mapper's analysis directly informs
the reward signal.
"""

import numpy as np
from scipy.stats import spearmanr


def persistence_reward(diagrams: list[np.ndarray], threshold: float = 0.1) -> float:
    """
    Sums lifetime of all topological features above threshold.
    Features that persist = signal. Features that die quickly = noise.
    High total persistence → stable, structured representation → high reward.
    """
    total = 0.0
    for dgm in diagrams:
        if len(dgm) == 0:
            continue
        lifetimes = dgm[:, 1] - dgm[:, 0]
        total += float(np.sum(lifetimes[lifetimes > threshold]))
    return total


def predictability_reward(activation_matrix: np.ndarray, lag: int = 1) -> float:
    """
    Mean absolute Spearman autocorrelation at `lag` across all dimensions.
    activation_matrix: [n_tokens, d_model]

    High autocorrelation → current token predicts next token in activation space
    → the model is building a trajectory, not random noise → high reward.
    """
    if activation_matrix.shape[0] < lag + 2:
        return 0.0
    scores = []
    for dim in range(activation_matrix.shape[1]):
        traj = activation_matrix[:, dim].astype(np.float32)
        r, _ = spearmanr(traj[:-lag], traj[lag:])
        if not np.isnan(r):
            scores.append(abs(float(r)))
    return float(np.mean(scores)) if scores else 0.0


def compression_reward(activation_matrix: np.ndarray, variance_threshold: float = 0.95) -> float:
    """
    Ratio of total dimensions to the number needed to explain variance_threshold
    of total variance (via PCA). Higher ratio = more compressed = higher reward.

    Example: 4096 dims, 95% variance explained by 40 PCs → ratio = 4096/40 = 102.4
    A random matrix would need ~3890 PCs → ratio ≈ 1.05
    """
    from sklearn.decomposition import PCA
    n = min(activation_matrix.shape[0] - 1, activation_matrix.shape[1])
    if n < 2:
        return 1.0
    pca = PCA(n_components=n)
    pca.fit(activation_matrix.astype(np.float32))
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_needed = int(np.searchsorted(cumvar, variance_threshold)) + 1
    return float(activation_matrix.shape[1] / n_needed)


def cross_lingual_alignment(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Mean mutual information between corresponding dimensions across two languages.
    matrix_a, matrix_b: [n_tokens, d_model] from parallel passages in lang A and B.

    High MI → dimension D encodes the same concept regardless of language
    → truly language-agnostic representation → high reward.

    Requires parallel corpora (same semantic content, different language).
    """
    from sklearn.metrics import mutual_info_score
    min_dims = min(matrix_a.shape[1], matrix_b.shape[1])
    min_len  = min(matrix_a.shape[0], matrix_b.shape[0])
    scores = []
    percentiles = np.linspace(0, 100, n_bins + 1)
    for dim in range(min_dims):
        a = matrix_a[:min_len, dim].astype(np.float32)
        b = matrix_b[:min_len, dim].astype(np.float32)
        a_bins = np.digitize(a, np.percentile(a, percentiles[1:-1]))
        b_bins = np.digitize(b, np.percentile(b, percentiles[1:-1]))
        scores.append(float(mutual_info_score(a_bins, b_bins)))
    return float(np.mean(scores)) if scores else 0.0


def coherence_reward(
    activation_matrix: np.ndarray,
    diagrams: list[np.ndarray],
    matrix_other_lang: np.ndarray = None,
    eps: float = 1e-8,
) -> dict:
    """
    Reward that approaches coherence — the state where persistence,
    predictability, compression (and cross-lingual alignment) are
    simultaneously maximized and mutually reinforcing.

    WHY geometric mean, not weighted sum:
    A weighted sum lets one strong signal compensate for a weak one.
    A model that is maximally compressed but chaotic still scores well.
    The geometric mean collapses to zero if ANY signal is zero — no
    signal can carry another. The maximum is only reached when all
    signals are jointly satisfied. That joint maximum IS coherence.

    WHY adaptive weights:
    Static weights embed a fixed opinion about what matters most.
    Instead, weights are inversely proportional to each signal's current
    saturation relative to its running maximum. The gradient always points
    toward whatever is most lacking. The fixed point where the adaptive
    weights stop shifting — where nothing is lacking — is coherence.
    The model finds it on its own.
    """
    r_persistence    = persistence_reward(diagrams)
    r_predictability = predictability_reward(activation_matrix)
    r_compression    = compression_reward(activation_matrix)
    r_alignment      = (
        cross_lingual_alignment(activation_matrix, matrix_other_lang)
        if matrix_other_lang is not None else None
    )

    signals = {
        "persistence":    r_persistence,
        "predictability": r_predictability,
        "compression":    r_compression,
    }
    if r_alignment is not None:
        signals["alignment"] = r_alignment

    # Normalize each signal to [0, 1] using a soft cap at its theoretical max.
    # Persistence and compression are unbounded; we use tanh to bring them in range.
    normed = {
        "persistence":    float(np.tanh(r_persistence / 10.0)),
        "predictability": float(np.clip(r_predictability, 0.0, 1.0)),
        "compression":    float(np.tanh(r_compression   / 100.0)),
    }
    if r_alignment is not None:
        normed["alignment"] = float(np.clip(r_alignment, 0.0, 1.0))

    # Adaptive weights: inversely proportional to current saturation.
    # High saturation → low weight (already doing well here, focus elsewhere).
    # All signals saturated → weights equal → gradient zero → coherence reached.
    raw_weights = {k: 1.0 / (v + eps) for k, v in normed.items()}
    total_w = sum(raw_weights.values())
    weights = {k: w / total_w for k, w in raw_weights.items()}

    # Weighted geometric mean: exp(sum(w_i * log(v_i))).
    # The adaptive weights make this signal converge toward coherence:
    # the gradient always points at whatever dimension is most lacking,
    # and the fixed point — where no dimension is lacking — is coherence.
    # Plain geometric mean would treat all signals equally regardless of state.
    geo_mean = float(np.exp(sum(
        weights[k] * np.log(v + eps) for k, v in normed.items()
    )))

    return {
        "total":          geo_mean,
        "normed":         normed,
        "adaptive_weights": weights,
        "raw": {
            "persistence":    r_persistence,
            "predictability": r_predictability,
            "compression":    r_compression,
            "alignment":      r_alignment,
        },
    }


# Alias so callers don't need to know the implementation detail
reward = coherence_reward
