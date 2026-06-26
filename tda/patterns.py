"""
Comprehensive pattern detection over activation trajectories.

Covers every pattern type humans know how to detect mathematically.
Each detector operates on a 1D trajectory T_d = activations of dimension d
across token positions, or on 2D arrays of [dimensions x tokens] for
cross-dimensional patterns.

These run alongside TDA and PySR:
- TDA finds topological shape (global structure)
- PySR finds governing equations (symbolic relationships)
- This module finds specific pattern archetypes (local signatures)

All detectors return a score in [0, 1] where 1 = pattern strongly present.
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.special import entr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# ── Linear / Polynomial ──────────────────────────────────────────────────────

def linear_score(trajectory: np.ndarray) -> dict:
    """R² of linear fit. High = dimension ramps predictably across tokens."""
    t = np.arange(len(trajectory)).reshape(-1, 1)
    r2 = LinearRegression().fit(t, trajectory).score(t, trajectory)
    return {"type": "linear", "score": float(max(r2, 0))}


def polynomial_score(trajectory: np.ndarray, degree: int = 3) -> dict:
    """R² of polynomial fit up to given degree."""
    t = np.arange(len(trajectory)).reshape(-1, 1)
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("reg", LinearRegression()),
    ])
    r2 = pipe.fit(t, trajectory).score(t, trajectory)
    return {"type": f"polynomial_deg{degree}", "score": float(max(r2, 0))}


# ── Exponential / Logarithmic ─────────────────────────────────────────────────

def exponential_score(trajectory: np.ndarray) -> dict:
    """Fits y = a*exp(b*t). Returns R² of fit in log space."""
    t = np.arange(len(trajectory))
    shifted = trajectory - trajectory.min() + 1e-8
    try:
        log_y = np.log(shifted)
        r2 = LinearRegression().fit(t.reshape(-1, 1), log_y).score(t.reshape(-1, 1), log_y)
        return {"type": "exponential", "score": float(max(r2, 0))}
    except Exception:
        return {"type": "exponential", "score": 0.0}


def logarithmic_score(trajectory: np.ndarray) -> dict:
    """Fits y = a*log(t+1) + b."""
    t = np.arange(len(trajectory))
    log_t = np.log(t + 1).reshape(-1, 1)
    r2 = LinearRegression().fit(log_t, trajectory).score(log_t, trajectory)
    return {"type": "logarithmic", "score": float(max(r2, 0))}


def power_law_score(trajectory: np.ndarray) -> dict:
    """Fits y = a * t^b in log-log space."""
    t = np.arange(1, len(trajectory) + 1)
    shifted = trajectory - trajectory.min() + 1e-8
    try:
        log_t = np.log(t).reshape(-1, 1)
        log_y = np.log(shifted)
        r2 = LinearRegression().fit(log_t, log_y).score(log_t, log_y)
        return {"type": "power_law", "score": float(max(r2, 0))}
    except Exception:
        return {"type": "power_law", "score": 0.0}


# ── Step / Gate ───────────────────────────────────────────────────────────────

def heaviside_score(trajectory: np.ndarray) -> dict:
    """
    Detects discrete gate switches — sudden jumps at specific token positions.
    Score = normalized max derivative magnitude.
    """
    if len(trajectory) < 2:
        return {"type": "heaviside", "score": 0.0, "jump_at": None}
    deriv = np.abs(np.diff(trajectory))
    max_jump = deriv.max()
    std = trajectory.std() + 1e-8
    score = float(np.tanh(max_jump / std))
    jump_at = int(np.argmax(deriv))
    return {"type": "heaviside", "score": score, "jump_at": jump_at}


def phase_transition_score(trajectory: np.ndarray, window: int = 5) -> dict:
    """
    Detects bifurcations — points where local variance suddenly changes,
    indicating the dimension shifted behavioral regime.
    """
    if len(trajectory) < window * 2:
        return {"type": "phase_transition", "score": 0.0, "transition_at": None}
    variances = [
        trajectory[i:i+window].var()
        for i in range(len(trajectory) - window)
    ]
    variances = np.array(variances)
    var_deriv = np.abs(np.diff(variances))
    score = float(np.tanh(var_deriv.max() / (variances.mean() + 1e-8)))
    transition_at = int(np.argmax(var_deriv))
    return {"type": "phase_transition", "score": score, "transition_at": transition_at}


# ── Fourier / Harmonic / Wavelet ──────────────────────────────────────────────

def harmonic_score(trajectory: np.ndarray) -> dict:
    """
    FFT-based. Detects oscillatory/cyclic patterns.
    Score = ratio of energy in top frequency peak to total energy.
    High = dimension oscillates at a stable frequency (recursive counter, stack manager).
    """
    n = len(trajectory)
    if n < 4:
        return {"type": "harmonic", "score": 0.0, "dominant_freq": None}
    spectrum = np.abs(fft(trajectory - trajectory.mean()))
    spectrum = spectrum[:n // 2]
    total_energy = spectrum.sum() + 1e-8
    peak_idx = np.argmax(spectrum[1:]) + 1
    peak_energy = spectrum[peak_idx]
    freqs = fftfreq(n)[:n // 2]
    return {
        "type": "harmonic",
        "score": float(peak_energy / total_energy),
        "dominant_freq": float(freqs[peak_idx]),
    }


def wavelet_score(trajectory: np.ndarray) -> dict:
    """
    Multi-scale pattern detection using continuous wavelet transform.
    Detects patterns that appear at different resolutions — not just
    a single frequency but structure that recurs across scales.
    Score = coefficient of variation of wavelet power across scales.
    """
    if len(trajectory) < 8:
        return {"type": "wavelet", "score": 0.0}
    widths = np.arange(1, min(len(trajectory) // 2, 32))
    cwt = signal.cwt(trajectory, signal.ricker, widths)
    power = np.abs(cwt) ** 2
    scale_energy = power.mean(axis=1)
    cv = scale_energy.std() / (scale_energy.mean() + 1e-8)
    score = float(np.tanh(cv))
    return {"type": "wavelet", "score": score}


# ── Autocorrelation / Self-similarity ─────────────────────────────────────────

def autocorrelation_score(trajectory: np.ndarray, max_lag: int = None) -> dict:
    """
    Measures self-similarity across token positions.
    High = dimension repeats its pattern, suggesting recurring structure.
    Score = mean absolute autocorrelation at lags 1..max_lag.
    """
    n = len(trajectory)
    if n < 4:
        return {"type": "autocorrelation", "score": 0.0}
    max_lag = max_lag or min(n // 2, 20)
    centered = trajectory - trajectory.mean()
    var = (centered ** 2).mean() + 1e-8
    acf = [
        float(np.mean(centered[:n-lag] * centered[lag:]) / var)
        for lag in range(1, max_lag + 1)
    ]
    score = float(np.mean(np.abs(acf)))
    return {"type": "autocorrelation", "score": score, "acf": acf}


def attractor_score(trajectory: np.ndarray, n_bins: int = 10) -> dict:
    """
    Detects whether a dimension repeatedly returns to the same value ranges —
    a fixed-point or limit-cycle attractor.
    Score = inverse of spread in visited value distribution (high = concentrated returns).
    """
    if len(trajectory) < 4:
        return {"type": "attractor", "score": 0.0}
    hist, _ = np.histogram(trajectory, bins=n_bins, density=True)
    hist = hist + 1e-8
    entropy = float(entr(hist / hist.sum()).sum())
    max_entropy = np.log(n_bins)
    score = float(1.0 - entropy / max_entropy)
    return {"type": "attractor", "score": score}


# ── Information / Entropy ─────────────────────────────────────────────────────

def entropy_score(trajectory: np.ndarray, n_bins: int = 20) -> dict:
    """
    Shannon entropy of the trajectory's value distribution.
    High entropy = dimension is doing a lot (noisy or complex).
    Low entropy = dimension is focused on a narrow range (specialized).
    Returns raw entropy and normalized score (low entropy = high score = specialized).
    """
    hist, _ = np.histogram(trajectory, bins=n_bins, density=True)
    hist = hist + 1e-8
    h = float(entr(hist / hist.sum()).sum())
    max_h = np.log(n_bins)
    return {
        "type": "entropy",
        "entropy": h,
        "score": float(1.0 - h / max_h),  # high score = low entropy = specialized
    }


def mutual_information_score(a: np.ndarray, b: np.ndarray, n_bins: int = 20) -> dict:
    """
    Non-linear dependence between two dimensions.
    High = these dimensions are coupled regardless of linear correlation.
    """
    hist2d, _, _ = np.histogram2d(a, b, bins=n_bins)
    hist2d = hist2d + 1e-8
    hist2d /= hist2d.sum()
    pa = hist2d.sum(axis=1)
    pb = hist2d.sum(axis=0)
    mi = float(np.sum(hist2d * np.log(hist2d / (pa[:, None] * pb[None, :]) + 1e-8)))
    return {"type": "mutual_information", "score": float(np.tanh(mi))}


# ── Fractal ───────────────────────────────────────────────────────────────────

def fractal_dimension_score(trajectory: np.ndarray) -> dict:
    """
    Estimates Higuchi fractal dimension. Values near 1 = smooth/predictable.
    Values near 2 = complex/fractal. Used to detect self-similar structure
    at multiple scales that TDA alone may miss.
    """
    n = len(trajectory)
    if n < 8:
        return {"type": "fractal", "score": 0.0, "dimension": None}

    k_max = min(8, n // 2)
    lk = []
    for k in range(1, k_max + 1):
        lengths = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            sub = trajectory[indices]
            length = np.sum(np.abs(np.diff(sub))) * (n - 1) / (((len(sub) - 1) * k) + 1e-8)
            lengths.append(length)
        if lengths:
            lk.append(np.mean(lengths))

    if len(lk) < 2:
        return {"type": "fractal", "score": 0.0, "dimension": None}

    ks = np.log(np.arange(1, len(lk) + 1))
    ls = np.log(np.array(lk) + 1e-8)
    slope, _, r, _, _ = stats.linregress(ks, ls)
    dimension = float(-slope)
    score = float(np.clip((dimension - 1.0) / 1.0, 0, 1))
    return {"type": "fractal", "score": score, "dimension": dimension, "r": float(r)}


# ── Cross-correlation ─────────────────────────────────────────────────────────

def cross_correlation_score(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Lagged correlation between two dimensions.
    Detects one dimension leading/following another — causal flow structure.
    Returns max cross-correlation and the lag at which it occurs.
    """
    if len(a) != len(b) or len(a) < 4:
        return {"type": "cross_correlation", "score": 0.0, "lag": None}
    a_n = (a - a.mean()) / (a.std() + 1e-8)
    b_n = (b - b.mean()) / (b.std() + 1e-8)
    xcorr = np.correlate(a_n, b_n, mode="full") / len(a)
    max_idx = np.argmax(np.abs(xcorr))
    lag = int(max_idx - (len(a) - 1))
    return {
        "type": "cross_correlation",
        "score": float(np.abs(xcorr[max_idx])),
        "lag": lag,
    }


# ── Master detector ───────────────────────────────────────────────────────────

def analyze_trajectory(trajectory: np.ndarray) -> dict:
    """
    Runs all single-dimension pattern detectors on a trajectory.
    Returns a dict of all pattern scores for one dimension across tokens.
    """
    return {
        "linear": linear_score(trajectory),
        "polynomial": polynomial_score(trajectory),
        "exponential": exponential_score(trajectory),
        "logarithmic": logarithmic_score(trajectory),
        "power_law": power_law_score(trajectory),
        "heaviside": heaviside_score(trajectory),
        "phase_transition": phase_transition_score(trajectory),
        "harmonic": harmonic_score(trajectory),
        "wavelet": wavelet_score(trajectory),
        "autocorrelation": autocorrelation_score(trajectory),
        "attractor": attractor_score(trajectory),
        "entropy": entropy_score(trajectory),
        "fractal": fractal_dimension_score(trajectory),
    }


def analyze_all_dimensions(
    activation_matrix: np.ndarray,
    top_k: int = 20,
    score_threshold: float = 0.6,
) -> list[dict]:
    """
    Runs analyze_trajectory on every dimension in an activation matrix.
    activation_matrix: [n_tokens, d_model]

    Returns the top_k most interesting dimensions — those where at least
    one pattern type scores above threshold.
    """
    n_tokens, d_model = activation_matrix.shape
    results = []

    for dim in range(d_model):
        trajectory = activation_matrix[:, dim]
        patterns = analyze_trajectory(trajectory)
        max_score = max(p["score"] for p in patterns.values())
        if max_score >= score_threshold:
            dominant = max(patterns.items(), key=lambda x: x[1]["score"])
            results.append({
                "dimension": dim,
                "max_score": float(max_score),
                "dominant_pattern": dominant[0],
                "all_patterns": patterns,
            })

    results.sort(key=lambda x: x["max_score"], reverse=True)
    return results[:top_k]


def cross_dimension_analysis(
    activation_matrix: np.ndarray,
    flagged_dims: list[int],
) -> list[dict]:
    """
    Runs mutual information and cross-correlation between all pairs
    of flagged dimensions. Finds which dimensions are coupled and how.
    """
    results = []
    for i, da in enumerate(flagged_dims):
        for db in flagged_dims[i+1:]:
            a = activation_matrix[:, da]
            b = activation_matrix[:, db]
            results.append({
                "dim_a": da,
                "dim_b": db,
                "mutual_information": mutual_information_score(a, b),
                "cross_correlation": cross_correlation_score(a, b),
            })
    results.sort(key=lambda x: x["mutual_information"]["score"], reverse=True)
    return results
