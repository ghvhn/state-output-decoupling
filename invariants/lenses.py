"""
Lenses — the pluggable MEASURE axis. Each lens detects one FAMILY of structure
in the difference between two matched activation sets. The engine runs every
registered lens on every transformation, each against its own null.

A pattern only "exists" relative to a named lens (design-consequence A, one level
up: the lens is the operational definition of "pattern"). And every lens is one
more independent chance to find structure in noise (design-consequence C), so
each clears its OWN null — generality is bought with stricter floors, not free.

A new pattern someone posits = a new Lens here. The engine never changes.

NOTE: lenses report the FORM of the change (translation / reallocation /
distributional / topological). They do NOT decide preserve-vs-break — a
constraint (a consistent translation) and a bridge (a consistent rotation) are
both "consistent maps." Preserve-vs-break is the CAUSAL verdict (ablate the
signature, does the conclusion move?), handled in engine.causal_effect.
"""

import torch


def _pca_reduce(X, k):
    Xc = X - X.mean(0)
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Xc @ Vt[:k].T


def _mmd2_rbf(A, B):
    Z = torch.cat([A, B], 0)
    d2 = torch.cdist(Z, Z).pow(2)
    pos = d2[d2 > 0]
    med = pos.median().clamp_min(1e-8) if pos.numel() else torch.tensor(1.0)
    K = torch.exp(-d2 / med)
    n = A.shape[0]
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def direction_at(A, B):
    """Unit mean-difference direction at one layer (the translation signature)."""
    d = A.mean(0) - B.mean(0)
    return d / d.norm().clamp_min(1e-8)


class Lens:
    name = "lens"; family = "generic"; paired = False
    def score(self, A, B) -> float:
        raise NotImplementedError


class MeanShift(Lens):
    """Translation: the centroid slides along one axis (Arditi-style)."""
    name = "mean_shift"; family = "translation"; paired = False

    def score(self, A, B):
        u = direction_at(A, B)
        pa, pb = A @ u, B @ u
        pooled = torch.sqrt(0.5 * (pa.var(unbiased=False) + pb.var(unbiased=False))).clamp_min(1e-8)
        return float((pa.mean() - pb.mean()).abs() / pooled)   # Cohen's d


class Reallocation(Lens):
    """
    Axis-reallocation: the concept re-encoded onto different dimensions — a
    ROTATION that a mean-shift can't see. Score = how much a fitted shared
    rotation improves alignment of matched pairs OVER identity (translation is
    removed by centering first). ~0 for a pure translation, large for genuine
    re-basing. This is the thing MeanShift structurally misses.
    """
    name = "reallocation"; family = "reallocation"; paired = True

    def __init__(self, max_dims=24):
        self.max_dims = max_dims

    def score(self, A, B):
        n = A.shape[0]
        k = min(self.max_dims, max(2, n - 2), A.shape[1])
        ABr = _pca_reduce(torch.cat([A, B], 0), k)
        Ar, Br = ABr[:n], ABr[n:]
        Ac, Bc = Ar - Ar.mean(0), Br - Br.mean(0)
        total = Bc.pow(2).sum().clamp_min(1e-8)
        resid_I = (Ac - Bc).pow(2).sum()                 # alignment under identity
        U, _, Vt = torch.linalg.svd(Ac.T @ Bc)
        R = U @ Vt                                        # best shared rotation
        resid_R = (Ac @ R - Bc).pow(2).sum()
        return float((resid_I - resid_R) / total)        # rotation's improvement


class Distributional(Lens):
    """Any distributional difference linear methods miss (kernel MMD)."""
    name = "mmd"; family = "distributional"; paired = False

    def score(self, A, B):
        return _mmd2_rbf(A, B)


class Topology(Lens):
    """
    Shape / connectivity (persistent homology) — the broadest geometric lens,
    and the project's original tool. Needs many points, so it activates only
    with per-token clouds, not one vector per condition item. Slot is wired;
    switch on by feeding per-token activations.
    """
    name = "topology"; family = "topological"; paired = False
    min_points = 40

    def score(self, A, B):
        if A.shape[0] < self.min_points:
            raise RuntimeError(
                f"topology needs >= {self.min_points} points (have {A.shape[0]}); "
                "use per-token extraction")
        import numpy as np
        from tda.homology import run as run_homology, persistent_features
        from tda.fingerprint import from_diagrams

        def fp(X):
            Xn = torch.nan_to_num(X).cpu().numpy().astype("float32")
            Xn = Xn[np.linalg.norm(Xn, axis=1) > 1e-6]   # drop ~zero rows (cosine NaN)
            dgms = persistent_features(
                run_homology(Xn, maxdim=1, metric="cosine"), threshold=0.1)
            return np.asarray(from_diagrams(dgms), dtype="float32")

        a, b = fp(A), fp(B)
        m = min(len(a), len(b))
        return float(np.linalg.norm(a[:m] - b[:m]))


# The registry. Add a phenomenon's detector here; the engine is untouched.
LENSES = [MeanShift(), Reallocation(), Distributional(), Topology()]
