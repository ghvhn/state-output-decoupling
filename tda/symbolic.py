"""
Symbolic regression over flagged topological features.

TDA identifies *where* structure exists — which domain, which layer band,
which homology group. PySR finds *what governs* that structure — the actual
equation relating activation dimensions that produces the observed topology.

Workflow:
1. TDA flags a topological feature as domain-specific (high contrast ratio)
2. extract_feature_points() pulls the activation vectors that contributed
   to that feature from the persistence diagram
3. fit() runs PySR on those vectors to find the symbolic equation
4. The result is a human-readable expression: e.g. sin(d_3 * d_117) + d_402

This is what makes the topology interpretable. The TDA fingerprint tells you
a domain signature exists. The symbolic equation tells you what the model
is actually computing when it's in that domain.
"""

import numpy as np
from pysr import PySRRegressor


def extract_feature_points(
    point_cloud: np.ndarray,
    diagram: np.ndarray,
    persistence_threshold: float = 0.1,
    top_n: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a point cloud and its persistence diagram, returns the points
    most associated with the most persistent topological features.

    These are the activation vectors where the interesting structure lives —
    the input to symbolic regression.

    Returns (X, y) where:
      X: [n_points, n_dims] — activation vectors
      y: [n_points] — persistence lifetime of the feature each point contributed to
    """
    if len(diagram) == 0:
        return np.array([]), np.array([])

    lifetimes = diagram[:, 1] - diagram[:, 0]
    top_indices = np.argsort(lifetimes)[-top_n:]
    top_features = diagram[top_indices]

    X_list, y_list = [], []
    for feature in top_features:
        birth, death = feature
        lifetime = death - birth
        # Find points whose pairwise distances fall within this feature's scale
        norms = np.linalg.norm(point_cloud, axis=1)
        mask = (norms >= birth) & (norms <= death)
        if mask.sum() > 0:
            X_list.append(point_cloud[mask])
            y_list.extend([lifetime] * mask.sum())

    if not X_list:
        return np.array([]), np.array([])

    return np.vstack(X_list), np.array(y_list)


def fit(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 40,
    population_size: int = 30,
    max_complexity: int = 12,
    unary_operators: list[str] = None,
    binary_operators: list[str] = None,
) -> PySRRegressor:
    """
    Runs symbolic regression to find the equation governing the relationship
    between activation dimensions that produces the observed topological feature.

    X: activation vectors [n_points, n_dims]
    y: target values (persistence lifetimes or other topological scalars)

    Returns a fitted PySRRegressor. Access results via:
      model.sympy()   — sympy expression
      model.latex()   — LaTeX
      model.equations_ — full Pareto front of equations
    """
    if unary_operators is None:
        unary_operators = ["sin", "cos", "exp", "log", "abs", "sqrt"]
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/", "^"]

    reg = PySRRegressor(
        niterations=n_iterations,
        population_size=population_size,
        maxsize=max_complexity,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
        verbosity=0,
        random_state=42,
    )
    reg.fit(X, y)
    return reg


def summarize(reg: PySRRegressor) -> list[dict]:
    """
    Returns the Pareto front of equations ordered by complexity vs. accuracy.
    Each entry: {complexity, loss, equation_str}
    """
    results = []
    for _, row in reg.equations_.iterrows():
        results.append({
            "complexity": int(row["complexity"]),
            "loss": float(row["loss"]),
            "equation": str(row["sympy_format"]),
        })
    return sorted(results, key=lambda r: r["loss"])


def run_on_flagged(
    point_cloud: np.ndarray,
    diagrams: list[np.ndarray],
    homology_dim: int = 1,
    persistence_threshold: float = 0.1,
    n_iterations: int = 40,
) -> dict:
    """
    Full pipeline: takes a point cloud and its persistence diagrams,
    extracts feature points from the specified homology dimension,
    runs symbolic regression, returns results.

    homology_dim: 0=components, 1=loops, 2=voids
    """
    if homology_dim >= len(diagrams):
        return {"error": f"No H{homology_dim} diagram available"}

    diagram = diagrams[homology_dim]
    X, y = extract_feature_points(point_cloud, diagram, persistence_threshold)

    if len(X) == 0:
        return {"error": "No feature points above persistence threshold"}

    if X.shape[0] < 5:
        return {"error": f"Too few feature points ({X.shape[0]}) for regression"}

    reg = fit(X, y, n_iterations=n_iterations)
    equations = summarize(reg)

    return {
        "homology_dim": homology_dim,
        "n_feature_points": len(X),
        "best_equation": equations[0] if equations else None,
        "pareto_front": equations,
    }
