"""
Latent variable detection.

Some dimensions don't just mark domains — they track continuous quantities
the model learned to represent because they're useful for prediction.

Two classes of latent variable:

PHYSICAL — measurable quantities: mass, temperature, size, time, probability.
  A "mass" dimension would activate differently for "feather" vs "boulder."

SALIENCE — human-relevant consequence. The model almost certainly doesn't
  encode raw mass. It encodes something closer to: how much would this matter
  to a person right now? A feather and a boulder differ in mass, but what the
  model actually tracks is the human-weighted consequence of each — threat,
  urgency, social weight, impact. These correlate with mass only because mass
  is a proxy for consequence. A thrown feather and a stationary boulder
  both score low on salience despite their mass difference.

  Salience is not a physical quantity. It's the projection of physical and
  social facts through a human relevance filter. The model has this because
  it was trained on human text, and humans weight everything by consequence.

This module:
1. Extracts both physical quantities and salience signals from token strings
2. Correlates each dimension against both scales independently
3. Flags dimensions encoding either type
4. Runs PySR to recover the governing function
"""

import re
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# ── Quantity extraction from text ─────────────────────────────────────────────

# Words that imply relative magnitude on a scale
MAGNITUDE_WORDS = {
    # Physical mass/weight
    "feather": 0.01, "dust": 0.02, "leaf": 0.05, "paper": 0.1,
    "pen": 0.2, "book": 0.5, "laptop": 2.0, "backpack": 5.0,
    "person": 70.0, "desk": 50.0, "car": 1500.0, "truck": 5000.0,
    "boulder": 10000.0, "mountain": 1e12,

    # Temperature
    "freezing": 0.0, "cold": 5.0, "cool": 15.0, "warm": 25.0,
    "hot": 40.0, "boiling": 100.0, "scorching": 60.0,

    # Size/scale
    "tiny": 0.01, "small": 0.1, "medium": 1.0, "large": 10.0,
    "huge": 100.0, "enormous": 1000.0, "vast": 10000.0,

    # Quantity/number feel
    "none": 0.0, "few": 2.0, "several": 5.0, "many": 20.0,
    "countless": 1000.0, "infinite": 1e9,

    # Probability/certainty
    "impossible": 0.0, "unlikely": 0.1, "possible": 0.3,
    "likely": 0.7, "probable": 0.8, "certain": 1.0, "definite": 1.0,

    # Temporal
    "instant": 0.001, "moment": 1.0, "minute": 60.0, "hour": 3600.0,
    "day": 86400.0, "week": 604800.0, "year": 3.15e7, "century": 3.15e9,

    # Comparative
    "lighter": -1.0, "heavier": 1.0, "smaller": -1.0, "larger": 1.0,
    "faster": 1.0, "slower": -1.0, "more": 1.0, "less": -1.0,
}

# Human-relevance salience scale.
# Not physical magnitude — consequence to a person in context.
# Things that are physically unrelated but share the property of mattering.
# Scale: 0.0 = irrelevant / 1.0 = maximally salient / can exceed 1.0 for extremes.
SALIENCE_WORDS = {
    # Threat / physical danger
    "bullet": 1.0, "knife": 0.9, "fire": 0.85, "explosion": 1.0,
    "poison": 0.95, "flood": 0.8, "fall": 0.75, "crash": 0.85,
    "disease": 0.8, "virus": 0.75, "injury": 0.7, "pain": 0.65,
    "death": 1.0, "murder": 1.0, "attack": 0.9, "threat": 0.8,

    # Social consequence
    "fired": 0.8, "arrested": 0.85, "divorce": 0.75, "bankrupt": 0.8,
    "shame": 0.6, "embarrassment": 0.4, "promotion": 0.5, "award": 0.45,
    "betrayal": 0.8, "lie": 0.5, "insult": 0.4, "praise": 0.3,
    "rejection": 0.6, "acceptance": 0.35, "failure": 0.55, "success": 0.4,

    # Urgency / time pressure
    "emergency": 0.95, "urgent": 0.8, "deadline": 0.6, "immediately": 0.75,
    "crisis": 0.9, "critical": 0.8, "routine": 0.1, "eventually": 0.05,
    "someday": 0.02, "never": 0.0,

    # Emotional weight
    "love": 0.7, "hate": 0.65, "grief": 0.75, "joy": 0.5,
    "fear": 0.8, "anger": 0.6, "guilt": 0.65, "pride": 0.4,
    "loneliness": 0.6, "hope": 0.45, "despair": 0.8, "relief": 0.4,

    # Proximity / directness
    "me": 0.9, "you": 0.8, "us": 0.75, "them": 0.2,
    "here": 0.85, "now": 0.9, "today": 0.7, "soon": 0.6,
    "there": 0.3, "then": 0.2, "distant": 0.1, "abstract": 0.05,

    # Irreversibility
    "permanent": 0.85, "forever": 0.9, "irreversible": 0.95,
    "temporary": 0.2, "fixable": 0.1, "recoverable": 0.15,
    "undo": 0.1, "delete": 0.5, "destroy": 0.9, "create": 0.3,

    # Objects by their consequence, not their mass
    # (a feather thrown at your face and a boulder in a museum differ here)
    "feather": 0.05,     # low consequence unless thrown hard
    "paper": 0.03,       # cut is surprising but minor
    "pen": 0.15,         # could hurt if aimed
    "rock": 0.55,        # depends on context
    "boulder": 0.3,      # usually stationary — low active threat
    "bullet": 1.0,       # always high consequence
    "needle": 0.5,       # small but penetrating
    "sword": 0.85,
    "shield": 0.2,       # protective, lowers consequence
    "helmet": 0.15,
    "pillow": 0.01,
    "blanket": 0.01,
    "bomb": 1.0,
    "match": 0.4,        # potential to ignite
    "water": 0.1,        # usually safe, context-dependent
    "acid": 0.95,

    # Null salience baseline
    "nothing": 0.0, "irrelevant": 0.0, "ignore": 0.0,
    "fine": 0.05, "okay": 0.05, "normal": 0.05,
}

NUMBER_PATTERN = re.compile(
    r"""
    (?:
        (\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)  # explicit number
        |
        (one|two|three|four|five|six|seven|eight|nine|ten|
         eleven|twelve|hundred|thousand|million|billion)  # word number
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "hundred": 100, "thousand": 1000,
    "million": 1e6, "billion": 1e9,
}


def extract_quantities(tokens_str: list[str]) -> dict[str, list[tuple[int, float]]]:
    """
    Scans token strings for continuous quantities — both physical and salience.
    Returns {quantity_type: [(token_position, value), ...]}

    quantity_type: "numeric", "magnitude", "salience"

    Physical magnitude and salience are tracked separately because a dimension
    may correlate with one but not the other — that distinction is the finding.
    A dimension correlated with magnitude but not salience is tracking physics.
    A dimension correlated with salience but not magnitude is tracking consequence.
    A dimension correlated with both probably exists — but the salience version
    is the more human-relevant one.
    """
    results = {
        "numeric": [],
        "magnitude": [],
        "salience": [],
    }

    for i, tok in enumerate(tokens_str):
        tok_lower = tok.lower().strip()

        # Explicit numbers
        m = NUMBER_PATTERN.match(tok_lower)
        if m:
            if m.group(1):
                results["numeric"].append((i, float(m.group(1))))
            elif m.group(2):
                results["numeric"].append((i, float(WORD_NUMBERS[m.group(2).lower()])))

        # Physical magnitude
        if tok_lower in MAGNITUDE_WORDS:
            results["magnitude"].append((i, MAGNITUDE_WORDS[tok_lower]))

        # Human-relevant salience
        if tok_lower in SALIENCE_WORDS:
            results["salience"].append((i, SALIENCE_WORDS[tok_lower]))

    return {k: v for k, v in results.items() if v}


# ── Correlation between dimensions and extracted quantities ───────────────────

def correlate_dimension_to_quantity(
    activation_trajectory: np.ndarray,
    quantity_positions: list[tuple[int, float]],
) -> dict:
    """
    Correlates a single dimension's activation trajectory against a
    set of (token_position, quantity_value) pairs.

    Returns Pearson r, Spearman r, and p-values.
    High correlation = this dimension is tracking the quantity.
    """
    if len(quantity_positions) < 3:
        return {"r_pearson": 0.0, "r_spearman": 0.0, "p": 1.0, "n": len(quantity_positions)}

    positions, values = zip(*quantity_positions)
    positions = list(positions)
    values = list(values)

    # Clamp positions to trajectory length
    valid = [(p, v) for p, v in zip(positions, values) if p < len(activation_trajectory)]
    if len(valid) < 3:
        return {"r_pearson": 0.0, "r_spearman": 0.0, "p": 1.0, "n": len(valid)}

    positions, values = zip(*valid)
    acts = activation_trajectory[list(positions)]
    values = np.array(values)

    r_pearson, p_pearson = stats.pearsonr(acts, values)
    r_spearman, p_spearman = stats.spearmanr(acts, values)

    return {
        "r_pearson": float(r_pearson),
        "r_spearman": float(r_spearman),
        "p_pearson": float(p_pearson),
        "p_spearman": float(p_spearman),
        "n": len(valid),
    }


def find_latent_variable_dimensions(
    activation_matrix: np.ndarray,
    token_strings: list[str],
    r_threshold: float = 0.6,
    p_threshold: float = 0.05,
) -> list[dict]:
    """
    activation_matrix: [n_tokens, d_model]
    token_strings: the string form of each token

    Returns list of dimensions that significantly correlate with
    any extracted continuous quantity.
    """
    quantities = extract_quantities(token_strings)
    if not quantities:
        return []

    n_tokens, d_model = activation_matrix.shape
    flagged = []

    for dim in range(d_model):
        trajectory = activation_matrix[:, dim]
        best = None

        for qty_type, positions_values in quantities.items():
            result = correlate_dimension_to_quantity(trajectory, positions_values)
            r = max(abs(result["r_pearson"]), abs(result["r_spearman"]))
            p = min(result.get("p_pearson", 1.0), result.get("p_spearman", 1.0))

            if r >= r_threshold and p <= p_threshold:
                if best is None or r > best["r"]:
                    best = {
                        "dimension": dim,
                        "quantity_type": qty_type,
                        "r": r,
                        "correlation": result,
                        "n_samples": result["n"],
                    }

        if best:
            flagged.append(best)

    flagged.sort(key=lambda x: x["r"], reverse=True)
    return flagged


# ── PySR integration: recover the function ────────────────────────────────────

def fit_latent_function(
    activation_trajectory: np.ndarray,
    quantity_positions: list[tuple[int, float]],
    n_iterations: int = 40,
) -> dict:
    """
    Given a dimension that correlates with a quantity, uses PySR to
    recover the exact function: activation = f(quantity_value).

    Returns the symbolic equation and Pareto front.
    """
    from pysr import PySRRegressor

    valid = [(p, v) for p, v in quantity_positions if p < len(activation_trajectory)]
    if len(valid) < 5:
        return {"error": f"Too few samples ({len(valid)}) for symbolic regression"}

    positions, values = zip(*valid)
    X = np.array(values).reshape(-1, 1)
    y = activation_trajectory[list(positions)]

    reg = PySRRegressor(
        niterations=n_iterations,
        population_size=30,
        maxsize=10,
        unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt"],
        binary_operators=["+", "-", "*", "/", "^"],
        verbosity=0,
        random_state=42,
    )
    reg.fit(X, y)

    best = reg.equations_.iloc[reg.equations_["loss"].argmin()]
    return {
        "equation": str(best["sympy_format"]),
        "loss": float(best["loss"]),
        "complexity": int(best["complexity"]),
        "pareto_front": [
            {"equation": str(row["sympy_format"]),
             "loss": float(row["loss"]),
             "complexity": int(row["complexity"])}
            for _, row in reg.equations_.iterrows()
        ],
    }


# ── Master function ───────────────────────────────────────────────────────────

def analyze(
    activation_matrix: np.ndarray,
    token_strings: list[str],
    r_threshold: float = 0.6,
    p_threshold: float = 0.05,
    run_pysr: bool = True,
    pysr_iterations: int = 40,
    run_discovery: bool = True,
) -> list[dict]:
    """
    Full latent variable analysis:
    1. Extract quantities from token strings
    2. Find dimensions that correlate with those quantities
    3. Optionally run PySR to recover the governing function

    Returns list of latent variable findings, each with:
    - dimension index
    - quantity type being tracked
    - correlation strength
    - symbolic function (if PySR enabled)
    """
    from tda.discovery import discover, compare_to_known, to_vocabulary

    quantities = extract_quantities(token_strings)

    # Dynamically discover new axes and merge into quantities
    if run_discovery:
        discovered = discover(activation_matrix, token_strings)
        compared = compare_to_known(
            discovered,
            {"magnitude": MAGNITUDE_WORDS, "salience": SALIENCE_WORDS}
        )
        novel = [c for c in compared if c.get("novel") and c.get("is_scale")]
        if novel:
            dynamic_vocab = to_vocabulary(novel)
            # Scan for tokens from the dynamic vocab in this conversation
            dynamic_hits = []
            for i, tok in enumerate(token_strings):
                tok_lower = tok.lower().strip()
                if tok_lower in dynamic_vocab:
                    dynamic_hits.append((i, dynamic_vocab[tok_lower]))
            if dynamic_hits:
                quantities["discovered"] = dynamic_hits

    flagged = find_latent_variable_dimensions(
        activation_matrix, token_strings, r_threshold, p_threshold
    )

    # Tag each finding with whether it came from a known or discovered axis
    for f in flagged:
        f["axis_source"] = "known" if f["quantity_type"] in ("magnitude", "salience", "numeric") \
                           else "discovered"

    if not flagged or not run_pysr:
        return flagged

    for finding in flagged:
        dim = finding["dimension"]
        qty_type = finding["quantity_type"]
        positions_values = quantities.get(qty_type, [])

        if positions_values:
            finding["symbolic_function"] = fit_latent_function(
                activation_matrix[:, dim],
                positions_values,
                n_iterations=pysr_iterations,
            )

    return flagged
