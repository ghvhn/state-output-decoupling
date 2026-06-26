"""
Confirmation mode: given a predefined domain label, measure how
consistently the topology appears within that domain vs. outside it.

A domain signature is confirmed if the contrast ratio
(in-domain persistence / out-of-domain persistence) exceeds the threshold.
"""

import numpy as np
from tda.fingerprint import distance


def contrast_ratio(
    in_domain: list[np.ndarray],
    out_domain: list[np.ndarray],
) -> float:
    """
    Measures how much tighter the in-domain fingerprints cluster
    relative to the out-of-domain fingerprints.
    Higher ratio = more domain-specific topology.
    """
    if len(in_domain) < 2 or len(out_domain) < 2:
        return 0.0

    in_centroid = np.mean(np.stack(in_domain), axis=0)
    out_centroid = np.mean(np.stack(out_domain), axis=0)

    in_spread = np.mean([distance(v, in_centroid) for v in in_domain])
    out_spread = np.mean([distance(v, out_centroid) for v in out_domain])

    if in_spread == 0:
        return float("inf")
    return out_spread / in_spread


def confirm(
    fingerprints: dict[str, dict[str, np.ndarray]],
    domain_hint_map: dict[str, str],
    domain: str,
    band: str,
    threshold: float = 3.0,
) -> dict:
    """
    fingerprints: {conv_id: {band: vector}}
    domain_hint_map: {conv_id: domain_hint}

    Returns confirmation result including contrast ratio and verdict.
    """
    in_domain = [
        fingerprints[cid][band]
        for cid, hint in domain_hint_map.items()
        if hint == domain and cid in fingerprints and band in fingerprints[cid]
    ]
    out_domain = [
        fingerprints[cid][band]
        for cid, hint in domain_hint_map.items()
        if hint != domain and cid in fingerprints and band in fingerprints[cid]
    ]

    ratio = contrast_ratio(in_domain, out_domain)
    confirmed = ratio >= threshold

    return {
        "domain": domain,
        "band": band,
        "in_domain_count": len(in_domain),
        "out_domain_count": len(out_domain),
        "contrast_ratio": round(ratio, 4),
        "threshold": threshold,
        "confirmed": confirmed,
        "verdict": "CONFIRMED" if confirmed else "WEAK",
    }
