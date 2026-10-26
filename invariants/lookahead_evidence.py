"""Online conditional estimates; every arrival has unit evidence weight.

No shared neural weights are updated: a sample can only affect its originating
semantic neighborhood, stream, and horizon. Centers stay fixed to prevent
chaining nearby states into an unbounded, slowly drifting environment.
"""
from __future__ import annotations

import math
import torch


class SemanticEvidence:
    def __init__(self, d_model, similarity=0.90):
        self.d_model = int(d_model)
        self.similarity = float(similarity)
        self.environments = []
        self.updates = 0
        self._center_cache = {}

    def clear_cache(self):
        self._center_cache.clear()

    @staticmethod
    def similarity_of(left, right):
        left, right = left.float().cpu(), right.float().cpu()
        denom = float(left.norm() * right.norm())
        if denom < 1e-12:
            return 1.0 if torch.equal(left, right) else -1.0
        return float(torch.dot(left, right)) / denom

    def locate(self, state, mode, namespace="default", create=False):
        if not math.isfinite(self.similarity) or not -1 <= self.similarity <= 1:
            raise ValueError("semantic similarity must be between -1 and 1")
        best = None
        state = state.detach().float()
        key = (mode, namespace, str(state.device))
        cached = self._center_cache.get(key)
        if cached is None:
            indices = [i for i, env in enumerate(self.environments)
                       if env["mode"] == mode and env["namespace"] == namespace]
            centers = torch.stack([self.environments[i]["center"] for i in indices]).to(state.device) if indices else None
            if centers is not None:
                norms = centers.norm(dim=1)
                centers = centers / norms.clamp_min(1e-12).unsqueeze(1)
            else:
                norms = None
            cached = indices, centers, norms
            self._center_cache[key] = cached
        indices, centers, norms = cached
        if centers is not None:
            norm = state.norm()
            scores = centers.mv(state / norm.clamp_min(1e-12))
            scores = torch.where((norm < 1e-12) & (norms < 1e-12), 1.0, scores)
            score, position = scores.max(0)
            if float(score) >= self.similarity:
                best = indices[int(position)]
        if best is None and create:
            best = len(self.environments)
            self.environments.append({"center": state.detach().float().cpu().clone(),
                                      "mode": mode, "namespace": namespace, "horizons": {}})
            self._center_cache.clear()
        return best

    def applies(self, environment, state, mode, namespace):
        if environment is None or state is None or not (0 <= environment < len(self.environments)):
            return False
        env = self.environments[environment]
        return (env["mode"] == mode and env["namespace"] == namespace
                and self.similarity_of(state, env["center"]) >= self.similarity)

    @staticmethod
    def _key(horizon, source_token):
        return f"{int(horizon)}:{source_token}"

    def stats(self, environment, horizon, source_token=None):
        if environment is None or not (0 <= environment < len(self.environments)):
            return None
        return self.environments[environment]["horizons"].get(self._key(horizon, source_token))

    def observe(self, environment, horizon, source, actual_state, actual_token, matched, source_token=None):
        # Correctness never filters, discounts, or negates an observation.
        if environment is None or not (0 <= environment < len(self.environments)):
            return None
        rows = self.environments[environment]["horizons"]
        row = rows.setdefault(self._key(horizon, source_token), {
            "count": 0, "state_count": 0, "matches": 0, "misses": 0,
            "mean_delta": torch.zeros(self.d_model), "m2": torch.zeros(self.d_model),
            "token_counts": {},
        })
        row["count"] += 1
        if actual_token is not None:
            token = int(actual_token)
            row["token_counts"][token] = row["token_counts"].get(token, 0) + 1
            row["matches" if matched else "misses"] += 1
        if actual_state is not None:
            change = actual_state.detach().float().cpu() - source.detach().float().cpu()
            row["state_count"] += 1
            delta = change - row["mean_delta"]
            row["mean_delta"] += delta / row["state_count"]
            row["m2"] += delta * (change - row["mean_delta"])
        self.updates += 1
        return row

    def certainty(self, environment, horizon, source_token=None):
        row = self.stats(environment, horizon, source_token)
        if row is None or row["state_count"] < 2:
            return torch.zeros(self.d_model)
        variance = row["m2"].clamp_min(0) / (row["state_count"] - 1)
        # A per-dimension dispersion score, not a claim of calibrated probability.
        return 1.0 / (1.0 + variance)

    def state_dict(self):
        return {"d_model": self.d_model, "similarity": self.similarity,
                "environments": self.environments, "updates": self.updates}

    def load_state_dict(self, state):
        if int(state["d_model"]) != self.d_model:
            raise ValueError("lookahead evidence width does not match this model")
        self.similarity = float(state["similarity"])
        self.environments = state["environments"]
        self.updates = int(state["updates"])
        self._center_cache.clear()
