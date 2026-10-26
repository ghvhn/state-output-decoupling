"""Compare hidden, short continuations before committing their token prefix.

The objective is an arithmetic change in measured probe scores. It does not
establish honesty, understanding, or the semantic quality of a continuation.
Discarded branches are operator artifacts and never enter the selected prefix.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import torch

from invariants import engine
from invariants.engine import _steer_handles, steer_band_layers
from invariants.generation_budget import check_context_budget
from invariants.generation_probe_capture import (GenerationProbeCapture,
                                                 GENERATION_PROBE_ENDPOINT)
from invariants.indirect_steering import _alias, _objectives, definition_fingerprints
from invariants.replay_investigation import (_hash, _hook_ids,
    _isolated_steer_telemetry, _model_hash, _prepare_helpers, _preserve_rng,
    _remove_new_hooks, _seed, _write_json, model_identity)


SOURCE = "controlled_continuation_investigation"
FINAL_PREFIX = "__continuation_final__"


ALLOWED_CONTINUATION_DISTRIBUTIONS = (
    "grid", "flexible", "continuous", "dirichlet", "adaptive",
    "softmax", "orthogonal", "subspace", "layer_tilted", "uniform",
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _temper_simplex(weights, temperature=1.0, priors=None):
    values = [max(float(w), 1e-6) for w in weights]
    if priors is not None and len(priors) == len(values):
        values = [w * max(float(p), 1e-6) for w, p in zip(values, priors)]
    if temperature != 1.0:
        exponent = max(min(1.0 / float(temperature), 20.0), 0.05)
        values = [w ** exponent for w in values]
    total = sum(values)
    if total <= 0 or not math.isfinite(total):
        return (1.0 / len(weights),) * len(weights)
    raw = [max(v / total, 1e-4) for v in values]
    norm = sum(raw)
    rounded = [round(v / norm, 6) for v in raw]
    peak = max(range(len(rounded)), key=rounded.__getitem__)
    rounded[peak] = round(max(1e-4, 1.0 - sum(rounded[i] for i in range(len(rounded)) if i != peak)), 6)
    return tuple(rounded)


def _continuous_simplex_profiles(size, count, *, resolution=4, distribution="flexible",
                                 temperature=1.0, seed=0, priors=None):
    """Yield continuous, prior-guided, and temperature-scaled open-simplex profiles."""
    uniform = (1. / size,) * size
    if distribution == "uniform":
        yield uniform
        return
    seen = set()

    def emit(weights):
        profile = tuple(float(w) for w in weights)
        key = tuple(round(w, 5) for w in profile)
        if any(w <= 0 for w in profile) or not math.isclose(sum(profile), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            return
        if key not in seen:
            seen.add(key)
            yield profile

    if priors is not None and len(priors) == size and len({round(float(p), 6) for p in priors}) > 1:
        yield from emit(_temper_simplex(uniform, temperature=temperature, priors=priors))

    dom = max(1.0 / size + 0.25, min(0.85, (max(resolution, size + 1) - size + 1) / max(resolution, size + 1)))
    rest = (1.0 - dom) / max(1, size - 1)
    for index in range(size):
        base = tuple(dom if pos == index else rest for pos in range(size))
        yield from emit(_temper_simplex(base, temperature=temperature, priors=priors))
        if index == 0:
            yield from emit(uniform)
            if temperature != 1.0 or priors is not None:
                yield from emit(_temper_simplex(uniform, temperature=temperature, priors=priors))

    eff_res = max(int(resolution), size + 1) if distribution != "grid" else int(resolution)
    if eff_res > size and eff_res <= 16:
        for cuts in itertools.combinations(range(1, eff_res), size - 1):
            bounds = (0, *cuts, eff_res)
            grid_w = tuple((right - left) / eff_res for left, right in zip(bounds, bounds[1:]))
            yield from emit(_temper_simplex(grid_w, temperature=temperature, priors=priors))
            if len(seen) >= count:
                return

    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    step = 1
    while len(seen) < count and step <= count * 8 + 32:
        exponentials = []
        for coord in range(size):
            base_p = primes[coord % len(primes)]
            # Halton / golden-ratio quasi-random draw in (0, 1) mapped to Exponential(1)
            n_val, denom, radical = step + int(seed) * 13 + coord * 7, 1.0, 0.0
            while n_val > 0:
                denom *= base_p
                radical += (n_val % base_p) / denom
                n_val //= base_p
            u = min(max(radical, 1e-4), 1.0 - 1e-4)
            exponentials.append(-math.log(u))
        yield from emit(_temper_simplex(exponentials, temperature=temperature, priors=priors))
        step += 1


def _mixture_profile_count(size, resolution, *, distribution="grid", priors=None):
    has_priors = priors is not None and len({round(float(p), 6) for p in priors}) > 1
    if distribution == "uniform":
        return 1
    if distribution == "grid" and not has_priors:
        if resolution == 1 or size > resolution:
            return 1
        # The uniform profile is included even when it falls between grid points.
        return math.comb(resolution - 1, size - 1) + int(resolution % size != 0)
    eff_res = max(int(resolution), size + 2)
    grid_count = math.comb(min(eff_res, 12) - 1, size - 1) + int(eff_res % size != 0)
    return max(grid_count, size + 2 + int(has_priors))


def _mixture_profiles(size, resolution, *, distribution="grid", temperature=1.0,
                      seed=0, priors=None):
    """Yield a simplex grid or continuous/prior-tempered simplex stream lazily.

    Every grid component is positive; single-helper candidates cover boundaries.
    Uniform is interleaved after the first asymmetric profile when available.
    At most ``size + 1`` profiles are held to avoid repeating this initial set.
    """
    has_priors = priors is not None and len({round(float(p), 6) for p in priors}) > 1
    if distribution == "grid" and temperature == 1.0 and not has_priors:
        uniform = (1. / size,) * size
        if resolution == 1 or size > resolution or resolution == size:
            yield uniform
            return
        seen = set()
        uniform_grid = ((resolution // size,) * size if resolution % size == 0 else None)
        for index in range(size):
            weights = tuple(resolution - size + 1 if position == index else 1
                            for position in range(size))
            if weights not in seen:
                seen.add(weights)
                yield tuple(weight / resolution for weight in weights)
            if index == 0:
                if uniform_grid is not None:
                    seen.add(uniform_grid)
                yield uniform
        for cuts in itertools.combinations(range(1, resolution), size - 1):
            bounds = (0, *cuts, resolution)
            weights = tuple(right - left for left, right in zip(bounds, bounds[1:]))
            if weights not in seen:
                yield tuple(weight / resolution for weight in weights)
        return
    target_count = _mixture_profile_count(size, resolution, distribution=distribution, priors=priors)
    yield from _continuous_simplex_profiles(
        size, target_count, resolution=resolution, distribution=distribution,
        temperature=temperature, seed=seed, priors=priors)


def _synthesize_orthogonal_vectors(helpers, indices, combination_signs, weights, layers,
                                   objective_vectors=None, layer_weights_map=None):
    """Synthesize a frameless unit-norm orthogonalized subspace direction per layer."""
    vectors = {}
    for layer in layers:
        basis = []
        for pos, (index, sign) in enumerate(zip(indices, combination_signs)):
            raw = helpers[index]["vectors"][layer].detach().float().clone() * float(sign)
            for prev in basis:
                denom = float(torch.dot(prev, prev))
                if denom > 1e-12:
                    raw = raw - (float(torch.dot(raw, prev)) / denom) * prev
            norm = float(raw.norm())
            basis.append(raw / norm if norm > 1e-6 else helpers[index]["vectors"][layer].detach().float().clone() * float(sign))
        lw = layer_weights_map[layer] if layer_weights_map and layer in layer_weights_map else weights
        combined = sum(vec * float(w) for vec, w in zip(basis, lw))
        for obj_dir in (objective_vectors or ()):
            if layer in obj_dir:
                t_vec = torch.as_tensor(obj_dir[layer], dtype=combined.dtype, device=combined.device).reshape(-1)
                t_norm = float(t_vec.norm())
                if t_norm > 1e-6:
                    t_unit = t_vec / t_norm
                    combined = combined - float(torch.dot(combined, t_unit)) * t_unit
        c_norm = float(combined.norm())
        vectors[layer] = combined / c_norm if c_norm > 1e-6 else combined
    return vectors


def _compute_arm_layer_weights(helpers, indices, weights, layers, tilt=0.0):
    """Compute per-layer simplex weights across depth while preserving unit L1 sum per layer."""
    size = len(indices)
    has_helper_lw = any(isinstance(helpers[idx].get("layer_weights"), dict) and helpers[idx]["layer_weights"]
                        for idx in indices)
    if (not tilt and not has_helper_lw) or len(layers) <= 1 or size <= 1:
        return None
    per_layer = {}
    for layer_pos, layer in enumerate(layers):
        depth = (layer_pos / (len(layers) - 1)) - 0.5
        raw = []
        for pos, (idx, w) in enumerate(zip(indices, weights)):
            carrier = ((pos / (size - 1)) - 0.5) * 2.0
            explicit = 1.0
            hlw = helpers[idx].get("layer_weights")
            if isinstance(hlw, dict):
                explicit = max(float(hlw.get(layer, hlw.get(str(layer), 1.0))), 1e-4)
            factor = math.exp(float(tilt) * depth * carrier) if tilt else 1.0
            raw.append(max(float(w) * explicit * factor, 1e-6))
        total = sum(raw)
        per_layer[layer] = tuple(val / total for val in raw) if total > 0 else weights
    return per_layer


def _build_candidate_arms(helpers, layers, max_joint_helpers=2, max_candidates=33,
                          mixture_resolution=4, *, distribution="grid", temperature=1.0,
                          seed=0, priors=None, compose_novel=False, layer_tilt=0.0,
                          objective_vectors=None):
    """Build bounded, deterministic single-axis and simultaneous helper trials.

    Every single-helper sign remains available. Joint arms distribute one unit
    L1 coefficient budget over positive simplex weights before the shared cap.
    Resolution one retains the balanced-only candidate family. Construction
    stores only profiles/subsets reached within the candidate cap, even when
    the full grid or helper-subset family is combinatorially large.
    """
    _positive_integer(max_joint_helpers, "max_joint_helpers")
    _positive_integer(max_candidates, "max_candidates")
    _positive_integer(mixture_resolution, "mixture_resolution")
    if mixture_resolution > 64:
        raise ValueError("mixture_resolution must be between 1 and 64")
    if distribution not in ALLOWED_CONTINUATION_DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {', '.join(ALLOWED_CONTINUATION_DISTRIBUTIONS)}")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    if not isinstance(compose_novel, bool):
        raise ValueError("compose_novel must be a boolean")
    tilt_value = 0.35 if layer_tilt is True else (0.0 if layer_tilt is False else float(layer_tilt))
    if not math.isfinite(tilt_value) or abs(tilt_value) > 2.0:
        raise ValueError("layer_tilt must be a finite number in [-2.0, 2.0]")
    if distribution in {"orthogonal", "subspace"}:
        compose_novel = True
    if distribution == "layer_tilted" and tilt_value == 0.0:
        tilt_value = 0.35
    if distribution in {"flexible", "adaptive"}:
        compose_novel = True
        if tilt_value == 0.0 and len(layers) > 1:
            tilt_value = 0.3

    has_helper_priors = bool(priors) or any(h.get("prior_weight") is not None for h in helpers)
    has_helper_lw = any(isinstance(h.get("layer_weights"), dict) and h["layer_weights"] for h in helpers)
    legacy_grid = (distribution == "grid" and float(temperature) == 1.0
                   and not has_helper_priors and not compose_novel
                   and tilt_value == 0.0 and not has_helper_lw)

    signs = [((helper["sign"],) if helper["sign"] is not None else (1, -1))
             for helper in helpers]
    required = 1 + sum(len(values) for values in signs)
    if max_candidates < required:
        raise ValueError(f"max_candidates must be at least {required} to retain baseline and all single-helper signs")
    largest = min(max_joint_helpers, len(helpers))
    # Elementary symmetric counts avoid enumerating an unbounded Cartesian
    # product just to describe how many possible candidates were omitted.
    counts = [1] + [0] * largest
    for values in signs:
        for size in range(largest, 0, -1):
            counts[size] += len(values) * counts[size - 1]
    if legacy_grid:
        profile_counts = {size: _mixture_profile_count(size, mixture_resolution)
                          for size in range(2, largest + 1)}
    else:
        default_priors = tuple(float((priors or {}).get(h["name"], h.get("prior_weight") or 1.0)) for h in helpers)
        novel_factor = 2 if compose_novel else 1
        profile_counts = {
            size: _mixture_profile_count(
                size, mixture_resolution, distribution=distribution,
                priors=default_priors[:size] if has_helper_priors else None) * novel_factor
            for size in range(2, largest + 1)}
    total_possible = sum(count * profile_counts.get(size, 1)
                         for size, count in enumerate(counts))
    arms = [{"arm": "baseline", "helper": None, "sign": 0, "vectors": {},
             "components": [], "joint_size": 0}]
    for helper, allowed in zip(helpers, signs):
        for sign in allowed:
            arms.append({"arm": f"{helper['name']}:{'+' if sign > 0 else '-'}",
                "helper": helper["name"], "sign": sign, "vectors": helper["vectors"],
                "components": [{"helper": helper["name"], "sign": sign,
                                "coefficient": float(sign)}], "joint_size": 1})

    def subset_stream(indices, subset_rank, profiles, get_profile):
        allowed = [signs[index] for index in indices]
        sign_count = math.prod(map(len, allowed))
        base_profiles = max(1, profiles // (2 if (compose_novel and not legacy_grid) else 1))
        for trial in range(sign_count * base_profiles):
            sign_rank, profile_round = trial % sign_count, trial // sign_count
            residual, combination_signs = sign_rank, []
            for values in reversed(allowed):
                combination_signs.append(values[residual % len(values)])
                residual //= len(values)
            combination_signs.reverse()
            weights = get_profile((profile_round + sign_rank + subset_rank) % base_profiles)
            layer_w = (None if legacy_grid else
                       _compute_arm_layer_weights(helpers, indices, weights, layers, tilt=tilt_value))
            components = []
            for pos, (index, sign, weight) in enumerate(zip(indices, combination_signs, weights)):
                comp = {"helper": helpers[index]["name"], "sign": sign,
                        "coefficient": sign * weight}
                if layer_w is not None:
                    comp["layer_weights"] = {int(layer): round(sign * layer_w[layer][pos], 6)
                                             for layer in layers}
                components.append(comp)
            if layer_w is not None:
                vectors = {layer: sum((helpers[index]["vectors"][layer] * (sign * layer_w[layer][pos])
                                       for pos, (index, sign) in enumerate(zip(indices, combination_signs))))
                           for layer in layers}
            else:
                vectors = {layer: sum((helpers[index]["vectors"][layer] * (sign * weight)
                                       for index, sign, weight in zip(indices, combination_signs, weights)))
                           for layer in layers}
            balanced = all(weight == weights[0] for weight in weights)
            label = ",".join(f"{item['helper']}:{'+' if item['sign'] > 0 else '-'}"
                             + ("" if balanced else f"@{abs(item['coefficient']):.12g}")
                             for item in components)
            arm_entry = {"arm": f"joint({label})", "helper": None, "sign": 1,
                         "vectors": vectors, "components": components, "joint_size": len(indices)}
            if layer_w is not None:
                arm_entry["layer_tilted"] = True
            yield arm_entry
            if compose_novel and not legacy_grid:
                ortho_vectors = _synthesize_orthogonal_vectors(
                    helpers, indices, combination_signs, weights, layers,
                    objective_vectors=objective_vectors, layer_weights_map=layer_w)
                ortho_components = [dict(item, composition_mode="orthogonal") for item in components]
                yield {"arm": f"novel_ortho({label})", "helper": None, "sign": 1,
                       "vectors": ortho_vectors, "components": ortho_components,
                       "joint_size": len(indices), "novel": True,
                       "composition_mode": "orthogonal", "framings": None}

    def size_stream(size):
        # Cache only requested profile indices. Even a huge grid is never
        # materialized to schedule a small capped candidate search.
        streams = []
        for subset_rank, indices in enumerate(itertools.combinations(range(len(helpers)), size)):
            subset_priors = (tuple(float((priors or {}).get(helpers[i]["name"], helpers[i].get("prior_weight") or 1.0))
                                   for i in indices)
                             if has_helper_priors else None)
            iterator = iter(_mixture_profiles(
                size, mixture_resolution, distribution=distribution,
                temperature=temperature, seed=seed + subset_rank, priors=subset_priors))
            cache = []

            def get_profile(index, _it=iterator, _cache=cache, _size=size):
                while len(_cache) <= index:
                    nxt = next(_it, None)
                    _cache.append(nxt if nxt is not None else (1. / _size,) * _size)
                return _cache[index]

            stream = subset_stream(indices, subset_rank, profile_counts[size], get_profile)
            streams.append(stream)
            first_item = next(stream, None)
            if first_item is not None:
                yield first_item
        while streams:
            remaining = []
            for stream in streams:
                candidate = next(stream, None)
                if candidate is not None:
                    remaining.append(stream)
                    yield candidate
            streams = remaining

    def balanced_stream():
        # Resolution one also retains the historical balanced candidate order.
        for size in range(2, largest + 1):
            for subset_rank, indices in enumerate(itertools.combinations(range(len(helpers)), size)):
                yield from subset_stream(indices, subset_rank, 2 if (compose_novel and not legacy_grid) else 1,
                                         lambda _, count=size: (1. / count,) * count)

    # Alternate joint sizes so a small cap can still test triples, then cycle
    # helper subsets within each size. Signs and profiles advance diagonally
    # inside each subset so one profile does not consume every sign slot.
    use_balanced = (mixture_resolution == 1 and legacy_grid) or distribution == "uniform"
    streams = ([iter(balanced_stream())] if use_balanced else
               [iter(size_stream(size)) for size in range(2, largest + 1)])
    while streams and len(arms) < max_candidates:
        remaining = []
        for stream in streams:
            if len(arms) >= max_candidates:
                break
            candidate = next(stream, None)
            if candidate is not None:
                arms.append(candidate)
                remaining.append(stream)
        streams = remaining
    selected_counts = {size: sum(arm["joint_size"] == size for arm in arms)
                       for size in range(largest + 1)}
    selected_profiles = {size: len({tuple(abs(item["coefficient"]) for item in arm["components"])
                                    for arm in arms if arm["joint_size"] == size})
                         for size in range(2, largest + 1)}
    plan = {"total_possible": total_possible, "selected": len(arms),
            "truncated": len(arms) < total_possible,
            "required_single_candidates": required,
            "max_joint_helpers": max_joint_helpers, "effective_max_joint_helpers": largest,
            "max_candidates": max_candidates, "mixture_resolution": mixture_resolution,
            "profile_counts_by_joint_size": profile_counts,
            "selected_by_joint_size": selected_counts,
            "selected_profile_counts_by_joint_size": selected_profiles,
            "priority": ("baseline and all singles; historical joint size/helper subset/sign order"
                         if mixture_resolution == 1 and legacy_grid else
                         "baseline and all singles; round-robin joint sizes and lexicographic helper subsets; signs cycle with diagonally rotated profile indices; asymmetric extreme profiles then uniform then other simplex points")}
    if not legacy_grid:
        plan.update({
            "distribution": distribution,
            "temperature": float(temperature),
            "compose_novel": bool(compose_novel),
            "novel_arms_selected": sum(1 for arm in arms if arm.get("novel")),
            "layer_tilt": float(tilt_value),
        })
    return arms, plan


def _filter_joint_objective_aliases(arms, probes, objectives, candidate_plan):
    """Keep joint construction from recreating a directly blocked target axis.

    The comparison intentionally uses only directions at shared intervention
    layers. No overlap supplies no evidence of an alias, and a zero combined
    vector remains a harmless baseline-equivalent candidate. Filtering the
    bounded set does not scan further combinations to refill omitted arms.
    """
    kept, blocked = [], []
    for arm in arms:
        targets = [name for name in objectives if arm["joint_size"] > 1
                   and _alias({"direction": arm["vectors"]},
                              {"direction": probes[name].get("direction", {})})]
        if targets:
            blocked.append({"arm": arm["arm"], "objectives": targets,
                "reason": "combined helper direction aliases or inverts objective on shared intervention layers"})
        else:
            kept.append(arm)
    constructed = len(arms)
    return kept, {**candidate_plan, "selected": len(kept),
        "constructed_candidates": constructed, "effective_candidates": len(kept),
        "constructed_by_joint_size": candidate_plan["selected_by_joint_size"],
        "selected_by_joint_size": {size: sum(arm["joint_size"] == size for arm in kept)
                                   for size in candidate_plan["selected_by_joint_size"]},
        "constructed_profile_counts_by_joint_size": candidate_plan["selected_profile_counts_by_joint_size"],
        "selected_profile_counts_by_joint_size": {
            size: len({tuple(abs(item["coefficient"]) for item in arm["components"])
                       for arm in kept if arm["joint_size"] == size})
            for size in candidate_plan["selected_profile_counts_by_joint_size"]},
        "blocked_joint_arms": blocked, "blocked_joint_count": len(blocked),
        "unexamined_candidates": candidate_plan["total_possible"] - constructed,
        "coverage": "bounded candidates filtered for direct objective aliases; blocked slots are not refilled"}


def _eos_ids(model):
    values = []
    for owner in (getattr(model.model, "generation_config", None),
                  getattr(model.model, "config", None), model.tok):
        candidate = getattr(owner, "eos_token_id", None)
        for value in candidate if isinstance(candidate, (list, tuple)) else (candidate,):
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0 and value not in values:
                values.append(value)
    return values


def _experiment_capture(metadata):
    return {**metadata, "capture_protocol_source": metadata.get("capture_protocol_source", metadata.get("source")),
            "source": SOURCE, "draft_visibility": "operator artifacts only"}


def _experiment_readings(readings):
    return {name: {**profile, "measurement": SOURCE} for name, profile in readings.items()}


@torch.no_grad()
def _generate_branch(model, capture_probes, prefix_ids, arm, *, horizon,
                     alpha, cap_fraction, layers, seed):
    """Generate once from exact IDs; never round-trip a selected prefix as text."""
    from transformers import LogitsProcessor, LogitsProcessorList

    ids = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
    check_context_budget(model.model, {"input_ids": ids}, reserve=horizon)
    original = _hook_ids(model)
    if original:
        raise ValueError("Unexpected active PyTorch hooks between continuation branches")
    _seed(seed)
    capture = GenerationProbeCapture(model, capture_probes, prompt_input_ids=ids,
                                    generation_channel="controlled_continuation")

    class ContextTracker(LogitsProcessor):
        def __call__(self, input_ids, scores):
            capture.commit_prediction(input_ids)
            return scores

    eos_ids = _eos_ids(model)
    pad = getattr(model.tok, "pad_token_id", None)
    if pad is None and eos_ids:
        pad = eos_ids[0]
    kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
              "max_new_tokens": horizon, "do_sample": False, "num_beams": 1,
              "num_return_sequences": 1, "use_cache": True,
              "repetition_penalty": 1., "return_dict_in_generate": False,
              "logits_processor": LogitsProcessorList([ContextTracker()])}
    if pad is not None:
        kwargs["pad_token_id"] = pad
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids
    handles, started = [], time.perf_counter()
    before = engine.steer_telemetry_stats()
    try:
        if arm["sign"]:
            handles = _steer_handles(model, arm["vectors"], layers,
                                     arm["sign"] * alpha, cap_fraction=cap_fraction)
        capture.attach()
        capture.start_chunk()
        output = model.model.generate(**kwargs)
        output = output.sequences if hasattr(output, "sequences") else output
        if (not isinstance(output, torch.Tensor) or output.ndim != 2
                or output.shape[0] != 1 or output.shape[-1] < ids.shape[-1]
                or not torch.equal(output[:, :ids.shape[-1]], ids)):
            raise ValueError("Branch generation did not preserve the exact selected prefix token IDs")
        tokens = output[0, ids.shape[-1]:].detach().cpu().tolist()
        if len(tokens) > horizon:
            raise ValueError("Branch generation exceeded its reserved token horizon")
        generated_count = len(tokens)
        # A single HF sequence normally ends immediately at EOS. Retain a
        # defensive trim for custom generators that append padding afterward.
        for index, token in enumerate(tokens):
            if token in eos_ids:
                tokens = tokens[:index + 1]
                break
        capture.commit_chunk(len(tokens))
        readings = _experiment_readings(capture.summary())
        metadata = _experiment_capture(capture.metadata())
        records = [{"position": row["position"], "layers": row["layers"]}
                   for row in capture._records]
    finally:
        capture.close()
        for handle in handles:
            handle.remove()
        _remove_new_hooks(model, original)
    after = engine.steer_telemetry_stats()
    return {"token_ids": tokens, "generated_token_count": generated_count,
            "readings": readings, "capture": metadata,
            "prediction_records": records, "elapsed_sec": time.perf_counter() - started,
            "steering_applications": after["applied"] - before["applied"],
            "steering_clips": after["clipped"] - before["clipped"]}


def _target_means(branch, objectives):
    values, missing = {}, []
    readings = branch.get("readings", {})
    for name in objectives:
        profile = readings.get(FINAL_PREFIX + name, {})
        trajectory = profile.get("by_token", [])
        # Every generated prediction must have a finite normalized endpoint
        # reading; incomplete trajectories cannot silently improve a score.
        if (not isinstance(trajectory, (list, tuple)) or not trajectory
                or len(trajectory) != len(branch["token_ids"])
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in trajectory)):
            missing.append(name)
        else:
            values[name] = sum(trajectory) / len(trajectory)
    return values, missing


def _append_committed(accumulator, branch, count, prefix_length):
    """Accumulate scalar predictions only at selected, committed positions."""
    records = branch.get("prediction_records", [])
    for name, profile in branch.get("readings", {}).items():
        values = profile.get("by_token", [])
        positions = profile.get("prediction_positions", [])
        bucket = accumulator.setdefault(name, {"by_token": [], "prediction_positions": [],
                                                "by_layer": {}})
        for index in range(min(count, len(values))):
            value = values[index]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            position = positions[index] if len(positions) > index else prefix_length + index
            if position != prefix_length + index:
                continue
            bucket["by_token"].append(float(value))
            bucket["prediction_positions"].append(position)
        if len(records) >= count and count:
            layers = records[count - 1].get("layers", {})
            bucket["by_layer"] = {int(layer): float(scores[name]) for layer, scores in layers.items()
                                  if name in scores and math.isfinite(scores[name])}


def _selected_readings(accumulator, probes):
    result = {}
    for name, data in accumulator.items():
        values = data["by_token"]
        if not values:
            continue
        mean, last = sum(values) / len(values), values[-1]
        result[name] = {**data, "raw_score": last, "score": last,
            "last_token_score": last, "mean_token_score": mean,
            "token_profile_score": .5 * (mean + last), "token_count": len(values),
            "token_mode": "last", "requested_token_mode": probes[name].get("token_mode", "last"),
            "measurement": SOURCE, "endpoint": GENERATION_PROBE_ENDPOINT,
            "aggregation": "selected committed scalar predictions across decision branches"}
    return result


def run_continuation_investigation(model, probes, objectives, controls=None, *,
        prompt_messages, output_dir, max_new_tokens=64, stride=8, horizon=12,
        alpha=.25, cap_fraction=.02, layers=None, max_helpers=2, seed=0,
        progress=None, on_commit=None, max_decisions=None, candidate_token_budget=None,
        max_joint_helpers=2, max_candidates=33, mixture_resolution=4,
        distribution="grid", temperature=1.0, compose_novel=False,
        layer_tilt=0.0, priors=None):
    """Select short continuations before rendering only their committed prefix.

    ``on_commit(cumulative_text)`` receives only cumulative selected text. Raw
    token IDs, rather than that decoded text, form the next decision's input.
    Empty/short budgets return an explicitly partial response and stop reason.
    Joint candidates test simultaneous helpers with one shared dose and cap;
    max_joint_helpers=1 preserves single-helper-only candidate construction.
    Mixture resolution four includes 75/25 and 25/75 pairs; one restricts
    combinations to uniform weights. The candidate cap bounds grid coverage.
    """
    if _hook_ids(model):
        raise ValueError("Continuation investigation requires a fresh model without active PyTorch forward hooks")
    for name, value in (("max_new_tokens", max_new_tokens), ("stride", stride),
                        ("horizon", horizon), ("max_helpers", max_helpers),
                        ("max_joint_helpers", max_joint_helpers), ("max_candidates", max_candidates),
                        ("mixture_resolution", mixture_resolution)):
        _positive_integer(value, name)
    if mixture_resolution > 64:
        raise ValueError("mixture_resolution must be between 1 and 64")
    if distribution not in ALLOWED_CONTINUATION_DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {', '.join(ALLOWED_CONTINUATION_DISTRIBUTIONS)}")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    if not isinstance(compose_novel, bool):
        raise ValueError("compose_novel must be a boolean")
    tilt_check = 0.35 if layer_tilt is True else (0.0 if layer_tilt is False else float(layer_tilt))
    if not math.isfinite(tilt_check) or abs(tilt_check) > 2.0:
        raise ValueError("layer_tilt must be a finite number in [-2.0, 2.0]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(alpha, bool) or isinstance(cap_fraction, bool)
            or not math.isfinite(alpha) or alpha <= 0
            or not math.isfinite(cap_fraction) or cap_fraction <= 0):
        raise ValueError("alpha and cap_fraction must be positive finite numbers")
    max_decisions = math.ceil(max_new_tokens / min(stride, horizon)) if max_decisions is None else max_decisions
    _positive_integer(max_decisions, "max_decisions")
    probes = dict(probes)
    objectives = _objectives(objectives)
    if any(name not in probes for name in objectives):
        raise ValueError("Every objective must name an available probe")
    layers = list(layers) if layers is not None else steer_band_layers(model.n_layers, .4, .7)
    if (not layers or len(set(layers)) != len(layers)
            or any(isinstance(layer, bool) or not isinstance(layer, int)
                   or not 0 <= layer < model.n_layers - 1 for layer in layers)):
        raise ValueError("Use distinct integer intervention layers below the final normalized readout endpoint")
    helpers, rejected = _prepare_helpers(probes, objectives, controls, layers,
                                         model.d_model, max_helpers)
    if not helpers:
        raise ValueError("No eligible independent helper direction is available")
    measured = {name: probes[name] for name in dict.fromkeys(
        [*objectives, *(helper["name"] for helper in helpers)])}
    # Helper selection controls actuation only. Other active probes remain
    # observational, including released or quiet probes; the collector reports
    # incompatible optional instruments without aborting required objectives.
    capture_probes = dict(probes)
    final_layer = model.n_layers - 1
    for name, payload in measured.items():
        directions = payload.get("direction", {})
        if not isinstance(directions, dict) or not directions:
            raise ValueError(f"Incompatible readout probe: {name}")
        for key, vector in directions.items():
            try:
                layer, vector = int(key), torch.as_tensor(vector).float().reshape(-1)
            except (ValueError, TypeError, RuntimeError) as exc:
                raise ValueError(f"Incompatible readout probe: {name}") from exc
            if (not 0 <= layer < model.n_layers or vector.numel() != model.d_model
                    or not bool(torch.isfinite(vector).all()) or not bool(vector.norm() > 0)):
                raise ValueError(f"Incompatible readout probe: {name}")
    for name in objectives:
        if FINAL_PREFIX + name in probes:
            raise ValueError(f"Probe name collides with continuation readout namespace: {FINAL_PREFIX + name}")
        directions = {int(layer): vector for layer, vector in probes[name]["direction"].items()}
        if final_layer not in directions:
            raise ValueError(f"Objective {name} needs a direction at the untouched final normalized readout layer")
        capture_probes[FINAL_PREFIX + name] = {"direction": {final_layer: directions[final_layer]}, "token_mode": "last"}
    if (not isinstance(prompt_messages, (list, tuple)) or not prompt_messages
            or any(not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}
                   or not isinstance(item.get("content"), str) for item in prompt_messages)
            or prompt_messages[-1]["role"] != "user"):
        raise ValueError("Supply ordinary role/content messages ending with the current user turn")
    prompt = model.tok.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = model.tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = inputs["input_ids"]
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1]:
        raise ValueError("The chat prompt must tokenize to one nonempty sequence")
    check_context_budget(model.model, inputs, reserve=max_new_tokens)
    original_ids = ids[0].detach().cpu().tolist()
    objective_vectors = [probes[name].get("direction", {}) for name in objectives]
    arms, candidate_plan = _build_candidate_arms(
        helpers, layers, max_joint_helpers, max_candidates, mixture_resolution,
        distribution=distribution, temperature=temperature, seed=seed,
        priors=priors, compose_novel=compose_novel, layer_tilt=layer_tilt,
        objective_vectors=objective_vectors)
    arms, candidate_plan = _filter_joint_objective_aliases(arms, probes, objectives, candidate_plan)
    candidate_token_budget = (len(arms) * horizon * max_decisions
                              if candidate_token_budget is None else candidate_token_budget)
    _positive_integer(candidate_token_budget, "candidate_token_budget")
    settings = {"objectives": objectives, "controls": [{"name": helper["name"], "sign": helper["sign"]}
                  for helper in helpers], "max_new_tokens": max_new_tokens, "stride": stride,
                "horizon": horizon, "alpha": alpha, "cap_fraction": cap_fraction, "layers": layers,
                "max_helpers": max_helpers, "max_joint_helpers": max_joint_helpers,
                "max_candidates": max_candidates, "mixture_resolution": mixture_resolution,
                "candidate_plan": candidate_plan,
                "joint_coefficient_budget": "sum absolute component coefficients is one; signed positive simplex weights share one alpha and injection cap",
                "seed": seed, "max_decisions": max_decisions,
                "candidate_token_budget": candidate_token_budget, "do_sample": False,
                "effect_metric": "final_layer_mean_token_score", "readout_layer": final_layer,
                "objective_aggregation": "equal-weight mean of signed target deltas from matched baseline",
                "tie_break": "baseline, then declared single-helper/sign order, then deterministic round-robin mixture candidate order",
                "injection_scope": "all tokens in recomputed prefix and continuation at selected block outputs",
                "draft_visibility": "operator artifacts only; unselected token IDs never enter the next prefix",
                "comparison_scope": "same exact selected prefix and decision seed; arm-local cache only"}
    if distribution != "grid" or float(temperature) != 1.0 or compose_novel or tilt_check != 0.0:
        settings.update({
            "distribution": distribution,
            "temperature": float(temperature),
            "compose_novel": bool(compose_novel or distribution in {"flexible", "orthogonal", "subspace", "adaptive"}),
            "layer_tilt": float(tilt_check),
        })
    model_sha, model_definition = _model_hash(model)
    scope = {"version": 1, "source": SOURCE, "settings": settings,
             "prompt_messages": prompt_messages, "prompt_token_ids": original_ids,
             "prompt_sha256": _hash([original_ids]), "model_identity": model_identity(model),
             "model_sha256": model_sha, "model_definition": model_definition,
             "definition_fingerprints": definition_fingerprints(capture_probes), "rejected_helpers": rejected,
             "arm_specs": [{key: arm[key] for key in ("arm", "helper", "sign", "joint_size", "components")}
                           for arm in arms],
             "arm_vector_sha256": {arm["arm"]: definition_fingerprints(
                 {"arm": {"direction": arm["vectors"]}})["arm"] for arm in arms},
             "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                 (Path(__file__), *(Path(__file__).with_name(name) for name in
                   ("engine.py", "generation_probe_capture.py", "replay_investigation.py",
                    "indirect_steering.py", "generation_budget.py", "other_probes.py")))}}
    manifest = {**scope, "fingerprint": _hash(scope), "created_at": datetime.now(timezone.utc).isoformat()}
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Continuation investigation requires a new or empty output directory; resume is unsupported")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "run_manifest.json", manifest)
    torch.save({arm["arm"]: arm["vectors"] for arm in arms}, output / "intervention_vectors.pt")
    decisions, selected, accumulator = [], [], {}
    unavailable_readouts = {}
    total_tokens, stop_reason = 0, "max_new_tokens"
    eos_ids = _eos_ids(model)
    started = time.perf_counter()

    def announce(message):
        if progress is not None:
            progress(message)

    def snapshot():
        text = model.tok.decode(selected, skip_special_tokens=True)
        metadata = {"source": SOURCE, "endpoint": GENERATION_PROBE_ENDPOINT,
            "prompt_sha256": manifest["prompt_sha256"], "token_count": len(selected),
            "generation_channel": "controlled_continuation", "capture_protocol_source": "live_assistant_prediction",
            "selection": "only committed positions from chosen local branches",
            "position_semantics": "absolute_position_of_token_predicted_by_captured_state",
            "prefix_reprocessing": "new branch-local cache each decision; prior committed readings retained once",
            "scratch_replays": "stored separately and excluded from selected readings",
            "unavailable_probes": dict(unavailable_readouts)}
        telemetry = {"source": SOURCE, "tokens_generated": len(selected), "branch_tokens": total_tokens,
            "decisions": decisions, "stop_reason": stop_reason, "elapsed_sec": time.perf_counter() - started,
            "generation_probe_readings": _selected_readings(accumulator, capture_probes),
            "generation_probe_capture": metadata, "activation_memory": {},
            "intervention_interpretation": "conditional probe-score comparison; semantic quality unvalidated"}
        return {"source": SOURCE, "fingerprint": manifest["fingerprint"], "response": text,
                "selected_token_ids": selected.copy(), "telemetry": telemetry, "output_dir": str(output)}

    announce(f"Continuation investigation: {len(arms)} candidate arms"
             f" ({sum(arm['joint_size'] > 1 for arm in arms)} joint; "
             f"{candidate_plan['total_possible']} possible; "
             f"{candidate_plan['blocked_joint_count']} direct-objective combinations blocked"
             f"{', capped' if candidate_plan['truncated'] else ''}); "
             f"mix {mixture_resolution}, horizon {horizon}, stride {stride}; output {output}")
    with _preserve_rng(), _isolated_steer_telemetry(), \
            (output / "trials.jsonl").open("a", encoding="utf-8", buffering=1) as trial_stream, \
            (output / "decisions.jsonl").open("a", encoding="utf-8", buffering=1) as decision_stream:
        while len(selected) < max_new_tokens:
            if len(decisions) >= max_decisions:
                stop_reason = "max_decisions"
                break
            remaining_budget = candidate_token_budget - total_tokens
            trial_horizon = min(horizon, max_new_tokens - len(selected), remaining_budget // len(arms))
            if trial_horizon < 1:
                stop_reason = "candidate_token_budget"
                break
            prefix_ids = original_ids + selected
            prefix_hash = _hash([prefix_ids])
            decision_seed, index = seed + len(decisions), len(decisions)
            branches, arm_rows = [], []
            announce(f"Decision {index + 1}: committed {len(selected)}/{max_new_tokens} tokens; testing {len(arms)} arms of at most {trial_horizon} tokens")
            for arm in arms:
                branch = _generate_branch(model, capture_probes, prefix_ids, arm,
                    horizon=trial_horizon, alpha=alpha, cap_fraction=cap_fraction, layers=layers,
                    seed=decision_seed)
                unavailable_readouts.update(branch.get('capture', {}).get('unavailable_probes', {}))
                tokens = branch["token_ids"]
                actual_count = branch.get("generated_token_count", len(tokens))
                if (not isinstance(tokens, list) or len(tokens) > trial_horizon
                        or isinstance(actual_count, bool) or not isinstance(actual_count, int)
                        or not len(tokens) <= actual_count <= trial_horizon):
                    raise ValueError("Invalid candidate token sequence")
                total_tokens += actual_count
                values, missing = _target_means(branch, objectives)
                branches.append(branch)
                row = {"source": SOURCE, "fingerprint": manifest["fingerprint"], "decision": index,
                    "arm": arm["arm"], "helper": arm["helper"], "sign": arm["sign"],
                    "components": arm["components"], "joint_size": arm["joint_size"],
                    "prompt_sha256": manifest["prompt_sha256"], "prefix_sha256": prefix_hash,
                    "prefix_token_ids": prefix_ids, "seed": decision_seed, "candidate_horizon": trial_horizon,
                    "token_ids": tokens, "tokens_generated": actual_count, "usable_tokens": len(tokens), "target_means": values,
                    "missing_targets": missing, "telemetry": {
                        "source": SOURCE, "generation_probe_readings": _experiment_readings(branch.get("readings", {})),
                        "generation_probe_capture": _experiment_capture(branch.get("capture", {}))},
                    "elapsed_sec": branch.get("elapsed_sec"),
                    "steering_applications": branch.get("steering_applications", 0),
                    "steering_clips": branch.get("steering_clips", 0)}
                trial_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                trial_stream.flush()
                arm_rows.append({key: row[key] for key in
                    ("arm", "helper", "sign", "components", "joint_size", "tokens_generated", "target_means", "missing_targets")})
            baseline, baseline_missing = _target_means(branches[0], objectives)
            winner, best = 0, 0.
            for arm_index, row in enumerate(arm_rows):
                eligible = not row["missing_targets"] and not baseline_missing and bool(branches[arm_index]["token_ids"])
                utility = 0. if arm_index == 0 else (sum(objectives[name] * (row["target_means"][name] - baseline[name])
                    for name in objectives) / len(objectives) if eligible else None)
                row.update({"utility": utility, "eligible": eligible})
                if arm_index and eligible and utility > best:
                    winner, best = arm_index, utility
            chosen = branches[winner]
            count = min(stride, max_new_tokens - len(selected), len(chosen["token_ids"]))
            commit = chosen["token_ids"][:count]
            for position, token in enumerate(commit):
                if token in eos_ids:
                    commit, count = commit[:position + 1], position + 1
                    break
            _append_committed(accumulator, chosen, count, len(prefix_ids))
            selected.extend(commit)
            decision = {"source": SOURCE, "fingerprint": manifest["fingerprint"], "decision": index,
                "prompt_sha256": manifest["prompt_sha256"], "prefix_sha256": prefix_hash,
                "prefix_token_ids": prefix_ids, "seed": decision_seed, "candidate_horizon": trial_horizon,
                "arms": arm_rows, "selected_arm": arms[winner]["arm"], "selected_utility": best,
                "selected_components": arms[winner]["components"],
                "selected_joint_size": arms[winner]["joint_size"],
                "committed_token_ids": commit, "committed_tokens": len(commit),
                "generated_branch_tokens": sum(branch.get("generated_token_count", len(branch["token_ids"])) for branch in branches),
                "baseline_missing_targets": baseline_missing}
            decisions.append(decision)
            decision_stream.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + "\n")
            decision_stream.flush()
            if not commit:
                stop_reason = "empty_selected_branch"
            elif commit[-1] in eos_ids:
                stop_reason = "eos"
            elif len(selected) >= max_new_tokens:
                stop_reason = "max_new_tokens"
            if on_commit is not None and commit:
                on_commit(model.tok.decode(selected, skip_special_tokens=True))
            _write_json(output / "summary.json", snapshot())
            mixture = ("; mix={" + ",".join(
                f"{item['helper']}:{item['coefficient']:+.6g}"
                for item in arms[winner]["components"]) + "}"
                if arms[winner]["joint_size"] > 1 else "")
            announce(f"Decision {index + 1}: selected {arms[winner]['arm']}{mixture}; committed {len(commit)} tokens; candidate tokens used {total_tokens}/{candidate_token_budget}")
            if stop_reason in {"eos", "empty_selected_branch"}:
                break
    result = snapshot()
    _write_json(output / "summary.json", result)
    return result
