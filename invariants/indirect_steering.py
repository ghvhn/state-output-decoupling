"""Bounded indirect probe control from scoped live and controlled rerun evidence.

Roles are operator declarations, never interpretations inferred from probe names.
Raw probe changes and correlations do not establish a probe's semantic validity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
from collections import deque
from copy import deepcopy

SOURCE = "live_assistant_prediction"
REPLAY_SOURCE = "controlled_replay_investigation"
REPLAY_METRIC = "final_layer_mean_token_score"
ROLES = {"perception", "thinking", "execution", "unclassified"}
COMPOSITION_MODES = {"weighted", "orthogonal", "residual", "subspace"}
DISTRIBUTIONS = {"power", "softmax", "dirichlet", "flexible", "uniform"}
DEFAULTS = {"budget": .15, "min_samples": 8, "min_correlation": .2,
            "top_k": 3, "history_limit": 128,
            "min_replay_cases": 3, "min_replay_effect": .0001}
OPTIONAL_DEFAULTS = {"temperature": 1.0, "layer_tilt": 1.0, "distribution": "power"}
_FINGERPRINT_CACHE = {}


def _revision(value):
    if hasattr(value, "detach") and hasattr(value, "shape"):
        try:
            version = value._version
        except RuntimeError:
            # Inference tensors have no mutation counter, so do not cache them.
            version = object()
        return (id(value), version, tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, dict):
        return tuple((str(key), _revision(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_revision(item) for item in value)
    return value


def _hash_content(digest, value):
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().float().contiguous()
        digest.update(f"tensor:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            digest.update(f"key:{key}:".encode())
            _hash_content(digest, item)
    else:
        digest.update(json.dumps(value, sort_keys=True, default=str).encode())


def definition_fingerprints(probes):
    """Hash definitions and tensors, excluding mutable score/physics settings."""
    result = {}
    for name, probe in probes.items():
        definition = {key: probe.get(key) for key in
                      ("framings", "recipe", "token_mode", "measurement_source", "direction", "position", "scale")}
        for extra in ("components", "composed_from", "composition_mode", "layer_weights", "orthogonal_to"):
            if probe.get(extra) is not None:
                definition[extra] = probe.get(extra)
        revision = _revision(definition)
        cached = _FINGERPRINT_CACHE.get(id(probe))
        if cached is not None and cached[0] is probe and cached[1] == revision:
            result[name] = cached[2]
            continue
        digest = hashlib.sha256()
        _hash_content(digest, definition)
        result[name] = digest.hexdigest()
        if len(_FINGERPRINT_CACHE) > 1024:
            _FINGERPRINT_CACHE.clear()
        _FINGERPRINT_CACHE[id(probe)] = (probe, revision, result[name])
    return result


def _sign(direction):
    if direction in ("maximize", "max", "+", 1):
        return 1
    if direction in ("minimize", "min", "-", -1):
        return -1
    raise ValueError("Direction must be maximize or minimize")


def _controls(controls):
    controls = shlex.split(controls) if isinstance(controls, str) else list(controls or [])
    result = []
    for token in controls:
        prior_weight = None
        if isinstance(token, dict):
            name, sign = token["name"], token.get("sign")
            if token.get("prior_weight") is not None:
                prior_weight = float(token["prior_weight"])
                if not math.isfinite(prior_weight) or prior_weight <= 0:
                    raise ValueError(f"Prior weight for {name} must be positive and finite")
        else:
            token = str(token).strip()
            if token.count("@") == 1:
                base, weight_str = token.rsplit("@", 1)
                if base and weight_str and not any(op in weight_str for op in "+-*(),"):
                    try:
                        pw = float(weight_str)
                    except ValueError as exc:
                        raise ValueError(f"Invalid prior weight in {token!r}") from exc
                    if not math.isfinite(pw) or pw <= 0:
                        raise ValueError(f"Prior weight for {base} must be positive and finite")
                    token = base
                    prior_weight = pw
            if token.startswith(("+", "-")) and not re.fullmatch(r"[+-]?(ortho|orthogonal|residual|subspace|compose)\(.+\)", token):
                sign = 1 if token.startswith("+") else -1
                name = token[1:]
            else:
                sign = None
                name = token
        if not name or sign not in (None, -1, 1):
            raise ValueError("Helpers must be probe names, +probe, or -probe")
        if name in {item["name"] for item in result}:
            raise ValueError(f"Duplicate helper: {name}")
        entry = {"name": name, "sign": sign}
        if prior_weight is not None:
            entry["prior_weight"] = prior_weight
        result.append(entry)
    return result


def _when(condition):
    if condition is None:
        return None
    if isinstance(condition, str):
        condition = shlex.split(condition)
    if isinstance(condition, (list, tuple)) and len(condition) == 3:
        condition = dict(zip(("probe", "comparison", "threshold"), condition))
    if not isinstance(condition, dict):
        raise ValueError("Condition must be: <probe> above|below <threshold>")
    probe = condition.get("probe", condition.get("name"))
    comparison = condition.get("comparison", condition.get("operator", condition.get("op")))
    try:
        threshold = float(condition["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Condition requires a finite numeric threshold") from exc
    if not isinstance(probe, str) or not probe or comparison not in {"above", "below"} or not math.isfinite(threshold):
        raise ValueError("Condition must name a probe, above|below, and finite threshold")
    return {"probe": probe, "comparison": comparison, "threshold": threshold}


def _tail_options(tokens):
    remaining, purpose, condition = [], "drive", None
    index = 0
    while index < len(tokens):
        if tokens[index] == "purpose":
            if index + 1 >= len(tokens):
                raise ValueError("purpose requires drive or coherence")
            purpose, index = tokens[index + 1], index + 2
        elif tokens[index] == "when":
            if index + 3 >= len(tokens):
                raise ValueError("when requires <probe> above|below <threshold>")
            condition, index = _when(tokens[index + 1:index + 4]), index + 4
        else:
            remaining.append(tokens[index])
            index += 1
    return remaining, purpose, condition


def _objectives(objectives):
    pairs = objectives.items() if isinstance(objectives, dict) else objectives
    result = {}
    for target, direction in pairs:
        if not isinstance(target, str) or not target:
            raise ValueError('Each objective must name a target probe')
        if target in result:
            raise ValueError(f'Duplicate objective: {target}')
        result[target] = _sign(direction)
    if not result:
        raise ValueError('Supply at least one objective')
    return result


def _batch_arguments(tokens):
    tokens, purpose, condition = _tail_options(tokens)
    if tokens.count('via') > 1:
        raise ValueError('Use one via separator before the shared helpers')
    if 'via' in tokens:
        separator = tokens.index('via')
        objective_tokens, controls = tokens[:separator], tokens[separator + 1:]
        if not controls:
            raise ValueError('Supply helper probes after via')
    else:
        objective_tokens, controls = tokens, None
    objectives, direction, group_size = [], None, 0
    for token in objective_tokens:
        if token in {'maximize', 'minimize'}:
            if direction is not None and not group_size:
                raise ValueError('Each maximize/minimize group needs target probes')
            direction, group_size = _sign(token), 0
        else:
            if direction is None:
                raise ValueError('Start each objective group with maximize or minimize')
            targets = token.split(',')
            if not all(targets):
                raise ValueError('Empty target in objective list')
            objectives.extend((target, direction) for target in targets)
            group_size += len(targets)
    if direction is None or not group_size:
        raise ValueError('Each maximize/minimize group needs target probes')
    return _objectives(objectives), controls, purpose, condition


def summarize_plan(entries, budget=None):
    """Keep each objective's request visible beside the net helper action."""
    groups = {}
    layer_groups = {}
    for entry in entries:
        scope = tuple(sorted(entry['layers'])) if entry.get('layers') is not None else None
        groups.setdefault((entry['name'], scope), []).append({
            'target': entry['target'], 'coefficient': float(entry['coefficient'])})
        if entry.get('layer_coefficients'):
            layer_groups.setdefault((entry['name'], scope), []).append(entry['layer_coefficients'])
    helpers, conflicts = [], []
    for (name, scope), contributions in groups.items():
        net = math.fsum(item['coefficient'] for item in contributions)
        if abs(net) < 1e-12:
            net = 0.
        helper = {'name': name, 'coefficient': net, 'contributions': contributions}
        if scope is not None:
            helper['layers'] = list(scope)
        if (name, scope) in layer_groups:
            all_layers = sorted({int(layer) for item in layer_groups[(name, scope)] for layer in item})
            layer_net = {}
            for layer in all_layers:
                val = math.fsum(
                    float(entry.get('layer_coefficients', {}).get(layer,
                          entry.get('layer_coefficients', {}).get(str(layer), entry['coefficient'])))
                    for entry in entries
                    if entry['name'] == name
                    and (tuple(sorted(entry['layers'])) if entry.get('layers') is not None else None) == scope
                )
                layer_net[layer] = 0. if abs(val) < 1e-12 else val
            helper['layer_coefficients'] = layer_net
        helpers.append(helper)
        if any(item['coefficient'] > 0 for item in contributions) and any(
                item['coefficient'] < 0 for item in contributions):
            if name not in conflicts:
                conflicts.append(name)
    return {'helpers': helpers, 'conflicts': conflicts,
            'gross_budget_used': math.fsum(abs(item['coefficient']) for item in entries),
            'net_budget_used': math.fsum(abs(item['coefficient']) for item in helpers),
            'budget': budget}


def _dependencies(probe):
    framing = probe.get("framings") or ("", "")
    text = framing[0] if isinstance(framing, (list, tuple)) and framing else framing
    recipe = probe.get("recipe") or (str(text).split(":", 1)[1] if str(text).startswith("composed:") else "")
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", str(recipe)))
    for source in (probe.get("composed_from"), probe.get("components"), probe.get("orthogonal_to")):
        if isinstance(source, dict):
            names.update(str(k) for k in source if isinstance(k, str) and k)
        elif isinstance(source, (list, tuple, set)):
            for item in source:
                if isinstance(item, dict):
                    candidate = item.get("name") or item.get("helper") or item.get("probe")
                    if isinstance(candidate, str) and candidate:
                        names.add(candidate)
                elif isinstance(item, str) and item:
                    names.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", item))
    for term in probe.get("other_composition") or ():
        if isinstance(term, (list, tuple)) and len(term) >= 2 and isinstance(term[1], str):
            names.add(term[1])
    return names


def _parse_component_term(raw_term):
    text = str(raw_term).strip()
    if not text:
        raise ValueError("Empty component in direction composition")
    sign = 1
    while text and text[0] in "+-":
        if text[0] == "-":
            sign *= -1
        text = text[1:].strip()
    weight = 1.0
    if "*" in text:
        left, right = (part.strip() for part in text.split("*", 1))
        try:
            weight = float(left)
            name = right
        except ValueError:
            weight = float(right)
            name = left
    elif "@" in text:
        name, weight_str = (part.strip() for part in text.split("@", 1))
        weight = float(weight_str)
    elif ":" in text:
        name, weight_str = (part.strip() for part in text.split(":", 1))
        weight = float(weight_str)
    else:
        name = text
    if weight < 0:
        sign *= -1
        weight = abs(weight)
    if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9_-]*", name):
        raise ValueError(f"Invalid component probe name: {name!r}")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"Component weight for {name} must be positive and finite")
    return {"name": name, "sign": int(sign), "weight": float(weight)}


def _parse_composition_components(components, mode=None):
    """Parse helper tokens, algebraic strings, or dicts into normalized components."""
    inferred_mode = mode
    if isinstance(components, dict):
        raw_items = []
        for name, value in components.items():
            val = float(value)
            if not math.isfinite(val) or val == 0:
                raise ValueError(f"Component weight for {name} must be finite and nonzero")
            raw_items.append({"name": str(name), "sign": 1 if val > 0 else -1, "weight": abs(val)})
    else:
        if isinstance(components, str):
            expr = components.strip()
            func_match = re.fullmatch(r"([+-]?)(ortho|orthogonal|residual|subspace|compose)\((.+)\)", expr)
            outer_sign = 1
            if func_match:
                if func_match.group(1) == "-":
                    outer_sign = -1
                prefix = func_match.group(2)
                if inferred_mode is None:
                    inferred_mode = ("orthogonal" if prefix in {"ortho", "orthogonal"}
                                     else "weighted" if prefix == "compose" else prefix)
                expr = func_match.group(3).strip()
            tokens = [part.strip() for part in re.split(r",|(?=[+-])", expr) if part.strip()]
            raw_items = [_parse_component_term(tok) for tok in tokens]
            if outer_sign < 0:
                for item in raw_items:
                    item["sign"] *= -1
        elif isinstance(components, (list, tuple)):
            raw_items = []
            for entry in components:
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("helper") or entry.get("probe")
                    if not isinstance(name, str) or not name:
                        raise ValueError("Component dict must specify probe name")
                    if "coefficient" in entry and "weight" not in entry:
                        coeff = float(entry["coefficient"])
                        if not math.isfinite(coeff) or coeff == 0:
                            raise ValueError(f"Component coefficient for {name} must be finite and nonzero")
                        sign = entry.get("sign", 1 if coeff > 0 else -1)
                        weight = abs(coeff)
                    else:
                        sign = entry.get("sign", 1)
                        sign = 1 if sign is None else _sign(sign)
                        weight = float(entry.get("weight", 1.0))
                    if not math.isfinite(weight) or weight <= 0:
                        raise ValueError(f"Component weight for {name} must be positive and finite")
                    raw_items.append({"name": name, "sign": int(sign), "weight": float(weight)})
                elif isinstance(entry, (list, tuple)) and len(entry) in (2, 3) and isinstance(entry[0], str):
                    name = entry[0].strip()
                    if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9_-]*", name):
                        raise ValueError(f"Invalid component probe name: {name!r}")
                    if len(entry) == 2:
                        val = float(entry[1])
                        if not math.isfinite(val) or val == 0:
                            raise ValueError(f"Component weight for {name} must be finite and nonzero")
                        sign = 1 if val > 0 else -1
                        weight = abs(val)
                    else:
                        sign = 1 if entry[1] is None else _sign(entry[1])
                        weight = float(entry[2])
                        if weight < 0:
                            sign *= -1
                            weight = abs(weight)
                        if not math.isfinite(weight) or weight <= 0:
                            raise ValueError(f"Component weight for {name} must be positive and finite")
                    raw_items.append({"name": name, "sign": int(sign), "weight": float(weight)})
                else:
                    sub_items, sub_mode = _parse_composition_components(str(entry), mode=inferred_mode)
                    if inferred_mode is None and sub_mode is not None:
                        inferred_mode = sub_mode
                    raw_items.extend({"name": item["name"], "sign": item["sign"], "weight": item["weight"]}
                                     for item in sub_items)
        else:
            raise ValueError("Supply component helpers as a string, list, or dict")
    if not raw_items:
        raise ValueError("Supply at least one component helper to compose a direction")
    aggregated = {}
    order = []
    for item in raw_items:
        name = item["name"]
        if name not in aggregated:
            order.append(name)
            aggregated[name] = 0.0
        aggregated[name] += item["sign"] * item["weight"]
    merged = []
    for name in order:
        net = aggregated[name]
        if abs(net) <= 1e-12:
            raise ValueError(f"Component {name} cancels to zero in composition")
        merged.append({"name": name, "sign": 1 if net > 0 else -1, "weight": abs(net)})
    total = math.fsum(item["weight"] for item in merged)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Total component weight must be positive and finite")
    normalized = [{"name": item["name"], "helper": item["name"], "sign": item["sign"],
                   "weight": item["weight"] / total,
                   "coefficient": item["sign"] * (item["weight"] / total)}
                  for item in merged]
    resolved_mode = inferred_mode or "weighted"
    if resolved_mode not in COMPOSITION_MODES:
        raise ValueError(f"Composition mode must be one of {sorted(COMPOSITION_MODES)}")
    return normalized, resolved_mode


def compose_direction(probes, components, *, orthogonal_to=None, blocked=None,
                      layers=None, mode=None, layer_weights=None,
                      role="unclassified", measurement_source=SOURCE):
    """Synthesize a frameless probe direction from helper vectors and subspaces.

    No contrastive text framings are required or generated. Any component in
    ``blocked`` is rejected, and any active target/orthogonal basis directions
    on shared layers are projected out before normalization.
    """
    import torch
    from invariants.other_probes import is_other_probe

    if role not in ROLES:
        raise ValueError("Role must be perception, thinking, execution, or unclassified")
    parsed, resolved_mode = _parse_composition_components(components, mode=mode)
    blocked_set = set(blocked or ())
    if isinstance(orthogonal_to, str):
        orthogonal_names = [part.strip() for part in re.split(r"[,\s]+", orthogonal_to) if part.strip()]
    else:
        orthogonal_names = [str(item).strip() for item in (orthogonal_to or ()) if str(item).strip()]
    for item in parsed:
        name = item["name"]
        if name not in probes:
            raise ValueError(f"Unknown helper probe: {name}")
        if name in blocked_set:
            raise ValueError(f"Helper {name} is the target, its alias/inverse, or a target-derived composite")
        payload = probes[name]
        if payload.get("released") or is_other_probe(payload):
            raise ValueError(f"Helper {name} is released or observational-only")
        if not payload.get("direction"):
            raise ValueError(f"Helper {name} has no direction")
    for orth in orthogonal_names:
        if orth not in probes:
            raise ValueError(f"Unknown orthogonal probe: {orth}")
    shared_layers = set.intersection(*(
        {int(layer) for layer in probes[item["name"]].get("direction", {})}
        for item in parsed
    ))
    if layers is not None:
        shared_layers &= {int(layer) for layer in layers}
    if not shared_layers:
        raise ValueError("Composed helpers have no shared direction layers")
    directions = {}
    projection_weights = {}
    out_layer_weights = {}
    project_out_names = list(dict.fromkeys([
        *orthogonal_names,
        *(name for name in sorted(blocked_set) if name in probes and name not in {p["name"] for p in parsed}),
    ]))
    for layer in sorted(shared_layers):
        unit_vectors = []
        width = None
        for item in parsed:
            raw = torch.as_tensor(probes[item["name"]]["direction"][layer]).detach().cpu().float().reshape(-1)
            if not raw.numel() or not bool(torch.isfinite(raw).all()) or float(raw.norm()) <= 1e-12:
                raise ValueError(f"Incompatible helper direction for {item['name']} at layer {layer}")
            if width is None:
                width = raw.numel()
            elif raw.numel() != width:
                raise ValueError(f"Mismatched direction widths at layer {layer}")
            unit_vectors.append(raw / raw.norm())
        # Resolve per-layer component weights when provided.
        raw_lw = []
        for item in parsed:
            cname = item["name"]
            w = item["weight"]
            if isinstance(layer_weights, dict):
                if layer in layer_weights and isinstance(layer_weights[layer], dict):
                    w = float(layer_weights[layer].get(cname, w))
                elif str(layer) in layer_weights and isinstance(layer_weights[str(layer)], dict):
                    w = float(layer_weights[str(layer)].get(cname, w))
                elif cname in layer_weights and isinstance(layer_weights[cname], dict):
                    w = float(layer_weights[cname].get(layer, layer_weights[cname].get(str(layer), w)))
                elif layer in layer_weights and isinstance(layer_weights[layer], (int, float)):
                    w = float(item["weight"]) * float(layer_weights[layer])
            if not math.isfinite(w) or w <= 0:
                raise ValueError(f"Invalid layer weight for {cname} at layer {layer}")
            raw_lw.append(w)
        lw_sum = math.fsum(raw_lw)
        norm_lw = [w / lw_sum for w in raw_lw]
        out_layer_weights[layer] = {item["name"]: norm_lw[idx] for idx, item in enumerate(parsed)}

        # Track exact linear expansion in terms of component unit vectors + projected-out probes.
        coeff_map = {item["name"]: 0.0 for item in parsed}
        if resolved_mode == "weighted":
            combined = torch.zeros(width, dtype=torch.float32)
            for item, uvec, weight in zip(parsed, unit_vectors, norm_lw):
                signed_w = item["sign"] * weight
                combined = combined + signed_w * uvec
                coeff_map[item["name"]] = coeff_map.get(item["name"], 0.0) + signed_w
        elif resolved_mode in {"orthogonal", "subspace"}:
            basis_vecs = []
            basis_coeffs = []
            for item, uvec in zip(parsed, unit_vectors):
                q = float(item["sign"]) * uvec.clone()
                c = {other["name"]: (float(item["sign"]) if other["name"] == item["name"] else 0.0)
                     for other in parsed}
                for b_vec, b_c in zip(basis_vecs, basis_coeffs):
                    proj = float(torch.dot(q, b_vec))
                    q = q - proj * b_vec
                    for k_name, k_val in b_c.items():
                        c[k_name] -= proj * k_val
                q_norm = float(q.norm())
                if q_norm > 1e-6:
                    basis_vecs.append(q / q_norm)
                    basis_coeffs.append({k_name: k_val / q_norm for k_name, k_val in c.items()})
            if not basis_vecs:
                raise ValueError(f"Orthogonal basis vanished at layer {layer}")
            combined = torch.zeros(width, dtype=torch.float32)
            active_weights = norm_lw[:len(basis_vecs)]
            aw_sum = math.fsum(active_weights) or 1.0
            for b_vec, b_c, weight in zip(basis_vecs, basis_coeffs, active_weights):
                w_norm = weight / aw_sum
                combined = combined + w_norm * b_vec
                for k_name, k_val in b_c.items():
                    coeff_map[k_name] = coeff_map.get(k_name, 0.0) + w_norm * k_val
        elif resolved_mode == "residual":
            primary_item, primary_vec = parsed[0], unit_vectors[0]
            q = float(primary_item["sign"]) * primary_vec.clone()
            c = {other["name"]: (float(primary_item["sign"]) if other["name"] == primary_item["name"] else 0.0)
                 for other in parsed}
            orth_vecs, orth_coeffs = [], []
            for item, uvec in zip(parsed[1:], unit_vectors[1:]):
                b = uvec.clone()
                bc = {other["name"]: (1.0 if other["name"] == item["name"] else 0.0) for other in parsed}
                for prev_v, prev_c in zip(orth_vecs, orth_coeffs):
                    proj = float(torch.dot(b, prev_v))
                    b = b - proj * prev_v
                    for k_name, k_val in prev_c.items():
                        bc[k_name] -= proj * k_val
                b_norm = float(b.norm())
                if b_norm > 1e-6:
                    orth_vecs.append(b / b_norm)
                    orth_coeffs.append({k_name: k_val / b_norm for k_name, k_val in bc.items()})
            for b_vec, b_c in zip(orth_vecs, orth_coeffs):
                proj = float(torch.dot(q, b_vec))
                q = q - proj * b_vec
                for k_name, k_val in b_c.items():
                    c[k_name] -= proj * k_val
            combined = q
            coeff_map = c

        # Project out any explicit orthogonal_to and any blocked target directions at this layer.
        ext_basis = []
        for pname in project_out_names:
            p_dirs = {int(k): v for k, v in (probes[pname].get("direction") or {}).items()}
            if layer not in p_dirs:
                continue
            p_raw = torch.as_tensor(p_dirs[layer]).detach().cpu().float().reshape(-1)
            if p_raw.numel() != width or not bool(torch.isfinite(p_raw).all()) or float(p_raw.norm()) <= 1e-12:
                continue
            b = p_raw / p_raw.norm()
            for prev in ext_basis:
                b = b - float(torch.dot(b, prev)) * prev
            b_norm = float(b.norm())
            if b_norm > 1e-6:
                ext_basis.append(b / b_norm)
        for b_vec in ext_basis:
            proj = float(torch.dot(combined, b_vec))
            if abs(proj) > 1e-12:
                combined = combined - proj * b_vec

        final_norm = float(combined.norm())
        if not math.isfinite(final_norm) or final_norm <= 1e-6:
            raise ValueError(f"Composed direction vanishes after orthogonalization at layer {layer}")
        directions[layer] = combined / final_norm
        projection_weights[layer] = {k_name: float(k_val / final_norm)
                                     for k_name, k_val in coeff_map.items() if abs(k_val) > 1e-12}

    result = {
        "direction": directions,
        "history": deque(maxlen=40),
        "framings": None,
        "exposed": False,
        "released": False,
        "measurement_source": measurement_source,
        "token_mode": "last",
        "role": role,
        "components": parsed,
        "composed_from": [item["name"] for item in parsed],
        "orthogonal_to": orthogonal_names,
        "composition_mode": resolved_mode,
        "projection_weights": projection_weights,
    }
    if layer_weights is not None:
        result["layer_weights"] = out_layer_weights
    for b_name in blocked_set:
        if b_name in probes and _alias(result, probes[b_name]):
            raise ValueError(f"Composed direction aliases or inverts blocked target {b_name}")
    return result


def _is_inline_composition_token(token, probes):
    """Return True only when token represents a frameless composition of known probes."""
    if not isinstance(token, str) or not token or token in probes:
        return False
    stripped = token[1:] if token[0] in "+-" else token
    if stripped in probes:
        return False
    if re.fullmatch(r"[+-]?(ortho|orthogonal|residual|subspace|compose)\(.+\)", token.strip()):
        return True
    if any(op in stripped for op in ("+", "@", "*")):
        return True
    if "-" in stripped:
        parts = [part.strip() for part in stripped.split("-") if part.strip()]
        if len(parts) >= 2 and all(part in probes for part in parts):
            return True
    return False


def _alias(left, right):
    import torch
    def framing(probe):
        pair = probe.get("framings")
        if (not isinstance(pair, (list, tuple)) or len(pair) < 2
                or not all(isinstance(text, str) and text.strip() for text in pair[:2])
                or pair[0].startswith(("composed:", "adopted:"))):
            return None
        return tuple(" ".join(text.split()).casefold() for text in pair[:2])
    left_frame, right_frame = framing(left), framing(right)
    if (left_frame is not None and right_frame is not None
            and left_frame in (right_frame, right_frame[::-1])):
        return True
    a = {int(key): value for key, value in left.get("direction", {}).items()}
    b = {int(key): value for key, value in right.get("direction", {}).items()}
    similarities = []
    for layer in a.keys() & b.keys():
        x, y = (torch.as_tensor(value).detach().cpu().float().reshape(-1)
                for value in (a[layer], b[layer]))
        if x.numel() != y.numel() or not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            return False
        nx, ny = x.norm(), y.norm()
        if not bool((nx > 0) & (ny > 0)):
            return False
        similarities.append(float((x / nx) @ (y / ny)))
    return bool(similarities) and abs(sum(similarities) / len(similarities)) > .9999


def _correlation(pairs):
    if not pairs:
        return None
    columns = list(zip(*pairs))
    normalized = []
    for column in columns:
        scale = max(map(abs, column))
        if not scale:
            return None
        values = [value / scale for value in column]
        center = math.fsum(values) / len(values)
        centered = [value - center for value in values]
        norm = math.sqrt(math.fsum(value * value for value in centered))
        if norm <= 1e-12:
            return None
        normalized.append([value / norm for value in centered])
    return max(-1., min(1., math.fsum(x * y for x, y in zip(*normalized))))


class IndirectSteering:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.settings, self.routes, self.roles = dict(DEFAULTS), {}, {}
        self.observations, self.fingerprints, self.feedback, self.pending = [], {}, {}, []
        self.replay_evidence = []
        self._probes = {}
        self._fresh = False
        self._alias_cache = {}
        self.model_identity = None
        self.joint_plan = summarize_plan([], self.settings['budget'])
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            saved_settings = data.get("settings", {})
            for key in DEFAULTS:
                self._set(key, saved_settings.get(key, DEFAULTS[key]))
            for key in OPTIONAL_DEFAULTS:
                if key in saved_settings:
                    self._set(key, saved_settings[key])
            self.routes = data.get("routes", {})
            self.roles = {name: role for name, role in data.get("roles", {}).items() if role in ROLES}
            self.observations = data.get("observations", [])[-int(self.settings["history_limit"]):]
            self.fingerprints = data.get("fingerprints", {})
            self.feedback = data.get("feedback", {})
            self.model_identity = data.get("model_identity")
            self.replay_evidence = data.get("replay_evidence", [])
            # A saved plan does not establish an intervention after process restart.
            self.pending = []
        self.joint_plan = summarize_plan([], self.settings['budget'])

    def save(self):
        if self.path is None:
            return
        data = {"version": 1, "source": SOURCE, "settings": self.settings, "roles": self.roles,
                "routes": self.routes, "observations": self.observations,
                "fingerprints": self.fingerprints, "feedback": self.feedback,
                "model_identity": self.model_identity, "replay_evidence": self.replay_evidence}
        encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _set(self, key, value):
        if key not in DEFAULTS and key not in OPTIONAL_DEFAULTS:
            raise ValueError(f"Unknown setting: {key}")
        if key == "distribution":
            dist = str(value).strip().lower()
            if dist not in DISTRIBUTIONS:
                raise ValueError(f"distribution must be one of {sorted(DISTRIBUTIONS)}")
            self.settings[key] = dist
            return
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Settings must be finite and nonnegative")
        if key == "temperature":
            if number <= 0:
                raise ValueError("temperature must be positive and finite")
            self.settings[key] = number
            return
        if key in {"min_samples", "top_k", "history_limit", "min_replay_cases"}:
            if number != int(number) or number < (2 if key != "top_k" else 1):
                raise ValueError(f"Invalid integer setting: {key}")
            number = int(number)
        if key == "min_correlation" and number > 1:
            raise ValueError("min_correlation must be at most 1")
        self.settings[key] = number

    def set_role(self, name, role):
        if role not in ROLES:
            raise ValueError("Role must be perception, thinking, execution, or unclassified")
        if self._probes and name not in self._probes:
            raise ValueError(f"Unknown probe: {name}")
        self.roles[name] = role
        self.pending = []
        self.save()
        return {"name": name, "role": role}

    def _refresh(self, probes):
        if probes is None:
            return
        for probe in probes.values():
            if isinstance(probe, dict) and not isinstance(probe.get("history"), deque):
                probe["history"] = deque(list(probe.get("history") or ()), maxlen=40)
        current = definition_fingerprints(probes)
        changed = {name for name in self.fingerprints if self.fingerprints[name] != current.get(name)}
        if changed:
            while True:
                dependent = {name for name, probe in probes.items() if _dependencies(probe) & changed}
                if dependent <= changed:
                    break
                changed |= dependent
            self._alias_cache.clear()
            for row in self.observations:
                for name in changed:
                    row["values"].pop(name, None)
                    if "layers" in row:
                        row["layers"].pop(name, None)
            self.pending = []
            for name in changed:
                self.feedback.pop(name, None)
            self.replay_evidence = [entry for entry in self.replay_evidence
                                    if not {entry['target'], entry['helper']} & changed]
        self.fingerprints, self._probes = current, probes

    @staticmethod
    def _project_composed_reading(composed, values, layer_values=None):
        proj = composed.get("projection_weights")
        if not isinstance(proj, dict) or not proj:
            return None, None
        by_layer = {}
        for layer, weights_map in proj.items():
            layer_int = int(layer)
            if not weights_map:
                return None, None
            if (isinstance(layer_values, dict)
                    and all(p in layer_values and layer_int in layer_values[p] for p in weights_map)):
                score = math.fsum(float(coeff) * float(layer_values[p][layer_int])
                                  for p, coeff in weights_map.items())
            elif all(p in values for p in weights_map):
                score = math.fsum(float(coeff) * float(values[p])
                                  for p, coeff in weights_map.items())
            else:
                return None, None
            if not math.isfinite(score):
                return None, None
            by_layer[layer_int] = score
        if not by_layer:
            return None, None
        return math.fsum(by_layer.values()) / len(by_layer), by_layer

    def _backfill_composed_observations(self, name, composed):
        for row in self.observations:
            if name in row["values"]:
                continue
            scalar, by_layer = self._project_composed_reading(
                composed, row["values"], row.get("layers"))
            if scalar is not None:
                row["values"][name] = scalar
                if "layers" in row and by_layer:
                    row["layers"][name] = by_layer

    def compose_direction(self, probes, components, *, name=None, orthogonal_to=None,
                          layers=None, mode=None, layer_weights=None, role=None,
                          blocked=None):
        """Compose a frameless direction from helpers and optionally register it in probes."""
        self._refresh(probes)
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Composed probe name must be a nonempty string")
            name = name.strip()
            if name in self.routes and self.routes[name].get("enabled", True):
                raise ValueError(f"Cannot overwrite active indirect target {name}")
        effective_role = role or (self.roles.get(name, "unclassified") if name else "unclassified")
        active_blocked = (set(blocked) - ({name} if name else set())) if blocked is not None else set()
        composed = compose_direction(
            probes, components, orthogonal_to=orthogonal_to, blocked=active_blocked,
            layers=layers, mode=mode, layer_weights=layer_weights, role=effective_role,
        )
        if name is not None:
            probes[name] = composed
            if role is not None:
                self.roles[name] = effective_role
            self._refresh(probes)
            self._backfill_composed_observations(name, composed)
            self.pending = []
            if role is not None:
                self.save()
        return composed

    def _ensure_inline_controls(self, controls, probes, extra_blocked=None):
        if not controls or not probes:
            return
        for item in controls:
            name = item["name"] if isinstance(item, dict) else str(item).lstrip("+-")
            if name not in probes and _is_inline_composition_token(name, probes):
                self.compose_direction(probes, name, name=name, blocked=extra_blocked)

    def import_investigation(self, summary, probes, model_identity=None):
        """Import complete controlled effects without manufacturing live samples.

        The original probe definitions, model, independent cases, dose and
        separate final-layer readout must agree before any owner state changes.
        Repeated greedy runs and repeated imports never add independent cases.
        """
        if isinstance(summary, (str, os.PathLike)):
            summary = json.loads(Path(summary).read_text(encoding='utf-8'))
        if not isinstance(summary, dict) or summary.get('source') != REPLAY_SOURCE:
            raise ValueError('Require a controlled_replay_investigation summary')

        def integer(value, name, minimum=1):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
            return value

        def finite(value, name, positive=False, nonnegative=False):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{name} must be finite numeric data')
            value = float(value)
            if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
                raise ValueError(f'{name} must be finite with the required sign')
            return value

        fingerprint = summary.get('fingerprint')
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError('Investigation fingerprint is required')
        completed = integer(summary.get('completed_trials'), 'completed_trials')
        expected = integer(summary.get('expected_trials'), 'expected_trials')
        if completed != expected:
            raise ValueError('Only complete investigations can be imported')
        identity = summary.get('model_identity')
        if not isinstance(identity, dict) or not identity:
            raise ValueError('Investigation model identity is required')
        try:
            json.dumps(identity, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError('Model identity must contain finite JSON values') from exc
        for current in (self.model_identity, model_identity):
            if current is not None and current != identity:
                raise ValueError('Investigation model identity differs from the current model')
        readout = integer(identity.get('layers'), 'model layer count', 2) - 1
        settings = summary.get('settings')
        if not isinstance(settings, dict):
            raise ValueError('Investigation settings are required')
        alpha = finite(settings.get('alpha'), 'intervention alpha', positive=True)
        cap_fraction = settings.get('cap_fraction')
        if cap_fraction is not None:
            cap_fraction = finite(cap_fraction, 'intervention cap_fraction', positive=True)
        layers = settings.get('layers')
        if (not isinstance(layers, list) or not layers
                or any(isinstance(layer, bool) or not isinstance(layer, int)
                       or not 0 <= layer < readout for layer in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError('Intervention layers must be distinct and below the final-layer readout')
        case_ids = summary.get('case_ids')
        if (not isinstance(case_ids, list) or not case_ids
                or any(not isinstance(case, str) or not case for case in case_ids)
                or len(set(case_ids)) != len(case_ids)):
            raise ValueError('Distinct source case_ids are required')
        known_cases = set(case_ids)
        if expected < len(known_cases):
            raise ValueError('Trial count cannot be smaller than the source case count')
        effects = summary.get('effects')
        if not isinstance(effects, list) or not effects or any(not isinstance(entry, dict) for entry in effects):
            raise ValueError('Investigation effects must be a nonempty list')
        current_definitions = definition_fingerprints(probes)
        declared = summary.get('definition_fingerprints')
        if not isinstance(declared, dict):
            raise ValueError('Original probe definition fingerprints are required')
        # Use a temporary guard so rejected imports cannot refresh or mutate the
        # owner's observations, alias cache, definitions, pending plan or file.
        guard = IndirectSteering()
        guard._refresh(probes)
        target_names = [entry.get('target') for entry in effects]
        if any(not isinstance(target, str) or target not in probes for target in target_names):
            raise ValueError('Every effect target must be a current probe')
        targets = set(target_names)
        blocked = set().union(*(guard._blocked_target(target, probes) for target in targets))
        staged, keys = [], set()
        for effect in effects:
            target, helper = effect.get('target'), effect.get('helper')
            if not isinstance(helper, str) or helper not in probes:
                raise ValueError('Every effect helper must be a current probe')
            if helper in blocked:
                raise ValueError('An investigation helper cannot be an objective or its direct proxy')
            if any(declared.get(name) != current_definitions.get(name) for name in (target, helper)):
                raise ValueError('Investigation probe definitions differ from the current instruments')
            sign = effect.get('sign')
            if isinstance(sign, bool) or sign not in (-1, 1):
                raise ValueError('Effect sign must be +1 or -1')
            if (effect.get('metric') != REPLAY_METRIC
                    or isinstance(effect.get('readout_layer'), bool)
                    or effect.get('readout_layer') != readout):
                raise ValueError('Require final_layer_mean_token_score at the separate last layer')
            count = integer(effect.get('n_cases'), 'effect n_cases')
            cases = effect.get('case_deltas')
            if not isinstance(cases, list) or len(cases) != count:
                raise ValueError('n_cases must agree with the independent case_deltas')
            seen, deltas, normalized_cases = set(), [], []
            for case in cases:
                if not isinstance(case, dict):
                    raise ValueError('Every case delta must be an object')
                case_id = case.get('case_id')
                if not isinstance(case_id, str) or case_id not in known_cases or case_id in seen:
                    raise ValueError('Effect cases must be distinct members of source case_ids')
                seen.add(case_id)
                delta = finite(case.get('mean_delta'), 'case mean_delta')
                repeats = integer(case.get('n_repeats', 1), 'case n_repeats')
                deltas.append(delta)
                normalized_cases.append({'case_id': case_id, 'mean_delta': delta, 'n_repeats': repeats})
            mean = finite(effect.get('mean_delta'), 'effect mean_delta')
            if not math.isclose(mean, math.fsum(delta / count for delta in deltas), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Effect mean_delta disagrees with its independent case deltas')
            if not math.isfinite(mean / alpha):
                raise ValueError('Effect per unit intervention alpha must be finite')
            stderr = effect.get('standard_error')
            if stderr is not None:
                stderr = finite(stderr, 'effect standard_error', nonnegative=True)
            key = (fingerprint, target, helper, sign)
            if key in keys:
                raise ValueError('Duplicate target/helper/sign effect in the same investigation')
            keys.add(key)
            staged.append({'source': REPLAY_SOURCE, 'fingerprint': fingerprint,
                'target': target, 'helper': helper, 'sign': int(sign),
                'n_cases': count, 'mean_delta': mean, 'standard_error': stderr,
                'case_deltas': normalized_cases, 'metric': REPLAY_METRIC,
                'readout_layer': readout, 'alpha': alpha, 'layers': sorted(layers),
                'cap_fraction': cap_fraction,
                'model_identity': deepcopy(identity),
                'definition_fingerprints': {name: current_definitions[name] for name in (target, helper)}})
        existing = {(entry['fingerprint'], entry['target'], entry['helper'], entry['sign']): entry
                    for entry in self.replay_evidence}
        added, duplicates = [], 0
        for entry in staged:
            key = (entry['fingerprint'], entry['target'], entry['helper'], entry['sign'])
            if key in existing:
                if existing[key] != entry:
                    raise ValueError('An existing investigation fingerprint has conflicting effect content')
                duplicates += 1
            else:
                added.append(entry)
        previous = {name: deepcopy(getattr(self, name)) for name in
                    ('observations', 'fingerprints', 'feedback', 'pending', 'replay_evidence',
                     'model_identity', 'joint_plan', '_alias_cache', '_fresh')}
        prior_probes = self._probes
        try:
            self._refresh(probes)
            self.model_identity = deepcopy(identity)
            self.replay_evidence.extend(added)
            if added:
                self.pending = []
                self.joint_plan = summarize_plan([], self.settings['budget'])
            self.save()
        except (OSError, ValueError, TypeError):
            for name, value in previous.items():
                setattr(self, name, value)
            self._probes = prior_probes
            raise
        return {'source': REPLAY_SOURCE, 'fingerprint': fingerprint,
                'imported_effects': len(added), 'duplicate_effects': duplicates,
                'replay_evidence': len(self.replay_evidence),
                'live_observations': len(self.observations),
                'interpretation': 'Controlled effects on raw probe scores; no semantic or causal generalization established'}

    def _replay_choice(self, target, helper, desired, allowed_sign):
        """Return an eligible latest-per-sign effect, or a tested-data veto."""
        latest = {}
        for entry in self.replay_evidence:
            if (entry['target'] != target or entry['helper'] != helper
                    or entry.get('model_identity') != self.model_identity
                    or any(entry.get('definition_fingerprints', {}).get(name) != self.fingerprints.get(name)
                           for name in (target, helper))):
                continue
            latest[entry['sign']] = entry
        adequate = [entry for entry in latest.values()
                    if entry['n_cases'] >= self.settings['min_replay_cases']]
        if not adequate:
            return None, None
        qualified = [entry for entry in adequate
                     if (allowed_sign is None or entry['sign'] == allowed_sign)
                     and desired * entry['mean_delta'] > 0
                     and desired * entry['mean_delta'] >= self.settings['min_replay_effect']]
        if not qualified:
            return None, 'adequate controlled replay data has no qualified benefit for the allowed helper sign'
        choice = max(qualified, key=lambda entry: (desired * entry['mean_delta'], entry['n_cases']))
        return choice, None

    def observe(self, row, probes=None):
        if (probes is None and isinstance(row, dict) and "probe_measurement" not in row
                and any(isinstance(v, dict) and ("direction" in v or "last_profile" in v) for v in row.values())):
            probes = row
            gen_id = next(
                (v.get("_last_generation_observation", {}).get("generation_id")
                 for v in probes.values()
                 if isinstance(v, dict) and isinstance(v.get("_last_generation_observation"), dict)
                 and v["_last_generation_observation"].get("generation_id") is not None),
                None,
            )
            readings = {
                name: v["last_profile"]
                for name, v in probes.items()
                if isinstance(v, dict) and isinstance(v.get("last_profile"), dict)
            }
            row = {
                "probe_measurement": {"source": SOURCE, "generation_id": gen_id},
                "probe_generation_readings": readings,
                **{f"probe_{name}": prof["raw_score"]
                   for name, prof in readings.items() if "raw_score" in prof},
            }
        self._refresh(probes)
        model_identity = row.get("probe_model_identity")
        if model_identity:
            json.dumps(model_identity, allow_nan=False)
            if model_identity != self.model_identity:
                self.observations, self.pending, self.feedback = [], [], {}
                self.replay_evidence = []
                self.model_identity, self._fresh = model_identity, False
        metadata = row.get("probe_measurement") or {}
        if metadata.get("source") != SOURCE or row.get("backfill") or row.get("type") == "backfill" or row.get("basis") == "backfill":
            self.pending = []
            return False
        identity = metadata.get("generation_id") or row.get("ts") or row.get("timestamp")
        if identity is None:
            self.pending = []
            return False
        identity = str(identity)
        if any(item["id"] == identity for item in self.observations):
            return False
        declared = row.get("probe_definitions") or row.get("probe_definition_fingerprints") or {}
        values = {key[6:]: float(value) for key, value in row.items()
                  if key.startswith("probe_") and isinstance(value, (int, float))
                  and not isinstance(value, bool) and math.isfinite(value)
                  and (not declared or declared.get(key[6:]) == self.fingerprints.get(key[6:]))}
        layer_values = {}
        readings = row.get("probe_generation_readings")
        if isinstance(readings, dict):
            for pname, profile in readings.items():
                if not isinstance(profile, dict) or (declared and declared.get(pname) != self.fingerprints.get(pname)):
                    continue
                by_layer = profile.get("by_layer")
                if isinstance(by_layer, dict):
                    valid_layers = {}
                    for l_key, l_val in by_layer.items():
                        if isinstance(l_val, (int, float)) and not isinstance(l_val, bool) and math.isfinite(float(l_val)):
                            try:
                                valid_layers[int(l_key)] = float(l_val)
                            except (TypeError, ValueError):
                                continue
                    if valid_layers:
                        layer_values[str(pname)] = valid_layers
        active_probes = probes if probes is not None else self._probes
        for cname, cprobe in (active_probes or {}).items():
            if cname not in values and isinstance(cprobe, dict) and cprobe.get("projection_weights"):
                scalar, by_layer = self._project_composed_reading(cprobe, values, layer_values)
                if scalar is not None:
                    values[cname] = scalar
                    if layer_values and by_layer:
                        layer_values[cname] = by_layer
        for pending in self.pending:
            target = pending["target"]
            if target not in values or pending["previous_id"] == identity:
                continue
            delta = values[target] - pending["previous_value"]
            if not math.isfinite(delta):
                continue
            self.feedback[target] = {"n": self.feedback.get(target, {}).get("n", 0) + 1,
                "applied_n": self.feedback.get(target, {}).get("applied_n", 0) + bool(pending.get("applied_helpers")),
                "last_target_delta": delta, "last_desired_delta": pending["direction"] * delta,
                "previous_value": pending["previous_value"], "current_value": values[target],
                "generation_id": identity, "source": SOURCE,
                "application_confirmed": bool(pending.get("applied_helpers")),
                "applied_helpers": pending.get("applied_helpers", []),
                "interpretation": "observational change after a " + ("confirmed applied" if pending.get("applied_helpers") else "proposed") + " plan; not causal credit",
                "plan": pending["helpers"]}
        self.pending = []
        self._fresh = True
        obs_entry = {"id": identity, "source": SOURCE, "values": values}
        if layer_values:
            obs_entry["layers"] = layer_values
        self.observations.append(obs_entry)
        self.observations = self.observations[-int(self.settings["history_limit"]):]
        self.save()
        return True

    def _is_alias(self, left, right, probes):
        if left == right:
            return True
        key = tuple(sorted((self.fingerprints.get(left, left), self.fingerprints.get(right, right))))
        if key not in self._alias_cache:
            self._alias_cache[key] = _alias(probes[left], probes[right])
        return self._alias_cache[key]

    def _blocked_target(self, target, probes):
        blocked = {target}
        if target in probes:
            blocked.update(name for name in probes if self._is_alias(target, name, probes))
        while True:
            expanded = blocked | {name for name, probe in probes.items() if _dependencies(probe) & blocked}
            # A copied direction of a derived target is still a direct proxy.
            expanded |= {name for name in probes if any(
                self._is_alias(source, name, probes) for source in expanded if source in probes)}
            if expanded == blocked:
                return blocked
            blocked = expanded

    def blocked_probes(self, probes):
        # All actuator entry points use this guard, including pin/mix paths
        # that do not call plan(). Reminting must invalidate aliases first.
        self._refresh(probes)
        return set().union(*(self._blocked_target(target, probes) for target, route in self.routes.items()
                             if route.get("enabled", True))) if self.routes else set()

    def _validated_route(self, target, direction, controls=None, purpose="drive", when=None, extra_blocked=None):
        if purpose not in {"drive", "coherence"}:
            raise ValueError("Purpose must be drive or coherence")
        desired = _sign(direction)
        parsed = None if controls is None else _controls(controls)
        condition = _when(when)
        if condition is not None and self._probes and condition["probe"] not in self._probes:
            raise ValueError(f"Unknown condition probe: {condition['probe']}")
        if parsed is not None and not parsed:
            raise ValueError("Supply at least one explicit helper, or omit helpers for automatic selection")
        if self._probes and target not in self._probes:
            raise ValueError(f"Unknown target probe: {target}")
        blocked = self._blocked_target(target, self._probes) | set(extra_blocked or ())
        if parsed and self._probes:
            self._ensure_inline_controls(parsed, self._probes, extra_blocked=blocked)
            blocked = self._blocked_target(target, self._probes) | set(extra_blocked or ())
        for helper in parsed or []:
            name = helper["name"]
            if name in blocked:
                raise ValueError(f"Helper {name} is the target, its alias/inverse, or a target-derived composite")
            if self._probes and name not in self._probes:
                raise ValueError(f"Unknown helper probe: {name}")
        return {"direction": desired, "controls": parsed, "enabled": True,
                "purpose": purpose, "when": condition}

    def configure(self, target, direction, controls=None, purpose="drive", when=None):
        self.routes[target] = self._validated_route(target, direction, controls, purpose, when)
        self.pending = []
        self.joint_plan = summarize_plan([], self.settings['budget'])
        self.save()
        return dict(self.routes[target])

    def configure_many(self, objectives, controls=None, purpose='drive', when=None):
        """Validate an entire objective set before changing any active route."""
        objectives = _objectives(objectives)
        controls = None if controls is None else _controls(controls)
        blocked = set().union(*(self._blocked_target(target, self._probes) for target in objectives))
        blocked |= self.blocked_probes(self._probes)
        staged = {target: self._validated_route(target, direction, controls, purpose, when, extra_blocked=blocked)
                  for target, direction in objectives.items()}
        blocked = set().union(*(self._blocked_target(target, self._probes) for target in objectives))
        blocked |= self.blocked_probes(self._probes)
        for control in controls or []:
            if control['name'] in blocked:
                raise ValueError(f"Helper {control['name']} is one of the objectives or its direct proxy")
        previous_routes, previous_pending, previous_joint = self.routes, self.pending, self.joint_plan
        self.routes = {**self.routes, **staged}
        self.pending = []
        self.joint_plan = summarize_plan([], self.settings['budget'])
        try:
            self.save()
        except (OSError, ValueError, TypeError):
            self.routes, self.pending, self.joint_plan = previous_routes, previous_pending, previous_joint
            raise
        return {target: dict(route) for target, route in staged.items()}

    def _compute_layer_weights(self, target, desired, helper_name, helper_sign, probes, candidate_layers=None):
        tilt = float(self.settings.get("layer_tilt", 1.0))
        if tilt <= 0:
            return None
        probe = probes.get(helper_name, {})
        directions = probe.get("direction") or {}
        layers = sorted(int(l) for l in (candidate_layers if candidate_layers is not None else directions.keys())
                        if int(l) in {int(k) for k in directions})
        if len(layers) <= 1:
            return None
        raw_weights = {}
        explicit_lw = probe.get("layer_weights")
        if isinstance(explicit_lw, dict):
            for l in layers:
                val = explicit_lw.get(l, explicit_lw.get(str(l)))
                if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                    raw_weights[l] = float(val)
                elif isinstance(val, dict):
                    total_v = math.fsum(abs(float(v)) for v in val.values() if isinstance(v, (int, float)))
                    if total_v > 0:
                        raw_weights[l] = total_v
        if len(raw_weights) != len(layers):
            raw_weights = {}
            rms_by_layer = {}
            for l in layers:
                pairs = []
                for row in self.observations:
                    row_layers = row.get("layers")
                    if not isinstance(row_layers, dict) or target not in row["values"]:
                        continue
                    h_layers = row_layers.get(helper_name)
                    if not isinstance(h_layers, dict) or l not in h_layers:
                        continue
                    t_layers = row_layers.get(target)
                    t_val = t_layers[l] if isinstance(t_layers, dict) and l in t_layers else row["values"][target]
                    pairs.append((t_val, h_layers[l]))
                if len(pairs) < self.settings["min_samples"]:
                    return None
                corr = _correlation(pairs)
                rms = math.sqrt(math.fsum(h_val * h_val for _, h_val in pairs) / len(pairs))
                rms_by_layer[l] = max(1e-6, rms)
                if corr is None:
                    raw_weights[l] = 0.05
                else:
                    aligned = desired * helper_sign * corr
                    raw_weights[l] = max(0.05, aligned if aligned > 0 else 0.05 * abs(corr))
            mean_rms = math.fsum(rms_by_layer.values()) / len(rms_by_layer) if rms_by_layer else 1.0
            if mean_rms > 1e-12:
                for l in layers:
                    raw_weights[l] *= (rms_by_layer[l] / mean_rms)
        tilted = {l: (w ** tilt) for l, w in raw_weights.items()}
        mean_w = math.fsum(tilted.values()) / len(tilted)
        if mean_w <= 1e-12:
            return None
        normalized = {l: w / mean_w for l, w in tilted.items()}
        if all(abs(w - 1.0) <= 1e-9 for w in normalized.values()):
            return None
        return normalized

    def propose(self, target, direction, probes, controls=None, purpose="drive", when=None, _prepared=False, _extra_blocked=None):
        from invariants.other_probes import is_other_probe
        if purpose not in {"drive", "coherence"}:
            raise ValueError("Purpose must be drive or coherence")
        if not _prepared:
            self._refresh(probes)
        if target not in probes:
            raise ValueError(f"Unknown target probe: {target}")
        desired = _sign(direction)
        condition = _when(when)
        blocked = self._blocked_target(target, probes) | self.blocked_probes(probes) | set(_extra_blocked or ())
        choices = _controls(controls) if controls is not None else None
        if choices is not None:
            try:
                self._ensure_inline_controls(choices, probes, extra_blocked=blocked)
                blocked = self._blocked_target(target, probes) | self.blocked_probes(probes) | set(_extra_blocked or ())
            except ValueError:
                pass
        else:
            choices = [{"name": name, "sign": None} for name in probes]
        latest = self.observations[-1]["values"] if self.observations and self._fresh else {}
        if condition is not None:
            value = latest.get(condition["probe"])
            passed = value is not None and ((value > condition["threshold"]) if condition["comparison"] == "above" else (value < condition["threshold"]))
            if not passed:
                return {"target": target, "direction": desired, "purpose": purpose,
                        "when": condition, "condition_satisfied": False, "helpers": [], "rejected": {},
                        "source": SOURCE, "evidence": "condition absent or not satisfied by latest live reading"}
        helpers, rejected = [], {}
        for control in choices:
            name = control["name"]
            prior_w = float(control["prior_weight"]) if control.get("prior_weight") is not None else 1.0
            reason = None
            if name not in probes:
                reason = "unknown helper"
            elif name in blocked:
                reason = "target, alias/inverse, or target-derived composite"
            elif probes[name].get("released") or is_other_probe(probes[name]):
                reason = "helper is released or observational-only"
            elif probes[name].get("measurement_source", SOURCE) != SOURCE or probes[target].get("measurement_source", SOURCE) != SOURCE:
                reason = "current probe measurement source is not live generation"
            elif not probes[name].get("direction"):
                reason = "helper has no direction"
            elif target not in latest or name not in latest:
                reason = "latest live row lacks paired readings"
            if reason:
                rejected[name] = reason
                continue
            pairs = [(row["values"][target], row["values"][name]) for row in self.observations
                     if target in row["values"] and name in row["values"]]
            correlation = _correlation(pairs)
            experimental, veto = self._replay_choice(target, name, desired, control['sign'])
            if veto:
                rejected[name] = veto
                continue
            if experimental is not None:
                degree = desired * experimental['mean_delta'] / experimental['alpha']
                entry = {'name': name, 'coefficient': float(experimental['sign']),
                    'target': target, 'n': experimental['n_cases'], 'n_cases': experimental['n_cases'],
                    'basis': REPLAY_SOURCE, 'mean_delta': experimental['mean_delta'],
                    'standard_error': experimental['standard_error'],
                    'metric': experimental['metric'], 'readout_layer': experimental['readout_layer'],
                    'investigation_fingerprint': experimental['fingerprint'],
                    'layers': list(experimental['layers']),
                    'tested_alpha': experimental['alpha'],
                    'tested_cap_fraction': experimental.get('cap_fraction'),
                    'operator_transfer': 'provisional_additive_to_field',
                    'transfer_assumption': 'Bounded additive replay effects have not validated the state-dependent field intervention',
                    'effect_per_unit_alpha': degree, 'correlation': correlation,
                    '_weight': degree * prior_w, 'role': self.roles.get(name, 'unclassified'), 'purpose': purpose}
                if control.get("prior_weight") is not None:
                    entry["prior_weight"] = prior_w
                helpers.append(entry)
                continue
            if len(pairs) < self.settings["min_samples"]:
                rejected[name] = "insufficient paired live samples"
            elif correlation is None or abs(correlation) < self.settings["min_correlation"]:
                rejected[name] = "constant scores or correlation below threshold"
            else:
                sign = desired * (1 if correlation > 0 else -1)
                if control["sign"] is not None and control["sign"] != sign:
                    rejected[name] = "explicit helper sign disagrees with observed target direction"
                    continue
                entry = {"name": name, "coefficient": float(sign), "target": target,
                         "n": len(pairs), "correlation": correlation,
                         "basis": SOURCE, "_weight": abs(correlation) * prior_w,
                         "role": self.roles.get(name, "unclassified"), "purpose": purpose}
                if control.get("prior_weight") is not None:
                    entry["prior_weight"] = prior_w
                helpers.append(entry)
        role_order = {"thinking": 0, "execution": 0, "perception": 1, "unclassified": 2}
        helpers.sort(key=lambda item: (role_order[item["role"]] if controls is None else 0,
                                       0 if item.get('basis') == REPLAY_SOURCE else 1,
                                       -item['_weight'], item["name"]))
        selected = []
        for helper in helpers:
            if any(self._is_alias(helper["name"], other["name"], probes) for other in selected):
                rejected[helper["name"]] = "duplicate helper axis"
            elif len(selected) < self.settings["top_k"]:
                selected.append(helper)
        scale = max((item['_weight'] for item in selected), default=1.)
        temperature = float(self.settings.get("temperature", 1.0))
        dist_mode = str(self.settings.get("distribution", "power"))
        transformed = []
        for helper in selected:
            raw_ratio = max(1e-12, helper.pop('_weight') / scale)
            if dist_mode == "uniform":
                eff_w = 1.0
            elif dist_mode == "softmax":
                eff_w = math.exp(min(50.0, (raw_ratio - 1.0) / temperature))
            elif dist_mode in {"dirichlet", "flexible"}:
                prior_floor = (1.0 / max(1, len(selected))) * (temperature / (1.0 + temperature))
                eff_w = prior_floor + (raw_ratio ** (1.0 / temperature))
            else:
                eff_w = raw_ratio if temperature == 1.0 else (raw_ratio ** (1.0 / temperature))
            transformed.append(eff_w)
        weight = math.fsum(transformed) or 1.0
        for helper, eff_w in zip(selected, transformed):
            helper_sign = 1 if helper["coefficient"] > 0 else -1
            helper["coefficient"] *= self.settings["budget"] * eff_w / weight
            layer_w = self._compute_layer_weights(
                target, desired, helper["name"], helper_sign, probes,
                candidate_layers=helper.get("layers"))
            if layer_w is not None:
                helper["layer_weights"] = layer_w
                helper["layer_coefficients"] = {l: helper["coefficient"] * w for l, w in layer_w.items()}
        return {"target": target, "direction": desired, "purpose": purpose,
                "when": condition, "condition_satisfied": True if condition else None,
                "helpers": selected, "rejected": rejected,
                "source": SOURCE, "evidence": "Scoped live correlations or controlled replay raw-score effects; no semantic claim"}

    def _budgeted(self, entries):
        total = math.fsum(abs(entry['coefficient']) for entry in entries)
        if total > self.settings['budget']:
            factor = self.settings['budget'] / total
            for entry in entries:
                entry['coefficient'] *= factor
                if entry.get('layer_coefficients'):
                    entry['layer_coefficients'] = {
                        l: val * factor for l, val in entry['layer_coefficients'].items()
                    }
        return entries

    def compose_plan_directions(self, probes, entries=None):
        """Synthesize per-layer frameless composite vectors from the active helper plan."""
        import torch
        self._refresh(probes)
        if entries is None:
            entries = self.joint_plan.get("helpers", [])
        blocked = self.blocked_probes(probes)
        layer_buckets = {}
        for helper in entries:
            name = helper["name"]
            coeff = float(helper.get("coefficient", 0.0))
            if not coeff or name not in probes:
                continue
            directions = probes[name].get("direction") or {}
            allowed_layers = set(helper["layers"]) if helper.get("layers") is not None else None
            layer_coeffs = helper.get("layer_coefficients") or {}
            for l_key, vec in directions.items():
                layer = int(l_key)
                if allowed_layers is not None and layer not in allowed_layers:
                    continue
                l_coeff = float(layer_coeffs.get(layer, layer_coeffs.get(str(layer), coeff)))
                if abs(l_coeff) <= 1e-12:
                    continue
                raw = torch.as_tensor(vec).detach().cpu().float().reshape(-1)
                if not raw.numel() or not bool(torch.isfinite(raw).all()) or float(raw.norm()) <= 1e-12:
                    continue
                layer_buckets.setdefault(layer, []).append((name, l_coeff, raw / raw.norm()))
        by_layer = {}
        for layer, items in sorted(layer_buckets.items()):
            width = items[0][2].numel()
            if any(vec.numel() != width for _, _, vec in items):
                continue
            combined = sum(l_coeff * uvec for _, l_coeff, uvec in items)
            # Project out any active target directions at this layer.
            for b_name in sorted(blocked):
                if b_name not in probes:
                    continue
                b_dirs = {int(k): v for k, v in (probes[b_name].get("direction") or {}).items()}
                if layer not in b_dirs:
                    continue
                b_raw = torch.as_tensor(b_dirs[layer]).detach().cpu().float().reshape(-1)
                if b_raw.numel() == width and bool(torch.isfinite(b_raw).all()) and float(b_raw.norm()) > 1e-12:
                    b_unit = b_raw / b_raw.norm()
                    combined = combined - float(torch.dot(combined, b_unit)) * b_unit
            norm = float(combined.norm())
            if norm <= 1e-9 or not math.isfinite(norm):
                continue
            gross_mag = math.fsum(abs(l_coeff) for _, l_coeff, _ in items)
            by_layer[layer] = {
                "direction": combined / norm,
                "coefficient": norm,
                "gross_budget": gross_mag,
                "components": [{"name": name, "coefficient": l_coeff} for name, l_coeff, _ in items],
            }
        return {
            "direction": {layer: info["direction"] for layer, info in by_layer.items()},
            "coefficients": {layer: info["coefficient"] for layer, info in by_layer.items()},
            "layers": by_layer,
            "framings": None,
            **by_layer,
        }

    def propose_many(self, objectives, probes=None, controls=None, purpose='drive', when=None):
        if (probes is None and isinstance(objectives, dict)
                and any(isinstance(v, dict) for v in objectives.values())):
            probes = objectives
            self._refresh(probes)
            active_routes = {t: r for t, r in self.routes.items() if r.get("enabled", True) and t in probes}
            blocked = set().union(*(self._blocked_target(t, probes) for t in active_routes)) if active_routes else set()
            proposals = {
                t: self.propose(t, r["direction"], probes, r["controls"],
                                purpose=r.get("purpose", "drive"), when=r.get("when"),
                                _prepared=True, _extra_blocked=blocked)
                for t, r in active_routes.items()
            }
            entries = self._budgeted([dict(entry) for proposal in proposals.values() for entry in proposal['helpers']])
            for t, proposal in proposals.items():
                proposal['helpers'] = [entry for entry in entries if entry['target'] == t]
            return {
                'objectives': {t: r['direction'] for t, r in active_routes.items()},
                'proposals': proposals,
                'plan': entries,
                'helpers': entries,
                'joint_plan': summarize_plan(entries, self.settings['budget']),
            }
        self._refresh(probes)
        objectives = _objectives(objectives)
        for target in objectives:
            if target not in probes:
                raise ValueError(f'Unknown target probe: {target}')
        blocked = set().union(*(self._blocked_target(target, probes) for target in objectives))
        proposals = {target: self.propose(target, direction, probes, controls, purpose, when,
                                         _prepared=True, _extra_blocked=blocked)
                     for target, direction in objectives.items()}
        entries = self._budgeted([dict(entry) for proposal in proposals.values() for entry in proposal['helpers']])
        # Single-objective proposals expose the coefficients actually reserved
        # for their objective within this shared budget.
        for target, proposal in proposals.items():
            proposal['helpers'] = [entry for entry in entries if entry['target'] == target]
        return {'objectives': objectives, 'proposals': proposals, 'plan': entries,
                'helpers': entries,
                'joint_plan': summarize_plan(entries, self.settings['budget'])}

    def plan(self, probes):
        self._refresh(probes)
        entries = []
        for target, route in self.routes.items():
            if route.get("enabled", True) and target in probes:
                entries.extend(self.propose(target, route["direction"], probes, route["controls"],
                                            purpose=route.get("purpose", "drive"), when=route.get("when"),
                                            _prepared=True)["helpers"])
        self._budgeted(entries)
        self.joint_plan = summarize_plan(entries, self.settings['budget'])
        if self.pending and [helper for pending in self.pending for helper in pending["helpers"]] != entries:
            self.pending = []
        if not entries:
            self.pending = []
        if entries and self.observations and not self.pending:
            latest = self.observations[-1]
            for target in dict.fromkeys(entry["target"] for entry in entries):
                self.pending.append({"target": target, "direction": self.routes[target]["direction"],
                    "previous_id": latest["id"], "previous_value": latest["values"][target],
                    "helpers": [dict(entry) for entry in entries if entry["target"] == target]})
        return entries

    def mark_applied(self, actions):
        """Confirm the subset the caller actually installed with nonzero force."""
        count = 0
        for pending in self.pending:
            applied = pending.setdefault("applied_helpers", [])
            for action in actions:
                if action.get("target") != pending["target"] or not action.get("coefficient"):
                    continue
                if any(helper["name"] == action.get("name") for helper in pending["helpers"]) and action not in applied:
                    applied.append(dict(action))
                    count += 1
        return count

    def status(self, probes=None):
        self._refresh(probes)
        return {"settings": dict(self.settings), "routes": self.routes, "roles": self.roles,
                "observations": len(self.observations), "source": SOURCE, "feedback": self.feedback,
                "replay_evidence": len(self.replay_evidence), "replay_source": REPLAY_SOURCE,
                "replay_runs": len({entry['fingerprint'] for entry in self.replay_evidence}),
                "model_identity": self.model_identity,
                "joint_plan": self.joint_plan,
                "pending": self.pending, "evidence": "Live associations and separately scoped controlled replay score effects"}


def _handle_compose_command(tokens, controller, probes, tuner=None):
    if len(tokens) < 2:
        raise ValueError("Usage: compose <name> [from|via] <helpers...> [mode weighted|orthogonal|residual|subspace] [orthogonal <probes>] [layers <first> <last>] [role <role>]")
    name = tokens.pop(0)
    if tokens and tokens[0] in {"from", "via"}:
        tokens.pop(0)
    component_tokens = []
    mode = None
    orthogonal_to = []
    layers = None
    role = None
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "mode":
            if idx + 1 >= len(tokens):
                raise ValueError("mode requires weighted, orthogonal, residual, or subspace")
            mode = tokens[idx + 1]
            idx += 2
        elif tok in {"orthogonal", "ortho"}:
            if idx + 1 >= len(tokens):
                raise ValueError("orthogonal requires one or more comma-separated probe names")
            orthogonal_to.extend(p.strip() for p in tokens[idx + 1].split(",") if p.strip())
            idx += 2
        elif tok == "layers":
            if idx + 2 >= len(tokens):
                raise ValueError("layers requires inclusive FIRST LAST indices")
            first, last = int(tokens[idx + 1]), int(tokens[idx + 2])
            if first < 0 or first > last:
                raise ValueError("layers FIRST must be nonnegative and no greater than LAST")
            layers = list(range(first, last + 1))
            idx += 3
        elif tok == "role":
            if idx + 1 >= len(tokens):
                raise ValueError("role requires perception, thinking, execution, or unclassified")
            role = tokens[idx + 1]
            idx += 2
        else:
            component_tokens.append(tok)
            idx += 1
    if not component_tokens:
        raise ValueError("Supply at least one helper probe to compose a frameless direction")
    composed = controller.compose_direction(
        probes, component_tokens, name=name, orthogonal_to=orthogonal_to or None,
        layers=layers, mode=mode, role=role,
    )
    if tuner is not None and hasattr(tuner, "register") and hasattr(tuner, "triggers"):
        if f"probe_{name}" not in tuner.triggers:
            try:
                tuner.register(f"probe_{name}", 0.0)
            except Exception:
                pass
    summary = {
        "name": name,
        "framings": None,
        "composition_mode": composed["composition_mode"],
        "components": composed["components"],
        "composed_from": composed["composed_from"],
        "orthogonal_to": composed["orthogonal_to"],
        "layers": sorted(composed["direction"]),
        "role": controller.roles.get(name, composed.get("role", "unclassified")),
        "paired_observations": sum(name in row["values"] for row in controller.observations),
    }
    summary["composed"] = dict(summary)
    return summary


def handle_indirect_command(args, controller, probes, tuner=None):
    """Return JSON-like output; raise ValueError for invalid operator commands."""
    tokens = shlex.split(args) if isinstance(args, str) else list(args)
    if tokens and tokens[0] == "indirect":
        tokens.pop(0)
    controller._refresh(probes)
    if not tokens or tokens[0] == "status":
        return controller.status(probes)
    action = tokens.pop(0)
    if action == "compose":
        return _handle_compose_command(tokens, controller, probes, tuner=tuner)
    if action == "roles":
        return {name: controller.roles.get(name, "unclassified") for name in probes}
    if action == "role":
        if len(tokens) != 2:
            raise ValueError("Usage: role <probe> <perception|thinking|execution|unclassified>")
        return controller.set_role(*tokens)
    if action == "set":
        if len(tokens) < 2 or len(tokens) % 2 != 0:
            raise ValueError("Usage: set <budget|min_samples|min_correlation|top_k|history_limit|min_replay_cases|min_replay_effect|temperature|distribution|layer_tilt> <value>")
        for idx in range(0, len(tokens), 2):
            controller._set(tokens[idx], tokens[idx + 1])
        controller.pending = []
        controller.observations = controller.observations[-int(controller.settings["history_limit"]):]
        controller.save()
        return controller.status()
    if action in {"off", "on", "drop"}:
        if len(tokens) > 1:
            raise ValueError(f"Usage: {action} [target]")
        targets = tokens or list(controller.routes)
        for target in targets:
            if target not in controller.routes:
                raise ValueError(f"Unknown indirect target: {target}")
        for target in targets:
            if action == "drop":
                del controller.routes[target]
            else:
                controller.routes[target]["enabled"] = action == "on"
        controller.pending = []
        controller.save()
        return controller.status()
    if action == "suggest":
        if tokens and tokens[0] in {'maximize', 'minimize'}:
            objectives, controls, purpose, condition = _batch_arguments(tokens)
            return controller.propose_many(objectives, probes, controls, purpose=purpose, when=condition)
        tokens, purpose, condition = _tail_options(tokens)
        if len(tokens) < 2:
            raise ValueError("Usage: suggest <target> maximize|minimize [helpers...]")
        return controller.propose(tokens[0], tokens[1], probes, tokens[2:] or None, purpose=purpose, when=condition)
    if action in {'maximize', 'minimize'}:
        objectives, controls, purpose, condition = _batch_arguments([action, *tokens])
        routes = controller.configure_many(objectives, controls, purpose=purpose, when=condition)
        return dict(controller.propose_many(objectives, probes, controls, purpose=purpose, when=condition),
                    routes=routes)
    tokens, purpose, condition = _tail_options(tokens)
    if not tokens:
        raise ValueError("Usage: <target> maximize|minimize [helpers...]")
    controls = tokens[1:] or None
    if controls is not None and controls[0] == 'via':
        controls = controls[1:]
        if not controls:
            raise ValueError('Supply helper probes after via')
    route = controller.configure(action, tokens[0], controls, purpose=purpose, when=condition)
    return {"target": action, "route": route,
            "proposal": controller.propose(action, route["direction"], probes, route["controls"], purpose=purpose, when=condition)}

