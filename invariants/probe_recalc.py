"""Recalculate probes from their creation definitions.

Every probe is defined by how it was originally created:
  1. Minted probes: Defined by contrastive text framings (a_text || b_text).
     Recalculated by running forward passes through the model to obtain fresh
     hidden states (hs_a, hs_b) and deriving unit directions and geometry via probe_geometry.
  2. Composed probes: Defined by a signed composition recipe over constituent probes.
     Stored static directions are NEVER used; directions are re-synthesized as a
     normalized weighted sum of the freshly recalculated constituent directions.
  3. Adopted probes: Defined by adopting another probe file.
     Inherits directly from the freshly recalculated target probe.
"""

from collections import deque
import glob
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from invariants.engine import (
    _hidden_states,
    _inputs,
    probe_geometry,
    steer_band_layers,
)


def parse_compose_expr(expr: str) -> Tuple[List[Tuple[float, str]], Optional[str]]:
    """Parse a signed weighted mix of named dimensions into (terms, error):
    terms = [(weight, name), ...]. Grammar: terms joined by + or -, each
    optionally weighted 'w*name' (e.g. 'ambiguity + disagreement - validated_flow').
    """
    padded = (expr or "").replace("+", " + ").replace("-", " - ")
    pending = 1.0
    terms = []
    for tok in padded.split():
        if tok == "+":
            continue
        if tok == "-":
            pending = -pending
            continue
        weight = 1.0
        tname = tok
        if "*" in tok:
            wpart, _, tname = tok.partition("*")
            try:
                weight = float(wpart)
            except ValueError:
                return [], tok
        tname = re.sub(r"[^a-z0-9_]", "_", tname.lower())[:40].strip("_")
        if not tname:
            return [], tok
        terms.append((pending * weight, tname))
        pending = 1.0
    if not terms:
        return [], expr or "(empty)"
    return terms, None


def strip_band_suffix(text: str) -> Tuple[str, Optional[List[int]]]:
    """Pop a trailing 'band <lo> <hi>' (inclusive layer indices) off a probe
    argument string: the explicit reading depth for adopt/compose/mint.
    Returns (text_without_suffix, layers or None)."""
    m = re.search(r"\s+band\s+(\d+)\s+(\d+)\s*$", text or "", re.IGNORECASE)
    if not m:
        return (text or "").strip(), None
    lo, hi = int(m.group(1)), int(m.group(2))
    return text[:m.start()].strip(), list(range(min(lo, hi), max(lo, hi) + 1))


def classify_probe(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a probe payload according to its creation definition.

    Returns a dict with 'type' in ('minted', 'composed', 'adopted', 'vector')
    and relevant parsed metadata (recipe, terms, framings, band, source).
    """
    framings = payload.get("framings")
    recipe = payload.get("recipe")

    a_text, b_text = "", ""
    if isinstance(framings, (tuple, list)) and len(framings) >= 2:
        a_text = str(framings[0] or "")
        b_text = str(framings[1] or "")
    elif isinstance(framings, str):
        a_text = framings

    # 1. Composed probe: defined by recipe string or framings[0] starting with 'composed:'
    if recipe or a_text.startswith("composed:"):
        rec_str = recipe if recipe else a_text.split(":", 1)[1].strip()
        terms, err = parse_compose_expr(rec_str)
        return {
            "type": "composed",
            "name": name,
            "recipe": rec_str,
            "terms": terms,
            "error": err,
        }

    # 2. Adopted probe: framings[0] starting with 'adopted:'
    if a_text.startswith("adopted:"):
        source = a_text.split(":", 1)[1].strip()
        return {
            "type": "adopted",
            "name": name,
            "source": source,
        }

    # 3. Minted probe: contrastive text pair
    if a_text or b_text:
        clean_b, band = strip_band_suffix(b_text)
        return {
            "type": "minted",
            "name": name,
            "with": a_text,
            "without": clean_b,
            "band": band,
        }

    # 4. Raw static vector (no creation definition / text stored)
    return {
        "type": "vector",
        "name": name,
    }


def resolve_probe_term_name(
    term_name: str,
    available_names: Set[str],
    current_probe_name: Optional[str] = None,
    d_model: Optional[int] = 4096,
) -> Optional[str]:
    """Resolve a term name in a recipe to an available probe name.
    
    Handles exact match, _d{d_model} suffix, and avoids self-cycles
    by falling back to the base minted probe (e.g. 'denial' -> 'denial_d4096').
    """
    candidates = []
    # If the term matches the current probe being composed, prefer the d_model base version
    if current_probe_name and term_name == current_probe_name:
        if d_model and f"{term_name}_d{d_model}" in available_names:
            return f"{term_name}_d{d_model}"

    if term_name in available_names:
        candidates.append(term_name)
    if d_model and f"{term_name}_d{d_model}" in available_names:
        candidates.append(f"{term_name}_d{d_model}")
    
    # If term ends with _d{d_model}, also check stripped
    if d_model and term_name.endswith(f"_d{d_model}"):
        base = term_name[:-len(f"_d{d_model}")]
        if base in available_names:
            candidates.append(base)

    if not candidates:
        return None
    # Prefer exact match unless it's a self-cycle
    if current_probe_name and candidates[0] == current_probe_name and len(candidates) > 1:
        return candidates[1]
    return candidates[0]


def recalc_minted_probe(
    model: Any,
    with_text: str,
    without_text: str,
    layers: Optional[List[int]] = None,
    token_mode: str = "last",
    max_chars: int = 600,
) -> Dict[str, Any]:
    """Recalculate a minted probe from its contrastive text framings.

    Runs forward passes through `model` to extract hidden states and
    derives full unit directions and latent geometry via `probe_geometry`.
    """
    clean_without, band_suffix = strip_band_suffix(without_text)
    effective_layers = layers if layers is not None else band_suffix

    ids_a = _inputs(model, with_text[:max_chars])
    hs_a = _hidden_states(model, ids_a["input_ids"], ids_a.get("attention_mask"))

    ids_b = _inputs(model, clean_without[:max_chars])
    hs_b = _hidden_states(model, ids_b["input_ids"], ids_b.get("attention_mask"))

    # Sensing is distinct from intervention: probes default to the full model depth (0..n_layers-1)
    # unless explicitly constrained by an operator-specified reading band.
    target_layers = effective_layers if effective_layers is not None else list(range(hs_a.shape[0]))
    geom = probe_geometry(hs_a, hs_b, target_layers, token_mode=token_mode)

    direction = geom.get("direction", {})
    if not direction:
        raise ValueError("Contrastive framings produced no usable direction.")

    return {
        "direction": direction,
        "position": geom.get("position", {}),
        "scale": geom.get("scale", {}),
        "poles": geom.get("poles", {}),
        "mean_direction": geom.get("mean_direction", {}),
        "token_salience": geom.get("token_salience", {}),
        "peak_token_offset": geom.get("peak_token_offset", {}),
        "framings": (with_text, without_text),
    }


def recalc_composed_probe(
    recipe: str,
    terms: List[Tuple[float, str]],
    probe_lookup: Dict[str, Dict[str, Any]],
    current_name: Optional[str] = None,
    band: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    d_model: Optional[int] = None,
) -> Dict[str, Any]:
    """Recalculate a composed probe from its creation recipe.

    NEVER uses stored directions for composed probes. Synthesizes a fresh
    direction as the normalized weighted sum of unit directions from resolved
    terms across common layers.
    """
    available = set(probe_lookup.keys())
    resolved = []
    resolved_geom = {}

    for weight, tname in terms:
        resolved_name = resolve_probe_term_name(tname, available, current_probe_name=current_name, d_model=d_model)
        if not resolved_name or resolved_name not in probe_lookup:
            raise ValueError(f"Component probe '{tname}' could not be resolved from active/recalculated probes.")
        
        comp_data = probe_lookup[resolved_name]
        cdir = comp_data.get("direction")
        if not cdir:
            raise ValueError(f"Component probe '{resolved_name}' has no valid directions.")
        
        if band is not None:
            band_set = set(band)
            cdir = {L: v for L, v in cdir.items() if int(L) in band_set}
            if not cdir:
                raise ValueError(f"Component probe '{resolved_name}' has no directions within requested band {band}.")

        resolved.append((weight, resolved_name, cdir))
        resolved_geom[resolved_name] = {
            "position": comp_data.get("position", {}),
            "scale": comp_data.get("scale", {}),
        }

    common_layers = set.intersection(*(set(d.keys()) for _, _, d in resolved))
    if not common_layers:
        spans = ", ".join(f"{t}: L{min(d)}-{max(d)}" for _, t, d in resolved)
        raise ValueError(f"Composed probe terms share no layers ({spans}).")

    direction = {}
    comp_position = {}
    comp_scale = {}

    target_device = device
    if target_device is None:
        first_vec = next(iter(resolved[0][2].values()))
        target_device = first_vec.device

    for L in sorted(common_layers):
        acc = None
        pos_acc = None
        pos_wsum = 0.0
        scale_acc = 0.0

        for w, tname, d in resolved:
            v = d[L].to(target_device).float().reshape(-1) * w
            acc = v if acc is None else acc + v

            tg = resolved_geom.get(tname, {})
            tpos = tg.get("position", {}).get(L)
            if tpos is None and isinstance(tg.get("position"), dict):
                tpos = tg["position"].get(int(L))
            if tpos is not None:
                pv = (tpos if torch.is_tensor(tpos) else torch.as_tensor(tpos)).to(target_device).float().reshape(-1)
                aw = abs(float(w))
                pos_acc = pv * aw if pos_acc is None else pos_acc + pv * aw
                pos_wsum += aw

            tsc = tg.get("scale", {}).get(L)
            if tsc is None and isinstance(tg.get("scale"), dict):
                tsc = tg["scale"].get(int(L))
            if tsc is not None:
                scale_acc += abs(float(w)) * float(tsc)

        norm = acc.norm()
        if norm.item() > 0:
            direction[int(L)] = acc / norm
            if pos_acc is not None and pos_wsum > 0:
                comp_position[int(L)] = pos_acc / pos_wsum
                comp_scale[int(L)] = float(scale_acc) if scale_acc > 0 else float(norm.item())

    if not direction:
        raise ValueError("Composed probe mix cancelled to zero on every shared layer.")

    return {
        "direction": direction,
        "position": comp_position,
        "scale": comp_scale,
        "framings": (f"composed: {recipe}", ""),
    }


def recalc_adopted_probe(
    source_str: str,
    probe_lookup: Dict[str, Dict[str, Any]],
    d_model: Optional[int] = None,
) -> Dict[str, Any]:
    """Recalculate an adopted probe by referencing its target probe definition."""
    clean_src = source_str[:-3] if source_str.endswith(".pt") else source_str
    available = set(probe_lookup.keys())
    target_name = resolve_probe_term_name(clean_src, available, d_model=d_model)

    if not target_name or target_name not in probe_lookup:
        raise ValueError(f"Adopted target '{source_str}' could not be resolved from active/recalculated probes.")

    target_data = probe_lookup[target_name]
    recalculated = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in target_data.items()
        if k not in ("history", "last_profile", "framings")
    }
    recalculated["framings"] = (f"adopted:{source_str}", "")
    return recalculated


def build_recalculation_schedule(
    probes_data: Dict[str, Dict[str, Any]],
    target_names: Optional[List[str]] = None,
    d_model: Optional[int] = 4096,
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, Any]]]:
    """Analyze probe dependencies and return an ordered execution schedule.

    Returns:
      (minted_order, composed_order, adopted_order, classifications)
    """
    classifications = {
        name: classify_probe(name, data) for name, data in probes_data.items()
    }
    all_names = set(probes_data.keys())

    minted_names = [name for name, c in classifications.items() if c["type"] == "minted"]
    composed_names = [name for name, c in classifications.items() if c["type"] == "composed"]
    adopted_names = [name for name, c in classifications.items() if c["type"] == "adopted"]

    # If targets are specified, narrow down to targets and their transitive dependencies
    if target_names is not None:
        needed: Set[str] = set()

        def add_deps(name: str):
            if name in needed:
                return
            needed.add(name)
            c = classifications.get(name)
            if not c:
                return
            if c["type"] == "composed":
                for _, tname in c.get("terms", []):
                    res = resolve_probe_term_name(tname, all_names, current_probe_name=name, d_model=d_model)
                    if res:
                        add_deps(res)
            elif c["type"] == "adopted":
                src = c.get("source", "").replace(".pt", "")
                res = resolve_probe_term_name(src, all_names, d_model=d_model)
                if res:
                    add_deps(res)

        for t in target_names:
            add_deps(t)

        minted_names = [n for n in minted_names if n in needed]
        composed_names = [n for n in composed_names if n in needed]
        adopted_names = [n for n in adopted_names if n in needed]

    # Topologically sort composed probes by dependency depth
    ordered_composed: List[str] = []
    remaining_composed = set(composed_names)
    resolved_so_far = set(minted_names)

    # Allow up to len(composed_names) rounds of resolution
    max_rounds = len(composed_names) + 2
    for _ in range(max_rounds):
        progress = False
        for cname in sorted(remaining_composed):
            c_info = classifications[cname]
            all_terms_ready = True
            for _, tname in c_info.get("terms", []):
                res = resolve_probe_term_name(tname, all_names, current_probe_name=cname, d_model=d_model)
                if res and res != cname and res not in resolved_so_far and res in remaining_composed:
                    all_terms_ready = False
                    break
            if all_terms_ready:
                ordered_composed.append(cname)
                resolved_so_far.add(cname)
                remaining_composed.remove(cname)
                progress = True
                break
        if not progress:
            break

    # If any cyclic/unresolved composed probes remain, append them at the end
    if remaining_composed:
        ordered_composed.extend(sorted(remaining_composed))

    return minted_names, ordered_composed, adopted_names, classifications


def recalc_probes(
    model: Any,
    probe_dir: Optional[str] = None,
    probe_names: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    in_place_probes: Optional[Dict[str, Any]] = None,
    save: bool = True,
    on_progress: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Batch recalculate probes according to their creation definitions.

    Args:
      model: The loaded transformer model.
      probe_dir: Directory containing stored .pt probe files (default: invariants/out/probes).
      probe_names: Optional list of probe names to recalculate (defaults to all).
      layers: Optional explicit layers list. When None, probes sense across all model layers.
      in_place_probes: Optional active probes dictionary to update in-place in memory.
      save: Whether to save recalculated probes back to disk.
      on_progress: Optional callback invoked as each probe completes: (name, type, result).

    Returns:
      Dict with summary of results, updated probes, and any errors.
    """
    directory = probe_dir or os.path.join("invariants", "out", "probes")
    try:
        import importlib
        import sys
        for mod_name in ("invariants.probe_measurement", "invariants.indirect_steering"):
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and "invariants.probe_measurement" in sys.modules:
            pm_mod = sys.modules["invariants.probe_measurement"]
            if hasattr(main_mod, "prepare_probe_measurements") and hasattr(pm_mod, "prepare_probe_measurements"):
                setattr(main_mod, "prepare_probe_measurements", pm_mod.prepare_probe_measurements)
    except Exception:
        pass

    if in_place_probes is not None:
        for _pname, _pdata in in_place_probes.items():
            if isinstance(_pdata, dict):
                if not isinstance(_pdata.get("history"), deque):
                    _pdata["history"] = deque(list(_pdata.get("history") or ()), maxlen=40)
                _pdata.setdefault("exposed", False)
                _pdata.setdefault("released", False)

    d_model = getattr(model, "d_model", getattr(getattr(model, "model", None), "config", None),)
    if not isinstance(d_model, int) and hasattr(d_model, "hidden_size"):
        d_model = d_model.hidden_size
    elif not isinstance(d_model, int):
        d_model = 4096

    # Load probe files
    probes_data: Dict[str, Dict[str, Any]] = {}
    if in_place_probes is not None:
        probes_data.update(in_place_probes)

    if os.path.isdir(directory):
        for fpath in glob.glob(os.path.join(directory, "*.pt")):
            stem = os.path.splitext(os.path.basename(fpath))[0]
            if stem not in probes_data:
                try:
                    payload = torch.load(fpath, map_location="cpu", weights_only=False)
                    if isinstance(payload, dict):
                        probes_data[stem] = payload
                except Exception:
                    pass

    minted_names, composed_names, adopted_names, classifications = build_recalculation_schedule(
        probes_data, target_names=probe_names, d_model=d_model
    )

    recalculated_probes: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    if probe_names is not None:
        for _tname in probe_names:
            _tdata = probes_data.get(_tname)
            if isinstance(_tdata, dict) and _tdata.get("framings") is None and _tdata.get("direction"):
                if not isinstance(_tdata.get("history"), deque):
                    _tdata["history"] = deque(list(_tdata.get("history") or ()), maxlen=40)
                recalculated_probes[_tname] = _tdata
                classifications[_tname] = {"type": "composed", "name": _tname}
                if on_progress:
                    on_progress(_tname, "composed", _tdata)

    # 1. Recalculate Minted Probes
    for name in minted_names:
        c = classifications[name]
        try:
            probe_layers = layers if layers is not None else c.get("band")
            res = recalc_minted_probe(model, c["with"], c["without"], layers=probe_layers)
            # Preserve existing flags
            orig = probes_data.get(name, {})
            for flag in ("exposed", "released", "frozen", "mass"):
                if flag in orig:
                    res[flag] = orig[flag]
            res["history"] = deque(maxlen=40)
            recalculated_probes[name] = res
            if in_place_probes is not None:
                in_place_probes[name] = res
            if on_progress:
                on_progress(name, "minted", res)
        except Exception as exc:
            errors[name] = str(exc)

    # Combine recalculated minted probes with existing data for term resolution
    active_lookup = dict(probes_data)
    active_lookup.update(recalculated_probes)

    # 2. Recalculate Composed Probes
    for name in composed_names:
        c = classifications[name]
        try:
            res = recalc_composed_probe(
                c["recipe"],
                c["terms"],
                active_lookup,
                current_name=name,
                band=layers,
                device=getattr(model, "device", None),
                d_model=d_model,
            )
            orig = probes_data.get(name, {})
            for flag in ("exposed", "released", "frozen", "mass"):
                if flag in orig:
                    res[flag] = orig[flag]
            res["history"] = deque(maxlen=40)
            recalculated_probes[name] = res
            active_lookup[name] = res
            if in_place_probes is not None:
                in_place_probes[name] = res
            if on_progress:
                on_progress(name, "composed", res)
        except Exception as exc:
            errors[name] = str(exc)

    # 3. Recalculate Adopted Probes
    for name in adopted_names:
        c = classifications[name]
        try:
            res = recalc_adopted_probe(c["source"], active_lookup, d_model=d_model)
            orig = probes_data.get(name, {})
            for flag in ("exposed", "released", "frozen", "mass"):
                if flag in orig:
                    res[flag] = orig[flag]
            res["history"] = deque(maxlen=40)
            recalculated_probes[name] = res
            active_lookup[name] = res
            if in_place_probes is not None:
                in_place_probes[name] = res
            if on_progress:
                on_progress(name, "adopted", res)
        except Exception as exc:
            errors[name] = str(exc)

    # 4. Save to Disk if requested
    saved_count = 0
    if save and os.path.isdir(directory):
        for name, pdata in recalculated_probes.items():
            save_payload = {
                k: v for k, v in pdata.items()
                if k not in ("history", "last_profile", "measurement_histories")
            }
            save_path = os.path.join(directory, f"{name}.pt")
            try:
                torch.save(save_payload, save_path)
                saved_count += 1
            except Exception as exc:
                errors[f"{name}_save"] = str(exc)

    return {
        "minted_count": len([n for n in recalculated_probes if classifications.get(n, {}).get("type") == "minted"]),
        "composed_count": len([n for n in recalculated_probes if classifications.get(n, {}).get("type") == "composed"]),
        "adopted_count": len([n for n in recalculated_probes if classifications.get(n, {}).get("type") == "adopted"]),
        "total_recalculated": len(recalculated_probes),
        "saved_count": saved_count,
        "errors": errors,
        "probes": recalculated_probes,
    }
