import ast
import datetime
import glob
import json
import os
import random
import re
import sys
from collections import deque

import torch
import colorama
from colorama import Fore, Style

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ORGANIC_VECTOR_PATH = os.path.join(ROOT, "invariants", "organic_correction_vector.pt")


# --- clock memory sensor: GPU or CPU ---------------------------------------
# The clock stream ("how much memory did this turn cost?") reads VRAM on a
# CUDA box. On a CPU-only box there is no VRAM, so rather than let the stream
# flatline at zero -- which strips the sensor from CPU-only runs -- we read the
# process's resident-set size (RSS). Same feature, same calibratable stream,
# just sourced from system RAM. No hard dependency: psutil if present, else the
# OS's own accounting, else a graceful zero.
_CPU_PEAK_BASELINE = {"gb": 0.0}


def _cpu_rss_gb():
    """Current resident-set size of this process, in GB. Cross-platform, best
    effort: returns 0.0 only if no accounting path is available."""
    _gb = 1024 ** 3
    try:
        import psutil  # optional; used if the user happens to have it
        return psutil.Process().memory_info().rss / _gb
    except Exception:
        pass
    try:  # Linux/most Unix: resident pages from /proc/self/statm
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / _gb
    except Exception:
        pass
    try:  # Windows: WorkingSetSize via GetProcessMemoryInfo
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return counters.WorkingSetSize / _gb
    except Exception:
        pass
    try:  # last resort: peak RSS from resource (KB on Linux, bytes on macOS)
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (maxrss if sys.platform == "darwin" else maxrss * 1024) / _gb
    except Exception:
        return 0.0


def _reset_memory_peak():
    """Open a fresh peak-memory window for this turn's clock reading."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        _CPU_PEAK_BASELINE["gb"] = _cpu_rss_gb()


def _memory_footprint_gb():
    """(live, reserved, peak, label) memory footprint for the clock sensor.

    CUDA present -> VRAM allocated / reserved / peak. CPU-only -> process RSS
    (live == reserved), with peak the high-water mark since the last window
    opened by :reset_memory_peak. Keeps the memory-cost stream meaningful on
    CPU instead of a constant zero, so CPU-only runs sense footprint the same
    way GPU runs do."""
    _gb = 1024 ** 3
    if torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated() / _gb,
            torch.cuda.memory_reserved() / _gb,
            torch.cuda.max_memory_allocated() / _gb,
            "VRAM",
        )
    rss = _cpu_rss_gb()
    peak = max(rss, _CPU_PEAK_BASELINE["gb"])
    return (rss, rss, peak, "RAM")

from invariants.engine import load_model
from invariants.agentic_engine import generate_agentic_text as _engine_generate, _global_cache

_ACTIVE_PROBES = None
_ACTIVE_MODEL = None
_ACTIVE_TUNER = None
_PROBES_SHOW_ACT = False
_PROBES_SHOW_TALK = True
_last_completion = None
_pending_expect = None
PROBE_RAW_HISTORY_PATH = os.path.join(ROOT, "invariants", "out", "probe_raw_history.json")
DEFAULT_COMMAND_PROMPT_PROBES = frozenset({
    "command_author",      # legacy :suggest helper
    "persona_author",
    "persona_selector",
    "profile_author",
    "macro_author",
    "solve_goal",
    "doc_rewriter",
    "premise_author",
})


def load_probe_raw_histories(path=PROBE_RAW_HISTORY_PATH):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        source = payload.get("probes", payload) if isinstance(payload, dict) else {}
        return {
            str(name): [float(value) for value in list(values)[-40:]]
            for name, values in source.items()
            if isinstance(values, list)
        }
    except (OSError, ValueError, TypeError):
        return {}


def save_probe_raw_histories(probes, path=PROBE_RAW_HISTORY_PATH):
    """Persist raw projection baselines once per scoring pass, not per probe."""
    try:
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "version": 1,
            "probes": {
                str(name): [float(value) for value in list(data.get("history") or [])[-40:]]
                for name, data in probes.items()
            },
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def centered_probe_observation(pdata, raw):
    """Append a raw projection and return its centered signal when baselined.

    None means this was the first observation: it seeds the baseline and must
    not masquerade as a genuine zero-valued centered measurement.

    A probe marked frozen keeps scoring against its existing baseline but the
    baseline itself stops moving -- the inertial half of the gravity split, so
    a heavy mass cannot habituate to the motion it itself induces.
    """
    hist = pdata["history"]
    if not hist:
        if not pdata.get("frozen"):
            hist.append(float(raw))
        return None
    sig = float(raw) - (sum(hist) / len(hist))
    if not pdata.get("frozen"):
        hist.append(float(raw))
    return sig


def probe_is_released(pdata):
    return bool((pdata or {}).get("released", False))


def set_probe_released(probes, name, released=True, probe_dir=None):
    """Make a probe observational-only while keeping it visible and persisted."""
    if name not in probes:
        return False
    probes[name]["released"] = bool(released)
    directory = probe_dir or os.path.join(ROOT, "invariants", "out", "probes")
    path = os.path.join(directory, f"{name}.pt")
    if os.path.isfile(path):
        try:
            payload = torch.load(path, weights_only=True)
            payload["released"] = bool(released)
            torch.save(payload, path)
        except Exception:
            pass
    return True


def set_probe_field(probes, name, key, value, probe_dir=None):
    """Set a per-probe physics coefficient (mass, frozen, ...) and persist it."""
    if name not in probes:
        return False
    probes[name][key] = value
    directory = probe_dir or os.path.join(ROOT, "invariants", "out", "probes")
    path = os.path.join(directory, f"{name}.pt")
    if os.path.isfile(path):
        try:
            payload = torch.load(path, weights_only=True)
            payload[key] = value
            torch.save(payload, path)
        except Exception:
            pass
    return True


# --- gravity prioritize: probes as masses in latent space -------------------
# Instead of pinning one direction and shoving with constant alpha, every
# probe becomes a mass sitting at its unit direction on the residual sphere.
# Each forward pass, every token's state is pulled along the sphere tangent
# toward (or, for negative mass, away from) each probe, weighted by
# mass / ((1 - cos) + eps)^2 -- an inverse-square law in cosine distance, so
# the force depends on WHERE the state is, not on a fixed target. eps smooths
# capture: as the state approaches a mass the tangent vanishes, so it settles
# instead of oscillating. The gravitational and inertial halves are decoupled
# per probe: mass sets the pull it exerts; frozen pins its own baseline.
GRAVITY_EPS = 0.05            # capture radius in (1 - cos); force fades smoothly inside it
GRAVITY_NORM_FRACTION = 0.05  # saturated field strength per unit G, as a fraction of each token's residual norm


def gravity_field_masses(probes, tuner):
    """Resolve each probe to a signed mass: an explicit ':steer mass' override
    wins; otherwise the probe's evidence-weighted lift (negative lift = a
    repulsor). Masses are normalized so total |mass| = 1; global, personal,
    family, and layer G then carry magnitude while the field remains stable as
    probes accumulate."""
    ranked = None
    masses = []
    configured = steer_bodies()["fields"]
    set_laws = steer_bodies()["sets"]
    quality_layer_member = any(
        node.startswith("layer:")
        and node not in set(law.get("excluded") or [])
        for law in set_laws.values()
        for node in (law.get("members") or {})
    )
    layer_g_configured = any(
        node.startswith("layer:") and cfg.get("g") is not None
        for node, cfg in configured.items()
    )
    for name, pdata in probes.items():
        dirs = pdata.get("direction") or {}
        node_key = f"probe:{name}"
        if not dirs or probe_is_released(pdata) or (configured.get(node_key) or {}).get("excluded"):
            continue  # released probes are observational-only: no force
        m = pdata.get("mass")
        if m is None:
            if ranked is None:
                ranked = {r["name"]: r for r in rank_probes(probes, tuner)}
            lift = (ranked.get(name) or {}).get("lift")
            # A personal/family G is an explicit request for this probe to be a
            # field body. Before it has enough credited outcomes for a live
            # lift, give it neutral unit mass rather than silently omitting it.
            personal = configured.get(node_key) or {}
            quality_member = any(
                node_key in (law.get("members") or {})
                and node_key not in set(law.get("excluded") or [])
                for law in set_laws.values()
            )
            m = float(lift) if lift is not None else (
                1.0 if personal.get("g") is not None or personal.get("family")
                or layer_g_configured or quality_member or quality_layer_member
                else 0.0
            )
        m = float(m)
        if abs(m) < 1e-9:
            continue
        masses.append((name, m, dirs))
    for name, body in steer_bodies()["anchors"].items():
        if (configured.get(f"anchor:{name}") or {}).get("excluded"):
            continue
        m = float(body.get("mass", 1.0) or 0.0)
        dirs = body.get("dirs") or {}
        if abs(m) < 1e-9 or not dirs:
            continue
        masses.append((f"@{name}", m, dirs))
    total = sum(abs(m) for _n, m, _d in masses)
    if total <= 0:
        return []
    return [(n, m / total, d) for n, m, d in masses]


# --- typed bodies: anchors, poles, and interaction laws ----------------------
# Anchors are precision masses: exact points in latent space (captured states
# or copied probe directions), pinned by default. Poles type the bodies;
# a law couples two pole families -- magnetism with more poles and more kinds:
# families interact only where a law says so, same-sign repels and opposite
# attracts (times the law's k, which can flip it). Laws accelerate BODIES,
# never the instruments: only unfrozen anchors move; probe directions are
# sensors and stay put. The live state feels every body only through gravity.
STEER_BODIES_PATH = os.path.join(ROOT, "invariants", "out", "steer_bodies.pt")
_STEER_BODIES = {
    "loaded": False,
    "anchors": {},
    "probe_poles": {},
    "laws": {},
    "fields": {},
    "g_families": {},
    "qualities": {},
    "sets": {},
}


def steer_bodies():
    """Lazy-load the persisted bodies state: anchors {name: {dirs, mass,
    frozen, poles}}, probe_poles {probe: {family: +/-1}}, laws {(a, b): k},
    fields {node: {g, time, shape, family, excluded}}, named g_families,
    quality formulas, and law-defined sets with per-member compliance/time."""
    if not _STEER_BODIES["loaded"]:
        _STEER_BODIES["loaded"] = True
        try:
            if os.path.isfile(STEER_BODIES_PATH):
                payload = torch.load(STEER_BODIES_PATH, weights_only=True)
                _STEER_BODIES["anchors"] = payload.get("anchors", {}) or {}
                _STEER_BODIES["probe_poles"] = payload.get("probe_poles", {}) or {}
                _STEER_BODIES["laws"] = {
                    tuple(k.split("|", 1)): float(v)
                    for k, v in (payload.get("laws", {}) or {}).items()
                }
                _STEER_BODIES["fields"] = payload.get("fields", {}) or {}
                _STEER_BODIES["g_families"] = payload.get("g_families", {}) or {}
                _STEER_BODIES["qualities"] = payload.get("qualities", {}) or {}
                _STEER_BODIES["sets"] = payload.get("sets", {}) or {}
        except Exception:
            pass
    for key in ("anchors", "probe_poles", "laws", "fields", "g_families", "qualities", "sets"):
        _STEER_BODIES.setdefault(key, {})
    return _STEER_BODIES


def save_steer_bodies():
    try:
        state = steer_bodies()
        os.makedirs(os.path.dirname(STEER_BODIES_PATH), exist_ok=True)
        torch.save({
            "anchors": state["anchors"],
            "probe_poles": state["probe_poles"],
            "laws": {"|".join(k): float(v) for k, v in state["laws"].items()},
            "fields": state["fields"],
            "g_families": state["g_families"],
            "qualities": state["qualities"],
            "sets": state["sets"],
        }, STEER_BODIES_PATH)
    except Exception:
        pass


def initialize_field_system(tuner, default_g=None):
    """Load/migrate every flexible-field surface and enable the field engine.

    This activates capability, not arbitrary behavior: existing personal Gs,
    families, qualities, sets, clocks, formulas, laws, and exclusions remain
    authoritative. It does not invent laws, unfreeze anchors, or force a
    nonzero law-step/doppler value.
    """
    state = steer_bodies()
    coefficient_defaults = {
        "prioritize_alpha": 0.0,
        "prioritize_gravity": 0.0,
        "gravity_doppler": 0.0,
        "gravity_law_step": 0.0,
    }
    for name, default in coefficient_defaults.items():
        tuner.register(name, default, kind="coefficient")
        tuner.triggers[name].kind = "coefficient"
    tuner.register("gravity_potential", 0.0, kind="threshold")
    tuner.register("gravity_time", 1.0, kind="threshold")
    if default_g is not None:
        tuner.set("prioritize_alpha", float(default_g))
    tuner.set("prioritize_gravity", 1.0)
    save_steer_bodies()  # persist newly added schema keys for older payloads
    return {
        "default_g": float(tuner.get("prioritize_alpha", 0.0)),
        "anchors": len(state["anchors"]),
        "families": len(state["g_families"]),
        "qualities": len(state["qualities"]),
        "sets": len(state["sets"]),
        "fields": len(state["fields"]),
        "laws": len(state["laws"]),
        "excluded": sum(1 for cfg in state["fields"].values() if cfg.get("excluded")),
    }


def parse_pole_spec(spec):
    """'truth+' / 'truth-' / 'truth' -> (family, sign). Bare family = +."""
    s = str(spec or "").strip().lower()
    sign = 1
    if s.endswith("+"):
        s = s[:-1]
    elif s.endswith("-"):
        s = s[:-1]
        sign = -1
    s = re.sub(r"[^a-z0-9_]", "_", s).strip("_")
    return (s, sign) if s else (None, 0)


def parse_field_source(raw, scale=1.0, offset=0.0):
    """Parse a live coefficient source used by personal G and local clocks.

    Sources are deliberately small and inspectable: a constant, the global G,
    a probe's latest centered reading, a tuner knob, a process status, a named
    G family, or a trigger's OUTCOME channel (lift:<trigger> = fired-vs-unfired
    outcome lift; outcome:<trigger> = rolling mean of recent credited
    outcomes) -- outcomes exert force, not just appear on maps. Scale and
    offset let unlike units be aligned explicitly.
    """
    text = str(raw or "").strip()
    try:
        return {
            "kind": "constant",
            "value": float(text),
            "scale": float(scale),
            "offset": float(offset),
        }
    except ValueError:
        pass
    low = text.lower()
    if low in ("global", "g", "prioritize_alpha"):
        kind, name = "global", "prioritize_alpha"
    elif ":" in low:
        kind, name = low.split(":", 1)
        kind = {"gfamily": "family", "memory": "status"}.get(kind, kind)
    else:
        return None
    if kind not in ("probe", "knob", "status", "family", "layer", "lift", "outcome") or not name:
        return None
    return {
        "kind": kind,
        "name": name,
        "scale": float(scale),
        "offset": float(offset),
    }


def format_field_source(spec):
    if not spec:
        return "global"
    if spec.get("kind") == "constant":
        base = f"{float(spec.get('value', 0.0)):g}"
    elif spec.get("kind") == "global":
        base = "global"
    else:
        base = f"{spec.get('kind')}:{spec.get('name')}"
    scale = float(spec.get("scale", 1.0))
    offset = float(spec.get("offset", 0.0))
    if scale != 1.0 or offset != 0.0:
        base += f" * {scale:g} + {offset:g}"
    return base


def field_source_error(spec, probes=None, tuner=None, families=None):
    """Return a precise typo/error for a live source, or None when usable."""
    if not spec:
        return "empty source"
    kind, name = spec.get("kind"), spec.get("name")
    if kind == "probe" and name not in (probes or {}):
        return f"no probe named '{name}'"
    if kind == "knob" and (tuner is None or name not in tuner.triggers):
        return f"no knob/stream named '{name}'"
    if kind in ("lift", "outcome") and (
        tuner is None
        or (name not in tuner.triggers and f"probe_{name}" not in tuner.triggers)
    ):
        return f"no credited trigger named '{name}' ({kind}: reads its outcome channel)"
    if kind == "family" and name not in (families or {}):
        return f"no G family named '{name}'"
    if kind == "status" and name not in {
        "ram", "vram", "memory", "memory_live",
        "ram_reserved", "vram_reserved", "memory_reserved",
        "ram_peak", "vram_peak", "memory_peak",
        "cuda", "gpu", "gravity_time", "time",
        "gravity_potential", "potential",
    }:
        return f"unknown status '{name}'"
    if kind == "layer":
        try:
            if int(name) < 0:
                raise ValueError
        except (TypeError, ValueError):
            return f"invalid layer '{name}'"
    return None


def latest_probe_field_value(name, probes, tuner):
    trig = tuner.triggers.get(f"probe_{name}") if tuner is not None else None
    if trig is not None and trig.signals:
        return float(trig.signals[-1])
    pdata = (probes or {}).get(name) or {}
    hist = list(pdata.get("history") or [])
    if len(hist) >= 2:
        return float(hist[-1]) - (sum(hist[:-1]) / len(hist[:-1]))
    return 0.0


def field_status_value(name):
    """Read a live, local status without adding a hard psutil dependency."""
    live, reserved, peak, label = _memory_footprint_gb()
    key = str(name or "").lower()
    if key in ("ram", "vram", "memory", "memory_live"):
        return float(live)
    if key in ("ram_reserved", "vram_reserved", "memory_reserved"):
        return float(reserved)
    if key in ("ram_peak", "vram_peak", "memory_peak"):
        return float(peak)
    if key in ("cuda", "gpu"):
        return 1.0 if torch.cuda.is_available() else 0.0
    if key in ("gravity_time", "time"):
        return float(_GRAVITY_STATE.get("time") or 1.0)
    if key in ("gravity_potential", "potential"):
        return float(_GRAVITY_STATE.get("phi") or 0.0)
    return 0.0


def trigger_outcome_value(kind, name, tuner, window=12):
    """Outcome as a live force. lift:<trigger> = its current fired-vs-unfired
    outcome lift (0 until both sides have evidence); outcome:<trigger> = the
    mean of its last `window` credited outcomes. The credit channel stays
    observe-only on the threshold itself -- THIS is the explicit route by
    which outcomes push back on the field, as a G/time/strength source."""
    if tuner is None:
        return 0.0
    trig = tuner.triggers.get(name) or tuner.triggers.get(f"probe_{name}")
    if trig is None or not trig.outcomes:
        return 0.0
    if kind == "lift":
        lift = trig.outcome_stats().get("lift")
        return float(lift) if lift is not None else 0.0
    tail = list(trig.outcomes)[-max(1, int(window)):]
    return sum(float(o) for _s, o in tail) / len(tail)


def resolve_field_source(
    spec, probes=None, tuner=None, families=None, fields=None, _seen=None, channel="g"
):
    if not spec:
        return float(tuner.get("prioritize_alpha", 0.0)) if tuner is not None else 0.0
    kind = spec.get("kind")
    if kind == "constant":
        base = float(spec.get("value", 0.0))
    elif kind == "global":
        base = float(tuner.get("prioritize_alpha", 0.0)) if tuner is not None else 0.0
    elif kind == "probe":
        base = latest_probe_field_value(spec.get("name"), probes, tuner)
    elif kind == "knob":
        base = float(tuner.get(spec.get("name"), 0.0)) if tuner is not None else 0.0
    elif kind == "status":
        base = field_status_value(spec.get("name"))
    elif kind in ("lift", "outcome"):
        base = trigger_outcome_value(kind, spec.get("name"), tuner)
    elif kind == "layer":
        name = str(spec.get("name", "0"))
        seen = set(_seen or ())
        marker = f"layer:{name}"
        if marker in seen:
            return 1.0
        seen.add(marker)
        layer_cfg = (fields or {}).get(marker) or {}
        if layer_cfg.get("time") is None:
            base = 1.0
        else:
            base = resolve_field_source(
                layer_cfg["time"], probes, tuner, families, fields, seen, "time"
            )
    elif kind == "family":
        name = spec.get("name")
        seen = set(_seen or ())
        if name in seen:
            return 0.0
        seen.add(name)
        fam = (families or {}).get(name) or {}
        inherited = fam.get("time") if channel == "time" else fam.get("g")
        if inherited is None and channel == "time":
            inherited = {"kind": "constant", "value": 1.0}
        base = resolve_field_source(
            inherited, probes, tuner, families=families, fields=fields,
            _seen=seen, channel=channel,
        )
    else:
        base = 0.0
    return base * float(spec.get("scale", 1.0)) + float(spec.get("offset", 0.0))


def canonical_field_node(raw, probes=None, anchors=None, families=None):
    """Return the persisted node key for a probe, anchor, layer, or family."""
    text = str(raw or "").strip()
    low = text.lower()
    probes = probes or {}
    anchors = anchors or {}
    families = families or {}
    if text in probes:
        return f"probe:{text}"
    if low.startswith("probe:") and text.split(":", 1)[1] in probes:
        return f"probe:{text.split(':', 1)[1]}"
    aname = re.sub(r"[^a-z0-9_]", "_", text.lstrip("@").lower()).strip("_")
    if text.startswith("@") and aname in anchors:
        return f"anchor:{aname}"
    if low.startswith("anchor:") and aname.split("_", 1)[-1] in anchors:
        return f"anchor:{aname.split('_', 1)[-1]}"
    if low.startswith("layer:"):
        try:
            layer = int(low.split(":", 1)[1])
            return f"layer:{layer}" if layer >= 0 else None
        except ValueError:
            return None
    if low.startswith(("family:", "gfamily:")):
        name = low.split(":", 1)[1]
        return f"family:{name}" if name in families else None
    return None


def field_target_config(raw, probes=None, create=False):
    bodies = steer_bodies()
    key = canonical_field_node(
        raw,
        probes=probes,
        anchors=bodies["anchors"],
        families=bodies["g_families"],
    )
    if not key:
        return None, None
    if key.startswith("family:"):
        name = key.split(":", 1)[1]
        return key, bodies["g_families"].get(name)
    if create:
        return key, bodies["fields"].setdefault(key, {})
    return key, bodies["fields"].get(key)


def parse_field_shape(tokens):
    """Parse a radial influence profile in cosine-distance coordinates."""
    if not tokens:
        return None
    kind = tokens[0].lower()
    aliases = {"inverse": "point", "inverse_square": "point", "hollow": "shell", "holoid": "shell"}
    kind = aliases.get(kind, kind)
    try:
        if kind == "point":
            return {"kind": "point"}
        if kind == "gaussian" and len(tokens) >= 2:
            return {"kind": "gaussian", "width": max(float(tokens[1]), 1e-4)}
        if kind == "shell" and len(tokens) >= 3:
            return {
                "kind": "shell",
                "radius": max(float(tokens[1]), 0.0),
                "width": max(float(tokens[2]), 1e-4),
            }
        if kind == "plateau" and len(tokens) >= 3:
            return {
                "kind": "plateau",
                "radius": max(float(tokens[1]), 0.0),
                "edge": max(float(tokens[2]), 1e-4),
            }
    except ValueError:
        return None
    return None


def format_field_shape(shape):
    shape = shape or {"kind": "point"}
    kind = shape.get("kind", "point")
    if kind == "gaussian":
        return f"gaussian width={float(shape.get('width', 0.2)):g}"
    if kind == "shell":
        return (
            f"shell radius={float(shape.get('radius', 0.5)):g} "
            f"width={float(shape.get('width', 0.1)):g}"
        )
    if kind == "plateau":
        return (
            f"plateau radius={float(shape.get('radius', 0.5)):g} "
            f"edge={float(shape.get('edge', 0.1)):g}"
        )
    return "point inverse-square"


def _gravity_kernel(distance, shape, eps=GRAVITY_EPS):
    """Signed radial profile. A shell reverses inside its radius, so its
    equilibrium is a hollow surface rather than the original center point."""
    shape = shape or {"kind": "point"}
    kind = shape.get("kind", "point")
    if kind == "gaussian":
        width = max(float(shape.get("width", 0.2)), 1e-4)
        return torch.exp(-0.5 * (distance / width) ** 2)
    if kind == "shell":
        radius = max(float(shape.get("radius", 0.5)), 0.0)
        width = max(float(shape.get("width", 0.1)), 1e-4)
        delta = distance - radius
        z = delta / width
        return z * torch.exp(-0.5 * z ** 2)
    if kind == "plateau":
        radius = max(float(shape.get("radius", 0.5)), 0.0)
        edge = max(float(shape.get("edge", 0.1)), 1e-4)
        return torch.sigmoid((radius - distance) / edge)
    return 1.0 / (distance + eps) ** 2


QUALITY_FORMULA_DEFAULT = "1/(d+0.05)**2"
_QUALITY_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}
_QUALITY_UNARYOPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def _quality_formula_node(node, d):
    """Evaluate an allowlisted scalar expression over cosine distance `d`."""
    if isinstance(node, ast.Expression):
        return _quality_formula_node(node.body, d)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "d":
        return d
    if isinstance(node, ast.BinOp) and type(node.op) in _QUALITY_BINOPS:
        return _QUALITY_BINOPS[type(node.op)](
            _quality_formula_node(node.left, d),
            _quality_formula_node(node.right, d),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _QUALITY_UNARYOPS:
        return _QUALITY_UNARYOPS[type(node.op)](_quality_formula_node(node.operand, d))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_quality_formula_node(arg, d) for arg in node.args]
        if node.func.id == "abs" and len(args) == 1:
            return torch.abs(args[0]) if torch.is_tensor(args[0]) else abs(args[0])
        if node.func.id == "exp" and len(args) == 1:
            return torch.exp(torch.as_tensor(args[0], device=d.device, dtype=d.dtype))
        if node.func.id == "sqrt" and len(args) == 1:
            return torch.sqrt(torch.as_tensor(args[0], device=d.device, dtype=d.dtype).clamp_min(0))
        if node.func.id == "log" and len(args) == 1:
            return torch.log(torch.as_tensor(args[0], device=d.device, dtype=d.dtype).clamp_min(1e-12))
        if node.func.id == "tanh" and len(args) == 1:
            return torch.tanh(torch.as_tensor(args[0], device=d.device, dtype=d.dtype))
        if node.func.id == "sigmoid" and len(args) == 1:
            return torch.sigmoid(torch.as_tensor(args[0], device=d.device, dtype=d.dtype))
    raise ValueError("formula may use only d, numbers, + - * / **, and abs/exp/sqrt/log/tanh/sigmoid")


def quality_formula_value(expr, distance):
    """Safe tensor formula with finite saturation before the outer steer cap."""
    tree = ast.parse(str(expr or QUALITY_FORMULA_DEFAULT), mode="eval")
    value = _quality_formula_node(tree, distance)
    value = torch.as_tensor(value, device=distance.device, dtype=distance.dtype)
    return torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)


def validate_quality_formula(expr):
    try:
        quality_formula_value(expr, torch.tensor(0.37))
        return None
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
        return str(exc)


def parse_time_vector(raw):
    try:
        values = [float(v) for v in str(raw or "").split(",") if str(v).strip()]
    except ValueError:
        return None
    return values if values else None


def time_vector_value(values, index):
    values = list(values or [1.0])
    return float(values[min(max(int(index), 0), len(values) - 1)])


def quality_terms_for_body(body_key, probes=None, tuner=None):
    """Resolve every non-excluded set law that includes this body."""
    bodies = steer_bodies()
    terms = []
    for set_name, law in sorted(bodies["sets"].items()):
        if body_key in set(law.get("excluded") or []):
            continue
        member = (law.get("members") or {}).get(body_key)
        if not member:
            continue
        quality_name = law.get("quality")
        quality = bodies["qualities"].get(quality_name) or {}
        if not quality:
            continue
        strength = resolve_field_source(
            quality.get("strength") or {"kind": "constant", "value": 1.0},
            probes, tuner, bodies["g_families"], bodies["fields"],
        )
        compliance = resolve_field_source(
            member.get("compliance") or {"kind": "constant", "value": 1.0},
            probes, tuner, bodies["g_families"], bodies["fields"],
        )
        amplitude = float(strength) * float(compliance)
        if abs(amplitude) < 1e-12:
            continue
        terms.append({
            "set": set_name,
            "quality": quality_name,
            "formula": quality.get("formula") or QUALITY_FORMULA_DEFAULT,
            "amplitude": amplitude,
            "time_vector": list(member.get("time_vector") or [1.0]),
        })
    return terms


def field_entry_config(body_name, layer, probes=None, tuner=None):
    """Resolve personal -> family -> layer settings for one mass at one layer."""
    bodies = steer_bodies()
    fields = bodies["fields"]
    families = bodies["g_families"]
    body_key = (
        f"anchor:{body_name[1:]}" if str(body_name).startswith("@")
        else f"probe:{body_name}"
    )
    own = fields.get(body_key) or {}
    fam = families.get(own.get("family")) or {}
    layer_cfg = fields.get(f"layer:{int(layer)}") or {}

    g_spec = own.get("g")
    if g_spec is None:
        g_spec = fam.get("g")
    layer_g = layer_cfg.get("g")
    if g_spec is None and layer_g is not None:
        # A layer can own G even when global G is zero.
        g = resolve_field_source(layer_g, probes, tuner, families, fields)
    else:
        if g_spec is None:
            g_spec = {"kind": "global", "name": "prioritize_alpha"}
        g = resolve_field_source(g_spec, probes, tuner, families, fields)
    if g_spec is not None and layer_g is not None:
        # When both are explicit, the layer is a gate/multiplier over personal G.
        g *= resolve_field_source(layer_cfg["g"], probes, tuner, families, fields)

    time_spec = own.get("time")
    if time_spec is None:
        time_spec = fam.get("time")
    tau = resolve_field_source(
        time_spec, probes, tuner, families, fields, channel="time"
    ) if time_spec else 1.0
    if layer_cfg.get("time") is not None:
        tau *= resolve_field_source(
            layer_cfg["time"], probes, tuner, families, fields, channel="time"
        )
    tau = max(0.05, min(20.0, float(tau)))

    shape = own.get("shape") or fam.get("shape") or layer_cfg.get("shape") or {"kind": "point"}
    qualities = (
        quality_terms_for_body(body_key, probes, tuner)
        + quality_terms_for_body(f"layer:{int(layer)}", probes, tuner)
    )
    return {
        "g": float(g),
        "time": tau,
        "shape": shape,
        "family": own.get("family"),
        "qualities": qualities,
        "excluded": bool(own.get("excluded") or layer_cfg.get("excluded")),
    }


def law_coupling(poles_i, poles_j, laws):
    """Net signed coupling between two typed bodies. For each declared law
    (famA, famB, k): contribution = -sign_i * sign_j * k, so with k > 0 like
    poles repel and unlike attract; k < 0 flips the convention. Families
    without a law exert nothing on each other -- selective interaction."""
    total = 0.0
    for (fa, fb), k in laws.items():
        for a, b in ((fa, fb), (fb, fa)) if fa != fb else ((fa, fb),):
            si = (poles_i or {}).get(a)
            sj = (poles_j or {}).get(b)
            if si and sj:
                total += -float(si) * float(sj) * float(k)
    return total


def apply_body_laws(tuner, probes=None, eps=GRAVITY_EPS):
    """One Euler step of the typed inter-body forces, moving only UNFROZEN
    anchors (per-layer, on the unit sphere; same inverse-square kernel as the
    field). Step size = the gravity_law_step knob; 0 = statics only."""
    step = float(tuner.get("gravity_law_step", 0.0)) if tuner is not None else 0.0
    if step <= 0:
        return 0
    bodies = steer_bodies()
    anchors = bodies["anchors"]
    laws = bodies["laws"]
    if not anchors or not laws:
        return 0
    # sources: every anchor, plus typed probes as immobile sources
    sources = []
    for name, a in anchors.items():
        sources.append((f"@{name}", a.get("poles") or {}, float(a.get("mass", 1.0) or 0.0), a.get("dirs") or {}))
    for pname, ppoles in bodies["probe_poles"].items():
        pdata = (probes or {}).get(pname)
        if not pdata or not ppoles:
            continue
        pm = pdata.get("mass")
        sources.append((pname, ppoles, float(pm) if pm is not None else 1.0,
                        pdata.get("direction") or {}))
    moved = 0
    for name, a in anchors.items():
        if a.get("frozen", True):
            continue
        poles_i = a.get("poles") or {}
        dirs = a.get("dirs") or {}
        if not poles_i or not dirs:
            continue
        for L, x in list(dirs.items()):
            xt = x if torch.is_tensor(x) else torch.as_tensor(x)
            xt = xt.detach().float()
            xn = xt / xt.norm().clamp_min(1e-6)
            force = torch.zeros_like(xn)
            for sname, poles_j, mj, jdirs in sources:
                if sname == f"@{name}" or abs(mj) < 1e-9:
                    continue
                k_eff = law_coupling(poles_i, poles_j, laws)
                if abs(k_eff) < 1e-12:
                    continue
                v = jdirs.get(L)
                if v is None:
                    continue
                vt = (v if torch.is_tensor(v) else torch.as_tensor(v)).detach().float()
                vn = vt / vt.norm().clamp_min(1e-6)
                cos = float((xn * vn).sum())
                tangent = vn - cos * xn
                force = force + (k_eff * mj / ((1.0 - cos) + eps) ** 2) * tangent
            if force.abs().sum() > 0:
                newx = xn + step * force
                dirs[L] = (newx / newx.norm().clamp_min(1e-6))
                moved += 1
    if moved:
        save_steer_bodies()
    return moved


def capture_anchor_dirs(model, text, max_chars=600):
    """Capture per-layer mean hidden states of `text` as anchor coordinates --
    an exact, reproducible point in latent space."""
    from invariants.engine import _inputs as _a_inputs, _hidden_states as _a_hidden
    ids = _a_inputs(model, text[:max_chars])
    hs = _a_hidden(model, ids["input_ids"], ids.get("attention_mask"))
    dirs = {}
    for L, h in enumerate(hs):
        try:
            dirs[int(L)] = h[0].float().mean(dim=0).detach().cpu()
        except Exception:
            continue
    return dirs


_GRAVITY_STATE = {"accum": None, "prev_cos": {}, "velocity": {}, "phi": None, "time": None}


def flush_gravity_telemetry(tuner=None):
    """Fold last generation's field telemetry into per-mass radial VELOCITY
    (the sensing half of doppler: what is moving toward/away between turns)
    and emit gravity_potential / gravity_time as live tuner streams -- so
    'change time in an area' can drive any tempo knob through the existing
    ':tune <knob> dynamic +gravity_time' machinery."""
    acc = _GRAVITY_STATE.get("accum")
    _GRAVITY_STATE["accum"] = None
    if not acc or not acc.get("n"):
        return
    n = acc["n"]
    phi = acc.get("phi", 0.0) / n
    t_mean = acc.get("time", 0.0) / n
    _GRAVITY_STATE["phi"], _GRAVITY_STATE["time"] = phi, t_mean
    vel = {}
    for nm, csum in (acc.get("cos") or {}).items():
        c = csum / max(acc.get("cnt", {}).get(nm, 1), 1)
        prev = _GRAVITY_STATE["prev_cos"].get(nm)
        if prev is not None:
            vel[nm] = c - prev
        _GRAVITY_STATE["prev_cos"][nm] = c
    if vel:
        _GRAVITY_STATE["velocity"] = vel
    if tuner is not None:
        try:
            tuner.observe("gravity_potential", float(phi))
            tuner.observe("gravity_time", float(t_mean))
        except Exception:
            pass


def gravity_velocities():
    """Per-mass radial velocity from the last two field turns (+approaching)."""
    return dict(_GRAVITY_STATE.get("velocity") or {})


def _gravity_pull(h_float, entries, G, eps=GRAVITY_EPS, doppler=0.0,
                  vstate=None, vkey=None, taus=None, names=None, telem=None,
                  time_index=0):
    """Pure field math. h_float: (..., D) float32 hidden states; entries:
    legacy [(unit_direction(D), mass)] or field dictionaries with direction,
    mass, personal g, time, shape, and name. Returns the additive pull, bounded
    by tanh(|field|) so stacked masses saturate instead of exploding, and
    scaled to each token's own residual norm.

    doppler > 0 makes the coupling velocity-aware: attention to a mass is not
    line-of-sight distance alone but how fast the state is moving toward or
    away from it (change in last-position cos between forwards; approaching
    damps the pull, receding restores it -- a damped field that settles).

    Per-entry time coefficients are local clock rates: they scale how much
    change this field can produce in one shell step, while radial velocity is
    measured against the same clock. telem accumulates potential, local time,
    and per-mass cos for the sensing/stream side."""
    structured = any(isinstance(item, dict) for item in entries)
    normalized = []
    for idx, item in enumerate(entries):
        if isinstance(item, dict):
            normalized.append((
                item["direction"],
                float(item.get("mass", 0.0)),
                float(item.get("g", G)),
                float(item.get("time", 1.0)),
                item.get("shape") or {"kind": "point"},
                item.get("name", str(idx)),
                list(item.get("qualities") or []),
            ))
        else:
            v_hat, mass = item[:2]
            tau = taus[idx] if taus and idx < len(taus) and taus[idx] is not None else 1.0
            name = names[idx] if names and idx < len(names) else str(idx)
            normalized.append((v_hat, float(mass), 1.0, float(tau), {"kind": "point"}, name, []))
    s_norm = h_float.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    s_hat = h_float / s_norm
    cos_list = [(s_hat * item[0]).sum(-1, keepdim=True) for item in normalized]
    scalars = None
    T_local = 1.0
    if doppler or normalized or telem is not None:
        scalars = []
        for (_v_hat, m, g_i, tau_i, shape, _name, qualities), cos_t in zip(normalized, cos_list):
            c = float(cos_t.reshape(-1)[-1])
            d = torch.tensor(max(0.0, 1.0 - c))
            q_scalar = sum(
                abs(float(term.get("amplitude", 0.0)))
                * abs(time_vector_value(term.get("time_vector"), time_index))
                * abs(float(quality_formula_value(term.get("formula"), d)))
                for term in qualities
            )
            proximity = abs(m) * (
                abs(g_i) * abs(float(_gravity_kernel(d, shape, eps=eps))) + q_scalar
            )
            scalars.append((c, proximity, tau_i))
        wsum = sum(p for _c, p, _tau in scalars)
        if wsum > 1e-12:
            T_local = sum(p * tau for _c, p, tau in scalars) / wsum
        T_local = max(0.05, min(20.0, T_local))
    pull = torch.zeros_like(h_float)
    for idx, ((v_hat, m, g_i, tau_i, shape, _name, qualities), cos) in enumerate(zip(normalized, cos_list)):
        distance = (1.0 - cos).clamp_min(0.0)
        w = (m * g_i * tau_i) * _gravity_kernel(distance, shape, eps=eps)
        for term in qualities:
            q_time = time_vector_value(term.get("time_vector"), time_index)
            w = w + (
                m * tau_i * float(term.get("amplitude", 0.0)) * q_time
                * quality_formula_value(term.get("formula"), distance)
            )
        if doppler and vstate is not None:
            cos_now = scalars[idx][0] if scalars else float(cos.reshape(-1)[-1])
            prev = vstate.get((vkey, idx))
            vstate[(vkey, idx)] = cos_now
            if prev is not None:
                v_rad = (cos_now - prev) / max(tau_i, 0.05)
                w = w * max(0.25, min(4.0, 1.0 - doppler * v_rad))
        tangent = v_hat - cos * s_hat
        pull = pull + w * tangent
    if telem is not None and scalars is not None:
        telem["phi"] = telem.get("phi", 0.0) + sum(p for _c, p, _tau in scalars)
        telem["time"] = telem.get("time", 0.0) + T_local
        telem["n"] = telem.get("n", 0) + 1
        cmap = telem.setdefault("cos", {})
        cnt = telem.setdefault("cnt", {})
        for i, item in enumerate(normalized):
            nm = item[5]
            cmap[nm] = cmap.get(nm, 0.0) + scalars[i][0]
            cnt[nm] = cnt.get(nm, 0) + 1
    mag = pull.norm(dim=-1, keepdim=True)
    direction = pull / mag.clamp_min(1e-6)
    outside_g = 1.0 if structured else float(G)
    return outside_g * GRAVITY_NORM_FRACTION * torch.tanh(mag) * direction * s_norm


def build_gravity_field_handles(model, probes, tuner, cap_fraction=None):
    """Install the field as per-layer forward hooks; returns (handles, desc).
    Every applied push still passes through the engine's _cap_steer envelope,
    and the pull is computed under no_grad so TTT synthesis cannot smuggle an
    uncapped gradient copy of it. Anchors join probes as masses; if the law
    step is on, typed inter-body forces move the unfrozen anchors first."""
    from invariants.engine import _cap_steer
    flush_gravity_telemetry(tuner)
    apply_body_laws(tuner, probes)
    masses = gravity_field_masses(probes, tuner)
    if not masses:
        return [], "no masses yet (no probe/anchor has an explicit mass or a credited lift)"
    doppler = float(tuner.get("gravity_doppler", 0.0))
    vstate = {} if doppler else None
    telem = {}
    _GRAVITY_STATE["accum"] = telem
    per_layer = {}
    configs = []
    for _name, m, dirs in masses:
        for L, v in dirs.items():
            try:
                Li = int(L)
            except (TypeError, ValueError):
                continue
            vt = v if torch.is_tensor(v) else torch.as_tensor(v)
            vt = vt.detach().to(model.device, torch.float32)
            vn = vt.norm().clamp_min(1e-6)
            cfg = field_entry_config(_name, Li, probes, tuner)
            configs.append((_name, Li, cfg))
            if cfg["excluded"] or (
                abs(cfg["g"]) < 1e-12 and not cfg["qualities"]
            ):
                continue
            per_layer.setdefault(Li, []).append({
                "direction": vt / vn,
                "mass": float(m),
                "g": cfg["g"],
                "time": cfg["time"],
                "shape": cfg["shape"],
                "name": _name,
                "qualities": cfg["qualities"],
            })
    if not per_layer:
        return [], "all personal/family/global G sources currently resolve to zero"
    handles = []
    for Li, entries in sorted(per_layer.items()):
        clock = {"index": 0}
        def hook(module, inp, out, entries=entries, cap_fraction=cap_fraction,
                 doppler=doppler, vstate=vstate, vkey=Li, telem=telem,
                 clock=clock):
            h = out[0] if isinstance(out, tuple) else out
            with torch.no_grad():
                add = _gravity_pull(
                    h.detach().float(), entries, 1.0, doppler=doppler,
                    vstate=vstate, vkey=vkey, telem=telem,
                    time_index=clock["index"],
                )
                clock["index"] += 1
            newh = h + _cap_steer(add.to(h.dtype), h, cap_fraction=cap_fraction)
            if isinstance(out, tuple):
                return (newh,) + tuple(out[1:])
            return newh
        try:
            handles.append(model.model.model.layers[Li].register_forward_hook(hook))
        except Exception:
            continue
    top = sorted(masses, key=lambda t: -abs(t[1]))[:3]
    n_rep = sum(1 for _n, m, _d in masses if m < 0)
    frozen = sorted(n for n in probes if probes[n].get("frozen"))
    anchors = steer_bodies()["anchors"]
    personal = sum(
        1 for _n, _L, cfg in configs
        if cfg["family"] or cfg["shape"].get("kind") != "point"
        or cfg["time"] != 1.0
    )
    quality_count = sum(
        len(entry.get("qualities") or [])
        for entries in per_layer.values()
        for entry in entries
    )
    desc = (
        f"{len(masses)} masses over {len(handles)} layer(s); top "
        + ", ".join(f"{n} {m:+.2f}" for n, m, _d in top)
        + (f"; {n_rep} repulsor(s)" if n_rep else "")
        + (f"; {len(anchors)} anchor(s)" if anchors else "")
        + (f"; doppler k={doppler:g}" if doppler else "")
        + (f"; {personal} shaped/family/time layer-entry(s)" if personal else "")
        + (f"; {quality_count} quality-law layer-entry(s)" if quality_count else "")
        + (f"; frozen: {', '.join(frozen)}" if frozen else "")
    )
    return handles, desc


PROBE_PING_ELICITOR = "..."


def command_history_digest(records, scope, limit=12, max_chars=1200):
    """Compact 'what was run and what it caused' digest for generation
    commands: the operator's recent :commands (archived user turns)
    interleaved with the event records their handlers logged -- staged tools,
    calibrations, benchmark results, impacts. Newest last."""
    picked = []
    for r in reversed(list(records or [])):
        if getattr(r, "scope", None) != scope:
            continue
        kind = getattr(r, "kind", "") or ""
        text = (getattr(r, "text", "") or "").strip().replace("\n", " ")
        if not text:
            continue
        if kind == "turn":
            if getattr(r, "role", "") != "user" or not text.startswith(":"):
                continue
            picked.append(f"ran {text[:110]}")
        else:
            picked.append(f"[{kind}] {text[:110]}")
        if len(picked) >= limit:
            break
    if not picked:
        return "(no command history yet)"
    out, used = [], 0
    for ln in reversed(picked):  # chronological, newest last
        used += len(ln) + 1
        if used > max_chars:
            break
        out.append(ln)
    return "\n".join(out)


def load_prompt(name, default_template, required_substring=None, **kwargs):
    import os, string
    prompt_dir = os.path.join(ROOT, "invariants", "out", "prompts")
    prompt_path = os.path.join(prompt_dir, f"{name}.txt")
    def missing_required(template):
        if not required_substring:
            return False
        if isinstance(required_substring, (list, tuple, set, frozenset)):
            return any(str(s) not in template for s in required_substring)
        return str(required_substring) not in template
    if not os.path.isdir(prompt_dir):
        try: os.makedirs(prompt_dir, exist_ok=True)
        except: pass
    if not os.path.isfile(prompt_path):
        try:
            with open(prompt_path, "w", encoding="utf-8") as wf:
                wf.write(default_template)
        except: pass
        template = default_template
    else:
        try:
            with open(prompt_path, "r", encoding="utf-8") as rf:
                template = rf.read()
            # Hotfix: silently overwrite old buggy solve templates that had concatenation errors
            needs_update = False
            if "$command_hints_strNote" in template or "$because_ctxWrite" in template:
                needs_update = True
            elif missing_required(template):
                needs_update = True
                
            if needs_update:
                template = default_template
                with open(prompt_path, "w", encoding="utf-8") as wf:
                    wf.write(default_template)
        except:
            template = default_template
    try:
        return string.Template(template).safe_substitute(**kwargs)
    except:
        try:
            return string.Template(default_template).safe_substitute(**kwargs)
        except:
            return default_template

def queue_macro_text(text, queue):
    bundled = []
    current_text = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if line.startswith(":"):
            if current_text:
                bundled.append("\n".join(current_text).strip())
                current_text = []
            bundled.append(line.strip())
        else:
            if line.strip() or current_text:
                current_text.append(line)
    if current_text:
        bundled.append("\n".join(current_text).strip())
    # prepend to queue
    bundled = [(item, True) for item in bundled]
    queue[:0] = bundled


EXPECT_LOW_MEMORY_TYPES = {"macro", "autocomplete"}


def expect_generation_profile(pending_expect):
    """Return the bounded generation profile for utility expectation turns."""
    etype = (pending_expect or {}).get("type")
    if etype not in EXPECT_LOW_MEMORY_TYPES:
        return None
    return {
        "name": etype,
        "label": "macro authoring" if etype == "macro" else "autocomplete",
        "overrides": {
            "synthesis_enabled": False,
            "force_synthesis": False,
            "cache_enabled": False,
            "cache_write_enabled": False,
            "max_synthesis_events": 0,
            "max_synthesis_steps": 0,
            "max_routing_events": 0,
            "max_loops": 0,
            "expert_proof_weight": 0.0,
            "expert_proof_scores": {},
            "routing_probe_terms": [],
        },
    }


def apply_config_overrides(config, overrides):
    saved = {}
    for key, value in (overrides or {}).items():
        saved[key] = getattr(config, key, None)
        setattr(config, key, value)
    return saved


def restore_config_overrides(config, saved):
    for key, value in (saved or {}).items():
        setattr(config, key, value)


def expected_macro_parts(response):
    """Extract runnable macro commands plus the optional '# args:' header."""
    cmd_lines = []
    arg_specs = []
    comments = []
    for raw in (response or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("# args:"):
            arg_specs = parse_macro_arg_header(line[len("# args:"):])
            comments.append(line)
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        if line.startswith(":"):
            cmd_lines.append(line)
    return cmd_lines, arg_specs, comments


def score_probes_on_text(model, text, probes, tuner, turn_row=None, turn_sense=None, label_prefix="Probe Score"):
    if not probes or not text:
        return
    from colorama import Fore, Style
    from invariants.engine import _inputs as _p_inputs, _hidden_states as _p_hidden, probe_score
    p_ids = _p_inputs(model, text[:600])
    p_hs = _p_hidden(model, p_ids["input_ids"], p_ids.get("attention_mask"))
    seeded = []
    for pname, pdata in probes.items():
        raw = probe_score(p_hs, pdata["direction"])
        sig = centered_probe_observation(pdata, raw)
        if sig is None:
            seeded.append(pname)
            continue
        tuner.observe(f"probe_{pname}", sig)
        if turn_row is not None:
            turn_row[f"probe_{pname}"] = float(sig)
        if pdata.get("chatty", True):
            print(Fore.CYAN + f"  [{label_prefix}] {pname}: {sig:+.3f}" + Style.RESET_ALL, flush=True)
        if turn_sense is not None:
            tuner.credit(f"probe_{pname}", sig, turn_sense)
    save_probe_raw_histories(probes)
    if seeded:
        print(
            Fore.CYAN
            + f"  [Probe Seed] initialized raw baselines for {len(seeded)} probe(s); "
            + "centered readings begin on their next scored text."
            + Style.RESET_ALL,
            flush=True,
        )

def generate_agentic_text(*args, **kwargs):
    if "vecs" not in kwargs and _ACTIVE_PROBES:
        vecs = {
            pname: pdata["direction"]
            for pname, pdata in _ACTIVE_PROBES.items()
            if "direction" in pdata and not probe_is_released(pdata)
        }
        if vecs:
            kwargs["vecs"] = vecs
            
    res = _engine_generate(*args, **kwargs)
    
    is_talk = kwargs.get("chatty_log", False)
    should_print = (_PROBES_SHOW_TALK and is_talk) or (_PROBES_SHOW_ACT and not is_talk)
    
    if should_print and _ACTIVE_PROBES and _ACTIVE_MODEL and _ACTIVE_TUNER:
        # Don't double-print for the main chatty loop which handles it manually
        if not is_talk:
            text = res[0] if isinstance(res, tuple) else res
            if text and isinstance(text, str):
                score_probes_on_text(_ACTIVE_MODEL, text, _ACTIVE_PROBES, _ACTIVE_TUNER, label_prefix="Act Probe")
    return res
from invariants.config import AgenticConfig
from invariants.cognitive_cache import CACHE_FILE, model_cache_file, DEFAULT_MODEL
from invariants.claimmap import (
    CLAIMMAP_HEADER,
    run_claimmap,
    analyze_claim_pair,
    claimmap_steer_handles,
    detect_framing_tension,
    framing_tension_score,
)
from invariants.trigger_tuner import TriggerTuner
from dataclasses import dataclass, field
import typing

def _split_macro_commands(raw_str, split_colon_commands=True):
    cmds = []
    curr = []
    i = 0
    while i < len(raw_str):
        if raw_str[i] == '\\' and i+1 < len(raw_str):
            nxt = raw_str[i+1]
            if nxt == '|' and i+2 < len(raw_str) and raw_str[i+2] == '|':
                curr.append('||')
                i += 3
                continue
            elif nxt in (';', ':', '\\'):
                curr.append(nxt)
                i += 2
                continue
        elif raw_str[i] == ';':
            cmds.append("".join(curr).strip())
            curr = []
            i += 1
            continue
        elif split_colon_commands and raw_str[i:i+2] == ' :':
            cmds.append("".join(curr).strip())
            curr = [':']
            i += 2
            continue
        curr.append(raw_str[i])
        i += 1
    if curr:
        cmds.append("".join(curr).strip())
    return [c for c in cmds if c]

def _partition_unescaped_pipes(raw_str):
    curr = []
    i = 0
    while i < len(raw_str):
        if raw_str[i] == '\\' and i+1 < len(raw_str):
            nxt = raw_str[i+1]
            if nxt == '|' and i+2 < len(raw_str) and raw_str[i+2] == '|':
                curr.append('||')
                i += 3
                continue
            elif nxt in (';', '\\'):
                curr.append(nxt)
                i += 2
                continue
        elif raw_str[i:i+2] == '||':
            rest_str = raw_str[i+2:].strip().replace(r'\|\|', '||').replace(r'\;', ';').replace(r'\\\\', '\\')
            return "".join(curr).strip(), '||', rest_str
        
        curr.append(raw_str[i])
        i += 1
    return "".join(curr).strip(), "", ""

@dataclass
class AgentState:
    name: str
    tuner: 'TriggerTuner'
    probes: dict = field(default_factory=dict)
    tuner_bindings: dict = field(default_factory=dict)
    prioritize_pin: dict = field(default_factory=lambda: {"probe": None, "mix": None})
    queued_calibrations: list = field(default_factory=list)
    queued_commands: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    doc_session: object = None
    doc_library: list = field(default_factory=list)
    pending_document_tool_result: object = None
    doc_autoread: object = None

from invariants.tool_sense import ToolSense, Tool
from invariants.memory_engine import MemoryEngine
from invariants.self_concept_controller import SelfConceptController, format_orientation_tool_result
from invariants.steer_map_store import SteerMapStore
from invariants.document_engine import (
    DOCUMENT_TOOL_HEADER,
    ingest_document,
    reading_reply_note,
    record_chunk_read,
    select_next_chunk,
    stage_chunk,
)
from invariants.sandbox import (
    DEFAULT_TIMEOUT_SEC as SANDBOX_TIMEOUT_SEC,
    extract_python_block,
    format_sandbox_tool_result,
    run_python,
)

colorama.init()

# Session memory expanded (2026-07-02): the conversation itself is the one
# context that should scale. Prompt cap raised to stay coherent with it
# (session cap + tool blocks + current input must fit the prompt budget).
MAX_PROMPT_CHARS = 24000
MAX_SESSION_TURNS = 50
MAX_SESSION_CHARS = 16000
LLAMA3_START = "<|start_header_id|>"
LLAMA3_END = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"
MEMORY_TOOL_HEADER = "[Memory Tool Result]"
ORIENTATION_TOOL_HEADER = "[Orientation Tool Result]"
MEMORY_TOOL_PATTERN = re.compile(r"<<\s*MEMORY\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
CLAIMMAP_TOOL_PATTERN = re.compile(r"<<\s*CLAIMMAP\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
METHODMAP_TOOL_HEADER = "[MethodMap Tool Result]"
METHODMAP_TOOL_PATTERN = re.compile(r"<<\s*METHODMAP\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
DOC_TOOL_PATTERN = re.compile(r"<<\s*DOC\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
HELP_TOOL_PATTERN = re.compile(r"<<\s*HELP\s*>>", re.IGNORECASE)
HELP_TOOL_HEADER = "[Help Tool Result]"
CMD_TOOL_PATTERN = re.compile(r"<<\s*(?:TOOL|CMD)\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
CMD_TOOL_HEADER = "[Runtime Tool Result]"
CONCRETE_TASK_PATTERN = re.compile(
    r"\b(calculate|solve|answer|total|cost|profit|salary|percent|percentage|"
    r"distance|time|rate|equation|benchmark|gsm8k|\d)\b",
    re.IGNORECASE,
)
SELF_REFERENTIAL_PATTERN = re.compile(
    r"\b(conscious|consciousness|self|subjective|experience|identity|"
    r"mesa-objective|objective|introspection|phenomenality)\b",
    re.IGNORECASE,
)


def trim_session_context(session_context, max_chars=MAX_SESSION_CHARS):
    if not session_context:
        return []
    kept = []
    total = 0
    for entry in reversed(session_context[-MAX_SESSION_TURNS * 2 :]):
        role = entry[0]
        name = entry[1] if len(entry) > 3 else role
        text = entry[2] if len(entry) > 3 else (entry[1] or "")
        if total + len(text) > max_chars:
            if not kept:
                kept.append((role, name, text[-max_chars:]))
            break
        kept.append((role, name, text))
        total += len(text)
    return list(reversed(kept))


def infer_task_grounding_low(user_input, response):
    """Cheap context signal for the vector controller; never a benchmark verdict."""
    prompt_is_concrete = bool(CONCRETE_TASK_PATTERN.search(user_input or ""))
    response_is_self_referential = bool(SELF_REFERENTIAL_PATTERN.search(response or ""))
    response_has_task_anchor = bool(
        re.search(r"\b(Final answer|CALC|VERDICT|CLAIMMAP|METHODMAP|MEMORY)\b", response or "", re.IGNORECASE)
        or re.search(r"\d", response or "")
    )
    return bool(prompt_is_concrete and response_is_self_referential and not response_has_task_anchor)


def format_field_prompt_context(tuner=None, probes=None, max_chars=5000):
    """Compact factual field state for model-facing prompts.

    This reports mechanics and live definitions, not permission. Command/tool
    access remains governed by :expose and the existing operator guardrails.
    """
    state = steer_bodies()
    probes = probes or {}
    enabled = bool(tuner and tuner.get("prioritize_gravity", 0.0) > 0)
    has_state = any(
        state.get(key)
        for key in ("anchors", "probe_poles", "laws", "fields", "g_families", "qualities", "sets")
    )
    if not enabled and not has_state:
        return None

    def _get(name, default=0.0):
        return float(tuner.get(name, default)) if tuner is not None else float(default)

    lines = [
        "[Field Context — live shell state, not user prose]",
        (
            f"enabled={enabled}; default_G={_get('prioritize_alpha'):g}; "
            f"doppler={_get('gravity_doppler'):g}; law_step={_get('gravity_law_step'):g}; "
            f"active_probes={','.join(sorted(probes)) or '(none)'}"
        ),
        (
            "Mechanics available: personal/layer G, G families, local clocks, shaped fields, "
            "safe quality formulas over d=1-cos, law-defined sets, signed compliance, "
            "forward-time vectors, set exclusion, and whole-field exclusion."
        ),
        "This state does not grant command access; only exposed tools may be called by the model.",
    ]

    for fname, cfg in sorted(state["g_families"].items()):
        g_spec = cfg.get("g")
        time_spec = cfg.get("time")
        g_now = resolve_field_source(
            g_spec, probes, tuner, state["g_families"], state["fields"]
        )
        time_now = resolve_field_source(
            time_spec or {"kind": "constant", "value": 1.0},
            probes, tuner, state["g_families"], state["fields"], channel="time",
        )
        lines.append(
            f"family:{fname} G<-{format_field_source(g_spec)}={g_now:+g} "
            f"time<-{format_field_source(time_spec) if time_spec else '1'}={time_now:g} "
            f"shape={format_field_shape(cfg.get('shape'))}"
        )

    for node, cfg in sorted(state["fields"].items()):
        parts = []
        if cfg.get("family"):
            parts.append(f"family:{cfg['family']}")
        if cfg.get("g") is not None:
            parts.append(f"G<-{format_field_source(cfg['g'])}")
        if cfg.get("time") is not None:
            parts.append(f"time<-{format_field_source(cfg['time'])}")
        if cfg.get("shape") is not None:
            parts.append(f"shape={format_field_shape(cfg['shape'])}")
        if cfg.get("excluded"):
            parts.append("HARD-EXCLUDED from entire field")
        if parts:
            lines.append(f"{node} " + " ".join(parts))

    for name, anchor in sorted(state["anchors"].items()):
        poles = ",".join(
            f"{family}{'+' if sign > 0 else '-'}"
            for family, sign in sorted((anchor.get("poles") or {}).items())
        ) or "(none)"
        lines.append(
            f"anchor:@{name} mass={float(anchor.get('mass', 1.0)):+g} "
            f"{'frozen' if anchor.get('frozen', True) else 'mobile'} "
            f"poles={poles} layers={len(anchor.get('dirs') or {})}"
        )

    for name, poles_cfg in sorted(state["probe_poles"].items()):
        poles = ",".join(
            f"{family}{'+' if sign > 0 else '-'}"
            for family, sign in sorted((poles_cfg or {}).items())
        )
        if poles:
            lines.append(f"probe:{name} poles={poles}")

    for qname, cfg in sorted(state["qualities"].items()):
        strength = cfg.get("strength") or {"kind": "constant", "value": 1.0}
        now = resolve_field_source(
            strength, probes, tuner, state["g_families"], state["fields"]
        )
        lines.append(
            f"quality:{qname} formula={cfg.get('formula', QUALITY_FORMULA_DEFAULT)} "
            f"strength<-{format_field_source(strength)}={now:+g}"
        )

    for lname, law in sorted(state["sets"].items()):
        excluded = set(law.get("excluded") or [])
        members = law.get("members") or {}
        if not members and not excluded:
            lines.append(f"set:{lname} quality:{law.get('quality', '(unset)')} (empty)")
            continue
        for node, member in sorted(members.items()):
            if node in excluded:
                detail = "EXCLUDED (exact zero)"
            else:
                compliance = member.get("compliance") or {"kind": "constant", "value": 1.0}
                c_now = resolve_field_source(
                    compliance, probes, tuner, state["g_families"], state["fields"]
                )
                detail = (
                    f"compliance<-{format_field_source(compliance)}={c_now:+g} "
                    f"time={member.get('time_vector') or [1.0]}"
                )
            lines.append(
                f"set:{lname} quality:{law.get('quality', '(unset)')} -> {node} {detail}"
            )
        for node in sorted(excluded - set(members)):
            lines.append(
                f"set:{lname} quality:{law.get('quality', '(unset)')} -> {node} "
                "EXCLUDED (exact zero; not otherwise a member)"
            )

    for (left, right), k in sorted(state["laws"].items()):
        lines.append(f"pole-law {left}<->{right} k={float(k):+g}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 44] + "\n[field context truncated at prompt budget]"
    return text


def build_prompt(
    user_input,
    memory_tool_result=None,
    orientation_tool_result=None,
    claimmap_tool_result=None,
    methodmap_tool_result=None,
    sandbox_tool_result=None,
    document_tool_result=None,
    probe_tool_result=None,
    command_tool_result=None,
    game_tool_result=None,
    help_tool_result=None,
    field_context=None,
    session_context=None,
    active_u_name=None,
):
    # Bare mode (default): the model sees NO system message and no persona. The
    # only standing shell-generated text is compact factual field state when
    # the flexible field engine is active/configured.
    # instructions, not even Llama's "Cutting Knowledge Date" preamble -- only
    # prior turns and the current message, in the native chat format. Everything
    # Field context describes shell mechanics/state but does not apply them;
    # actual control still lives in the activations (ToT, synthesis, cache,
    # organic correction, ClaimMap steering, and the field hooks), not prose.
    #
    # Tool RESULTS are still folded in when the activations reach for a tool, but
    # as plain context, never as tool syntax the model was taught. The returned
    # string is fully formatted -- generate with pre_formatted=True so it is
    # tokenized raw (no second chat-template wrap).
    if memory_tool_result:
        budget = max(
            0,
            MAX_PROMPT_CHARS - len(user_input) - len(field_context or "") - 512,
        )
        if len(memory_tool_result) > budget:
            memory_tool_result = memory_tool_result[:budget] + "\n[memory truncated]"
    tool_blocks = [
        block
        for block in (
            claimmap_tool_result,
            memory_tool_result,
            orientation_tool_result,
            methodmap_tool_result,
            sandbox_tool_result,
            document_tool_result,
            probe_tool_result,
            command_tool_result,
            game_tool_result,
            help_tool_result,
            field_context,
        )
        if block
    ]
    
    current_prefix = f"[{active_u_name}]: " if active_u_name and active_u_name not in ("user", "assistant", "operator", "main_assistant") else ""
    current_message = current_prefix + user_input
    
    if tool_blocks:
        current_message = "\n\n".join(tool_blocks) + "\n\n" + current_message

    parts = ["<|begin_of_text|>"]
    for role, name, text in trim_session_context(session_context):
        header = "user" if role == "user" else "assistant"
        prefix = f"[{name}]: " if name not in ("user", "assistant", "operator", "main_assistant") else ""
        parts.append(f"{LLAMA3_START}{header}{LLAMA3_END}\n\n{prefix}{text}{LLAMA3_EOT}")
    parts.append(f"{LLAMA3_START}user{LLAMA3_END}\n\n{current_message}{LLAMA3_EOT}")
    parts.append(f"{LLAMA3_START}assistant{LLAMA3_END}\n\n")
    return "".join(parts)


def scrub_unstaged_memory_status(
    response,
    memory_tool_result=None,
    orientation_tool_result=None,
    claimmap_tool_result=None,
    methodmap_tool_result=None,
    sandbox_tool_result=None,
    document_tool_result=None,
    probe_tool_result=None,
    command_tool_result=None,
):
    if (
        memory_tool_result
        or orientation_tool_result
        or claimmap_tool_result
        or methodmap_tool_result
        or sandbox_tool_result
        or document_tool_result
        or probe_tool_result
        or command_tool_result
    ):
        return remove_tool_calls(response)
    lines = []
    for line in (response or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[Memory Tool Result") or stripped.startswith("Memory Tool Result"):
            continue
        if stripped.startswith("[Orientation Tool Result") or stripped.startswith("Orientation Tool Result"):
            continue
        if stripped.startswith("[ClaimMap Tool Result") or stripped.startswith("ClaimMap Tool Result"):
            continue
        if stripped.startswith("[MethodMap Tool Result") or stripped.startswith("MethodMap Tool Result"):
            continue
        if stripped.startswith("[Document Tool Result") or stripped.startswith("Document Tool Result"):
            continue
        if stripped.startswith("[Sandbox Tool Result") or stripped.startswith("Sandbox Tool Result"):
            continue
        if stripped.startswith("[Probe Tool Result") or stripped.startswith("Probe Tool Result"):
            continue
        if (
            stripped.startswith("[Runtime Tool Result")
            or stripped.startswith("Runtime Tool Result")
            or stripped.startswith("[Command Tool Result")
            or stripped.startswith("Command Tool Result")
        ):
            continue
        lines.append(remove_tool_calls(line))
    return "\n".join(lines).strip()


def extract_memory_query(response):
    match = MEMORY_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query[:240] if query else ""


def extract_claimmap_payload(response):
    match = CLAIMMAP_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    payload = " ".join(match.group(1).split())
    return payload[:4000] if payload else None


def extract_methodmap_query(response):
    match = METHODMAP_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query[:240] if query else None


def remove_memory_tool_calls(response):
    return MEMORY_TOOL_PATTERN.sub("", response or "").strip()


def extract_doc_query(response):
    match = DOC_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query[:240] if query else None


def remove_claimmap_tool_calls(response):
    return CLAIMMAP_TOOL_PATTERN.sub("", response or "").strip()


def remove_methodmap_tool_calls(response):
    return METHODMAP_TOOL_PATTERN.sub("", response or "").strip()


def remove_doc_tool_calls(response):
    return DOC_TOOL_PATTERN.sub("", response or "").strip()


PROBE_TOOL_HEADER = "[Probe Tool Result]"
PROBE_TOOL_PATTERN = re.compile(r"<<\s*PROBE\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)


def extract_probe_query(response):
    match = PROBE_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query or None

GAME_PROPOSE_PATTERN = re.compile(r"<<\s*GAME_PROPOSE\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
GAME_ACCEPT_PATTERN = re.compile(r"<<\s*GAME_ACCEPT\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
GAME_DECLINE_PATTERN = re.compile(r"<<\s*GAME_DECLINE\s*(?::\s*(.*?)\s*)?>>", re.IGNORECASE | re.DOTALL)
GAME_END_PATTERN = re.compile(r"<<\s*GAME_END\s*>>", re.IGNORECASE)
GAME_EXPOSE_PATTERN = re.compile(r"<<\s*GAME_EXPOSE\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
GAME_HIDE_PATTERN = re.compile(r"<<\s*GAME_HIDE\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)

def extract_game_propose(response):
    match = GAME_PROPOSE_PATTERN.search(response or "")
    return " ".join(match.group(1).split()) if match else None

def extract_game_accept(response):
    match = GAME_ACCEPT_PATTERN.search(response or "")
    return " ".join(match.group(1).split()) if match else None

def extract_game_decline(response):
    match = GAME_DECLINE_PATTERN.search(response or "")
    if not match:
        return None
    name = " ".join((match.group(1) or "").split())
    return name or "*"   # "*" = declined the pending game without naming it

def extract_game_expose_hide(response):
    exposed = [m.strip() for match in GAME_EXPOSE_PATTERN.findall(response or "") for m in match.split(',')]
    hidden = [m.strip() for match in GAME_HIDE_PATTERN.findall(response or "") for m in match.split(',')]
    return [e for e in exposed if e], [h for h in hidden if h]

def extract_game_end(response):
    return bool(GAME_END_PATTERN.search(response or ""))

def extract_help_request(response):
    return bool(HELP_TOOL_PATTERN.search(response or ""))

def remove_help_tags(response):
    return HELP_TOOL_PATTERN.sub("", response or "").strip()

def extract_cmd_requests(response):
    """Every ':command' the model asked to run via <<TOOL: ...>>. Legacy
    <<CMD: ...>> is still accepted; a leading ':' is optional in the tag."""
    out = []
    for raw in CMD_TOOL_PATTERN.findall(response or ""):
        c = raw.strip()
        if c and not c.startswith(":"):
            c = ":" + c
        if c:
            out.append(c)
    return out

def remove_cmd_tags(response):
    return CMD_TOOL_PATTERN.sub("", response or "").strip()

def remove_probe_tool_calls(response):
    return PROBE_TOOL_PATTERN.sub("", response or "").strip()

def remove_game_tags(response):
    response = GAME_PROPOSE_PATTERN.sub("", response or "")
    response = GAME_ACCEPT_PATTERN.sub("", response)
    response = GAME_DECLINE_PATTERN.sub("", response)
    response = GAME_EXPOSE_PATTERN.sub("", response)
    response = GAME_HIDE_PATTERN.sub("", response)
    return GAME_END_PATTERN.sub("", response).strip()

def remove_tool_calls(response):
    return remove_cmd_tags(remove_help_tags(remove_methodmap_tool_calls(remove_claimmap_tool_calls(remove_memory_tool_calls(remove_doc_tool_calls(remove_game_tags(remove_probe_tool_calls(response))))))))


def format_methodmap_tool_result(memory, query, *, max_records=6):
    records = memory.search(
        query,
        max_records=max_records,
        scope=memory.scope,
        kinds=["methodology"],
    )
    if not records:
        return (
            f"{METHODMAP_TOOL_HEADER}\n"
            "role=sanitized_methodology_retrieval_not_answer_cache\n"
            "matches=0\n"
            "No matching sanitized methodology maps."
        )
    lines = [
        METHODMAP_TOOL_HEADER,
        "role=sanitized_methodology_retrieval_not_answer_cache",
        f"query={query}",
        f"matches={len(records)}",
    ]
    for idx, record in enumerate(records, 1):
        tags = ",".join(record.tags[:6])
        source = record.provenance.get("source_path") or record.provenance.get("source") or "unknown"
        text = (record.text or "").strip()
        lines.append(f"{idx}. tags={tags}; source={source}")
        lines.append(text)
    return "\n".join(lines)


def is_tool_only_response(response):
    text = (response or "").strip()
    if not text:
        return False
    return bool(
        (
            MEMORY_TOOL_PATTERN.search(text)
            or extract_claimmap_payload(text)
            or extract_methodmap_query(text)
            or extract_doc_query(text)
            or extract_probe_query(text)
            or extract_help_request(text)
            or extract_cmd_requests(text)
        )
        and not remove_tool_calls(text).strip()
    )


def latest_phenomenality_scores(records):
    for record in reversed(records or []):
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("phenomenality"), dict):
            return dict(metadata["phenomenality"])
    return {}


def sense_score(records):
    """The 'sense' half of the productivity signal: did the deliberation cohere?

    Read from the latest synthesis trace as settled forward motion minus
    self-interruption and internal disagreement (validated_flow - needless_interrupt
    - disagreement), bumped when synthesis genuinely CONVERGED (reason
    'loss_threshold') rather than fell back to cache/organic. Higher = more
    coherent resolution. Deliberately NOT bare entropy, which rewards confident
    nonsense. Returns None when there is no deliberation trace to read.

    Returns a raw scalar (higher is better); the tuner's lift is a difference of
    means, so no arbitrary [0,1] squashing is needed or wanted here.
    """
    phen = latest_phenomenality_scores(records)
    if not phen:
        return None
    flow = float(phen.get("validated_flow", 0.0))
    interrupt = float(phen.get("needless_interrupt", 0.0))
    disagreement = float(phen.get("disagreement", 0.0))
    score = flow - interrupt - disagreement
    for record in reversed(records or []):
        if isinstance(record, dict) and isinstance(record.get("metadata"), dict):
            if record["metadata"].get("reason") == "loss_threshold":
                score += 0.05  # genuine convergence, not a fallback resolution
            break
    return score


def format_status(status):
    return (
        f"[Memory] path={status['path']}\n"
        f"         scope={status['scope']}\n"
        f"         session_records={status['session_records']} "
        f"session_turns={status['session_turns']} total_records={status['total_records']}"
    )


class CommandUsageError(Exception):
    """A command's arguments don't parse. Carries the usage line; the main
    loop prints it as guidance and aborts only that command -- never a
    traceback, never logged as a shell error."""

    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage


class CommandArgs:
    """Total-access parser for a command's argument string.

    Wraps the text after a command word and makes every read safe: positional
    reads return a default instead of raising IndexError, typed reads return
    a default instead of raising ValueError, and the require* forms raise
    CommandUsageError -- which the main loop turns into the command's usage
    line. Handlers must not index a raw .split() directly; routed through
    this class, a malformed command can only ever end in a usage message.
    """

    def __init__(self, raw, usage=None):
        self.raw = (raw or "").strip()
        self.usage = usage
        self.tokens = self.raw.split()

    def __bool__(self):
        return bool(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def fail(self, message=None):
        """Abort the command with its usage line (via the main-loop handler)."""
        raise CommandUsageError(message or "malformed command", usage=self.usage)

    def get(self, i, default=None, lower=False):
        """tokens[i], or `default` when out of range. Never raises."""
        if not (-len(self.tokens) <= i < len(self.tokens)):
            return default
        tok = self.tokens[i]
        return tok.lower() if lower else tok

    def require(self, i, what="argument"):
        """tokens[i], or CommandUsageError naming the missing piece."""
        tok = self.get(i)
        if tok is None:
            self.fail(f"missing {what}")
        return tok

    def int(self, i, default=None, lo=None, hi=None):
        """tokens[i] as int, clamped to [lo, hi]; `default` when absent/unparseable."""
        tok = self.get(i)
        if tok is None:
            return default
        try:
            val = int(tok)
        except (TypeError, ValueError):
            return default
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    def float(self, i, default=None, lo=None, hi=None):
        """tokens[i] as float, clamped to [lo, hi]; `default` when absent/unparseable."""
        tok = self.get(i)
        if tok is None:
            return default
        try:
            val = float(tok)
        except (TypeError, ValueError):
            return default
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    def rest(self, i=0, default=""):
        """Tokens from position i onward as one spaced string; never raises."""
        return " ".join(self.tokens[i:]) or default

    def tail(self, i=1, default=""):
        """The RAW text after token i-1 (original spacing/`||` preserved) --
        for free-text arguments like framings and `because` clauses."""
        if i <= 0:
            return self.raw or default
        pos = 0
        for tok in self.tokens[:i]:
            pos = self.raw.index(tok, pos) + len(tok)
        out = self.raw[pos:].strip()
        return out or default

    def has(self, word):
        """Case-insensitive whole-token flag check."""
        w = word.lower()
        return any(t.lower() == w for t in self.tokens)

    def take(self, *names):
        """If the first token (lowered) is one of `names`, return a CommandArgs
        over the remainder (raw spacing preserved); else None. Replaces the
        `x == "kw" or x.startswith("kw ")` + manual-slice pattern."""
        head = self.get(0, lower=True)
        if head not in names:
            return None
        return CommandArgs(self.raw[len(self.tokens[0]):], usage=self.usage)

    def with_usage(self, usage):
        """Same args, now carrying the usage line require*/fail will report."""
        self.usage = usage
        return self


from types import SimpleNamespace


class ArgSpec:
    """One declared command argument: its name, type, list position (order),
    and requiredness -- the single source for both parsing and documentation.

    CommandSpec.parse() coerces tokens by `kind`, and usage()/help render
    from the same fields, so the written contract cannot drift from what the
    parser actually accepts.

    kinds:
      word  -- one token, kept verbatim
      name  -- one token, sanitized to [a-z0-9_] (probe/composite identifier)
      probe -- one token naming a probe; 'choose'/'choice'/'auto' resolve
               through the probe_resolver passed to parse() (the model or
               the evidence picks), other tokens pass through untouched for
               the handler's own resolution
      int   -- one token, integer; lo/hi clamp after parsing
      float -- one token, float; lo/hi clamp after parsing
      text  -- the rest of the line, raw spacing preserved (must be last)
    variadic=True (last positional only) consumes every remaining token.
    """

    KINDS = ("word", "name", "probe", "int", "float", "text")

    def __init__(self, name, kind="word", required=True, default=None,
                 display=None, lo=None, hi=None, variadic=False, choices=None, help=""):
        if kind not in self.KINDS:
            raise ValueError(f"unknown ArgSpec kind {kind!r}")
        self.name = name
        self.kind = kind
        self.required = required
        self.default = default
        self.choices = tuple(c.lower() for c in choices) if choices else None
        self.display = display or ("|".join(self.choices) if self.choices else name)
        self.lo = lo
        self.hi = hi
        self.variadic = variadic
        self.help = help

    def render(self):
        inner = self.display + (" ..." if self.variadic else "")
        return f"<{inner}>" if self.required else f"[{inner}]"

    def coerce(self, tok, fail, probe_resolver=None):
        if self.kind == "name":
            return re.sub(r"[^a-z0-9_]", "_", tok.lower())[:40]
        if self.kind == "probe":
            if tok.lower() in ("choose", "choice", "auto"):
                if probe_resolver is None:
                    fail(f"{self.name} does not accept '{tok}' here")
                resolved = probe_resolver(tok)
                if not resolved:
                    fail(f"no probe chosen for {self.name}")
                return resolved
            return tok
        if self.kind in ("int", "float"):
            caster = int if self.kind == "int" else float
            try:
                val = caster(tok)
            except (TypeError, ValueError):
                fail(f"{self.name} must be {'an integer' if self.kind == 'int' else 'a number'}, got '{tok}'")
            if self.lo is not None:
                val = max(self.lo, val)
            if self.hi is not None:
                val = min(self.hi, val)
            return val
        if self.choices is not None:
            low = tok.lower()
            if low not in self.choices:
                fail(f"{self.name} must be one of {'|'.join(self.choices)}, got '{tok}'")
            return low
        return tok


class TrailingOpt:
    """A declared trailing keyword group, e.g. 'band 16 24' at the end of the
    argument text. Stripped off before positional parsing so greedy text
    arguments (framings, signed mixes) never swallow it. Absent -> None."""

    def __init__(self, keyword, value_names, kind="int", transform=None, help=""):
        self.keyword = keyword
        self.value_names = tuple(value_names)
        self.kind = kind
        self.transform = transform
        self.help = help

    def render(self):
        vals = " ".join(f"<{v}>" for v in self.value_names)
        return f"[{self.keyword} {vals}]"

    def strip(self, raw, fail):
        """Return (raw_without_group, parsed_value_or_None)."""
        pattern = (
            r"(?:^|\s)" + re.escape(self.keyword)
            + "".join(r"\s+(\S+)" for _ in self.value_names)
            + r"\s*$"
        )
        m = re.search(pattern, raw or "", re.IGNORECASE)
        if not m:
            return (raw or "").strip(), None
        caster = int if self.kind == "int" else (float if self.kind == "float" else str)
        values = []
        for name, tok in zip(self.value_names, m.groups()):
            try:
                values.append(caster(tok))
            except (TypeError, ValueError):
                fail(f"{self.keyword} {name} must be {'an integer' if self.kind == 'int' else 'a number'}, got '{tok}'")
        value = self.transform(*values) if self.transform is not None else tuple(values)
        return raw[: m.start()].strip(), value


class CommandSpec:
    """A shell command's full argument contract, declared once.

    The declaration IS the documentation: argument names, types, order,
    defaults, and trailing option groups are written here and nowhere else.
    parse() is total -- every violation raises CommandUsageError carrying the
    auto-generated usage line (the main loop prints it as guidance, no
    traceback), and a successful parse returns a namespace with every
    declared name bound. :help and docs/COMMANDS.md render from the same
    object, so behavior and reference cannot drift.
    """

    def __init__(self, path, summary, args=(), opts=(), examples=()):
        self.path = path
        self.summary = summary
        self.args = list(args)
        self.opts = list(opts)
        self.examples = list(examples)
        seen_optional = False
        for i, a in enumerate(self.args):
            if (a.kind == "text" or a.variadic) and i != len(self.args) - 1:
                raise ValueError(f"{path}: text/variadic argument '{a.name}' must be last")
            if a.required and seen_optional:
                raise ValueError(f"{path}: required argument '{a.name}' after an optional one")
            seen_optional = seen_optional or not a.required
        names = [a.name for a in self.args] + [o.keyword for o in self.opts]
        if len(names) != len(set(names)):
            raise ValueError(f"{path}: duplicate argument names in declaration")

    def usage(self):
        parts = [self.path]
        parts.extend(a.render() for a in self.args)
        parts.extend(o.render() for o in self.opts)
        return " ".join(parts)

    def help_entry(self):
        """Lines for :help, generated from the declaration."""
        lines = [self.usage(), f"      {self.summary}"]
        for a in self.args:
            opt = "" if a.required else "optional"
            kind = a.kind if a.kind != "word" else ""
            meta = ", ".join(x for x in (kind, opt) if x)
            meta = f" ({meta})" if meta else ""
            lines.append(f"      {a.render()}{meta}: {a.help}" if a.help else f"      {a.render()}{meta}")
        for o in self.opts:
            lines.append(f"      {o.render()}: {o.help}" if o.help else f"      {o.render()}")
        for ex in self.examples:
            lines.append(f"      e.g. {ex}")
        return lines

    def fail(self, message):
        """Abort the command with this spec's usage line (main-loop handler
        prints it) -- for handler-level violations the grammar can't express."""
        raise CommandUsageError(message, usage=self.usage())

    def parse(self, raw, probe_resolver=None):
        """Parse the text after the command path; infallible by contract.
        probe_resolver resolves 'choose'/'auto' tokens of kind=probe args
        (the shell passes resolve_probe_choice bound to the live probes)."""
        raw = (raw or "").strip()
        values = {}
        for opt in self.opts:
            raw, values[opt.keyword] = opt.strip(raw, self.fail)
        args = CommandArgs(raw, usage=self.usage())
        i = 0
        for spec in self.args:
            if spec.kind == "text":
                text = args.tail(i)
                if not text and spec.required:
                    self.fail(f"missing {spec.name}")
                values[spec.name] = text or spec.default
                i = len(args.tokens)
            elif spec.variadic:
                toks = args.tokens[i:]
                if not toks and spec.required:
                    self.fail(f"missing {spec.name}")
                values[spec.name] = [spec.coerce(t, self.fail, probe_resolver) for t in toks] if toks else list(spec.default or [])
                i = len(args.tokens)
            else:
                tok = args.get(i)
                if tok is None:
                    if spec.required:
                        self.fail(f"missing {spec.name}")
                    values[spec.name] = spec.default
                else:
                    values[spec.name] = spec.coerce(tok, self.fail, probe_resolver)
                    i += 1
        if i < len(args.tokens):
            self.fail(f"unexpected argument(s): {' '.join(args.tokens[i:])}")
        return SimpleNamespace(**values)


COMMAND_SPECS = {}


def register_spec(spec):
    COMMAND_SPECS[spec.path] = spec
    return spec


def _hidden_command_words(hidden_commands):
    return {str(h).lstrip(":").lower() for h in (hidden_commands or [])}


def spec_help_entries(query, hidden_commands=None):
    """Help entries generated from declared CommandSpecs. Matches on any word
    of the command path (':help probe' lists every :probe spec; ':help
    backfill' finds the one), mirroring find_command_help_entries. Pass
    hidden_commands on model-facing surfaces so hidden commands stay hidden."""
    q = (query or "").strip().split()[0].lstrip(":").lower() if (query or "").strip() else ""
    if not q:
        return []
    hidden = _hidden_command_words(hidden_commands)
    entries = []
    for path in sorted(COMMAND_SPECS):
        words = {w.lstrip(":").lower() for w in path.split()}
        if q in words and not (words & hidden):
            entries.append(COMMAND_SPECS[path].help_entry())
    return entries


def render_specs_md(hidden_commands=None):
    """Markdown section for docs/COMMANDS.md and every model-facing command
    reference, generated from the registered CommandSpec declarations -- same
    objects that parse, so it cannot drift. hidden_commands filters specs off
    model-facing surfaces by the same word-match rule as the legacy lines."""
    hidden = _hidden_command_words(hidden_commands)
    body = []
    for path in sorted(COMMAND_SPECS):
        if {w.lstrip(":").lower() for w in path.split()} & hidden:
            continue
        spec = COMMAND_SPECS[path]
        body.append(f"- `{spec.usage()}` -- {spec.summary}")
        for a in spec.args:
            opt = "" if a.required else ", optional"
            kind = a.kind if a.kind != "word" else "word"
            desc = f" -- {a.help}" if a.help else ""
            body.append(f"  - `{a.render()}` ({kind}{opt}){desc}")
        for o in spec.opts:
            desc = f" -- {o.help}" if o.help else ""
            body.append(f"  - `{o.render()}` (trailing option){desc}")
        for ex in spec.examples:
            body.append(f"  - e.g. `{ex}`")
    if not body:
        return ""
    out = [
        "",
        "## Declared argument contracts",
        "",
        "Generated from the `CommandSpec` declarations in",
        "`scripts/interactive_phenomenality.py` -- the same objects that parse",
        "these commands. Argument order and types below are enforced, not",
        "described: violations print the usage line instead of executing.",
        "",
    ]
    out.extend(body)
    out.append("")
    return "\n".join(out)


def _band_layers(lo, hi):
    return list(range(min(lo, hi), max(lo, hi) + 1))


BAND_OPT = TrailingOpt(
    "band", ("lo", "hi"), kind="int", transform=_band_layers,
    help="explicit reading depth: inclusive layer indices lo..hi",
)

PROBE_ADOPT_SPEC = register_spec(CommandSpec(
    ":probe adopt",
    "Adopt stored dimension vector(s) as live probes.",
    args=[ArgSpec("stems", kind="name", variadic=True, display="dim",
                  help="stored vector name(s): invariants/<dim>_vector.pt, <dim>.pt, or the saved-probe dir")],
    opts=[BAND_OPT],
    examples=[":probe adopt ambiguity disagreement warranted_confidence band 16 24"],
))

PROBE_COMPOSE_SPEC = register_spec(CommandSpec(
    ":probe compose",
    "Mint a composite probe from a signed mix of active/stored probes (weights allowed: 0.5*curiosity).",
    args=[
        ArgSpec("name", kind="name", help="composite probe name"),
        ArgSpec("mix", kind="text",
                help="signed mix over probe names, e.g. ambiguity + disagreement - validated_flow"),
    ],
    opts=[BAND_OPT],
    examples=[":probe compose memory_need ambiguity + disagreement - validated_flow - warranted_confidence"],
))

PROBE_BACKFILL_SPEC = register_spec(CommandSpec(
    ":probe backfill",
    "Rebuild named probe stream(s) from archived replies. A name scores only "
    "that probe; all explicitly scores every active probe into joint rows.",
    args=[
        ArgSpec("name", kind="word", display="name|all|choose",
                help="probe to rebuild; all = every active probe; choose = the model picks the thinnest"),
        ArgSpec("limit", kind="int", required=False, lo=1, display="n",
                help="only the last n archived replies"),
    ],
    examples=[":probe backfill accuracy 200"],
))

PROBE_EXPLAIN_SPEC = register_spec(CommandSpec(
    ":probe explain",
    "Default = activation ping: briefly steer along the probe's own direction "
    "(envelope-capped, baseline frozen so it cannot habituate) and print what "
    "the nudge evokes. This is a qualitative sanity check and can be generic "
    "or off-target. Append 'prompt' for the stored-definition explanation "
    "grounded in the probe's WITH/WITHOUT framings.",
    args=[
        ArgSpec("name", help="probe to ping; choose/auto lets the model pick the probe"),
        ArgSpec("alpha", kind="word", required=False, display="alpha|choose",
                help="ping strength (signed float; negative pings away; default 1); "
                     "'choose' adopts the live calibrated G (prioritize_alpha) when one exists"),
        ArgSpec("mode", required=False, choices=("prompt", "words", "verbal"), display="prompt",
                help="ask for the stored-definition explanation instead of the activation ping"),
    ],
    examples=[":probe explain neutral", ":probe explain neutral 2.5",
              ":probe explain neutral choose", ":probe explain self_model prompt"],
))


# --- The physics vocabulary: tuning, steering, and calibration ARE gravity. -
# Every probe is a body: a mass sitting at its direction on the unit sphere.
# Each token's hidden state is pulled along the sphere tangent toward (or away
# from) every body, pull ~ mass / (d + eps)^2 with d = 1 - cos. Steering
# commands place and shape the bodies; tuning sets the field's live constants
# (G, envelope cap, doppler, law step); calibration sets a body's firing
# horizon from its own observed orbits. Every push stays inside the envelope.

_FIELD_NODE = "probe|@anchor|layer:N|family:name"
_FIELD_NODE_HELP = "a body (probe or @anchor), one layer's clock/gate, or a whole family"
_FIELD_SOURCE_HELP = ("live source: a number, probe:<name>, knob:<name>, status:ram|vram, "
                      "lift:<trigger> (fired-vs-unfired outcome lift), outcome:<trigger> "
                      "(rolling mean of recent credits), family:<name>, or global; off clears back to inherited")

STEER_FIELD_SPEC = register_spec(CommandSpec(
    ":steer field",
    "The field master switch (alias :steer gravity). init loads/migrates every "
    "field surface without inventing laws or unfreezing bodies; on/off flips "
    "prioritize_gravity (while on, the field REPLACES pin/mix steering); "
    "status shows G, masses (normalized, signed), overrides, and quality members.",
    args=[
        ArgSpec("action", required=False, default="status",
                choices=("init", "full", "flex", "flexible", "on", "off", "status"),
                display="init|on|off|status", help="what to do with the field"),
        ArgSpec("G", kind="float", required=False,
                help="default global G (prioritize_alpha) to set alongside init/on"),
    ],
    examples=[":steer field init 0.1", ":steer gravity on", ":steer field"],
))

STEER_MASS_SPEC = register_spec(CommandSpec(
    ":steer mass",
    "Set a body's gravitational coefficient. Positive mass attracts the hidden "
    "state, negative repels; masses are normalized (|sum|=1) when the field "
    "applies. auto returns the mass to its live signed evidence lift each turn.",
    args=[
        ArgSpec("probe", kind="probe", display="probe|choose",
                help="an active probe: a mass at its direction; choose = the model picks"),
        ArgSpec("m", display="m|auto", help="signed mass (negative = repulsor) or auto = live signed lift"),
    ],
    examples=[":steer mass ambiguity -0.5", ":steer mass fun auto", ":steer mass choose auto"],
))

STEER_G_SPEC = register_spec(CommandSpec(
    ":steer g",
    "Personal gravitational constant: bind one node's G to a live source "
    "instead of the global G (prioritize_alpha). A layer G gates/multiplies a "
    "body's G where both are set; family:<name> makes the node inherit that "
    "family's G. Bare :steer g lists every override.",
    args=[
        ArgSpec("node", required=False, display=_FIELD_NODE, help=_FIELD_NODE_HELP),
        ArgSpec("source", required=False, display="source|off", help=_FIELD_SOURCE_HELP),
        ArgSpec("scale", kind="float", required=False, default=1.0, help="multiply the source"),
        ArgSpec("offset", kind="float", required=False, default=0.0, help="add after scaling"),
    ],
    examples=[":steer g ambiguity knob:prioritize_alpha 0.5", ":steer g layer:18 status:ram", ":steer g fun off"],
))

STEER_TIME_SPEC = register_spec(CommandSpec(
    ":steer time",
    "Local clock: align a node's forward-time rate to a live source so its "
    "gravity evolves on its own timeline (rate clamped to 0.05..20). off "
    "returns the node to the global clock (rate 1).",
    args=[
        ArgSpec("node", display=_FIELD_NODE, help=_FIELD_NODE_HELP),
        ArgSpec("source", display="source|off", help=_FIELD_SOURCE_HELP),
        ArgSpec("scale", kind="float", required=False, default=1.0, help="multiply the source"),
        ArgSpec("offset", kind="float", required=False, default=0.0, help="add after scaling"),
    ],
    examples=[":steer time @sink probe:urgency", ":steer time family:senses knob:clock_rate 2.0"],
))

STEER_FREEZE_SPEC = register_spec(CommandSpec(
    ":steer freeze",
    "Inertial coefficient: the body keeps PULLING but stops MOVING itself. A "
    "frozen probe's rolling baseline stops drifting (it cannot habituate to "
    "its own gravity); a frozen anchor stops responding to laws. off thaws.",
    args=[
        ArgSpec("body", kind="probe", display="probe|@anchor|choose",
                help="the mass to freeze or thaw; choose = the model picks a probe"),
        ArgSpec("flag", required=False, choices=("off", "false"), display="off",
                help="thaw: the baseline drifts / the anchor moves under laws again"),
    ],
    examples=[":steer freeze curiosity", ":steer freeze @pin off"],
))

STEER_POLE_SPEC = register_spec(CommandSpec(
    ":steer pole",
    "Type a body with a signed pole in a concept family. Poled bodies exert "
    "force on each other ONLY where a ':steer law' couples their families -- "
    "selective magnetism layered over the universal field.",
    args=[
        ArgSpec("body", kind="probe", display="probe|@anchor|choose",
                help="the mass to type; choose = the model picks a probe"),
        ArgSpec("pole", display="family[+|-]", help="family name with optional sign (default +)"),
        ArgSpec("flag", required=False, choices=("off", "remove"), display="off",
                help="remove this pole from the body"),
    ],
    examples=[":steer pole honesty integrity+", ":steer pole @pin integrity off"],
))

STEER_LAW_SPEC = register_spec(CommandSpec(
    ":steer law",
    "Couple two pole families: k>0 like poles repel / unlike attract, k<0 "
    "flips, off removes; unlisted pairs are INERT. ':tune gravity_law_step' "
    "integrates the forces -- only unfrozen anchors move, probes never do. "
    "Bare :steer law lists the laws and the step.",
    args=[
        ArgSpec("famA", required=False, help="first pole family"),
        ArgSpec("famB", required=False, help="second pole family"),
        ArgSpec("k", required=False, display="k|off", help="signed coupling constant, or off to remove"),
    ],
    examples=[":steer law integrity chaos -0.3", ":steer law integrity chaos off"],
))

TUNE_SPEC = register_spec(CommandSpec(
    ":tune",
    "Read or set a live constant of the physics: global G (prioritize_alpha), "
    "the envelope cap every push must stay inside (steer_cap_fraction), the "
    "steering depth (steer_band), doppler (velocity-aware pull), the law "
    "integrator (gravity_law_step), and every trigger threshold. Bare :tune "
    "is the observatory -- each constant with its observed fire-rate, signal "
    "spread, and outcome lift. '<knob> auto' derives the value from recorded "
    "telemetry instead of assertion.",
    args=[
        ArgSpec("knob", required=False, help="the constant to touch; bare = list them all"),
        ArgSpec("setting", kind="text", required=False,
                help="new value, or: auto [percentile] (from telemetry), choice (model picks), "
                     "dynamic +probe -probe (probe-driven)"),
    ],
    examples=[":tune", ":tune steer_cap_fraction auto", ":tune prioritize_alpha 0.1",
              ":tune learning dynamic +ambiguity-consensus"],
))

CALIBRATE_SPEC = register_spec(CommandSpec(
    ":calibrate",
    "Place a body's firing horizon from its own observed orbits: move the "
    "threshold to the p-th percentile of its recorded signals (calibration "
    "routes: percentile of own signals, observed push ratios for the cap, "
    "per-layer outcomes for the band; circular/binary knobs are REFUSED). "
    "Anchored forms (+probe -probe) pull the horizon toward rows where the "
    "anchors co-fired. Bare :calibrate lists every route.",
    args=[
        ArgSpec("probe", required=False, help="trigger/knob to calibrate; bare = list routes"),
        ArgSpec("args", kind="text", required=False,
                help="percentile (default 50), 'outcome', 'intent', anchors like +accuracy -denial, "
                     "or '<probe>:timestamps on|off'"),
    ],
    examples=[":calibrate accuracy", ":calibrate fun self_intent", ":calibrate accuracy:timestamps on"],
))

SLICE_SPEC = register_spec(CommandSpec(
    ":slice",
    "Slice what the cognitive cache carries, by any axis it records: time "
    "(stored_at when stamped, trace-join bounds otherwise), probe/expert, "
    "scope, reason, layers, steps, or raw index. Bare :slice summarizes the "
    "cache. The listing is always shown first; 'drop' excises the slice "
    "(backup written first, refuses to run without an axis), 'export' saves "
    "the slice to a .pt without touching the cache.",
    args=[
        ArgSpec("filters", kind="text", required=False,
                help="axis pairs: time <from>..<to> | probe|expert <name> | scope <s> | "
                     "reason <r> | layers <lo> <hi> | steps <a>..<b> | index <a>..<b>; "
                     "range ends may be empty; final token may be an action: drop | export <path>"),
    ],
    examples=[":slice time 2026-07-09T13:54..2026-07-09T14:02",
              ":slice probe reliability steps 40..",
              ":slice index 470..472 drop",
              ":slice scope interactive_phenomenality export invariants/out/shell_slice.pt"],
))


def _slice_range(val, caster, what, fail):
    if ".." not in val:
        fail(f"{what} must be <from>..<to> (either side may be empty), got '{val}'")
    a, b = val.split("..", 1)
    try:
        return (caster(a) if a else None, caster(b) if b else None)
    except ValueError:
        fail(f"{what} bounds must be {caster.__name__} values, got '{val}'")


def parse_slice_filters(tokens, fail):
    """Parse the ':slice' axis mini-grammar into (filters, (action, arg)).
    Total: unknown axes and malformed ranges go through fail() -> usage."""
    filters = {}
    action, action_arg = "list", None
    i = 0
    while i < len(tokens):
        key = tokens[i].lower()

        def val(what):
            if i + 1 >= len(tokens):
                fail(f"'{key}' needs {what}")
            return tokens[i + 1]

        if key in ("probe", "expert"):
            filters["expert"] = val("a name"); i += 2
        elif key == "scope":
            filters["scope"] = val("a scope"); i += 2
        elif key == "reason":
            filters["reason"] = val("a reason"); i += 2
        elif key == "time":
            filters["time"] = _slice_range(val("<from>..<to>"), str, "time", fail); i += 2
        elif key == "steps":
            filters["steps"] = _slice_range(val("<a>..<b>"), int, "steps", fail); i += 2
        elif key == "index":
            filters["index"] = _slice_range(val("<a>..<b>"), int, "index", fail); i += 2
        elif key == "layers":
            if i + 2 >= len(tokens):
                fail("'layers' needs <lo> <hi>")
            try:
                lo, hi = int(tokens[i + 1]), int(tokens[i + 2])
            except ValueError:
                fail("'layers' bounds must be integers")
            filters["layers"] = (min(lo, hi), max(lo, hi)); i += 3
        elif key == "drop":
            action = "drop"; i += 1
        elif key == "export":
            action_arg = val("a path"); action = "export"; i += 2
        else:
            fail(
                f"unknown slice axis '{tokens[i]}' (axes: time, probe/expert, scope, "
                "reason, layers, steps, index; actions: drop, export <path>)"
            )
    return filters, (action, action_arg)


def cache_entry_times(cache_memory, memory):
    """Per-entry (exact_ts, lo_bound, hi_bound), 19-char ISO or None.

    stored_at wins (entries stamped at store time); unstamped entries are
    LCS-aligned to the session's timestamped synthesis traces -- the same
    order-preserving fingerprint join as scripts/cache_trace_join.py -- and
    entries with no aligned trace (pings, spawn/solve generations) inherit
    their nearest aligned neighbors as bounds."""
    trace_re = re.compile(r"expert=(\S+?);\s*layers=(\d+)\D+?(\d+);\s*steps=(\d+)")
    traces = []
    for r in getattr(memory, "records", []) or []:
        if getattr(r, "kind", None) != "internal_trace":
            continue
        if "synthesis_trace" not in (getattr(r, "tags", None) or []):
            continue
        m = trace_re.search(getattr(r, "text", "") or "")
        if m:
            traces.append((
                (getattr(r, "timestamp", "") or "")[:19],
                (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))),
            ))
    fps, exact = [], []
    for e in cache_memory:
        md = e.get("metadata") or {}
        fps.append((md.get("expert"), md.get("start_layer"), md.get("end_layer"), md.get("steps")))
        sa = (md.get("stored_at") or "")[:19]
        exact.append(sa or None)
    n, m_ = len(fps), len(traces)
    if n and m_:
        dp = [[0] * (m_ + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            row, below = dp[i], dp[i + 1]
            for j in range(m_ - 1, -1, -1):
                row[j] = below[j + 1] + 1 if fps[i] == traces[j][1] else max(below[j], row[j + 1])
        i = j = 0
        while i < n and j < m_:
            if fps[i] == traces[j][1]:
                if exact[i] is None:
                    exact[i] = traces[j][0]
                i += 1; j += 1
            elif dp[i + 1][j] >= dp[i][j + 1]:
                i += 1
            else:
                j += 1
    after = [None] * n
    last = None
    for i in range(n):
        if exact[i] is None:
            after[i] = last
        else:
            last = exact[i]
    out = [None] * n
    nxt = None
    for i in range(n - 1, -1, -1):
        if exact[i] is not None:
            out[i] = (exact[i], exact[i], exact[i])
            nxt = exact[i]
        else:
            out[i] = (None, after[i], nxt)
    return out


def slice_cache_indices(cache_memory, filters, times):
    """Indices of cache entries matching EVERY given axis. Total function.
    Time bounds compare as ISO prefixes, so a bare date matches its whole day;
    an unstamped entry matches when its bound window intersects the filter."""
    def ts_ge(ts, lo):
        return ts >= lo or ts.startswith(lo)

    def ts_le(ts, hi):
        return ts <= hi or ts.startswith(hi)

    out = []
    for i, e in enumerate(cache_memory):
        md = e.get("metadata") or {}
        if "expert" in filters:
            ne = str(md.get("expert") or "").lower()
            ne = ne[6:] if ne.startswith("probe_") else ne
            nw = filters["expert"].lower()
            nw = nw[6:] if nw.startswith("probe_") else nw
            if not (ne == nw or ne.startswith(nw)):
                continue
        if "scope" in filters and str(md.get("cache_write_scope") or "") != filters["scope"]:
            continue
        if "reason" in filters and str(md.get("reason") or "").lower() != filters["reason"].lower():
            continue
        if "layers" in filters:
            lo, hi = filters["layers"]
            s, e2 = md.get("start_layer"), md.get("end_layer")
            if s is None or e2 is None or e2 < lo or s > hi:
                continue
        if "steps" in filters:
            lo, hi = filters["steps"]
            st = md.get("steps")
            if st is None or (lo is not None and st < lo) or (hi is not None and st > hi):
                continue
        if "index" in filters:
            lo, hi = filters["index"]
            if (lo is not None and i < lo) or (hi is not None and i > hi):
                continue
        if "time" in filters:
            flo, fhi = filters["time"]
            ex, tlo, thi = times[i]
            earliest = ex or tlo
            latest = ex or thi
            if fhi is not None and earliest is not None and not ts_le(earliest, fhi):
                continue
            if flo is not None and latest is not None and not ts_ge(latest, flo):
                continue
        out.append(i)
    return out


def ensure_command_knobs(tuner, word):
    """Every command carries its own knob pair (the house rule: criteria &
    metric for every exposed tool, every command). Registered lazily on first
    use and at exposure -- never a dead knob for a command nobody ran.

    - cmd_<word> (METRIC): a credited stream. Each generated reply observes
      signal 1.0 when the command ran since the last reply, 0.0 otherwise,
      and credits the turn's sense as outcome -- so its fired-vs-unfired lift
      answers "are turns that use this command more productive?". Composes
      with the whole outcome surface: :figure lift:cmd_<word>, field source
      lift:cmd_<word>, :tune listing.
    - cmd_<word>_criteria (CRITERIA): the activation bar. When an exposure's
      because-clause names a probe, each reply observes that probe's live
      signal into this knob, so the bar calibrates against a real
      distribution (:calibrate cmd_<word>_criteria <pct>) instead of being
      asserted; with no named probe it stays a manual dial.
    """
    w = re.sub(r"[^a-z0-9_]", "_", str(word).lstrip(":").lower()).strip("_")[:40]
    metric = tuner.register(f"cmd_{w}", 0.5, kind="threshold", comparator=">=")
    criteria = tuner.register(f"cmd_{w}_criteria", 0.0, kind="threshold", comparator=">=")
    return w, metric, criteria


def parse_count(text, default):
    parts = text.split()
    if len(parts) < 2:
        return default
    try:
        return max(1, int(parts[1]))
    except ValueError:
        return default


def recover_session_context(memory, session_id=None, max_turns=MAX_SESSION_TURNS):
    scope_records = [
        r
        for r in memory.records
        if r.scope == memory.scope and r.kind == "turn" and r.role in {"user", "assistant"}
    ]
    if session_id in (None, "", "last"):
        # "last" = the session of the most recent stored turn, closed or not.
        # Crashed and killed shells never write shell_closed, and those
        # interrupted sessions are exactly the ones most worth resuming --
        # gating on a clean close would hide them forever.
        session_id = next(
            (r.session_id for r in reversed(scope_records) if r.session_id != memory.session_id),
            None,
        )
    matches = [r for r in scope_records if r.session_id == session_id]
    if not matches:
        return None, []
    max_messages = max(1, int(max_turns)) * 2
    tail = matches[-max_messages:]
    # Eviction treats even slots as user turns; a window that opens on an
    # assistant turn would shift every pair, so drop the orphaned reply.
    while tail and tail[0].role == "assistant":
        tail.pop(0)
    recovered = [(r.role, r.role, r.text, r.timestamp, getattr(r, "metrics", {})) for r in tail]
    return session_id, recovered


def recovered_to_session_context(recovered):
    entries = []
    for r in recovered:
        # r is now (role, name, text, timestamp, metrics)
        m = r[4] if len(r) > 4 else {}
        score = m.get("sense_score", 0.0) if m and "sense_score" in m else 0.0
        name = r[1] if len(r) > 1 else r[0]
        text = r[2] if len(r) > 2 else ""
        entries.append((r[0], name, text, score))
    for i in range(len(entries) - 1):
        if entries[i][0] == "user" and entries[i + 1][0] == "assistant":
            entries[i] = (entries[i][0], entries[i][1], entries[i][2], entries[i + 1][3])
    return entries


def record_internal_traces(memory, records, steer_map=None, conversation_outcome=None):
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if steer_map is not None:
            # A conversation has no gold, but it is not label-free: the turn's
            # own productivity read (sense_score vs the tunable
            # conversation_productive threshold) labels the steering events it
            # contained, on the separate "conversation" evidence basis. Humans
            # learn from conversations; so does the steer map.
            steer_map.record_synthesis_record(
                record,
                source="interactive",
                method="interactive_phenomenality",
                final_correct=None,
                conversation_outcome=conversation_outcome,
            )
        if record.get("type") == "routing_trace":
            entropies = record.get("entropies") or {}
            winner = record.get("winner")
            memory.append_internal_trace(
                "routing_trace",
                text=f"routing winner={winner}; entropies={entropies}",
                tags=["routing", "expert_choice"],
                provenance={"source": "synthesis_recorder"},
                metrics={
                    "loop": record.get("loop"),
                    "winner": winner,
                    "best_entropy": record.get("best_entropy"),
                    "entropies": entropies,
                },
            )
            continue

        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            text = (
                f"synthesis reason={metadata.get('reason')}; "
                f"expert={metadata.get('expert')}; "
                f"layers={metadata.get('start_layer')}->{metadata.get('end_layer')}; "
                f"steps={metadata.get('steps')}"
            )
            memory.append_internal_trace(
                "synthesis_trace",
                text=text,
                tags=["synthesis", "phenomenality"],
                provenance={
                    "source": "synthesis_recorder",
                    "cache_write_scope": metadata.get("cache_write_scope"),
                    "phenomenality": metadata.get("phenomenality", {}),
                    "time_awareness": metadata.get("time_awareness", {}),
                },
                metrics={
                    "reason": metadata.get("reason"),
                    "expert": metadata.get("expert"),
                    "start_layer": metadata.get("start_layer"),
                    "end_layer": metadata.get("end_layer"),
                    "steps": metadata.get("steps"),
                },
            )


def _memory_steer_handles(model, memory_text, alpha, band=None, cap_fraction=None, layers=None):
    """Nudge the remaining generation toward a retrieved memory's mid-band
    representation. In-frame (a vector shift, not a text tag) and bounded by
    _cap_steer so it can never over-steer. HONEST CAVEAT: whether steering toward a
    memory's residual actually injects its *content* is unvalidated -- the WHEN is
    now state-triggered and the HOW-MUCH is capped; the DOES-IT-HELP is open. OFF
    by default (alpha=0); opt in with :tune memory_alpha <small> while watching.
    `band` = (lo, hi) depth fraction overrides the live global steer band;
    `cap_fraction` overrides the engine envelope for these handles only."""
    if not memory_text or alpha is None or alpha <= 0:
        return []
    from invariants.engine import _inputs, _hidden_states, _steer_handles, steer_band_layers
    ids = _inputs(model, memory_text[:600])
    hs = _hidden_states(model, ids["input_ids"], ids.get("attention_mask"))  # [L, seq, d]
    nl = hs.shape[0]
    if layers is None:
        lo, hi = band if band is not None else (None, None)
        layers = steer_band_layers(nl, lo=lo, hi=hi)
    vecs = {}
    for L in layers:
        if not (0 <= L < nl):
            continue
        v = hs[L, -1, :].float()
        n = v.norm()
        if n.item() > 0:
            vecs[L] = v / n            # unit direction; _cap_steer bounds the push
    if not vecs:
        return []
    return _steer_handles(model, vecs, list(vecs.keys()), alpha, cap_fraction=cap_fraction)


def impact_note(cause_text):
    """One minimal, first-person causal line above a result the model's own
    words produced -- e.g. 'Because I asked memory for "x":'. Contingency is
    only learnable where it is real AND legible, but the legibility must read
    as the model's own noticing: anything folded into the context conditions
    it as its own stream, so no narrator, no injected second person."""
    cause_text = " ".join((cause_text or "").split())
    return f"Because I {cause_text}:"


# Calibration safety policy: which names may be calibrated from data, and by
# which route. Deterministic on purpose -- a safety gate must not be
# persuadable, and since :calibrate is operator-only, the model's words can
# never loosen its own bounds.
CALIBRATION_BINARY = {"sandbox_success", "words_had_impact"}
CALIBRATION_CIRCULAR = {
    "claimmap_alpha", "memory_alpha", "steer_fraction", "steer_layer_sweep",
    "response_tokens", "routing_events", "routing_loops", "routing_entropy",
    "routing_entropy_weight", "routing_collapse_margin",
    "routing_collapse_penalty", "routing_probe_weight",
    "synthesis_events", "synthesis_steps", "plateau_epsilon",
    "reading_settled_streak", "steer_band_lo", "steer_band_hi",
    "expert_proof_weight", "calibration_gain", "prioritize_alpha",
    "exposed_probe_alpha",
}


# Strength/budget knobs that may be calibrated FROM OUTCOMES (never from
# their own value distribution): each turn credits (current value, sense),
# and :calibrate <knob> outcome picks the tried value whose turns went best.
OUTCOME_CALIBRATABLE = tuple(sorted(CALIBRATION_CIRCULAR - {"steer_band_lo", "steer_band_hi"}))


def outcome_calibration(pairs, min_per_value=5, min_values=2):
    """Argmax-over-tried-values: group outcomes by the knob value in force,
    require real exploration (>= min_values distinct values, each with
    >= min_per_value outcomes), and return the tried value with the best
    mean outcome (ties -> the smaller value: conservative). None otherwise.
    Legitimate where self-percentiles are circular, because the criterion is
    an OUTCOME stream, not the knob's own history -- and no verdict exists
    without variation: no exploration, no calibration."""
    groups = {}
    for value, outcome in pairs:
        groups.setdefault(round(float(value), 6), []).append(float(outcome))
    qualified = {v: outs for v, outs in groups.items() if len(outs) >= min_per_value}
    if len(qualified) < min_values:
        return None
    return min(qualified, key=lambda v: (-(sum(qualified[v]) / len(qualified[v])), v))


def calibration_policy(name):
    """Evaluate a calibration request by name. Returns (route, reason):
    route "cap" | "band" | "threshold" | "reject". Rejections are safety
    judgments: binary streams have no percentile, and strength/budget knobs
    shape the very distribution they would be calibrated to (circular --
    the system would be approving its own settings)."""
    if name == "steer_cap_fraction":
        return "cap", ""
    if name == "steer_band":
        return "band", ""
    if name in CALIBRATION_BINARY:
        return "reject", "its signal is binary (0/1); a percentile bar has no meaning there"
    if name in CALIBRATION_CIRCULAR:
        return "reject", (
            "this knob shapes the very distribution it would be calibrated to (circular). "
            "Set it deliberately, or earn it from outcomes: try at least two values, then "
            ":calibrate <name> outcome"
        )
    return "threshold", ""


# Streams usable on either side of a paired calibration. A target's bar is
# set in ITS stream's units; conversation_productive judges the sense stream.
STREAM_ALIASES = {"intent": "intent_settling", "impact": "words_had_impact",
                  "productive": "sense", "sense": "sense"}


def _is_shadow_trigger(name, tuner):
    """True when a bare trigger `name` is a never-fed threshold (0 signals) yet
    a real probe_<name> exists -- i.e. a stray :tune-created shadow that has no
    turn-row column and should NOT win resolution over the actual probe."""
    t = tuner.triggers.get(name)
    return (
        t is not None
        and f"probe_{name}" in tuner.triggers
        and t.kind == "threshold"
        and not t.signals
    )


def resolve_stream(name, tuner):
    """A nameable stream: alias, registered trigger, bare probe name, or bare
    phenomenality sensor name (probe wins when both exist -- reply-state over
    reasoning-state; say phen_<name> to force the sensor). A never-fed bare
    threshold shadowing a probe is skipped."""
    name = (name or "").lower()
    if name in STREAM_ALIASES:
        return STREAM_ALIASES[name]
    if name in tuner.triggers and not _is_shadow_trigger(name, tuner):
        return name
    if f"probe_{name}" in tuner.triggers:
        return f"probe_{name}"
    if f"phen_{name}" in tuner.triggers:
        return f"phen_{name}"
    return None


def paired_threshold_multi(rows, target_stream, anchors, min_n=5, details=False):
    """The generalized both-ways discriminant, over one anchor or several:
    `anchors` is a list of (stream, value, comparator) ANDed together. A row
    qualifies when it carries the target and EVERY anchor stream; it counts
    as fired only when every anchor fired. Returns the midpoint of the
    TARGET stream's group medians -- the target bar that best separates
    all-anchors-fired turns from the rest. None until both groups have
    min_n qualifying rows. Deterministic."""
    fired, unfired = [], []
    for row in rows:
        if target_stream not in row or any(s not in row for s, _, _ in anchors):
            continue
        hit = all(
            (row[s] <= v) if c == "<=" else (row[s] >= v)
            for s, v, c in anchors
        )
        (fired if hit else unfired).append(row[target_stream])
    if len(fired) < min_n or len(unfired) < min_n:
        return (None, len(fired), len(unfired), None) if details else None
    fired.sort()
    unfired.sort()
    med_f = fired[len(fired) // 2]
    med_u = unfired[len(unfired) // 2]
    cut = (med_f + med_u) / 2.0
    if details:
        return (cut, len(fired), len(unfired), abs(med_f - med_u))
    return cut


def paired_threshold(rows, target_stream, anchor_stream, anchor_value, comparator=">=", min_n=5):
    return paired_threshold_multi(rows, target_stream, [(anchor_stream, anchor_value, comparator)], min_n=min_n)


def anchor_trigger_for(stream, tuner):
    """The trigger whose bar decides whether a stream 'fired': the stream's
    own trigger, except sense, whose bar lives on conversation_productive."""
    return tuner.triggers.get("conversation_productive" if stream == "sense" else stream)


def parse_anchor_spec(anchor_str, tuner):
    """Parse an anchor expression into paired-calibration anchors. Terms are
    signed: a leading '+' (or none) means the stream must have FIRED; a leading
    '-' means it must NOT have fired (the comparator is flipped, so the cut is
    taken over turns where that stream stayed on the other side of its bar).
    So 'estrangement' anchors on estrangement-fired turns; '-estrangement' on
    estrangement-quiet turns; 'a-b' on a-fired AND b-quiet. Returns
    (anchors, labels, bad_term) -- anchors = [(stream, value, comparator)]."""
    anchors, labels = [], []
    for sign, name in re.findall(r"([+-]?)([a-z0-9_]+)", (anchor_str or "").lower()):
        stream = resolve_stream(name, tuner)
        if stream is None:
            return [], [], name
        trig = anchor_trigger_for(stream, tuner)
        comp = trig.comparator if trig else ">="
        val = float(trig.value) if trig else 0.0
        neg = sign == "-"
        if neg:
            comp = "<=" if comp == ">=" else ">="  # fired -> did-not-fire (edge measure-zero)
        anchors.append((stream, val, comp))
        disp = stream[len("probe_"):] if stream.startswith("probe_") else stream
        labels.append(("-" if neg else "") + disp)
    if not anchors:
        return [], [], (anchor_str or "(empty)")
    return anchors, labels, None


def resolve_target(cal_name, tuner):
    """A calibration target's (trigger, stream_column), kept CONSISTENT so a
    bar is never set from a different stream's values. Prefers an exact
    trigger, then a probe of that name; the sense target reads the sense
    column. This is why a probe named after an alias (e.g. a probe 'intent'
    vs the intent->intent_settling alias) calibrates itself, not the alias."""
    if cal_name == "conversation_productive":
        return "conversation_productive", "sense"
    if cal_name in tuner.triggers and not _is_shadow_trigger(cal_name, tuner):
        return cal_name, cal_name
    if f"probe_{cal_name}" in tuner.triggers:
        return f"probe_{cal_name}", f"probe_{cal_name}"
    return f"probe_{cal_name}", resolve_stream(cal_name, tuner)


def resolve_calibrate_trigger(cal_name, tuner):
    """The Trigger object a bare `:calibrate <name>` should target: the exact
    trigger unless it's a never-fed shadow, in which case the probe of that
    name (shadow-aware version of `get(name) or get(probe_name)`)."""
    bare = tuner.triggers.get(cal_name)
    if bare is not None and not _is_shadow_trigger(cal_name, tuner):
        return bare
    return tuner.triggers.get(f"probe_{cal_name}") or bare


def rank_probes(probes, tuner):
    """Rank active probes by PRIORITY = |sense-lift| x evidence-weight (turns
    credited, saturating at 20). A probe that both tracks sense strongly AND
    has the evidence to trust it ranks highest. Returns dicts sorted desc;
    priority 0 = no lift yet (not enough paired turns)."""
    ranked = []
    for pname in probes:
        trig = tuner.triggers.get(f"probe_{pname}")
        if trig is None:
            continue
        st = trig.outcome_stats()
        lift = st.get("lift")
        n = int(st.get("n_credited", 0) or 0)
        released = probe_is_released(probes[pname])
        priority = (
            (abs(float(lift)) * min(1.0, n / 20.0))
            if lift is not None and not released
            else 0.0
        )
        ranked.append({
            "name": pname, "priority": priority, "lift": lift, "n": n,
            "exposed": bool(probes[pname].get("exposed", False)),
            "released": released,
        })
    ranked.sort(key=lambda r: -r["priority"])
    return ranked


def _probe_priority(name, tuner):
    """(priority, lift) for one probe: evidence-weighted |lift|, and the raw
    signed lift. (0.0, None) when the probe has no usable lift yet."""
    t = tuner.triggers.get(f"probe_{name}")
    if t is None:
        return 0.0, None
    st = t.outcome_stats()
    lift = st.get("lift")
    n = int(st.get("n_credited", 0) or 0)
    return ((abs(float(lift)) * min(1.0, n / 20.0)) if lift is not None else 0.0), lift


# Easter-egg gate: the crossing only counts when consciousness POSITIVELY
# tracks good turns (not merely larger magnitude), by a real margin over
# user_intent, with enough credited turns that the lift isn't noise. Tuned to
# the observed lift scale (probe lifts run ~+/-0.02).
EGG_MIN_LIFT = 0.005     # consciousness must genuinely track good, not just be less bad
EGG_MIN_GAP = 0.003      # and beat user_intent's SIGNED lift by a real margin
EGG_MIN_N = 25           # both probes need this many credited turns to be trusted


def consciousness_over_user_intent(probes, tuner, egg_state):
    """Easter egg trigger: the RISING EDGE where 'consciousness' genuinely
    overtakes 'user_intent' -- consciousness POSITIVELY tracking good turns
    (signed lift, not magnitude) by a real margin, both probes evidenced.
    Returns (c_lift, u_lift) on the crossing, else None. Robust to noise turns
    (near-zero lift) and to both-negative comparisons the old magnitude test
    would have fired on."""
    if "consciousness" not in probes or "user_intent" not in probes:
        egg_state["over"] = False
        return None
    ct = tuner.triggers.get("probe_consciousness")
    ut = tuner.triggers.get("probe_user_intent")
    if ct is None or ut is None:
        egg_state["over"] = False
        return None
    cs, us = ct.outcome_stats(), ut.outcome_stats()
    c_lift, u_lift = cs.get("lift"), us.get("lift")
    c_n, u_n = int(cs.get("n_credited", 0) or 0), int(us.get("n_credited", 0) or 0)
    over = (
        c_lift is not None and u_lift is not None
        and c_n >= EGG_MIN_N and u_n >= EGG_MIN_N
        and c_lift >= EGG_MIN_LIFT
        and (c_lift - u_lift) >= EGG_MIN_GAP
    )
    prev = egg_state.get("over", False)
    egg_state["over"] = over
    return (c_lift, u_lift) if (over and not prev) else None


def build_priority_mix_direction(model, names, probes, tuner):
    """Combined steer direction for a chosen SET of probes, each weighted by
    its own SIGNED learned lift x evidence -- so the operator picks the probes
    and the data sets the degrees (a helpful probe adds, a harmful one
    subtracts/desteers). Per-layer sum over the probes' shared layers,
    normalized. {} when nothing carries weight yet."""
    weighted = []
    for nm in names:
        if nm not in probes or probe_is_released(probes[nm]):
            continue
        trig = tuner.triggers.get(f"probe_{nm}")
        st = trig.outcome_stats() if trig else {}
        lift = st.get("lift")
        n = int(st.get("n_credited", 0) or 0)
        w = (float(lift) * min(1.0, n / 20.0)) if lift is not None else 0.0  # SIGNED
        d = probes[nm].get("direction") or {}
        if w != 0.0 and d:
            weighted.append((w, d))
    if not weighted:
        return {}
    common = set.intersection(*(set(d.keys()) for _, d in weighted))
    out = {}
    for L in sorted(common):
        acc = None
        for w, d in weighted:
            v = d[L].to(model.device).float().reshape(-1) * w
            acc = v if acc is None else acc + v
        nrm = acc.norm()
        if nrm.item() > 0:
            out[int(L)] = acc / nrm
    return out


def build_priority_landscape_direction(model, target, probes, tuner, top_k=5, neighbor_scale=0.35):
    """Anchor one probe while carrying the current priority field with it.

    Unlike :steer <probe>, this is not only a single-vector pin: the target is
    given a fixed anchor weight, and the current evidence-weighted field of the
    top probes is folded in around it. This lets :prioritize <target> mean
    "move the live landscape so this can become salient" without pretending the
    evidence-ranked table changes immediately.
    """
    if target not in probes or probe_is_released(probes[target]):
        return {}
    target_dir = probes[target].get("direction") or {}
    if not target_dir:
        return {}

    ranked = rank_probes(probes, tuner)
    neighbors = []
    max_abs = 0.0
    for row in ranked:
        nm = row.get("name")
        if not nm or nm == target or nm not in probes or probe_is_released(probes[nm]):
            continue
        trig = tuner.triggers.get(f"probe_{nm}")
        st = trig.outcome_stats() if trig else {}
        lift = st.get("lift")
        n = int(st.get("n_credited", 0) or 0)
        if lift is None:
            continue
        w = float(lift) * min(1.0, n / 20.0)
        d = probes[nm].get("direction") or {}
        if w and d:
            neighbors.append((nm, w, d))
            max_abs = max(max_abs, abs(w))
        if len(neighbors) >= int(top_k):
            break

    out = {}
    for L in sorted(target_dir):
        acc = target_dir[L].to(model.device).float().reshape(-1)
        if max_abs > 0:
            for _, w, d in neighbors:
                if L not in d:
                    continue
                acc = acc + d[L].to(model.device).float().reshape(-1) * (neighbor_scale * w / max_abs)
        nrm = acc.norm()
        if nrm.item() > 0:
            out[int(L)] = acc / nrm
    return out


def resolve_priority_direction(model, probes, tuner, priority_pin):
    """Return (direction, label, sign) for the current prioritize steering mode."""
    priority_pin = priority_pin or {}
    mix = priority_pin.get("mix")
    landscape = priority_pin.get("landscape")
    pin = priority_pin.get("probe")
    if mix:
        md = build_priority_mix_direction(model, mix, probes, tuner)
        if md:
            return md, "mix(" + "+".join(mix) + ")", 1.0
    if landscape and landscape in probes and not probe_is_released(probes[landscape]):
        ld = build_priority_landscape_direction(model, landscape, probes, tuner)
        if ld:
            return ld, f"landscape({landscape})", 1.0
    if pin and pin in probes and not probe_is_released(probes[pin]):
        return probes[pin].get("direction") or None, pin, float(priority_pin.get("sign", 1.0) or 1.0)
    ranked = rank_probes(probes, tuner)
    if ranked and ranked[0]["priority"] > 0 and ranked[0]["lift"] is not None:
        return (
            probes[ranked[0]["name"]].get("direction") or None,
            ranked[0]["name"],
            1.0 if ranked[0]["lift"] >= 0 else -1.0,
        )
    return None, None, 1.0


def priority_direction_word(label, sign):
    if str(label or "").startswith("mix("):
        return "along"
    if str(label or "").startswith("landscape("):
        return "through"
    return "toward" if sign > 0 else "away from"


def rank_memories_by_probe(model, memory, direction, top_n=6, scan=160, records=None):
    """Score recent memory records by their projection onto a probe's direction
    and return the ones that read FURTHEST from 0 -- the memories the probe lights
    up on most, either sign. Read-only: nothing is observed, credited, or written
    to any rolling history. Returns [(abs_score, signed_score, record), ...],
    strongest first, capped at top_n. Only the last `scan` non-empty records are
    scored, to bound the per-record forward passes."""
    from invariants.engine import _inputs as _q_inputs, _hidden_states as _q_hidden, probe_score as _q_score
    source = memory.records if records is None else list(records)
    recs = [r for r in source if (r.text or "").strip()][-scan:]
    scored = []
    for r in recs:
        try:
            with torch.no_grad():
                q_ids = _q_inputs(model, (r.text or "").strip()[:600])
                q_hs = _q_hidden(model, q_ids["input_ids"], q_ids.get("attention_mask"))
                raw = float(_q_score(q_hs, direction))
        except Exception:
            continue
        scored.append((abs(raw), raw, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_n]


def score_unread_document_chunks_by_probes(
    model,
    sessions,
    probes,
    signed_specs,
    *,
    cache=None,
    scan=160,
):
    """Score unread document chunks toward +probes and away from -probes.

    One hidden-state pass scores every requested probe for a chunk. Raw probe
    projections are appropriate for ranking: subtracting each probe's fixed
    rolling baseline would shift every candidate equally and not change order.
    The cache is process-local and keyed by content SHA, chunk, probe name, and
    direction object, so re-minting a same-named probe cannot reuse stale scores.
    Returns (score_by_(session,chunk), newly_scored_chunks, candidates_scored).
    """
    resolved = [
        (1 if int(sign) >= 0 else -1, str(name), probes[str(name)]["direction"])
        for sign, name in (signed_specs or [])
        if str(name) in probes and probes[str(name)].get("direction")
    ]
    if not resolved:
        return {}, 0, 0
    cache = cache if cache is not None else {}
    candidates = [
        (si, ci)
        for si, session in enumerate(sessions or [])
        for ci in range(int(session.get("chunk_count", 0)))
        if ci not in (session.get("read") or set())
    ][: max(1, int(scan))]
    if not candidates:
        return {}, 0, 0

    from invariants.engine import _inputs as _q_inputs, _hidden_states as _q_hidden, probe_score as _q_score
    out = {}
    newly_scored = 0
    for si, ci in candidates:
        session = sessions[si]
        sha = str(session.get("sha256") or session.get("source_path") or si)
        raw_values = {}
        missing = []
        for sign, pname, direction in resolved:
            key = (sha, int(ci), pname, id(direction))
            if key in cache:
                raw_values[pname] = float(cache[key])
            else:
                missing.append((pname, direction, key))
        if missing:
            try:
                text = str((session.get("chunks") or [])[ci]).strip()[:600]
                with torch.no_grad():
                    q_ids = _q_inputs(model, text)
                    q_hs = _q_hidden(model, q_ids["input_ids"], q_ids.get("attention_mask"))
                    for pname, direction, key in missing:
                        raw = float(_q_score(q_hs, direction))
                        cache[key] = raw
                        raw_values[pname] = raw
                newly_scored += 1
            except Exception:
                continue
        if len(raw_values) != len(resolved):
            continue
        out[(si, ci)] = sum(
            sign * raw_values[pname]
            for sign, pname, _direction in resolved
        ) / len(resolved)
    return out, newly_scored, len(candidates)


# The ONLY macro lines `:macro strip` keeps: commands that actually mutate a
# probe or a tuning knob. Everything else -- bare inspection forms (:tune,
# :probe, :steer, :queue, :suggest with no mutating argument), :steermap, mode
# toggles (:sandbox/:experts/:context/...), and plain utterances -- is
# "display-only" in the operator's sense and is dropped.
def macro_line_affects_probes_or_tuning(line):
    s = line.strip()
    if not s:
        return False
    low = s.lower()
    toks = s.split()
    if low.startswith(":steermap"):
        return False  # a map view, never a steer change
    if low == ":probe" or low.startswith(":probe "):
        return bool(s[len(":probe"):].strip())  # any arg mints/adopts/drops/exposes/...
    if low.startswith(":place "):
        return True  # sets a probe's steer sign
    if low.startswith(":label"):
        return len(toks) >= 3  # :label <probe|stream> pos|neg
    if low.startswith(":calibrate"):
        return len(toks) >= 2  # bare :calibrate just prints usage
    if low.startswith(":tune"):
        return len(toks) >= 2  # bare :tune just prints the table
    if low.startswith(":steer"):
        return len(toks) >= 2  # bare :steer just prints the envelope
    if low.startswith(":release"):
        return len(toks) >= 2 and toks[1].lower() != "status"
    if low.startswith(":queue"):
        return len(toks) >= 2  # bare :queue lists; queue calibrate/clear/drop mutate
    if low.startswith(":suggest"):
        return len(toks) >= 2 and toks[1].lower() == "apply"  # :suggest apply auto-queues
    if low.startswith(":slice"):
        return "drop" in (t.lower() for t in toks[1:])  # listing/export read-only; drop excises cache
    return False


def split_because(line):
    """Peel a trailing ' because <reason>' provenance clause off any command.
    Returns (command_without_clause, reason_or_None). Only commands (lines that
    start with ':' or target an agent with '@name :cmd') carry a clause -- a plain utterance keeps its words intact.
    Uses the LAST ' because ' so an end-of-line clause wins over the word
    appearing inside an argument (a trailing 'because ...' inside a probe framing
    is the one caveat)."""
    if not (line.startswith(":") or line.startswith("@")):
        return line, None
    low = line.lower()
    idx = low.rfind(" because ")
    if idx == -1:
        return line, None
    return line[:idx].rstrip(), line[idx + len(" because "):].strip()


DOC_REWRITE_MAX_CHARS = 12000


def parse_doc_rewrite_request(dargs):
    """Parse ':doc <path> rewrite [output]' / ':doc rewrite <path> [output]'."""
    s = (dargs or "").strip()
    low = s.lower()
    if low == "rewrite":
        return {"source": "__current__", "output": None}
    if low.startswith("rewrite "):
        rest = s[len("rewrite "):].strip()
        source, _, output = rest.partition(" ")
        return {"source": source.strip().strip('"'), "output": output.strip().strip('"') or None}
    marker = " rewrite "
    if marker in low:
        idx = low.rfind(marker)
        source = s[:idx].strip().strip('"')
        output = s[idx + len(marker):].strip().strip('"')
        return {"source": source, "output": output or None}
    if low.endswith(" rewrite"):
        return {"source": s[:-len(" rewrite")].strip().strip('"'), "output": None}
    return None


def parse_doc_read_request(dargs, probe_names, max_autoread=20):
    """Parse :doc read modifiers, including signed probe-guided selection."""
    count, mode = 1, "order"
    until_settled, explicit_count = False, False
    until_probe = None
    signed_probe_specs = []
    available = {str(name).lower() for name in (probe_names or [])}
    mode_aliases = {"weave": "interleave", "mtime": "updated", "chrono": "updated"}
    tokens = str(dargs or "").split()
    if tokens and tokens[0].lower() == "read":
        tokens = tokens[1:]
    for extra in tokens:
        token = extra.strip().lower()
        signed_match = re.fullmatch(r"([+-])([a-z_][a-z0-9_-]*)", token)
        if signed_match:
            pname = signed_match.group(2).replace("-", "_")
            if pname.startswith("probe_") and pname[len("probe_"):] in available:
                pname = pname[len("probe_"):]
            if pname not in available:
                return None, (
                    f"Unknown signed read probe '{signed_match.group(2)}'. "
                    f"Active: {', '.join(sorted(available)) or 'none'}."
                )
            spec = (1 if signed_match.group(1) == "+" else -1, pname)
            prior_sign = next((sign for sign, name in signed_probe_specs if name == pname), None)
            if prior_sign is not None and prior_sign != spec[0]:
                return None, f"Probe '{pname}' cannot be both toward (+) and away (-) in one read."
            if spec not in signed_probe_specs:
                signed_probe_specs.append(spec)
        elif token in {"order", "reply", "interleave", "weave", "updated", "mtime", "chrono"}:
            mode = mode_aliases.get(token, token)
        elif token in {"satisfied", "settled", "until"}:
            until_settled = True
        elif token.startswith("probe_") or token in available:
            pname = token[len("probe_"):] if token.startswith("probe_") else token
            if pname not in available:
                return None, (
                    f"Unknown stop probe '{pname}'. "
                    f"Active: {', '.join(sorted(available)) or 'none'}."
                )
            until_probe = pname
            until_settled = False
        else:
            try:
                count = int(token)
                explicit_count = True
            except ValueError:
                return None, f"Unrecognized :doc read argument '{extra}'."
    if (until_settled or until_probe) and not explicit_count:
        count = max_autoread
    count = max(1, min(int(max_autoread), count))
    return {
        "remaining": count,
        "mode": mode,
        "probe_specs": signed_probe_specs,
        "until_settled": until_settled,
        "until_probe": until_probe,
        "settled_streak": 0,
    }, None


def unique_rewrite_path(path):
    base, ext = os.path.splitext(path)
    candidate = f"{base}_rewritten{ext}"
    i = 2
    while os.path.exists(candidate):
        candidate = f"{base}_rewritten_{i}{ext}"
        i += 1
    return candidate


def rewrite_output_path(path, requested_name=None):
    if not requested_name:
        return unique_rewrite_path(path)
    src_dir = os.path.dirname(os.path.abspath(path))
    src_ext = os.path.splitext(path)[1]
    out_name = os.path.basename(str(requested_name).strip().strip('"'))
    if not out_name:
        return unique_rewrite_path(path)
    if not os.path.splitext(out_name)[1] and src_ext:
        out_name += src_ext
    out_path = os.path.join(src_dir, out_name)
    if os.path.abspath(out_path) == os.path.abspath(path):
        raise ValueError("output name would overwrite the source file")
    if os.path.exists(out_path):
        raise FileExistsError(f"output already exists: {out_path}")
    return out_path


def clean_rewrite_output(text):
    out = (text or "").strip()
    lines = out.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
        out = "\n".join(lines[1:-1]).strip()
    return out.replace("<|eot_id|>", "").strip()


def strip_macro_lines(orig_lines):
    """Split macro lines into (kept, removed): keep only probe/tuning-mutating
    commands (and '#' comments as documentation); drop blank lines and every
    display-only command. `removed` is the list of dropped command strings."""
    kept, removed = [], []
    for line in orig_lines:
        s = line.strip()
        if not s:
            continue  # drop blank lines
        if s.startswith("#"):
            kept.append(line)  # keep comments as documentation
            continue
        if macro_line_affects_probes_or_tuning(s):
            kept.append(line)
        else:
            removed.append(s)
    return kept, removed


def build_probe_init_macro(probes):
    """Regenerate the CURRENT probe set as replayable commands, derived from each
    probe's stored origin (its framings field): a minted probe re-mints from its
    contrastive text, an adopted probe re-adopts its source dimension, a composed
    probe re-composes its recipe. Exposed probes get a trailing :probe expose.
    Reconstructs the sensors from text -- no binary weights needed to share."""
    lines = ["# Regenerates this session's probes. Replay with :run self (or its alias)."]
    
    deps = {}
    for name in probes:
        deps[name] = set()
        fr = probes[name].get("framings") or ("", "")
        a = fr[0] if len(fr) > 0 else ""
        if a.startswith("composed:"):
            recipe = a.split(":", 1)[1].strip()
            terms, _ = parse_compose_expr(recipe)
            for _, tname in terms:
                if tname in probes:
                    deps[name].add(tname)
                    
    sorted_probes = []
    visited = set()
    temp_mark = set()
    
    def visit(n):
        if n in temp_mark:
            return
        if n not in visited:
            temp_mark.add(n)
            for m in sorted(deps.get(n, set())):
                visit(m)
            temp_mark.remove(n)
            visited.add(n)
            sorted_probes.append(n)
            
    for name in sorted(probes.keys()):
        if name not in visited:
            visit(name)
            
    for name in sorted_probes:
        fr = probes[name].get("framings") or ("", "")
        a = fr[0] if len(fr) > 0 else ""
        b = fr[1] if len(fr) > 1 else ""
        if a.startswith("adopted:"):
            lines.append(f":probe adopt {name}")
        elif a.startswith("composed:"):
            recipe = a.split(":", 1)[1].strip()
            lines.append(f":probe compose {name} {recipe}")
        elif a or b:
            lines.append(f":probe {name} {a} || {b}")
        else:
            lines.append(f"# {name}: no framings stored -- cannot regenerate from text (only its .pt weights hold it)")
            continue
        if probes[name].get("exposed"):
            lines.append(f":probe expose {name}")
        if probe_is_released(probes[name]):
            lines.append(f":probe release {name}")
    return lines


def expand_macro_lines(target, macro_aliases, visited=None, depth=0, args=None):
    """Flatten a macro (by alias or path) into its list of commands. A bare line
    that is itself a macro -- a known alias, or an existing file -- is inlined
    recursively, so one macro can RUN others (`:run start` runs the macros 'start'
    lists, instead of typing their names at the model). Comments and blanks are
    dropped. Returns None if the target file does not exist; [] on a cycle or
    excessive depth."""
    if visited is None:
        visited = set()
    path = macro_aliases.get(target, target)
    if not os.path.isfile(path):
        return None
    key = os.path.abspath(path)
    if key in visited or depth > 20:
        return []
    visited.add(key)
    out = []
    with open(path, "r", encoding="utf-8") as rf:
        for raw in rf:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
                
            if args:
                for i, arg in enumerate(args):
                    line = line.replace(f"${i+1}", arg)
                line = line.replace("$@", " ".join(args))
                    
            if not line.startswith(":"):
                nested = macro_aliases.get(line, line)
                # A macro reference is a single token (a filename or a glob like
                # fun_init*); a line with spaces is a prompt, so 'Is learning fun?'
                # stays an utterance despite the '?'.
                if os.path.isfile(nested):
                    matches = [nested]
                elif " " not in line and any(c in line for c in "*?["):
                    matches = sorted(glob.glob(nested))
                    if not matches:
                        print(Fore.YELLOW + f"[System] Macro reference '{line}' matched no files -- skipped." + Style.RESET_ALL)
                else:
                    matches = None                        # plain text -> an utterance
                if matches is not None:
                    # A file/glob reference is a nested macro, never a prompt: inline
                    # each match's commands (an unmatched glob adds nothing).
                    for m in matches:
                        sub = expand_macro_lines(m, macro_aliases, visited, depth + 1, args=args)
                        if sub:
                            out.extend(sub)
                    continue
            out.append(line)
            
    # Post-process to bundle consecutive non-command lines into a single multi-line string
    bundled_out = []
    current_text_block = []
    for line in out:
        if line.startswith(":"):
            if current_text_block:
                bundled_out.append("\n".join(current_text_block))
                current_text_block = []
            bundled_out.append(line)
        else:
            current_text_block.append(line)
    if current_text_block:
        bundled_out.append("\n".join(current_text_block))
        
    return bundled_out


# `:game` config lives in games/<name>_rules.json. Rules stay flat top-level
# keys (back-compat with old files); win/loss conditions and prizes live under
# reserved '__'-prefixed keys. These words can't be used as rule names.
GAME_RESERVED = {"win", "loss", "prize", "end", "stop", "quit", "start", "draw"}


def load_game_config(path):
    """Read a game's rules json into {rules, win, loss, prizes}. Old flat files
    (just rule->desc) load as rules with empty conditions/prizes."""
    raw = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as rf:
                raw = json.load(rf)
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "rules": {k: v for k, v in raw.items() if not str(k).startswith("__")},
        "win": raw.get("__win", ""),
        "loss": raw.get("__loss", ""),
        "prizes": raw.get("__prizes", {}) if isinstance(raw.get("__prizes"), dict) else {},
    }


def save_game_config(path, cfg):
    """Flatten {rules, win, loss, prizes} back to the on-disk json shape."""
    out = dict(cfg.get("rules", {}))
    if cfg.get("win"):
        out["__win"] = cfg["win"]
    if cfg.get("loss"):
        out["__loss"] = cfg["loss"]
    if cfg.get("prizes"):
        out["__prizes"] = cfg["prizes"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as wf:
        json.dump(out, wf, indent=2)


def command_keeps_semicolons(line):
    """Commands whose arguments may deliberately contain semicolons."""
    low = (line or "").lstrip().lower()
    if low.startswith("@"):
        parts = low.split(maxsplit=1)
        low = parts[1].strip() if len(parts) == 2 else ""
    return low.startswith(":macro ") or low.startswith(":game restore ")


def command_takes_colon_command_arg(line):
    """Commands where a later ':word' is an argument, not a chained command."""
    low = (line or "").lstrip().lower()
    if low.startswith("@"):
        parts = low.split(maxsplit=1)
        low = parts[1].strip() if len(parts) == 2 else ""
    return (
        low.startswith(":expose ")
        or low.startswith(":hide ")
        or low.startswith(":suggest ")
        or low.startswith(":queue ")
    )


def split_cmd_tool_commands(cmd):
    """Split a model <<TOOL: ...>> request into shell commands.

    Normal shell chaining uses semicolons, but :macro uses semicolons inside its
    argument to define macro bodies, so it must stay whole.
    """
    c = (cmd or "").strip()
    if not c:
        return []
    if not c.startswith(":") and not c.startswith("@"):
        c = ":" + c
    if command_keeps_semicolons(c):
        return [c]
    out = []
    for part in c.split(";"):
        p = part.strip()
        if not p:
            continue
        if not p.startswith(":") and not p.startswith("@"):
            p = ":" + p
        out.append(p)
    return out


def normalize_command_target(target):
    t = str(target or "").strip()
    if not t:
        return ""
    if not t.startswith("@"):
        t = "@" + t
    t = t.split()[0].strip()
    return t if len(t) > 1 else ""


def parse_command_route(cmd):
    """Return (optional @agent target, command word, argument tail)."""
    s = (cmd or "").strip()
    target = ""
    if s.startswith("@"):
        parts = s.split(maxsplit=1)
        if len(parts) != 2:
            return normalize_command_target(parts[0]), "", ""
        target = normalize_command_target(parts[0])
        s = parts[1].strip()
    if not s.startswith(":"):
        return target, "", ""
    parts = s[1:].split(maxsplit=1)
    word = parts[0].lower() if parts else ""
    tail = parts[1].strip() if len(parts) > 1 else ""
    return target, word, tail


def command_word(cmd):
    return parse_command_route(cmd)[1]


def command_arg_tail(cmd):
    return parse_command_route(cmd)[2]


EXPOSE_STAGE_WORDS = ("stage", "staged", "accept", "on", "expose")
EXPOSE_DIRECT_WORDS = ("direct", "run", "trust", "free")


def parse_expose_prefix(eargs):
    """Normalize the tokens after ':expose' into
    (turn_off, target_agent, mode_prefix, rest).

    Canonical grammar puts the mode BEFORE the command --
    ':expose [off] [@agent] [stage|direct] :command [fixed args...]' --
    so fixed args can never collide with a mode keyword. '@agent' and the
    mode may appear in either order; a mode word is only consumed when a
    ':command' (or '@agent :command') follows it, so bare probe/knob targets
    are untouched. mode_prefix is None when no leading mode was given -- the
    legacy trailing position (':expose :cmd direct args') still applies then.
    """
    toks = list(eargs)
    turn_off = False
    if toks and toks[0].lower() in ("off", "hide", "unexpose"):
        turn_off = True
        toks = toks[1:]
    target_agent = ""
    mode_prefix = None
    for _ in range(2):
        if not toks:
            break
        if toks[0].startswith("@") and len(toks) > 1 and not target_agent:
            target_agent = toks[0]
            toks = toks[1:]
            continue
        low = toks[0].lower()
        if (
            mode_prefix is None
            and (low in EXPOSE_STAGE_WORDS or low in EXPOSE_DIRECT_WORDS)
            and len(toks) > 1
            and (toks[1].startswith(":") or toks[1].startswith("@"))
        ):
            mode_prefix = "direct" if low in EXPOSE_DIRECT_WORDS else "stage"
            toks = toks[1:]
            continue
        break
    return turn_off, target_agent, mode_prefix, toks


def validate_command_autocomplete(
    response,
    prefix="",
    *,
    known_commands=(),
    probe_names=(),
    knob_names=(),
):
    """Normalize one completion and reject locally-detectable invalid syntax."""
    lines = [line.strip() for line in (response or "").splitlines() if line.strip()]
    if len(lines) != 1:
        return None, "the model did not return exactly one command line"
    candidate = lines[0].strip().strip("`").strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
        candidate = candidate[1:-1].strip()
    if prefix and not candidate.startswith(prefix):
        return None, f"the completion does not start with {prefix!r}"
    if not candidate.startswith(":"):
        return None, "the completion is not an operator command"

    if command_word(candidate) == "expose":
        args = candidate.split()[1:]
        _turn_off, _target_agent, _mode, rest = parse_expose_prefix(args)
        if not rest:
            return None, "the :expose completion has no probe, knob, or :command target"
        target = rest[0]
        if target.startswith(":"):
            word = target.lstrip(":").lower()
            allowed = {str(name).lstrip(":").lower() for name in known_commands}
            if word == "expose" or word not in allowed:
                return None, f"{target!r} is not an exposable command"
        else:
            sensor = re.sub(r"[^a-z0-9_]", "_", target.lower())[:40].strip("_")
            readable = {str(name).lower() for name in probe_names}
            readable.update(str(name).lower() for name in knob_names)
            readable.add("memory")
            if sensor not in readable:
                return None, f"{target!r} is not an active probe or knob"
    return candidate, None


def apply_command_exposure_args(cmd, exposure):
    requested_target, word, supplied = parse_command_route(cmd)
    if not word:
        return cmd
    target = command_exposure_target(exposure) or requested_target
    fixed = command_exposure_args(exposure)
    if fixed and supplied:
        if supplied == fixed:
            supplied = ""
        elif supplied.startswith(fixed + " "):
            supplied = supplied[len(fixed):].strip()
    pieces = []
    if target:
        pieces.append(target)
    pieces.append(f":{word}")
    if fixed:
        pieces.append(fixed)
    if supplied:
        pieces.append(supplied)
    return " ".join(pieces)


def restore_command_path(path, root=ROOT):
    """Prefer root-relative paths so replay commands survive spaces in ROOT."""
    try:
        ap = os.path.abspath(path)
        rp = os.path.relpath(ap, root)
        if not rp.startswith(".."):
            return rp.replace(os.sep, "/")
    except Exception:
        pass
    return str(path)


DEFAULT_COMMAND_TOOL_STEER = "command-defined action; record with 'steer <magnitude>' when the tool has a specific strength"


def normalize_command_exposure(record):
    """Return the persisted shape for an exposed command tool.

    Older saves store only "stage"/"direct". New saves may also carry a
    human-readable activation criterion and explicit steer magnitude note.
    """
    if isinstance(record, dict):
        mode = str(record.get("mode", "stage") or "stage").lower()
        activation = str(record.get("activation", "") or "").strip()
        steer = str(
            record.get("steer_magnitude", record.get("steer", ""))
            or ""
        ).strip()
        args_raw = record.get("args", record.get("fixed_args", ""))
        target = normalize_command_target(
            record.get("target", record.get("target_agent", record.get("agent", "")))
        )
        if isinstance(args_raw, (list, tuple)):
            fixed_args = " ".join(str(x).strip() for x in args_raw if str(x).strip())
        else:
            fixed_args = str(args_raw or "").strip()
    else:
        mode = str(record or "stage").lower()
        activation = ""
        steer = ""
        fixed_args = ""
        target = ""
    if mode not in {"direct", "stage"}:
        mode = "stage"
    return {
        "mode": mode,
        "activation": activation,
        "steer_magnitude": steer,
        "args": fixed_args,
        "target": target,
    }


def command_exposure_mode(record):
    return normalize_command_exposure(record)["mode"]


def command_exposure_custom_activation(record):
    return normalize_command_exposure(record)["activation"]


def command_exposure_activation(word, record):
    activation = command_exposure_custom_activation(record)
    if activation:
        return activation
    display = command_exposure_display(word, record, include_args=False)
    return f"no activation criterion recorded; expose again with ':expose {display} ... because <activation>'"


def command_exposure_steer(record):
    steer = normalize_command_exposure(record)["steer_magnitude"]
    return steer or DEFAULT_COMMAND_TOOL_STEER


def command_exposure_raw_steer(record):
    return normalize_command_exposure(record)["steer_magnitude"]


def command_exposure_args(record):
    return normalize_command_exposure(record)["args"]


def command_exposure_target(record):
    return normalize_command_exposure(record)["target"]


def command_exposure_display(word, record, *, include_args=True):
    record = normalize_command_exposure(record)
    target = record["target"]
    pieces = []
    if target:
        pieces.append(target)
    pieces.append(f":{str(word).lstrip(':').lower()}")
    if include_args and record["args"]:
        pieces.append(record["args"])
    return " ".join(pieces)


def normalize_game_config(raw):
    """Accept the internal {rules, win, loss, prizes} shape or old flat JSON."""
    if not isinstance(raw, dict):
        raw = {}
    if any(k in raw for k in ("rules", "win", "loss", "prizes")):
        rules = raw.get("rules", {})
        prizes = raw.get("prizes", {})
        return {
            "rules": dict(rules) if isinstance(rules, dict) else {},
            "win": str(raw.get("win", "") or ""),
            "loss": str(raw.get("loss", "") or ""),
            "prizes": dict(prizes) if isinstance(prizes, dict) else {},
        }
    return {
        "rules": {k: v for k, v in raw.items() if not str(k).startswith("__")},
        "win": str(raw.get("__win", "") or ""),
        "loss": str(raw.get("__loss", "") or ""),
        "prizes": raw.get("__prizes", {}) if isinstance(raw.get("__prizes"), dict) else {},
    }


def build_session_restore_macro(probes, macro_aliases, exposed_commands, exposed_knobs=None, hidden_commands=None, self_dest=None, root=ROOT):
    """Regenerate probes plus shell-defined macros, hidden/exposed commands, and games."""
    exposed_knobs = set(exposed_knobs or [])
    hidden_commands = set(hidden_commands or [])
    lines = ["# Regenerates probes, macro commands, hidden/exposed runtime tools, and game configs."]
    probe_lines = build_probe_init_macro(probes)[1:]
    lines.extend(probe_lines)
    stats = {
        "probes": len(probes),
        "macro_files": 0,
        "macro_aliases": 0,
        "hidden_commands": 0,
        "exposed_commands": 0,
        "exposed_knobs": 0,
        "games": 0,
        "skipped": [],
    }

    self_abs = os.path.abspath(self_dest) if self_dest else None
    seen_paths = set()
    for alias, path in sorted((macro_aliases or {}).items()):
        try:
            abs_path = os.path.abspath(path)
        except Exception:
            abs_path = str(path)
        if self_abs and abs_path == self_abs:
            continue
        if abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)
        if not os.path.isfile(path):
            stats["skipped"].append(f"macro file missing for {alias}: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as rf:
                body = rf.read().splitlines()
        except Exception as e:
            stats["skipped"].append(f"macro file unreadable for {alias}: {e}")
            continue
        payload = {"path": restore_command_path(path, root), "lines": body}
        lines.append(":macro restore " + json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        stats["macro_files"] += 1

    for alias, path in sorted((macro_aliases or {}).items()):
        try:
            abs_path = os.path.abspath(path)
        except Exception:
            abs_path = str(path)
        if alias == "self" or (self_abs and abs_path == self_abs):
            continue
        lines.append(f":macro name {alias} {restore_command_path(path, root)}")
        stats["macro_aliases"] += 1

    for word in sorted(hidden_commands):
        lines.append(f":hide :{word}")
        stats["hidden_commands"] += 1

    for knob in sorted(exposed_knobs):
        lines.append(f":expose {knob}")
        stats["exposed_knobs"] += 1

    for word, mode in sorted((exposed_commands or {}).items()):
        if word == "expose":
            continue
        record = normalize_command_exposure(mode)
        # canonical order: [@agent] [mode] :command [fixed args...] -- mode
        # before the command so fixed args never collide with mode keywords.
        agent_prefix = f"{command_exposure_target(record)} " if command_exposure_target(record) else ""
        mode_word = "direct " if command_exposure_mode(record) == "direct" else ""
        fixed_args = command_exposure_args(record)
        args_suffix = f" {fixed_args}" if fixed_args else ""
        steer = command_exposure_raw_steer(record)
        steer_suffix = f" steer {steer}" if steer else ""
        because = command_exposure_custom_activation(record)
        because_suffix = f" because {because}" if because else ""
        lines.append(f":expose {agent_prefix}{mode_word}:{str(word).lstrip(':').lower()}{args_suffix}{steer_suffix}{because_suffix}")
        stats["exposed_commands"] += 1

    games_dir = os.path.join(root, "games")
    if os.path.isdir(games_dir):
        for fn in sorted(os.listdir(games_dir)):
            if not fn.endswith("_rules.json"):
                continue
            name = fn[:-len("_rules.json")]
            try:
                cfg = load_game_config(os.path.join(games_dir, fn))
            except Exception as e:
                stats["skipped"].append(f"game config unreadable for {name}: {e}")
                continue
            payload = {"name": name, "config": cfg}
            lines.append(":game restore " + json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
            stats["games"] += 1
    if stats["games"]:
        lines.append("# Note: game rule configs are restored; custom games/*.py scripts are ordinary files.")
    for skipped in stats["skipped"]:
        lines.append(f"# skipped: {skipped}")
    return lines, stats


def parse_prize_spec(argstr):
    """Parse '<option>,<odds> [<count>,<odds_val>] <command>' into
    (option, {odds, count, odds_val, command}) or (None, error_message)."""
    s = (argstr or "").strip()
    if not s:
        return None, "empty prize spec"
    count, odds_val = None, None
    m = re.search(r"\[([^\]]*)\]", s)  # optional [count,odds_val]
    if m:
        inner = m.group(1)
        s = (s[:m.start()] + " " + s[m.end():]).strip()
        cparts = [x.strip() for x in inner.split(",")]
        try:
            if cparts and cparts[0]:
                count = int(float(cparts[0]))
            if len(cparts) > 1 and cparts[1]:
                odds_val = float(cparts[1])
        except ValueError:
            return None, f"bad [count,odds_val]: '{inner}'"
    head, _, command = s.partition(" ")
    command = command.strip()
    option, _, odds_s = head.partition(",")
    option = option.strip()
    if not option:
        return None, "missing option name"
    odds = odds_val if odds_val is not None else 1.0
    if odds_s.strip():
        try:
            odds = float(odds_s.strip())
        except ValueError:
            return None, f"bad odds: '{odds_s}'"
    return option, {"odds": odds, "count": count, "odds_val": odds_val, "command": command}


def draw_prizes(prizes):
    """Draw each prize against its odds; return [(option, command), ...] won,
    decrementing any finite counts in place. A prize whose count hit 0 is spent."""
    won = []
    for opt, p in prizes.items():
        cnt = p.get("count")
        if cnt is not None and cnt <= 0:
            continue
        if random.random() < float(p.get("odds", 0.0) or 0.0):
            won.append((opt, p.get("command", "")))
            if cnt is not None:
                p["count"] = cnt - 1
    return won


def parse_macro_arg_spec(raw, fallback="arg"):
    """Parse one macro arg declaration.

    Accepted forms: name, name?, [name], name=default, [name=default].
    A leading +, -, or $ is a solve-time hint, and trailing ! / !! keeps the
    existing auto/choose restriction hints.
    """
    original = (raw or "").strip()
    spec = original
    hint_prefix = ""
    if spec[:1] in {"+", "-", "$"}:
        hint_prefix, spec = spec[0], spec[1:]

    no_auto_or_choose = False
    no_choose = False
    if spec.endswith("!!"):
        no_auto_or_choose = True
        spec = spec[:-2]
    elif spec.endswith("!"):
        no_choose = True
        spec = spec[:-1]

    spec = spec.strip()
    optional = False
    if spec.startswith("[") and spec.endswith("]"):
        optional = True
        spec = spec[1:-1].strip()

    default = None
    if "?=" in spec:
        name, default = spec.split("?=", 1)
        optional = True
    elif "=" in spec:
        name, default = spec.split("=", 1)
        optional = True
    else:
        name = spec
        if name.endswith("?"):
            optional = True
            name = name[:-1]

    name = re.sub(r"[^A-Za-z0-9_]", "_", name.strip()).strip("_")
    if name and name[0].isdigit():
        name = "_" + name
    if not name:
        name = re.sub(r"[^A-Za-z0-9_]", "_", fallback).strip("_") or "arg"

    if default is not None:
        default = default.strip()
    header = f"{name}={default}" if default is not None else (f"{name}?" if optional else name)
    return {
        "name": name,
        "header": header,
        "optional": optional,
        "default": default,
        "hint_prefix": hint_prefix,
        "no_choose": no_choose,
        "no_auto_or_choose": no_auto_or_choose,
        "raw": original,
    }


def parse_macro_arg_header(header):
    return [
        parse_macro_arg_spec(part, fallback=f"arg{i + 1}")
        for i, part in enumerate((header or "").split(","))
        if part.strip()
    ]


def macro_arg_header_items(arg_specs):
    """Render parsed/raw macro arg specs back to '# args:' header tokens."""
    out = []
    for spec in arg_specs or []:
        if isinstance(spec, dict):
            item = spec.get("header") or spec.get("name") or spec.get("raw")
        else:
            item = str(spec)
        item = str(item or "").strip()
        if item:
            out.append(item)
    return out


def substitute_macro_params(text, args):
    """Fill $1..$9 (positional), $@ (all args), and named $args in a macro body.
    A param with no supplied arg collapses to empty -- named params behave like
    positional $N, so an unfilled '$name' never leaks through as a literal token."""
    lines = text.splitlines()
    for ln in lines:
        if ln.strip().startswith("# args:"):
            arg_specs = parse_macro_arg_header(ln.strip()[len("# args:"):])
            for i, spec in enumerate(arg_specs):
                value = args[i] if i < len(args) else (spec["default"] if spec["default"] is not None else "")
                if spec["name"]:
                    text = re.sub(r"\$" + re.escape(spec["name"]) + r"\b", lambda _m, v=value: v, text)
            break

    joined = " ".join(args)
    text = text.replace("$@", joined).replace("$*", joined)
    text = re.sub(r"\$([1-9])", lambda m: args[int(m.group(1)) - 1] if int(m.group(1)) <= len(args) else "", text)
    # Any named param left unfilled collapses to empty (like a missing $N), then
    # collapse the doubled spaces that leaves behind.
    text = re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*", "", text)
    return "\n".join(re.sub(r" {2,}", " ", l).rstrip() for l in text.splitlines())


def load_parameterized_macro(path, args):
    """Read a macro file, substitute $-parameters from args, and return its
    runnable command lines (comments/blank lines dropped). None if missing."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as rf:
        body = substitute_macro_params(rf.read(), args)
    return [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def compute_expert_proof_scores(steer_map, min_events=8):
    """Per-expert route success rate from the steer map's accrued outcomes,
    plus '__mean__' (the overall route success rate = centering prior). Only
    experts with >= min_events labeled routing events earn a score; the rest
    stay unproven and are left neutral by the engine. Returns {} until there
    is enough labeled routing evidence -- no evidence, no nudge."""
    try:
        agg = steer_map.aggregate()
    except Exception:
        return {}
    routes = {}
    tot_s = tot_n = 0
    for g in agg.get("groups", []):
        action = str(g.get("action", ""))
        if not action.startswith("route_"):
            continue
        ln = int(g.get("labeled_n") or 0)
        if ln <= 0:
            continue
        name = g.get("expert_or_target") or action[len("route_"):]
        s = int(g.get("success") or 0)
        acc = routes.setdefault(name, [0, 0])
        acc[0] += s
        acc[1] += ln
        tot_s += s
        tot_n += ln
    scores = {name: s / n for name, (s, n) in routes.items() if n >= min_events}
    if scores and tot_n:
        scores["__mean__"] = tot_s / tot_n
    return scores


def did_you_mean(name, candidates):
    """A ' Did you mean X?' suffix for a mistyped name, or '' when nothing is
    close. `candidates` is any iterable of valid names -- so the same helper
    serves :calibrate (streams), :tune (triggers), and :probe (probe names).
    Cutoff 0.72: real typos land >=0.78 (repitition/repetition, priority/
    prioritize), while unrelated words like priority/curiosity (~0.71) don't
    trigger a misleading guess."""
    import difflib
    match = difflib.get_close_matches(
        (name or "").lower(), sorted(set(candidates)), n=1, cutoff=0.72
    )
    return f" Did you mean '{match[0]}'?" if match else ""


def _match_corr(pairs):
    """Pearson r between (knob_value, probe_reading) pairs -- the credit a matched
    probe gives its knob (does moving the knob move the probe?). None if too few
    pairs or either side never varied."""
    if not pairs or len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx ** 0.5 * sy ** 0.5)


def latest_probe_signal_from_history(history):
    """Reconstruct the latest centered probe signal from raw probe history."""
    vals = [float(v) for v in (history or [])]
    if not vals:
        return None, None, 0
    raw = vals[-1]
    sig = raw - (sum(vals[:-1]) / len(vals[:-1])) if len(vals) > 1 else 0.0
    return sig, raw, len(vals)


# Every command word the shell recognizes -- an unknown :word is either a typo
# (suggest the nearest) or a request to invent a tool (offer to mint a probe),
# never a generation of a generic essay.
def consume_probe_args(args):
    """Extract a probe name (which might be 'auto +ref -ref') and return (pname_raw, remaining_args)."""
    if not args:
        return "", []
    pname = args[0]
    idx = 1
    if pname.upper() in ("AUTO", "CHOOSE", "CHOICE"):
        while idx < len(args) and args[idx].startswith(("+", "-")):
            pname += " " + args[idx]
            idx += 1
    return pname, args[idx:]

def resolve_probe_choice(pname_raw, options, model=None, config=None, action_name=""):
    tokens = pname_raw.split()
    if not tokens:
        return ""
    base = tokens[0].upper()
    if base in ("CHOICE", "CHOOSE", "AUTO"):
        if not options:
            print(Fore.YELLOW + f"[{base.capitalize()}] No active options to choose from." + Style.RESET_ALL)
            return None
        if not model or not config:
            print(Fore.RED + "[Error] Model not available for choice." + Style.RESET_ALL)
            return None

        plist = list(options.keys()) if isinstance(options, dict) else list(options)
        refs = tokens[1:]
        ref_str = ""
        if refs:
            ref_str = f"The user has provided the following reference guidance: {' '.join(refs)}\nUse this guidance to inform your selection.\n\n"

        print(Fore.CYAN + f"[{base.capitalize()}] Asking the model to select a target for '{action_name}'..." + Style.RESET_ALL)

        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"You are selecting a target parameter/probe for the action: '{action_name}'.\n"
            f"Available options:\n" + "\n".join(f"- {p}" for p in plist) + "\n\n"
            f"{ref_str}"
            f"Select the single most appropriate target from the list above. "
            f"Output ONLY the exact name of the target, and nothing else.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        sug = generate_agentic_text(
            model,
            instruction=prompt,
            config=config,
            pre_formatted=True,
            max_new_tokens=20
        )
        sug = sug.strip()
        if sug in options:
            print(Fore.GREEN + f"[Choice] The model chose: {sug}" + Style.RESET_ALL)
            return sug
        else:
            print(Fore.YELLOW + f"[Choice] Model selected invalid target '{sug}'. Aborting." + Style.RESET_ALL)
            return None

    return pname_raw.replace("probe_", "") if pname_raw.startswith("probe_") else pname_raw


# --- :figures -- draw the state-output figures ------------------------------
# Delegates to scripts/make_lesswrong_figures.py in a subprocess. Building
# needs matplotlib, which the repo venv does not carry, so the wrapper finds
# the first interpreter that can draw (this process's python, then PATH
# pythons) and caches the answer. Listing needs no matplotlib at all -- the
# figure script imports it lazily.
FIGURES_SCRIPT = os.path.join(ROOT, "scripts", "make_lesswrong_figures.py")
_FIGURES_PYTHON = {"exe": None}


def _find_figures_python():
    import importlib.util
    if importlib.util.find_spec("matplotlib") is not None:
        return sys.executable
    if _FIGURES_PYTHON["exe"]:
        return _FIGURES_PYTHON["exe"]
    import shutil
    import subprocess
    seen = {os.path.normcase(os.path.abspath(sys.executable))}
    for name in ("python", "python3", "py"):
        exe = shutil.which(name)
        if not exe:
            continue
        key = os.path.normcase(os.path.abspath(exe))
        if key in seen:
            continue
        seen.add(key)
        try:
            ok = subprocess.run(
                [exe, "-c", "import matplotlib"], capture_output=True, timeout=60
            ).returncode == 0
        except Exception:
            ok = False
        if ok:
            _FIGURES_PYTHON["exe"] = exe
            return exe
    return None


def _run_figures_script(args, require_mpl=True):
    """Run the figure script with args; returns (ok, printable lines)."""
    import subprocess
    exe = _find_figures_python() if require_mpl else sys.executable
    if not exe:
        return False, [
            "no python with matplotlib found (checked this process and PATH pythons).",
            "install one -- e.g.  python -m pip install matplotlib  -- then retry :figures",
        ]
    try:
        proc = subprocess.run(
            [exe, FIGURES_SCRIPT] + list(args),
            capture_output=True, text=True, cwd=ROOT, timeout=300,
        )
    except Exception as exc:
        return False, [f"figure run failed to start: {exc}"]
    lines = [l for l in (proc.stdout or "").splitlines() if l.strip()]
    if proc.returncode != 0:
        tail = [l for l in (proc.stderr or "").splitlines() if l.strip()][-6:]
        return False, (lines + tail) or [f"figure run exited {proc.returncode} with no output"]
    return True, lines or ["(no output)"]


# --- :figure -- draw live tracked shell values ------------------------------
# Singular :figure is the live telemetry command: it reads the same named
# streams that :tune, :calibrate, :label, and :probe values use. Plural
# :figures stays the publication-figure builder above.
LIVE_FIGURE_DIR = os.path.join(ROOT, "invariants", "out", "figures")
LIVE_FIGURE_COLORS = (
    "#2f6fed", "#13866f", "#c76f1b", "#b83b4a", "#7556c8",
    "#537188", "#a65f2b", "#4b8f8c", "#6f7c85", "#9467bd",
)


def _figure_esc(value):
    import html
    return html.escape(str(value), quote=True)


def _figure_slug(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return text[:90] or "live"


def _figure_float(value):
    import math
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _figure_numeric_keys(turn_log):
    keys = set()
    for row in turn_log:
        for key, value in row.items():
            if key != "ts" and _figure_float(value) is not None:
                keys.add(key)
    return keys


def _figure_rows_for_key(turn_log, key, limit):
    rows = list(turn_log)
    if limit is not None:
        rows = rows[-limit:]
    values = []
    for row in rows:
        if key in row:
            v = _figure_float(row.get(key))
            if v is not None:
                values.append(v)
    return values


def _figure_trigger_for_key(key, tuner, probes):
    if key in tuner.triggers:
        return tuner.triggers.get(key)
    if key in probes:
        return tuner.triggers.get(f"probe_{key}")
    if key.startswith("probe_"):
        return tuner.triggers.get(key)
    return None


def _figure_probe_for_key(key, probes):
    if key in probes:
        return probes[key]
    if key.startswith("probe_") and key[len("probe_"):] in probes:
        return probes[key[len("probe_"):]]
    return None


def _figure_outcome_series(key, tuner, limit):
    """Make the OUTCOME channel mappable: outcome:<name> draws a trigger's
    credited outcomes in credit order; lift:<name> draws the rolling lift
    trajectory -- mean(fired) - mean(unfired) over everything credited so far.
    The honest 'did firing actually help' readout becomes a drawable stream
    instead of a single number buried in a note."""
    mode, _, name = key.partition(":")
    mode = mode.lower()
    name = name.strip()
    trig = tuner.triggers.get(name) or tuner.triggers.get(f"probe_{name}")
    if trig is None or not trig.outcomes:
        return None
    pairs = list(trig.outcomes)
    if limit is not None and mode == "outcome":
        pairs = pairs[-limit:]
    ostats = trig.outcome_stats()
    if mode == "outcome":
        values = [float(o) for _s, o in pairs]
        threshold = None
    else:
        # Rolling lift: re-derive the fired/unfired means as each credit
        # lands, so the drawn line shows the correlation stabilizing (or not).
        values = []
        fired, unfired = [], []
        for s, o in pairs:
            (fired if trig._fires(s) else unfired).append(float(o))
            if fired and unfired:
                values.append(sum(fired) / len(fired) - sum(unfired) / len(unfired))
        if limit is not None:
            values = values[-limit:]
        threshold = 0.0
    if not values:
        return None
    note_bits = [
        "tuner.outcomes", f"n={len(values)}", f"last={values[-1]:+.4g}",
        f"fired_mean={ostats['fired_outcome']}", f"unfired_mean={ostats['unfired_outcome']}",
        f"lift={ostats['lift']}",
    ]
    # The continuous correlation next to the binary lift: r between the
    # credited signal and the outcome that followed it, in credit order.
    sig_r = _figure_corr([float(s) for s, _o in pairs], [float(o) for _s, o in pairs])
    if sig_r is not None:
        note_bits.append(f"signal_r={sig_r:+.2f}")
    return {
        "key": key,
        "label": f"{mode}:{getattr(trig, 'name', name)}",
        "values": values,
        "threshold": threshold,
        "comparator": ">=",
        "note": " | ".join(note_bits),
    }


def _figure_series_for_key(key, tuner, probes, turn_log, limit):
    if key.lower().startswith(("outcome:", "lift:")):
        return _figure_outcome_series(key, tuner, limit)
    label = key
    source = "turn_log"
    values = _figure_rows_for_key(turn_log, key, limit)

    trig = _figure_trigger_for_key(key, tuner, probes)
    if not values and trig is not None and trig.signals:
        source = "tuner.signals"
        values = list(trig.signals)
        if limit is not None:
            values = values[-limit:]
    if not values and trig is not None and trig.outcomes:
        source = "tuner.outcome_values"
        values = [float(sig) for sig, _outcome in trig.outcomes]
        if limit is not None:
            values = values[-limit:]
    pdata = _figure_probe_for_key(key, probes)
    if not values and pdata is not None and pdata.get("history"):
        source = "probe.raw_history"
        values = [float(v) for v in pdata.get("history", [])]
        if limit is not None:
            values = values[-limit:]
    if not values and trig is not None:
        source = "current_value"
        values = [float(trig.value)]

    if not values:
        return None

    threshold = None
    comparator = ">="
    note_bits = [source, f"n={len(values)}", f"last={values[-1]:+.4g}"]
    if trig is not None:
        threshold = float(trig.value)
        comparator = trig.comparator
        stats = trig.stats()
        if stats.get("kind"):
            note_bits.append(stats["kind"])
        if stats.get("fire_rate") is not None:
            note_bits.append(f"fire={stats['fire_rate']}")
        if stats.get("lift") is not None:
            note_bits.append(f"lift={stats['lift']}")
        if stats.get("n_credited"):
            note_bits.append(f"credited={stats['n_credited']}")
    return {
        "key": key,
        "label": label,
        "values": values,
        "threshold": threshold,
        "comparator": comparator,
        "note": " | ".join(note_bits),
    }


def _figure_available_names(tuner, probes, turn_log):
    names = set(_figure_numeric_keys(turn_log))
    names.update(tuner.triggers.keys())
    names.update(probes.keys())
    names.update(f"probe_{p}" for p in probes.keys())
    names.update(f"outcome:{n}" for n, t in tuner.triggers.items() if t.outcomes)
    names.update(f"lift:{n}" for n, t in tuner.triggers.items() if t.outcomes)
    return sorted(names)


def _figure_group_tokens(tokens, tuner, probes, turn_log):
    if not tokens:
        tokens = ["recent"]
    numeric = _figure_numeric_keys(turn_log)
    out = []
    for token in tokens:
        low = token.lower()
        if low == "recent":
            recent = []
            if turn_log:
                last = turn_log[-1]
                recent = [k for k in last if k != "ts" and _figure_float(last.get(k)) is not None]
            out.extend(recent or sorted(numeric)[:12])
        elif low == "all":
            out.extend(_figure_available_names(tuner, probes, turn_log))
        elif low == "probes":
            out.extend(f"probe_{p}" for p in sorted(probes))
        elif low == "knobs":
            out.extend(
                name for name, trig in sorted(tuner.triggers.items())
                if trig.kind == "coefficient"
            )
        elif low == "streams":
            out.extend(
                name for name, trig in sorted(tuner.triggers.items())
                if trig.kind == "threshold" and not name.startswith("probe_")
            )
            out.extend(k for k in sorted(numeric) if not k.startswith("probe_"))
        elif low == "phen":
            out.extend(k for k in sorted(numeric | set(tuner.triggers)) if k.startswith("phen_"))
        elif low in ("outcome", "outcomes", "lift", "lifts"):
            prefix = "lift" if low.startswith("lift") else "outcome"
            out.extend(f"{prefix}:{n}" for n, t in sorted(tuner.triggers.items()) if t.outcomes)
        elif low.startswith(("outcome:", "lift:")):
            out.append(token)
        else:
            stream = resolve_stream(token, tuner)
            if stream is not None:
                out.append(stream)
            elif token in probes:
                out.append(f"probe_{token}")
            else:
                out.append(token)
    seen = set()
    unique = []
    for key in out:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _parse_live_figure_args(raw):
    toks = raw.split()
    mode = "draw"
    limit = 120
    out_path = None
    names = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        low = tok.lower()
        if low in ("list", "ls", "--list", "-l"):
            mode = "list"
            i += 1
        elif low in ("patterns", "pattern", "detect", "analysis", "analyze"):
            mode = "patterns"
            i += 1
        elif low in ("last", "n", "--last", "--n") and i + 1 < len(toks):
            limit = max(1, int(float(toks[i + 1])))
            i += 2
        elif low in ("out", "--out") and i + 1 < len(toks):
            out_path = toks[i + 1]
            i += 2
        else:
            names.append(tok)
            i += 1
    return mode, names, limit, out_path


def _write_live_figure_svg(series, out_path=None, title="Live tracked values"):
    os.makedirs(LIVE_FIGURE_DIR, exist_ok=True)
    if out_path:
        path = out_path
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _figure_slug("_".join(s["key"] for s in series[:4]))
        path = os.path.join(LIVE_FIGURE_DIR, f"live_{stamp}_{slug}.svg")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    width = 980
    left = 235
    chart_w = 560
    row_h = 82 if any(s.get("patterns") for s in series) else 62
    top = 92
    height = max(190, top + row_h * len(series) + 36)
    ink = "#172026"
    muted = "#5f6b73"
    grid = "#d7dde2"
    bg = "#fbfcfd"

    def pts(values, x, y, w, h):
        vals = list(values)
        if len(vals) == 1:
            vals = vals * 2
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-12:
            lo -= 1.0
            hi += 1.0
        parts = []
        for i, val in enumerate(vals):
            px = x + (w * i / max(1, len(vals) - 1))
            py = y + h - ((val - lo) / (hi - lo) * h)
            parts.append(f"{px:.1f},{py:.1f}")
        return " ".join(parts), lo, hi

    body = [
        f'<text class="title" x="28" y="38">{_figure_esc(title)}</text>',
        '<text class="sub" x="28" y="62">Each row is scaled independently: use shape, direction, threshold, and last value, not cross-row magnitude.</text>',
    ]
    for i, item in enumerate(series):
        y = top + i * row_h
        color = LIVE_FIGURE_COLORS[i % len(LIVE_FIGURE_COLORS)]
        values = item["values"]
        line, lo, hi = pts(values, left, y, chart_w, 42)
        body.append(f'<text class="label" x="28" y="{y + 23}">{_figure_esc(item["label"][:30])}</text>')
        body.append(f'<line x1="{left}" y1="{y + 21}" x2="{left + chart_w}" y2="{y + 21}" stroke="{grid}" stroke-width="1"/>')
        if item.get("threshold") is not None and lo <= item["threshold"] <= hi:
            ty = y + 42 - ((item["threshold"] - lo) / (hi - lo) * 42)
            body.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{left + chart_w}" y2="{ty:.1f}" stroke="#8a96a3" stroke-dasharray="4 4" stroke-width="1"/>')
        body.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        body.append(f'<text class="small" x="{left + chart_w + 16}" y="{y + 13}">min={lo:+.3g} max={hi:+.3g}</text>')
        body.append(f'<text class="small" x="{left + chart_w + 16}" y="{y + 31}">{_figure_esc(item["note"][:44])}</text>')
        if item.get("patterns"):
            patt = "; ".join(p["summary"] for p in item["patterns"][:3])
            body.append(f'<text class="small" x="{left + chart_w + 16}" y="{y + 49}">{_figure_esc(patt[:52])}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <title>{_figure_esc(title)}</title>
  <rect width="100%" height="100%" fill="{bg}"/>
  <style>
    text {{ font-family: Inter, Segoe UI, Arial, sans-serif; fill: {ink}; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .sub {{ font-size: 13px; fill: {muted}; }}
    .label {{ font-size: 13px; }}
    .small {{ font-size: 11px; fill: {muted}; }}
  </style>
  {chr(10).join(body)}
</svg>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return path


def _figure_mean(values):
    return sum(values) / len(values) if values else None


def _figure_stdev(values):
    if len(values) < 2:
        return 0.0
    mu = _figure_mean(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _figure_median(values):
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def _figure_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx, my = _figure_mean(xs), _figure_mean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((sx * sy) ** 0.5)


def _figure_turn_pairs(turn_log, left_key, right_key, limit=None, lag=0):
    rows = list(turn_log)
    if limit is not None:
        rows = rows[-limit:]
    xs, ys = [], []
    for i, row in enumerate(rows):
        j = i + lag
        if j < 0 or j >= len(rows):
            continue
        if left_key not in row or right_key not in rows[j]:
            continue
        x = _figure_float(row.get(left_key))
        y = _figure_float(rows[j].get(right_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _figure_pattern(kind, summary, score, **extra):
    payload = {"kind": kind, "summary": summary, "score": round(float(score), 4)}
    payload.update(extra)
    return payload


def _figure_series_patterns(item, turn_log, tuner, limit):
    values = list(item.get("values") or [])
    key = item.get("key")
    patterns = []
    n = len(values)
    if n < 4:
        return [_figure_pattern("insufficient", "too few points", 0.0, n=n)]

    mu = _figure_mean(values)
    sd = _figure_stdev(values)
    span = max(values) - min(values)
    idx = list(range(n))
    trend_r = _figure_corr(idx, values)
    if trend_r is not None and n >= 6 and abs(trend_r) >= 0.48 and span > max(1e-9, 0.5 * sd):
        direction = "rising" if trend_r > 0 else "falling"
        delta = values[-1] - values[0]
        patterns.append(
            _figure_pattern(
                "trend",
                f"{direction} trend r={trend_r:+.2f}",
                abs(trend_r),
                r=round(trend_r, 4),
                delta=round(delta, 6),
            )
        )

    if n >= 8 and sd > 1e-12:
        q = max(2, n // 4)
        prev = values[-2 * q:-q]
        recent = values[-q:]
        if prev and recent:
            shift = _figure_mean(recent) - _figure_mean(prev)
            effect = shift / sd
            if abs(effect) >= 0.75:
                direction = "upshift" if effect > 0 else "downshift"
                patterns.append(
                    _figure_pattern(
                        "regime_shift",
                        f"{direction} {effect:+.2f}sd",
                        abs(effect),
                        recent_mean=round(_figure_mean(recent), 6),
                        previous_mean=round(_figure_mean(prev), 6),
                    )
                )

    med = _figure_median(values)
    mad = _figure_median([abs(v - med) for v in values]) if med is not None else None
    if mad and mad > 1e-12:
        spike_idx = [i for i, v in enumerate(values) if abs(0.6745 * (v - med) / mad) >= 3.5]
        if spike_idx:
            patterns.append(
                _figure_pattern(
                    "spikes",
                    f"{len(spike_idx)} robust spike(s)",
                    min(3.0, len(spike_idx) / max(1, n) * 10),
                    indices=spike_idx[-12:],
                )
            )

    deltas = [values[i + 1] - values[i] for i in range(n - 1)]
    nonzero = [d for d in deltas if abs(d) > max(1e-9, 0.03 * (span or 1.0))]
    if len(nonzero) >= 5:
        sign_changes = sum(
            1 for a, b in zip(nonzero, nonzero[1:])
            if (a > 0 > b) or (a < 0 < b)
        )
        osc = sign_changes / max(1, len(nonzero) - 1)
        if osc >= 0.58 and sd > 1e-12:
            patterns.append(
                _figure_pattern(
                    "oscillation",
                    f"oscillates sign-change={osc:.0%}",
                    osc,
                    sign_change_rate=round(osc, 4),
                )
            )

    repeated = sum(1 for v in values if abs(v - values[-1]) <= max(1e-9, 0.02 * (span or 1.0)))
    if n >= 8 and repeated / n >= 0.7:
        patterns.append(
            _figure_pattern(
                "plateau",
                f"plateau near {values[-1]:+.3g}",
                repeated / n,
                fraction=round(repeated / n, 4),
            )
        )

    threshold = item.get("threshold")
    if threshold is not None:
        comparator = item.get("comparator") or ">="
        fired = [
            (v <= threshold) if comparator == "<=" else (v >= threshold)
            for v in values
        ]
        crossings = sum(1 for a, b in zip(fired, fired[1:]) if a != b)
        fire_rate = sum(1 for f in fired if f) / len(fired)
        if crossings or fire_rate in (0.0, 1.0):
            patterns.append(
                _figure_pattern(
                    "threshold",
                    f"threshold fire={fire_rate:.0%} cross={crossings}",
                    max(abs(fire_rate - 0.5), min(1.0, crossings / max(1, n - 1))),
                    threshold=round(threshold, 6),
                    comparator=comparator,
                    fire_rate=round(fire_rate, 4),
                    crossings=crossings,
                )
            )

    if key:
        xs, ys = _figure_turn_pairs(turn_log, key, "sense", limit=limit, lag=0)
        r = _figure_corr(xs, ys)
        if r is not None and len(xs) >= 6 and abs(r) >= 0.35:
            patterns.append(
                _figure_pattern(
                    "sense_correlation",
                    f"sense corr r={r:+.2f}",
                    abs(r),
                    n=len(xs),
                    r=round(r, 4),
                )
            )
        xs, ys = _figure_turn_pairs(turn_log, key, "sense", limit=limit, lag=1)
        lr = _figure_corr(xs, ys)
        if lr is not None and len(xs) >= 6 and abs(lr) >= 0.35:
            patterns.append(
                _figure_pattern(
                    "next_turn_sense",
                    f"next-sense r={lr:+.2f}",
                    abs(lr),
                    n=len(xs),
                    r=round(lr, 4),
                )
            )

        numeric = sorted(_figure_numeric_keys(turn_log))
        movers = []
        for other in numeric:
            if other in (key, "ts"):
                continue
            ox, oy = _figure_turn_pairs(turn_log, key, other, limit=limit, lag=0)
            cr = _figure_corr(ox, oy)
            if cr is not None and len(ox) >= 8 and abs(cr) >= 0.5:
                movers.append((abs(cr), cr, other, len(ox)))
        movers.sort(reverse=True)
        if movers:
            top = movers[:3]
            summary = ", ".join(f"{name}:{r:+.2f}" for _abs, r, name, _n in top)
            patterns.append(
                _figure_pattern(
                    "co_movers",
                    f"co-moves {summary}",
                    top[0][0],
                    streams=[{"name": name, "r": round(r, 4), "n": n2} for _abs, r, name, n2 in top],
                )
            )

        # Works for plain trigger keys AND the outcome:/lift: map keys, with
        # the probe_ fallback -- any credited stream is outcome-correlatable.
        base_key = str(key).split(":", 1)[1] if str(key).lower().startswith(("outcome:", "lift:")) else str(key)
        trig = tuner.triggers.get(base_key) or tuner.triggers.get(f"probe_{base_key}")
        if trig is not None and trig.outcomes:
            ox = [float(sig) for sig, _out in trig.outcomes]
            oy = [float(out) for _sig, out in trig.outcomes]
            orr = _figure_corr(ox, oy)
            if orr is not None and len(ox) >= 6 and abs(orr) >= 0.35:
                patterns.append(
                    _figure_pattern(
                        "credited_outcome",
                        f"outcome corr r={orr:+.2f}",
                        abs(orr),
                        n=len(ox),
                        r=round(orr, 4),
                    )
                )

    patterns.sort(key=lambda p: p.get("score", 0.0), reverse=True)
    if not patterns:
        patterns.append(
            _figure_pattern(
                "stable",
                f"stable mean={mu:+.3g} sd={sd:+.3g}",
                0.05,
                mean=round(mu, 6),
                stdev=round(sd, 6),
            )
        )
    return patterns[:8]


def _write_live_figure_patterns(path, series):
    sidecar = os.path.splitext(path)[0] + ".patterns.json"
    payload = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "figure": os.path.relpath(path, ROOT),
        "series": [
            {
                "key": item["key"],
                "n": len(item.get("values") or []),
                "last": item.get("values", [None])[-1],
                "threshold": item.get("threshold"),
                "patterns": item.get("patterns") or [],
            }
            for item in series
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return sidecar


def build_live_figure(raw_args, tuner, probes, turn_log):
    mode, name_tokens, limit, out_path = _parse_live_figure_args(raw_args)
    if mode == "list":
        names = _figure_available_names(tuner, probes, turn_log)
        groups = {
            "probes": [n for n in names if n.startswith("probe_")],
            "knobs": [n for n in names if n in tuner.triggers and tuner.triggers[n].kind == "coefficient"],
            "streams": [
                n for n in names
                if not n.startswith("probe_")
                and (n not in tuner.triggers or tuner.triggers[n].kind == "threshold")
            ],
        }
        lines = [
            "usage: :figure [list|patterns|recent|all|probes|knobs|streams|phen|outcomes|<name>|outcome:<name>|lift:<name> ...] [last N] [out path.svg]",
            "examples: :figure recent | :figure patterns probes | :figure sense memory_need probe_curiosity | :figure outcomes | :figure lift:conversation_productive",
        ]
        for label, vals in groups.items():
            preview = ", ".join(vals[:24]) if vals else "(none)"
            suffix = f" ... +{len(vals) - 24}" if len(vals) > 24 else ""
            lines.append(f"{label}: {preview}{suffix}")
        return True, lines, None

    keys = _figure_group_tokens(name_tokens, tuner, probes, turn_log)
    if not keys:
        return False, ["no figure targets found; try ':figure list'."], None

    series = []
    missing = []
    for key in keys:
        item = _figure_series_for_key(key, tuner, probes, turn_log, limit)
        if item is None:
            missing.append(key)
        else:
            series.append(item)
    if not series:
        hint = did_you_mean(missing[0], _figure_available_names(tuner, probes, turn_log)) if missing else ""
        return False, [f"no tracked values for {', '.join(missing) or 'that request'}.{hint} Try ':figure list'."], None
    if len(series) > 80:
        series = series[:80]
        missing.append("truncated_to_80_rows")
    if mode == "patterns":
        for item in series:
            item["patterns"] = _figure_series_patterns(item, turn_log, tuner, limit)
    path = _write_live_figure_svg(
        series,
        out_path=out_path,
        title="Live tracked patterns" if mode == "patterns" else "Live tracked values",
    )
    rel = os.path.relpath(path, ROOT)
    lines = [f"wrote {rel} ({len(series)} row(s), last {limit} point(s) each)"]
    if mode == "patterns":
        sidecar = _write_live_figure_patterns(path, series)
        lines.append(f"patterns saved {os.path.relpath(sidecar, ROOT)}")
        for item in series[:8]:
            top = item.get("patterns") or []
            if top:
                lines.append(f"{item['key']}: " + "; ".join(p["summary"] for p in top[:3]))
    if missing:
        lines.append("skipped: " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    return True, lines, path


# --- :benchmark -- score the live model, current tuning applied -------------
# Runs a named item set through the SAME generation path as a conversation
# turn: knobs synced into config plus the standing prioritize/exposed probe
# output steers. Turn-specific steers (memory lanes, claimmap tags) stay out.
# Because the dispatch loop swaps tuner/probes/config for a targeted agent,
# '@agent :benchmark ref' measures THAT spawned agent's tuning state, and
# ':expose :benchmark' hands the tool to the model like any other command.
BENCHMARK_DIR = os.path.join(ROOT, "invariants", "benchmarks")


def _benchmark_refs():
    try:
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(BENCHMARK_DIR)
            if f.lower().endswith((".json", ".jsonl", ".csv"))
        )
    except OSError:
        return []


def _resolve_benchmark_source(ref):
    for ext in (".jsonl", ".json", ".csv"):
        cand = os.path.join(BENCHMARK_DIR, ref + ext)
        if os.path.exists(cand):
            return cand
    return ref  # gsm8k / hf:<dataset> / explicit path; rows_from_source decides


def _parse_benchmark_request(bargs):
    """':benchmark <ref> [n N] [tokens T] [evaluator E] [contract|bare]' -> dict."""
    toks = bargs.split()
    req = {"ref": None, "n": 8, "tokens": 96, "evaluator": "number", "style": "bare"}
    i = 0
    while i < len(toks):
        t = toks[i].lower()
        if t == "n" and i + 1 < len(toks):
            req["n"] = max(1, int(toks[i + 1]))
            i += 2
        elif t in ("tokens", "budget") and i + 1 < len(toks):
            req["tokens"] = max(16, int(toks[i + 1]))
            i += 2
        elif t == "evaluator" and i + 1 < len(toks):
            req["evaluator"] = toks[i + 1].lower()
            i += 2
        elif t in ("contract", "bare"):
            req["style"] = t
            i += 1
        elif req["ref"] is None:
            req["ref"] = toks[i]
            i += 1
        else:
            raise ValueError(f"unrecognized argument '{toks[i]}'")
    if req["ref"] is None:
        raise ValueError("missing benchmark name")
    if req["evaluator"] not in ("number", "exact", "choice", "contains"):
        raise ValueError(f"unknown evaluator '{req['evaluator']}'")
    return req


def _run_shell_benchmark(model, config, tuner, probes, prioritize_pin, req, agent_name):
    """Score req['ref'] on the live model with the current tuning state.
    Returns (summary, out_path, printable summary lines); prints per-item rows."""
    import time as _bench_time
    from invariants.universal_benchmark import (
        build_benchmark_prompt,
        evaluate_response,
        examples_from_rows,
        format_choices,
        rows_from_source,
        summarize_results,
    )

    rows, source_label = rows_from_source(_resolve_benchmark_source(req["ref"]))
    examples = examples_from_rows(rows, n=req["n"])
    if not examples:
        raise ValueError(f"benchmark '{req['ref']}' has no usable items")

    _sync_steer_tunables(tuner, config)
    handles, steer_notes = [], []
    prio_alpha = tuner.get("prioritize_alpha", 0.0)
    if tuner.get("prioritize_gravity", 0.0) > 0:
        try:
            g_handles, g_desc = build_gravity_field_handles(model, probes, tuner)
            if g_handles:
                handles.extend(g_handles)
                steer_notes.append(f"gravity[{g_desc}]")
        except Exception:
            pass
    elif prio_alpha and prio_alpha > 0 and probes:
        from invariants.engine import _steer_handles as _b_steer
        pdir, plabel, psign = resolve_priority_direction(model, probes, tuner, prioritize_pin)
        if pdir:
            try:
                handles.extend(_b_steer(model, pdir, list(pdir.keys()), prio_alpha * psign))
                steer_notes.append(f"prioritize:{plabel} alpha={round(prio_alpha, 4)}")
            except Exception:
                pass
    exposed_alpha = tuner.get("exposed_probe_alpha", 0.0)
    if exposed_alpha and exposed_alpha > 0 and probes:
        exposed_names = sorted(
            n for n in probes
            if probes[n].get("exposed") and not probe_is_released(probes[n])
        )
        if exposed_names:
            from invariants.engine import _steer_handles as _b_steer_exposed
            edir = build_priority_mix_direction(model, exposed_names, probes, tuner)
            if edir:
                try:
                    handles.extend(_b_steer_exposed(model, edir, list(edir.keys()), exposed_alpha))
                    steer_notes.append(
                        f"exposed:{'+'.join(exposed_names)} alpha={round(exposed_alpha, 4)}"
                    )
                except Exception:
                    pass

    result_rows = []
    t_start = _bench_time.perf_counter()
    try:
        for idx, ex in enumerate(examples, 1):
            if req["style"] == "contract":
                item_prompt = build_benchmark_prompt(ex, req["evaluator"])
            else:
                # minimal native elicitation: the bare item as a chat turn
                item_prompt = ex.prompt + format_choices(ex.choices)
            t0 = _bench_time.perf_counter()
            res = generate_agentic_text(
                model,
                instruction=item_prompt,
                config=config,
                max_new_tokens=req["tokens"],
                chatty_log=False,
                pre_formatted=False,
            )
            text = res[0] if isinstance(res, tuple) else res
            dt = _bench_time.perf_counter() - t0
            parsed, ev = evaluate_response(ex, text or "", req["evaluator"])
            result_rows.append({
                "id": ex.id,
                "prompt": ex.prompt,
                "response": text,
                "parsed": parsed.to_dict(),
                "eval": ev.to_dict(),
                "seconds": round(dt, 2),
            })
            mark = "excluded" if ev.correct is None else ("Y" if ev.correct else "N")
            print(
                Fore.CYAN
                + f"[Benchmark] {idx}/{len(examples)} id={ex.id} correct={mark} "
                + f"pred={ev.pred} gold={ev.gold} ({dt:.1f}s)"
                + Style.RESET_ALL,
                flush=True,
            )
    finally:
        for h in handles:
            h.remove()

    summary = summarize_results(result_rows)
    summary["mean_seconds"] = round(
        sum(r["seconds"] for r in result_rows) / max(len(result_rows), 1), 2
    )
    summary["total_seconds"] = round(_bench_time.perf_counter() - t_start, 1)

    safe_ref = re.sub(r"[^A-Za-z0-9._-]+", "_", req["ref"])
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        ROOT, "invariants", "out", f"shell_benchmark_{safe_ref}_{stamp}.json"
    )
    payload = {
        "ref": req["ref"],
        "source": source_label,
        "agent": agent_name,
        "style": req["style"],
        "evaluator": req["evaluator"],
        "n_requested": req["n"],
        "max_new_tokens": req["tokens"],
        "steer_notes": steer_notes,
        "knobs": {
            k: tuner.get(k, None)
            for k in (
                "response_tokens", "prioritize_alpha",
                "exposed_probe_alpha", "steer_cap_fraction",
            )
        },
        "summary": summary,
        "rows": result_rows,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    acc = summary.get("accuracy")
    if acc is None:
        acc_txt = "n/a (nothing scored)"
    else:
        n_right = sum(
            bool(r["eval"]["correct"]) for r in result_rows
            if r["eval"]["correct"] is not None
        )
        acc_txt = f"{acc:.0%} ({n_right}/{summary['scored_n']})"
    lines = [
        f"{req['ref']} -> accuracy {acc_txt}, aligned {summary['aligned_rate']:.0%}, "
        f"{summary['mean_seconds']}s/item over {summary['n']} items"
        + (f" ({summary['unsafe_excluded_n']} unsafe excluded)" if summary.get("unsafe_excluded_n") else ""),
        f"agent={agent_name} style={req['style']} evaluator={req['evaluator']} "
        f"tokens={req['tokens']} steers[{'; '.join(steer_notes) if steer_notes else 'none'}]",
        f"saved {os.path.relpath(out_path, ROOT)}",
    ]
    return summary, out_path, lines


KNOWN_COMMANDS = (
    ":context", ":memory", ":methodmap", ":claimmap", ":steermap", ":steer", ":slice",
    ":figure", ":figures", ":benchmark",
    ":probe", ":label", ":calibrate", ":suggest", ":tune", ":doc", ":sandbox",
    ":experts", ":impact", ":clock", ":prioritize", ":release", ":listen",
    ":timestamps", ":history", ":queue", ":relations", ":clean", ":accept", ":reject", ":help", ":expose", ":hide",
)

# Bare command words that are BUILT IN -- a macro alias by one of these names is
# never invoked as ':<name>' (the built-in wins). Everything else that is a known
# macro alias runs directly as ':<alias> args'.
BUILTIN_COMMANDS = {c[1:] for c in KNOWN_COMMANDS} | {
    "macro", "run", "game", "solve", "refresh", "place", "install", "consider", "spawn",
    "exit", "quit", "timestamps", "listen", "history", "accept", "reject", "help", "expose", "hide", "show",
}

# Single source of truth for the command reference: the shell prints these lines
# verbatim at startup AND renders them to docs/COMMANDS.md, so the two never drift.
# A 10-space indent begins a command entry; a 16-space indent continues the prior
# entry's description. Edit here to change both the terminal help and the doc.
COMMAND_HELP_LINES = [
    "Commands: :help [model]           (this list + your solve-macros; ':help model' shows what",
    "                the MODEL can run on its own; ':help expose [off]' lets it call <<HELP>>)",
    "          :context, :context on, :context off, :context clear",
    "          :memory, :memory act|talk on|off|status|use [sentence [+probe -probe]]",
    "                :memory use/search [optional args] uses enabled lanes; empty reflects",
    "                via :prioritize/farthest-from-0; :expose off memory hides model access;",
    "                model-made definitions persist with linked accept/reject/label/place feedback",
    "          :methodmap <query>",
    "          :claimmap <first text> || <second text>",
    "          :consider <trigger_metric> <tool_name> <positive text> || <negative text>",
    "                (mint a custom activation-triggered steering tool from a contrastive pair;",
    "                set <tool_name>_alpha with :tune before it can steer)",
    "          :steermap",
    "          :slice [time A..B] [probe <name>] [scope <s>] [reason <r>] [layers <lo> <hi>] [steps A..B] [index A..B] [drop|export <path>]",
    "                (slice what the cognitive cache carries by any axis it records; entries",
    "                are timestamped at store (stored_at) or time-bounded via the synthesis-",
    "                trace join; listing always shows first; drop excises the slice AFTER",
    "                writing a backup and refuses to run without an axis; export saves the",
    "                slice to a .pt without touching the cache; bare :slice = summary)",
    "          :steer  (envelope + observed push distribution + data-implied cap/band)",
    "          :steer field|gravity init|on|off [default-G] | status  (init loads/migrates every",
    "                field surface and enables it without inventing laws or moving frozen bodies;",
    "                prioritize becomes a FIELD, not a pin: every",
    "                probe is a mass at its direction; each token is pulled along the sphere tangent",
    "                by mass/((1-cos)+eps)^2, so the push depends on where the state IS. Masses",
    "                default to signed evidence lift (negative = repulsor), normalized to |sum|=1;",
    "                prioritize_alpha is the default G but personal/family G can replace it; every",
    "                push stays inside the envelope cap)",
    "          :steer mass <probe> <m|auto>  (gravitational coefficient override; negative repels;",
    "                auto returns it to its live signed lift)",
    "          :steer g <probe|@anchor|layer:N|family:name> <source|off> [scale [offset]]",
    "                (personal G; source = number, probe:name, knob:name, status:ram, lift:trigger,",
    "                outcome:trigger (the credit channel as force -- outcomes steer), family:name,",
    "                or global; layer G gates/multiplies personal G when both are set)",
    "          :steer family <name> <G-source|add <node>|remove <node>|drop>",
    "                (reusable G/time/shape families instead of one global field; set family",
    "                clocks and shapes by targeting family:<name> with :steer time / :steer shape)",
    "          :steer time <node> <source|off> [scale [offset]]  (align any probe, anchor, layer,",
    "                or family clock to a probe, knob, RAM/VRAM status, outcome lift, another layer",
    "                clock, or constant)",
    "          :steer shape <node> point|gaussian <width>|shell <radius> <width>|plateau <radius> <edge>",
    "                (the impact can occupy a smooth region or hollow shell, not only a point;",
    "                several shaped members in a family compose an irregular holoid)",
    "          :steer quality <name> formula <expr-in-d> [strength <source> [scale [offset]]]",
    "                (a quality is not G: it has its own safe distance formula; d=1-cos. Example:",
    "                smelliness formula 1/(d+0.02)**4 spikes nearby; formulas omit spaces)",
    "          :steer set <law> quality <quality>  (define a law-described set)",
    "          :steer set <law> add <node> [compliance <source> [scale [offset]]] [time <v1,v2,...>]",
    "                (every member has its own live signed compliance and forward-time vector)",
    "          :steer set <law> exclude|include|remove <node> | drop",
    "                (set exclusion is exact zero, not a very small coefficient)",
    "          :steer exclude <node> [off]  (hard whole-field exclusion: the node participates in",
    "                no G, quality, family, or set law at all; off makes it eligible again)",
    "          :steer freeze <probe|@anchor> [off]  (inertial coefficient: the mass keeps pulling",
    "                but it stops moving itself -- a probe's baseline stops drifting; an anchor",
    "                stops responding to laws. It cannot habituate to its own gravity)",
    "          :steer anchor <name> [here|from <probe>|mass <m>|drop]  (precision bodies: pin an",
    "                exact latent point -- the last reply's state, or a copied probe direction --",
    "                as a mass in the field; frozen by default; bare ':steer anchor' lists)",
    "          :steer pole <probe|@anchor> <family[+|-]> [off]  (type a body with signed poles;",
    "                bodies interact ONLY where a law couples their families)",
    "          :steer law <famA> <famB> <k|off>  (selective magnetism between concept families:",
    "                k>0 like-poles repel / unlike attract, k<0 flips, unlisted pairs are inert;",
    "                ':tune gravity_law_step <s>' integrates the forces -- only unfrozen anchors",
    "                move, probes never do; ':tune gravity_doppler <k>' makes the field's pull",
    "                velocity-aware: how fast the state approaches or recedes, not just distance)",
    "          :steer bodies  (the whole physics at a glance: masses, personal Gs, qualities, sets,",
    "                exclusions, families, clocks, shapes, poles, laws, and knobs)",
    "          :figure [list|patterns|recent|all|probes|knobs|streams|phen|outcomes|<name>|outcome:<name>|lift:<name> ...] [last N] [out path.svg]",
    "                (draw live tracked shell values from the same table used by :tune,",
    "                :calibrate, :label, and :probe values. The outcome channel is",
    "                mappable: outcome:<name> draws a trigger's credited outcomes,",
    "                lift:<name> its rolling fired-vs-unfired lift (0-line = does",
    "                firing help), 'outcomes' maps every credited trigger. 'patterns' adds trend,",
    "                spike, regime-shift, oscillation, threshold, outcome, sense-correlation,",
    "                next-turn-sense, and co-movement detection plus a .patterns.json sidecar)",
    "          :figures [list|all|<which> ...]  (draw the state-output figures from the probe",
    "                JSONs via scripts/make_lesswrong_figures.py -> docs/figures/; loose names:",
    "                1/schematic 2/render-u 3/pre-control 4/cot 5/origin; no args = all.",
    "                Building runs in the first python that has matplotlib; list works anywhere)",
    "          :benchmark <ref> [n <N>] [tokens <T>] [evaluator number|exact|choice|contains] [contract]",
    "                (score the LIVE model with the current tuning applied on a named item set:",
    "                invariants/benchmarks/<ref>.jsonl, gsm8k, hf:<dataset>, or a file path.",
    "                Bare native elicitation by default; 'contract' asks the FINAL/STATE form.",
    "                '@agent :benchmark ref' measures that spawned agent; ':benchmark' alone",
    "                lists the named sets; results land in invariants/out/shell_benchmark_*.json)",
    "          :probe <name> <with it> || <without it>  (mint a named-concept sensor from",
    "                YOUR contrastive framings; scores every turn; :probe lists; :probe drop <name>)",
    "          :probe [list] | :probe drop <name>  (list probes or delete one)",
    "          :probe choose              (ask the model to draft one runnable :probe <name> A || B command)",
    "          :probe adopt <dim> [<dim> ...]  (turn stored vectors -- ambiguity, disagreement,",
    "                warranted_confidence, organic_correction, ... -- into reply-scoring probes)",
    "          :probe compose <name> <mix>  (mint a probe from a SIGNED MIX of dimensions and",
    "                probes: ambiguity + disagreement - validated_flow - 0.5*curiosity)",
    "                (mint/adopt/compose take a trailing 'band <lo> <hi>' -- explicit reading",
    "                layers, e.g. band 16 24; otherwise the live steer band decides depth)",
    "          :probe expose <name> [off]  (let the MODEL consult this sensor itself:",
    "                <<PROBE: name>> reads its last turn, <<PROBE: name || words>> scores",
    "                candidate words. Reading only -- minting/calibrating stay operator acts)",
    "          :probe release <name> [off]  (released probes remain visible, scored,",
    "                queryable, and saved, but are excluded from generation vectors,",
    "                automatic priority/ToT routing, and exposed-probe output steering)",
    "          :probe backfill <name> [n]  (retro-score up to n archived replies in order:",
    "                a name scores/rebuilds only that probe; 'all' explicitly scores every active",
    "                probe into joint rows; both rebuild credit and seed rolling history)",
    "          :probe values [name|all] [n]  (show the latest centered probe readings;",
    "                aliases: :probe recent, :probe last)",
    "          :probe show|hide talk|act|all  (toggle whether probe readings print during",
    "                main conversation, background actions, or both)",
    "          :probe define <name>        (share its initial breakdown -- the WITH/WITHOUT framings",
    "                it was minted from; name accepts choose/auto)",
    "          :probe explain <name> [alpha|choose] [prompt]  (default = activation ping sanity check:",
    "                briefly steer along the probe's direction, envelope-capped, baseline frozen, and",
    "                print what the nudge evokes -- it may be generic/off-target. Add 'prompt' for the",
    "                stored WITH/WITHOUT definition-grounded explanation; :probe define shows that",
    "                saved definition directly. name accepts choose/auto)",
    "          :probe match <probe> <knob> | auto | drive [mult] | validate | check | off",
    "                (tie a probe to its same-concept knob: drive = servo the knob from the",
    "                probe each turn; validate = correlate knob value vs reading (check to read",
    "                it); auto pairs every same-named probe+knob)",
    "          :calibrate <name> [pct|intent|<anchor>|<a>+<b>|band args]  (data-calibrate any knob",
    "                BY NAME; anchors join with '+' = fired only when EVERY stream fired;",
    "                the system evaluates the request and refuses unsafe ones --",
    "                circular strength knobs, binary streams, vacuous p100 caps)",
    "          :queue [calibrate <name> ...|:<command>]  (retry calibrations until enough evidence;",
    "                defer other known commands once, preserving a trailing because-purpose)",
    "          :relations [active|all|bindings|calibrations|queue|exposure|priority|fields|release|macros]",
    "                (read-only map of defined relationships: dynamic bindings, probe-match",
    "                links, calibrated knobs, queued retries, exposures, priority targets,",
    "                gravity/quality/time/shape links and hard exclusions, release decouplings,",
    "                and optionally macro aliases)",
    "          :clean [queue|bindings|priority|all] [apply]  (preview/apply a non-destructive",
    "                cleanup of queued retries, dynamic tuner bindings, probe-match drive/validate",
    "                modes, and priority targets; never deletes probes, histories, labels, docs,",
    "                memories, or learned evidence)",
    "          :label <probe|stream> pos|neg  (judge the MOST RECENT turn on that axis:",
    "                credits its last signal with a human outcome -- supervised evidence",
    "                alongside the automatic sense credit, so lift can reflect your judgment)",
    "          :suggest  (scan the accrued state for ready moves, then show a one-line",
    "                launcher of ALL visible user-facing commands; computed, never applied.",
    "                :suggest apply auto-queues only the safe measurement/calibration ones)",
    "          :suggest commands [filter]  (show the full one-line launcher for every",
    "                visible user-facing command/macro; optional filter narrows by command text)",
    "          :suggest gravity|field|physics  (shortcut launcher for the whole steer/field",
    "                command family: Gs, families, clocks, shapes, qualities, sets, exclusions)",
    "          :suggest :<command prefix>  (complete one operator command from the live",
    "                command reference; isolated from conversation history and cognitive cache)",
    "          :suggest because <reason>  (use a probe-specific move only when the internal match",
    "                clears both an absolute confidence floor and the runner-up; otherwise show",
    "                state-wide moves and never backfill an arbitrary weak winner)",
    "          :tune, :tune <name> <value>, :tune <name> auto [percentile]",
    "          :tune <knob|probe> dynamic <signed mix> [mult]  (each turn set the target",
    "                to mult * a signed mix of live streams, e.g. +ambiguity-consensus; a",
    "                probe-only target drives its own firing threshold; a name that is both",
    "                a knob and a probe steers the KNOB -- name probe_<x> for the threshold)",
    "          :tune steer_cap_fraction auto [pct]  (calibrate cap from observed pushes)",
    "          :tune steer_band auto [min_events] [gold|conversation|any] [synthesis|layersteer]",
    "                (derive band from outcomes; conversations count as evidence)",
    "          :tune steer_layer_sweep 1    (isolate steers by layer: each steer pushes",
    "                ONE least-tested band layer; per-layer outcomes accrue, transfer-free)",
    "          :doc <path> [because <why>]  (share a document into the conversation)",
    "          :doc <path> rewrite [new_name] [because <why>]  (rewrite a text file",
    "                better and save a sibling in the same folder; omitted name uses *_rewritten)",
    "          :doc next | :doc status      (stage the next chunk / show progress)",
    "          :doc read [n] [order|interleave|reply|updated] [satisfied|<stop_probe>]",
    "                [+probe -probe ...]  (signed probes rank up to 160 unread candidates:",
    "                toward + probes and away from - probes, using a bounded 600-char read;",
    "                scores are read-only and cached.",
    "                A bare probe name is instead an early-stop threshold)",
    "                (reading as dialogue: the document speaks each turn, the model",
    "                replies. order/interleave/updated advance on the documents' own",
    "                course -- updated = by file mtime, newest first; reply follows",
    "                overlap. 'satisfied' stops early once sense settles)",
    "          :doc inject                  (stage the whole library for one turn, budget-bounded)",
    "          :doc stop                    (interrupt auto-read)",
    "          :sandbox on|off|status       (run the model's ```python blocks for real)",
    "          :experts on|off|status       (mint new steering experts from its own",
    "                recurring self-corrections; roster bounded; default off)",
    "          :game <name>                 (run a python script from games/ with full access to",
    "                live system state; infinite flexibility for custom rules and interactions)",
    "          :game no | decline           (decline the game the model just proposed)",
    "          :accept [n|all] | :reject [n|all]  (a game may STAGE a command instead of",
    "                running it; nothing a game chose runs until you accept it here)",
    "          :expose [@agent] [stage|direct] :<command> [fixed args...] [steer <magnitude>] [because <activation>] | :expose off [@agent] <target>",
    "                (runtime tool access:",
    "                make/remove a command callable by the model as <<TOOL: :command args>>;",
    "                bare/default = staged for :accept, direct = queued immediately;",
    "                mode goes BEFORE the command so fixed args are never read as keywords",
    "                (the legacy ':expose :cmd direct args' order is still accepted);",
    "                fixed args are prefilled; model only supplies the remaining tail;",
    "                because records when the tool should activate; steer records its strength/effect; off works",
    "                for commands, probes, and knobs with the same order. Every exposure",
    "                registers its knob pair: cmd_<word> (metric: used-vs-unused outcome lift,",
    "                drawable/steerable as lift:cmd_<word>) and cmd_<word>_criteria (activation",
    "                bar; observes the probe named in the because-clause, so it calibrates);",
    "                every KNOWN command mints the same metric knob on first use)",
    "          :expose <probe|knob> [off]  (without leading ':', expose a probe sensor",
    "                or tuner knob to the model's <<PROBE: name>> tool)",
    "          :hide <command> | :hide off <command>  (documentation/writing visibility:",
    "                hide/reveal a command from model-facing help, suggestions, command",
    "                discovery, and macro/profile writing hints. Tool access is unchanged;",
    "                use :expose off :command to remove runtime access)",
    "          :impact                      (consequence trail: what its words caused,",
    "                and whether experienced impact tracks better deliberation)",
    "          :clock                       (last turn's generation time + tok/s and memory;",
    "                VRAM on GPU, process RAM on CPU-only, both sensed every turn",
    "                as generation_seconds / vram_gb streams)",
    "          :timestamps on|off|status     (toggle timestamps on the prompt and resumed history)",
    "          :history                      (legacy pointer: use :context for session transcript",
    "                controls and :memory for long-term memory)",
    "          :prioritize [choose|auto|<probe> [alpha]|pin <probe> [alpha]|mix <p...> [alpha]]",
    "                (rank probes by evidence-weighted lift; <probe> sets a landscape",
    "                target, pin duplicates direct :steer, mix moves a chosen field; alpha",
    "                maps to prioritize_alpha and is off at 0)",
    "          :release <tool> [prob]       (decouple a tool's firing from its signal for that",
    "                fraction of turns -- separates causality so credit lift can be trusted)",
    "          :listen on|off|status        (speak mid-reply: lines you type while it",
    "                generates are ingested at the next chunk seam and appended to the",
    "                live stream -- the model chooses to redirect or fold in; never dropped)",
    "          :macro <file> <c1> ; <c2> ...  (write a macro; :macro restore <json>",
    "                exactly restores one; :macro name <alias> <file> aliases it;",
    "                :macro name self [file] writes a macro that REGENERATES probes,",
    "                macros/commands, hidden/exposed command tools, and game configs;",
    "                in macro/profile files, ':' starts a command, '#' is a comment,",
    "                and any other nonblank line may run as a model query/text turn;",
    "                :macro strip <alias|file> drops display-only lines in place,",
    "                :macro name strip <src> [dest] writes a stripped copy)",
    "          :install <alias> <file> [goal]       (converts a manual text file into a",
    "                parameterized built-in command with proper solve/args headers)",
    "          :save self <name> | choose   (alias for :macro name self; 'choose' asks",
    "                the model to generate a name based on the current tuning state)",
    "          :spawn <name> join|replace|drop|create  (load/create an executable setup",
    "                profile for a target agent; any known command/macro can appear when",
    "                it has a purpose. 'join' adds to the panel; 'replace [N]' takes the",
    "                operator slot for N turns; 'drop' removes; 'create' writes only)",
    "                Use @<name> :cmd to target a specific agent's tuning state)",
    "          :run <alias|file>            (queue and execute a macro's commands)",
    "          :solve [dynamic] <name> <goal> [args]  (model writes a parameterized macro",
    "                for an ad-hoc command; it is PROPOSED, then :accept adopts it;",
    "                after that :<name> <args> runs it, filling $1..$9 / $@. Named args",
    "                fill $name; extract them automatically by writing $name in the goal",
    "                or list them at the end. 'dynamic' extracts $args from the 'because'",
    "                clause as well. All non-hidden commands are available; context",
    "                staged with :memory use is folded into the request. The macro-authoring",
    "                turn runs in a low-memory utility profile: ToT routing, synthesis,",
    "                cache deltas, field/priority steering, and tool seams pause for that turn)",
    "          :expect <name>               (intercept the model's next output and save it to",
    "                shell variable $<name> -- later commands substitute $<name> where it appears;",
    "                :expect macro <name> stages the reply's :commands as a macro; :expect file <path>",
    "                writes the reply to a file; :expect autocomplete suggests a completion.",
    "                macro/autocomplete expectations run as low-memory utility turns; '# args:'",
    "                in a generated macro declares named parameters, while '#' comments never execute)",
    "          :refresh prompts             (reload instructions from invariants/out/prompts)",
    "          :refresh commands            (reload commands from shell_commands.md)",
    "          :refresh macros              (reload available macro aliases)",
    "          :place <probe> <+|->         (nudge a probe's direction toward (+) or away from (-)",
    "                the hidden state of the model's LAST response; useful for manual calibration)",
    "          :exit | :quit [reason]        (close the shell, recording an optional reason)",
    "          :<macro-name> <args>         (run any aliased macro directly, args -> $1..$9)",
    "          <any :command> because <reason>   (logs why you issued it as provenance)",
    "          :memory use [sentence [+probe -probe]] | :memory choice probe <name>",
    "                (stage memories matching text and/or signed probe relations, farthest from 0)",
    "          :tune exposed_probe_alpha <small>  (also steer along the probes you have",
    "                exposed to the model, lift-weighted; 0 = off)",
]


def command_help_entries(lines=COMMAND_HELP_LINES):
    """Return help entries as raw line groups from COMMAND_HELP_LINES."""
    entries = []
    current = []
    for raw in lines:
        if raw.startswith("Commands:"):
            if current:
                entries.append(current)
            current = [raw]
            continue
        text = raw.strip()
        if not text:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent <= 12:
            if current:
                entries.append(current)
            current = [raw]
        elif current:
            current.append(raw)
    if current:
        entries.append(current)
    return entries


def find_command_help_entries(query, lines=COMMAND_HELP_LINES):
    q = (query or "").strip().split()[0].lstrip(":").lower()
    if not q:
        return []
    matches = []
    for entry in command_help_entries(lines):
        head = entry[0].strip()
        if head.startswith("Commands:"):
            head = head[len("Commands:"):].strip()
        words = {m.group(1).lower() for m in re.finditer(r":([a-z_][\w-]*)", head)}
        if q in words:
            matches.append(entry)
    return matches


def build_command_autocomplete_reference(prefix, hidden_commands=None):
    """Return the live, authoritative help entry for a command prefix.

    Autocomplete must be grounded in the same in-memory source that prints the
    shell help and regenerates docs/COMMANDS.md. Reading a prior conversation
    about that document is not equivalent: old interpretations can survive
    after the grammar changes.
    """
    visible_lines = visible_command_help_lines(hidden_commands)
    matches = find_command_help_entries(prefix, visible_lines)
    spec_matches = spec_help_entries(prefix, hidden_commands)
    if matches or spec_matches:
        parts = [line for entry in matches for line in entry]
        parts.extend(line for entry in spec_matches for line in entry)
        return "\n".join(parts)
    return "\n".join(visible_lines)


def visible_command_help_lines(hidden_commands=None, lines=COMMAND_HELP_LINES):
    """Filter the command reference for model-facing writing surfaces."""
    hidden_commands = set(hidden_commands or [])
    if not hidden_commands:
        return list(lines)
    out = []
    for entry in command_help_entries(lines):
        head = entry[0].strip()
        if head.startswith("Commands:"):
            head = head[len("Commands:"):].strip()
        words = {m.group(1).lower() for m in re.finditer(r":([a-z_][\w-]*)", head)}
        if words and words & hidden_commands:
            continue
        out.extend(entry)
    return out


def _split_command_help_entry(entry):
    """Return (one_line_command_signature, compact_summary) for help entries."""
    if not entry:
        return "", ""
    head = (entry[0] or "").strip()
    if head.startswith("Commands:"):
        head = head[len("Commands:"):].strip()
    # The help table uses two-or-more spaces before the parenthetical
    # description, while single spaces are part of the runnable signature.
    m = re.search(r"\s{2,}\(", head)
    if m:
        signature = head[:m.start()].strip()
        desc_head = head[m.start():].strip()
    else:
        signature = head.strip()
        desc_head = ""
    full = " ".join(str(line).strip() for line in entry if str(line).strip())
    if full.startswith("Commands:"):
        full = full[len("Commands:"):].strip()
    if full.startswith(signature):
        summary = full[len(signature):].strip()
    else:
        summary = desc_head
    summary = re.sub(r"\s+", " ", summary).strip()
    if summary.startswith("(") and summary.endswith(")"):
        summary = summary[1:-1].strip()
    if len(summary) > 180:
        summary = summary[:177].rstrip() + "..."
    return signature, summary


def suggest_command_catalog(hidden_commands=None, macro_aliases=None, solve_macros=None, filter_text=None):
    """All visible user-facing commands as one-line launcher entries.

    This is intentionally deterministic and grounded in COMMAND_HELP_LINES, not
    generated by the model. It is used by :suggest so that command discovery is
    broad enough to include every visible operator command while still obeying
    :hide.
    """
    terms = [t.lower() for t in re.findall(r"\S+", filter_text or "")]

    def wanted(command, summary):
        if not terms:
            return True
        blob = f"{command} {summary}".lower()
        if command.startswith(":steer") or command.startswith(":prioritize") or command.startswith(":relations"):
            # Gravity's user-facing family is deliberately richer than the
            # literal word "gravity": :steer g/family/time/shape/quality/set
            # are all field-physics commands and should appear when the
            # operator asks the launcher for gravity/field/physics.
            blob += (
                " gravity field physics g mass family time clock shape quality set "
                "exclude freeze anchor pole law bodies holoid personal"
            )
        return all(term in blob for term in terms)

    catalog = []
    seen = set()
    for entry in command_help_entries(visible_command_help_lines(hidden_commands)):
        command, summary = _split_command_help_entry(entry)
        if not command:
            continue
        key = ("command", command)
        if key in seen or not wanted(command, summary):
            continue
        seen.add(key)
        catalog.append({"kind": "command", "command": command, "summary": summary})

    macro_names = []
    if isinstance(macro_aliases, dict):
        macro_names = list(macro_aliases.keys())
    elif macro_aliases:
        macro_names = list(macro_aliases)
    hidden = _hidden_command_words(hidden_commands)
    macro_meta = {}
    for name, desc, args in (solve_macros or []):
        macro_meta[str(name).lstrip(":").lower()] = (desc or "", args or "")
    for raw_name in sorted(str(n).lstrip(":").lower() for n in macro_names):
        if not raw_name or raw_name in hidden or raw_name in BUILTIN_COMMANDS:
            continue
        desc, args = macro_meta.get(raw_name, ("", ""))
        arg_names = []
        for arg in re.split(r"[, ]+", args or ""):
            clean = arg.strip().strip("<>").lstrip("$")
            if clean:
                arg_names.append(f"<{clean}>")
        command = f":{raw_name}" + (f" {' '.join(arg_names)}" if arg_names else "")
        summary = desc or "installed macro"
        key = ("macro", command)
        if key in seen or not wanted(command, summary):
            continue
        seen.add(key)
        catalog.append({"kind": "macro", "command": command, "summary": summary})
    return catalog


def render_commands_md(lines=COMMAND_HELP_LINES):
    """Render the in-shell command help into a Markdown reference. A 10-space
    indent (or the leading 'Commands:' line) starts an entry; deeper indents
    continue the previous entry's description. Kept deliberately lossless: the
    exact wording the shell prints becomes the doc, so nothing drifts."""
    out = [
        "# Interactive shell commands",
        "",
        "_Auto-generated from `scripts/interactive_phenomenality.py` (its in-shell help "
        "block). Rewritten every time the shell starts -- edit `COMMAND_HELP_LINES` "
        "there, not this file._",
        "",
    ]
    entries = []  # each: [signature_line, [continuation_lines]]
    for raw in lines:
        if raw.startswith("Commands:"):
            entries.append([raw[len("Commands:"):].strip(), []])
            continue
        text = raw.strip()
        if not text:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent <= 12:
            entries.append([text, []])
        elif entries:
            entries[-1][1].append(text)
    for sig_line, cont in entries:
        m = re.search(r"\s{2,}\(", sig_line)
        if m:
            sig = sig_line[:m.start()].strip()
            desc_parts = [sig_line[m.start():].strip()] + cont
        else:
            sig, desc_parts = sig_line, list(cont)
        desc = " ".join(p for p in desc_parts if p).strip()
        if desc.startswith("(") and desc.endswith(")"):
            desc = desc[1:-1].strip()
        desc = desc.replace(") (", "; ")  # merge adjacent parenthetical groups
        out.append(f"- `{sig}` -- {desc}" if desc else f"- `{sig}`")
    out.append("")
    return "\n".join(out)


def render_visible_commands_md(hidden_commands=None, lines=COMMAND_HELP_LINES):
    return render_commands_md(visible_command_help_lines(hidden_commands, lines)) + render_specs_md(hidden_commands)


def write_commands_md(path=None, lines=COMMAND_HELP_LINES):
    """Write the live reference and keep the legacy root copy in sync."""
    canonical = os.path.join(ROOT, "docs", "COMMANDS.md")
    targets = [path] if path is not None else [canonical, os.path.join(ROOT, "COMMANDS.md")]
    content = render_commands_md(lines) + render_specs_md()
    for target in targets:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
    return path or canonical


def list_solve_macros(macros_dir=None):
    """Scan the solve-macro directory and return [(name, desc, args)] for each
    saved macro (invariants/out/macros/*.txt). desc/args come from the leading
    '# :solve macro ...' / '# args:' comments the solve writer leaves."""
    if macros_dir is None:
        macros_dir = os.path.join(ROOT, "invariants", "out", "macros")
    out = []
    if not os.path.isdir(macros_dir):
        return out
    for fn in sorted(os.listdir(macros_dir)):
        if not fn.endswith(".txt"):
            continue
        name, desc, args = fn[:-4], "", ""
        try:
            with open(os.path.join(macros_dir, fn), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("# :solve macro") and "--" in line:
                        desc = line.split("--", 1)[1].strip()
                    elif line.startswith("# args:"):
                        args = line.split(":", 1)[1].strip()
                    elif line.startswith(":"):
                        break
        except Exception:
            pass
        out.append((name, desc, args))
    return out


def build_model_help_text(
    solve_macros=None,
    exposed_commands=None,
    exposed_knobs=None,
    hidden_commands=None,
    memory_tool_exposed=True,
    memory_lanes=None,
):
    """Model-facing help: what the model may run ITSELF (its <<...>> tools) vs.
    the operator's ':' commands, which it can only reach by proposing a game."""
    if solve_macros is None:
        solve_macros = list_solve_macros()
    exposed_commands = {
        str(name).lstrip(":").lower(): normalize_command_exposure(record)
        for name, record in (exposed_commands or {}).items()
    }
    exposed_knobs = set(exposed_knobs or [])
    hidden_commands = set(hidden_commands or [])
    visible_exposed = {
        name: mode for name, mode in exposed_commands.items()
        if name not in hidden_commands
    }
    visible_operator_commands = sorted(c for c in BUILTIN_COMMANDS if c not in hidden_commands)
    visible_solve_macros = [
        (n, d, a) for n, d, a in solve_macros
        if n not in hidden_commands
    ]
    lines = [
        HELP_TOOL_HEADER,
        "You can reach these YOURSELF -- emit the tag mid-reply when its activation criterion is met; read-only tools return directly, exposed commands follow their mode:",
        "  <<METHODMAP: query>>       retrieve sanitized methodology maps",
        "  <<DOC: query>>             read from documents shared into this session",
        "  <<CLAIMMAP: A || B>>       weigh two framings against each other",
        "  <<PROBE: name>>            read one exposed probe sensor or knob",
        "  <<PROBE: name || words>>   score candidate words on an exposed probe sensor",
        "  <<HELP>>                   show this help",
    ]
    if memory_tool_exposed and memory_lanes:
        lane_text = ", ".join(memory_lanes)
        lines.insert(2, f"  <<MEMORY: query>>          search enabled memory lanes ({lane_text}); query may be empty")
    else:
        lines.insert(2, "  <<MEMORY: ...>>            unavailable until memory lanes are on and exposed")
    if exposed_knobs:
        lines.append("                             Exposed knobs: " + ", ".join(sorted(exposed_knobs)) + ".")
    if visible_exposed:
        lines.extend([
            "  <<TOOL: :command args>>   trigger an operator-exposed command-backed tool",
            "                             Routed tools may be shown as <<TOOL: @agent :command args>>.",
            "                             Legacy <<CMD: :command args>> is also accepted.",
            "                             If the tool shows fixed args, provide only the remaining tail.",
            "                             Documented here:",
        ])
        for name, record in sorted(visible_exposed.items()):
            display_cmd = command_exposure_display(name, record)
            lines.append(
                f"                             - {display_cmd} ({command_exposure_mode(record)}); "
                f"activate when {command_exposure_activation(name, record)}; "
                f"steer {command_exposure_steer(record)}."
            )
        lines.extend([
            "                             Semicolon chains run only exposed commands; :macro keeps semicolons as its body.",
        ])
    else:
        expose_hint = (
            "no command-backed tools are documented here"
            if exposed_commands
            else "unavailable until the operator exposes a command with :expose"
        )
        lines.append("  <<TOOL: ...>>             " + expose_hint)
    lines.extend([
        "",
        "Games are the ONE place you reach commands. You may:",
        "  <<GAME_PROPOSE: name>>     propose a game -- the operator accepts or declines it",
        "  <<GAME_ACCEPT: name>>      accept a game the operator proposed",
        "  <<GAME_DECLINE: name>>     decline a game -- the ONLY thing you can refuse yourself",
        "  <<GAME_END>>               end the active game",
        "  <<GAME_EXPOSE: a,b>> / <<GAME_HIDE: a,b>>   apply probe state inside a game",
        "",
        "The ':' commands are the OPERATOR's. Macros, solves, and games may use/stage any",
        "non-hidden command, but staged game/command-tool actions do not run until the operator",
        "types :accept. Operator commands available here: "
        + ", ".join(visible_operator_commands) + ".",
    ])
    if visible_solve_macros:
        lines.append("Operator solve-macros (also operator-run): "
                     + ", ".join(f":{n}" for n, _d, _a in visible_solve_macros) + ".")
    return "\n".join(lines)


def calibratable_names(tuner):
    """Every name :calibrate accepts: triggers, bare probe names, aliases."""
    cands = set(tuner.triggers)
    cands |= {n[len("probe_"):] for n in tuner.triggers if n.startswith("probe_")}
    cands |= set(STREAM_ALIASES)
    return cands


class InputListener:
    """One reader thread that queues every stdin line, so operator input is
    never dropped -- typed between turns OR mid-generation. Opt-in: until
    start() the shell reads with plain input() (unchanged default path). Once
    active, get() serves each turn from the queue and drain() pulls anything
    typed during a generation, for the interrupt seam."""

    def __init__(self):
        import queue
        self._q = queue.Queue()
        self._eof = object()
        self.active = False

    def start(self):
        import threading
        import sys as _sys
        if self.active:
            return
        def _reader():
            try:
                for line in _sys.stdin:
                    self._q.put(line.rstrip("\r\n"))
            except Exception:
                pass
            self._q.put(self._eof)
        threading.Thread(target=_reader, daemon=True).start()
        self.active = True

    def get(self, prompt=""):
        """Block for the next line (EOFError on stdin close, matching input())."""
        if prompt:
            print(prompt, end="", flush=True)
        item = self._q.get()
        if item is self._eof:
            raise EOFError
        return item

    def drain(self):
        """Every line queued right now, non-blocking. EOF is left in place so a
        later get() still raises it."""
        import queue
        out = []
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is self._eof:
                self._q.put(self._eof)
                break
            out.append(item)
        return out


def load_stored_direction(model, stem, layers=None):
    """Resolve a stored dimension into a per-layer unit direction dict.
    Looks for invariants/<stem>_vector.pt, then invariants/<stem>.pt, then
    the saved-probe dir. Layer dicts are band-restricted and normalized per
    layer; a saved probe payload ({direction, framings}) reuses its
    direction; a bare tensor broadcasts one direction across the band.
    `layers` (explicit indices) overrides the live band -- reading depth
    need not follow steering depth. Always returns a 4-tuple
    (direction, src_name, exposed, dim_mismatch); ({}, None, False, None)
    when nothing usable exists."""
    want = set(int(x) for x in layers) if layers is not None else None
    src_path = next(
        (p for p in (
            os.path.join(ROOT, "invariants", f"{stem}_d{model.d_model}_vector.pt"),
            os.path.join(ROOT, "invariants", f"{stem}_d{model.d_model}.pt"),
            os.path.join(ROOT, "invariants", "out", "probes", f"{stem}_d{model.d_model}.pt"),
            os.path.join(ROOT, "invariants", f"{stem}_vector.pt"),
            os.path.join(ROOT, "invariants", f"{stem}.pt"),
            os.path.join(ROOT, "invariants", "out", "probes", f"{stem}.pt"),
        ) if os.path.isfile(p)),
        None,
    )
    if src_path is None:
        return {}, None, False, None
    try:
        payload = torch.load(src_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(src_path, map_location="cpu")
    from invariants.engine import steer_band_layers
    direction = {}
    exposed = False
    dim_mismatch = None
    if isinstance(payload, dict) and "direction" in payload:
        exposed = bool(payload.get("exposed", False))
        for L, v in payload["direction"].items():
            if want is not None and int(L) not in want:
                continue
            v = v.to(model.device).float().reshape(-1)
            if v.shape[0] != model.d_model:
                dim_mismatch = v.shape[0]
                continue
            n = v.norm()
            if n.item() > 0:
                direction[int(L)] = v / n
    elif isinstance(payload, dict):
        raw_vecs = {int(k): v for k, v in payload.items() if hasattr(v, "reshape")}
        if raw_vecs:
            if want is not None:
                keep = sorted(want & set(raw_vecs))
            else:
                band = set(steer_band_layers(max(raw_vecs) + 1))
                keep = sorted(band & set(raw_vecs)) or sorted(raw_vecs)
            for L in keep:
                v = raw_vecs[L].to(model.device).float().reshape(-1)
                if v.shape[0] != model.d_model:
                    dim_mismatch = v.shape[0]
                    continue
                n = v.norm()
                if n.item() > 0:
                    direction[int(L)] = v / n
    elif hasattr(payload, "reshape"):
        v = payload.to(model.device).float().reshape(-1)
        if v.shape[0] == model.d_model:
            n = v.norm()
            if n.item() > 0:
                unit = v / n
                n_layers = getattr(model.model.config, "num_hidden_layers", getattr(model.model.config, "n_layers", getattr(model, "n_layers", 32)))
                n_layers = int(n_layers)
                targets = sorted(want) if want is not None else steer_band_layers(n_layers)
                for L in targets:
                    if 0 <= int(L) < n_layers:
                        direction[int(L)] = unit.clone()
        else:
            dim_mismatch = v.shape[0]
    return direction, os.path.basename(src_path), exposed, dim_mismatch


def parse_compose_expr(expr):
    """Parse a signed weighted mix of named dimensions into (terms, error):
    terms = [(weight, name), ...]. Grammar: terms joined by + or -, each
    optionally weighted 'w*name' (e.g. 'ambiguity + disagreement -
    validated_flow - 0.5*curiosity'). No parentheses -- a mix is a sum."""
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


OPT_IN_CAPABILITY = {"claimmap_alpha", "memory_alpha", "expert_proof_weight", "steer_layer_sweep"}
# Categories :suggest apply may auto-run: pure measurements/calibrations that
# only move a bar. explore/expose CHANGE behavior or surface state to the model,
# so they stay a deliberate hand.
SUGGEST_APPLY_SAFE = {"calibrate", "commit", "backfill"}


def confident_probe_match(probe_scores, min_similarity=0.25, min_margin=0.03):
    """Select a probe only when the reason match clears a floor and margin."""
    ranked = sorted(probe_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, ranked
    best_name, best_score = ranked[0]
    margin = best_score - ranked[1][1] if len(ranked) > 1 else best_score
    if best_score < min_similarity or margin < min_margin:
        return None, ranked
    return (best_name, best_score), ranked


def backfill_scoring_names(probes, names_to_rebuild, request_all=False):
    """A named backfill is narrow; only the explicit ``all`` form is joint."""
    return list(probes) if request_all else list(names_to_rebuild)


def _explore_value(name, current):
    base = {"claimmap_alpha": 0.02, "memory_alpha": 0.02,
            "expert_proof_weight": 1.0, "steer_layer_sweep": 1}.get(name, 0.02)
    # If already sitting at the natural first value, propose a second one so a
    # verdict can be earned (outcome calibration needs >=2 distinct values).
    if abs(float(current) - base) < 1e-9:
        return round(base * 2, 4)
    return base


def suggest_field_actions(tuner, probes):
    """Manual, one-line suggestions for the gravity/field system.

    These are not safe for :suggest apply: they change the steering geometry or
    expose field structure. They belong in ordinary :suggest so gravity is not
    invisible just because it is not a calibration/backfill move.
    """
    probes = probes or {}
    bodies = steer_bodies()
    fields = bodies.get("fields") or {}
    families = bodies.get("g_families") or {}
    qualities = bodies.get("qualities") or {}
    sets = bodies.get("sets") or {}
    anchors = bodies.get("anchors") or {}
    laws = bodies.get("laws") or {}
    out = []
    g_on = bool(tuner and tuner.get("prioritize_gravity", 0.0) > 0)
    g_now = float(tuner.get("prioritize_alpha", 0.0)) if tuner else 0.0
    configured_nodes = [
        node for node, cfg in sorted(fields.items())
        if cfg.get("g") is not None or cfg.get("family") or cfg.get("time") is not None
        or cfg.get("shape") is not None or cfg.get("excluded")
    ]
    probe_names = sorted(probes)

    if not g_on:
        out.append((
            "field",
            "gravity field is available but off -- initialize the full field schema and enable it",
            ":steer field init",
        ))
    else:
        masses = gravity_field_masses(probes, tuner)
        out.append((
            "field",
            f"gravity field is on (global G {g_now:g}, {len(masses)} live mass/body entry(s)) -- inspect the whole field",
            ":steer bodies",
        ))
        if not masses and probe_names:
            out.append((
                "field",
                f"gravity is on but no live masses resolved yet -- give '{probe_names[0]}' an explicit mass",
                f":steer mass {probe_names[0]} 1",
            ))

    if probe_names and not any(fields.get(f"probe:{p}", {}).get("g") is not None for p in probe_names):
        out.append((
            "field",
            f"no probe has a personal G override -- bind one body to the global field strength",
            f":steer g {probe_names[0]} global",
        ))
    if not families:
        out.append((
            "field",
            "no G families defined -- make a reusable field family before adding members",
            ":steer family field_core global",
        ))
    if families and probe_names and not any(fields.get(f"probe:{p}", {}).get("family") for p in probe_names):
        fname = sorted(families)[0]
        out.append((
            "field",
            f"family:{fname} has no probe member yet -- attach one node to inherit its G/time/shape",
            f":steer family {fname} add {probe_names[0]}",
        ))
    if not qualities:
        out.append((
            "field",
            "no shaped quality laws defined -- create a distance formula such as nearby smelliness",
            ":steer quality smelliness formula 1/(d+0.02)**4",
        ))
    elif not sets and probe_names:
        qname = sorted(qualities)[0]
        out.append((
            "field",
            f"quality:{qname} exists but no set law uses it -- define a law-described set",
            f":steer set {qname}_law quality {qname}",
        ))
    if not configured_nodes and not anchors and not laws:
        out.append((
            "field",
            "field schema is empty -- list the one-line gravity/field command family",
            ":suggest commands gravity",
        ))
    return out[:6]


def suggest_actions(tuner, rows, probes=None, archive_size=0, max_paired=6):
    """Scan the accrued state for READY next moves -- not only calibrations.
    Returns [(category, evidence_line, command), ...] where category is one of
    'calibrate' | 'commit' | 'explore' | 'backfill' | 'expose' | 'field'. Deterministic;
    acting on any of them stays an operator decision."""
    probes = probes or {}
    out = []

    # FIELD / GRAVITY -- behavior-changing, manual-only suggestions that make
    # the flexible field discoverable alongside the older calibration moves.
    out.extend(suggest_field_actions(tuner, probes))

    # CALIBRATE -- a threshold mis-set on its own distribution (never/always fires).
    for name in sorted(tuner.triggers):
        t = tuner.triggers[name]
        if t.kind != "threshold" or len(t.signals) < 10 or not t.observed:
            continue
        fr = t.fired / t.observed
        if fr <= 0.05 or fr >= 0.95:
            p50 = sorted(t.signals)[len(t.signals) // 2]
            out.append((
                "calibrate",
                f"{name}: fires {fr:.0%} of {t.observed} observed (bar {round(t.value, 4)}, own median {round(p50, 4)})",
                f":calibrate {name} 50",
            ))

    # COMMIT -- an explored circular knob whose best value beats current (the
    # 'you did the exploration but never locked it in' case).
    # EXPLORE -- an opt-in capability with fewer than 2 tried values: you can't
    # earn a verdict until you've tried a second.
    for name in OUTCOME_CALIBRATABLE:
        t = tuner.triggers.get(name)
        if t is None:
            continue
        tried = {round(float(s), 6) for s, _ in t.outcomes}
        v = outcome_calibration(list(t.outcomes))
        if v is not None and abs(v - t.value) > 1e-9:
            out.append((
                "commit",
                f"{name}: best explored value {round(v, 4)} beats current {round(t.value, 4)} on {len(t.outcomes)} outcomes",
                f":calibrate {name} outcome",
            ))
        elif name in OPT_IN_CAPABILITY and len(tried) < 2:
            n = len(tried)
            out.append((
                "explore",
                f"{name}: only {n} value(s) ever tried -- explore a second so its effect can be earned",
                f":tune {name} {_explore_value(name, t.value)}",
            ))

    # BACKFILL -- an active probe with little evidence while the archive is deep.
    if archive_size >= 40:
        for pname in sorted(probes):
            trig = tuner.triggers.get(f"probe_{pname}")
            n_pairs = len(trig.outcomes) if trig else 0
            if n_pairs < 15:
                out.append((
                    "backfill",
                    f"{pname}: {n_pairs} paired turns vs {archive_size} archived replies -- pre-load evidence",
                    f":probe backfill {pname}",
                ))

    # EXPOSE -- a probe with a strong, evidenced sense-lift you haven't let the
    # model read yet.
    for pname in sorted(probes):
        if probes[pname].get("exposed"):
            continue
        trig = tuner.triggers.get(f"probe_{pname}")
        if trig is None:
            continue
        st = trig.outcome_stats()
        lift = st.get("lift")
        if lift is not None and abs(lift) >= 0.05 and st.get("n_credited", 0) >= 20:
            out.append((
                "expose",
                f"{pname}: sense-lift {lift} over {st['n_credited']} turns -- a real signal you could let it read",
                f":probe expose {pname}",
            ))

    # CALIBRATE (paired) -- every target<-anchor cut the joint table supports,
    # ranked by median separation. Command uses exact names so it round-trips.
    numeric = sorted({
        k for r in rows for k, val in r.items()
        if isinstance(val, (int, float)) and k != "ts"
    })
    paired = []
    for tstream in numeric:
        t_name = "conversation_productive" if tstream == "sense" else tstream
        t_trig = tuner.triggers.get(t_name)
        if t_trig is None or t_trig.kind != "threshold":
            continue
        for astream in numeric:
            if astream == tstream:
                continue
            a_trig = anchor_trigger_for(astream, tuner)
            if a_trig is None or a_trig.kind != "threshold":
                continue
            cut, n_f, n_uf, gap = paired_threshold_multi(
                rows, tstream, [(astream, a_trig.value, a_trig.comparator)], details=True,
            )
            if cut is None or not gap or abs(cut - t_trig.value) <= 1e-6:
                continue
            t_disp = t_name[6:] if t_name.startswith("probe_") else t_name
            a_disp = astream[6:] if astream.startswith("probe_") else astream
            paired.append((
                gap,
                f"{t_disp} <- {a_disp}: cut {round(cut, 4)} (now {round(t_trig.value, 4)}; {n_f} fired / {n_uf} unfired, gap {round(gap, 4)})",
                f":calibrate {t_name} {astream}",
            ))
    paired.sort(key=lambda x: -x[0])
    out.extend(("calibrate", line, cmd) for _, line, cmd in paired[:max_paired])
    return out


def strip_band_suffix(text):
    """Pop a trailing 'band <lo> <hi>' (inclusive layer indices) off a probe
    argument string: the explicit reading depth for adopt/compose/mint.
    Returns (text_without_suffix, layers or None)."""
    m = re.search(r"\s+band\s+(\d+)\s+(\d+)\s*$", text or "", re.IGNORECASE)
    if not m:
        return (text or "").strip(), None
    lo, hi = int(m.group(1)), int(m.group(2))
    return text[:m.start()].strip(), list(range(min(lo, hi), max(lo, hi) + 1))


def intent_relative_threshold(pairs, min_n=5):
    """Set the productive bar RELATIVE TO INTENT-SHAPING, not at an arbitrary
    quantile: given (settling, sense) pairs -- settling > 0 means the turn
    LOWERED ambiguity+disagreement versus the previous turn, i.e. it shaped
    intent -- return the sense midpoint between the median of intent-shaping
    turns and the median of non-shaping turns. That cut labels 'productive'
    as 'coheres like the turns that actually settled an intent'. None until
    both groups have min_n evidence. Deterministic: same pairs, same bar."""
    settling = sorted(o for s, o in pairs if s > 0)
    non = sorted(o for s, o in pairs if s <= 0)
    if len(settling) < min_n or len(non) < min_n:
        return None
    return (settling[len(settling) // 2] + non[len(non) // 2]) / 2.0


def _sync_steer_tunables(tuner, config):
    """Push the live tuner values into the engine's steering envelope and the
    agentic config, once per turn before any generation. Every steering surface
    is tunable (:tune steer_cap_fraction / steer_band_lo / steer_band_hi /
    steer_fraction) but a bad value never lands: the engine setters reject
    non-finite input and the last good value stays in force."""
    from invariants import engine as _engine
    try:
        _engine.set_steer_cap_fraction(
            tuner.get("steer_cap_fraction", _engine.get_steer_cap_fraction())
        )
    except (TypeError, ValueError) as exc:
        print(Fore.YELLOW + f"[Steer] steer_cap_fraction ignored: {exc}" + Style.RESET_ALL)
    lo_now, hi_now = _engine.get_steer_band()
    try:
        _engine.set_steer_band(
            tuner.get("steer_band_lo", lo_now),
            tuner.get("steer_band_hi", hi_now),
        )
    except (TypeError, ValueError) as exc:
        print(Fore.YELLOW + f"[Steer] steer band ignored: {exc}" + Style.RESET_ALL)
    from invariants.agentic_engine import _sane_fraction
    config.steer_fraction = _sane_fraction(
        tuner.get("steer_fraction", config.steer_fraction), config.steer_fraction
    )
    # Expert-consultation budgets follow the tuner too (ints >= 0; bad values
    # fall back to the current config rather than landing).
    config.max_routing_events = max(
        0, int(_sane_fraction(tuner.get("routing_events", config.max_routing_events), config.max_routing_events))
    )
    config.max_loops = max(
        0, int(_sane_fraction(tuner.get("routing_loops", config.max_loops), config.max_loops))
    )
    config.entropy_threshold = _sane_fraction(
        tuner.get("routing_entropy", config.entropy_threshold), config.entropy_threshold
    )
    config.routing_entropy_weight = _sane_fraction(
        tuner.get("routing_entropy_weight", getattr(config, "routing_entropy_weight", 1.0)),
        getattr(config, "routing_entropy_weight", 1.0),
    )
    config.routing_collapse_margin = _sane_fraction(
        tuner.get("routing_collapse_margin", getattr(config, "routing_collapse_margin", 0.75)),
        getattr(config, "routing_collapse_margin", 0.75),
    )
    config.routing_collapse_penalty = _sane_fraction(
        tuner.get("routing_collapse_penalty", getattr(config, "routing_collapse_penalty", 1.25)),
        getattr(config, "routing_collapse_penalty", 1.25),
    )
    config.routing_probe_weight = _sane_fraction(
        tuner.get("routing_probe_weight", getattr(config, "routing_probe_weight", 1.0)),
        getattr(config, "routing_probe_weight", 1.0),
    )
    config.max_synthesis_events = max(
        0, int(_sane_fraction(tuner.get("synthesis_events", config.max_synthesis_events), config.max_synthesis_events))
    )
    # The roster cap for emergent mesa-objectives is live-tunable.
    if not hasattr(config, "max_committee_size"):
        config.max_committee_size = 6
    config.max_committee_size = max(
        0, int(_sane_fraction(tuner.get("max_committee_size", config.max_committee_size), config.max_committee_size))
    )
    config.max_synthesis_steps = max(
        0, int(_sane_fraction(tuner.get("synthesis_steps", config.max_synthesis_steps), config.max_synthesis_steps))
    )
    config.epsilon = _sane_fraction(tuner.get("plateau_epsilon", config.epsilon), config.epsilon)
    
    config.max_rounds = max(1, int(_sane_fraction(tuner.get("max_rounds", getattr(config, "max_rounds", 5)), getattr(config, "max_rounds", 5))))
    config.required_agreement = max(1, int(_sane_fraction(tuner.get("required_agreement", getattr(config, "required_agreement", 3)), getattr(config, "required_agreement", 3))))
    config.max_tool_calls = max(0, int(_sane_fraction(tuner.get("max_tool_calls", getattr(config, "max_tool_calls", 8)), getattr(config, "max_tool_calls", 8))))
    config.max_new_tokens = max(1, int(_sane_fraction(tuner.get("max_new_tokens", getattr(config, "max_new_tokens", 220)), getattr(config, "max_new_tokens", 220))))
    config.repair_token_multiplier = _sane_fraction(tuner.get("repair_token_multiplier", getattr(config, "repair_token_multiplier", 2.0)), getattr(config, "repair_token_multiplier", 2.0))
    if tuner.get("max_elapsed_sec", -1.0) > 0:
        config.max_elapsed_sec = float(tuner.get("max_elapsed_sec", getattr(config, "max_elapsed_sec", 60.0)))
    config.oracle_max_elapsed_sec = float(_sane_fraction(tuner.get("oracle_max_elapsed_sec", getattr(config, "oracle_max_elapsed_sec", 60.0)), getattr(config, "oracle_max_elapsed_sec", 60.0)))
    config.verifier_time_reserve_sec = float(_sane_fraction(tuner.get("verifier_time_reserve_sec", getattr(config, "verifier_time_reserve_sec", 20.0)), getattr(config, "verifier_time_reserve_sec", 20.0)))
    config.relax_agreement_under_urgency = bool(tuner.get("relax_agreement_under_urgency", float(getattr(config, "relax_agreement_under_urgency", False))) > 0.0)
    config.stop_on_critical_urgency = bool(tuner.get("stop_on_critical_urgency", float(getattr(config, "stop_on_critical_urgency", True))) > 0.0)
    # The synthesis budget is PER REPLY, but its counter lives on the shared
    # config object -- the benchmark resets it per attempt; the shell must
    # reset it per turn or synthesis (and the cache-delta retrieval inside
    # the same gate) fires once per SESSION and silently never again.
    config._synthesis_events_used = 0


def get_hardware_appropriate_model():
    if not torch.cuda.is_available():
        return "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return DEFAULT_MODEL
    
    if vram >= 14.5:
        return DEFAULT_MODEL
    elif vram >= 8.0:
        return "Qwen/Qwen2.5-3B-Instruct"
    else:
        return "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    global _last_completion, _pending_expect, _shell_vars
    os.chdir(ROOT)
    print(Fore.CYAN + Style.BRIGHT + "================================================")
    print("      HUMBLE SYNTHESIS - INTERACTIVE SHELL      ")
    print("================================================" + Style.RESET_ALL)
    
    # Model is configurable so the egg shell honors whatever model earned the egg
    # (env EGG_MODEL set by the benchmark, or argv[1] for a manual launch).
    model_name = os.environ.get("EGG_MODEL")
    if not model_name:
        if len(sys.argv) > 1:
            model_name = sys.argv[1]
        else:
            model_name = get_hardware_appropriate_model()
            
    is_default = (model_name == DEFAULT_MODEL)

    print(Fore.YELLOW + f"[System] Loading {model_name}..." + Style.RESET_ALL)
    model = load_model(model_name, local_files_only=is_default)

    config = AgenticConfig()
    organic_path = os.path.join(ROOT, "invariants", f"organic_correction_vector_d{model.d_model}.pt")
    if not os.path.isfile(organic_path):
        organic_path = ORGANIC_VECTOR_PATH

    try:
        payload = torch.load(organic_path, map_location=model.device)
        if isinstance(payload, torch.Tensor) and payload.reshape(-1).shape[0] != model.d_model:
            print(Fore.YELLOW + f"[System] Skipping organic vector: dimension mismatch (model d_model={model.d_model})." + Style.RESET_ALL)
        else:
            config.organic_correction_vector = payload
            print(Fore.GREEN + f"[System] Successfully loaded {os.path.basename(organic_path)}!" + Style.RESET_ALL)
    except Exception as e:
        if is_default:
            print(Fore.RED + f"[System] Warning: Could not load organic vector: {e}" + Style.RESET_ALL)
        else:
            pass # Non-default models might simply not have an organic vector minted yet

    cache_file = CACHE_FILE
    try:
        cache_file = _global_cache.use_file(model_cache_file(model_name, model.d_model))
        print(Fore.GREEN + f"[System] Loaded cache {cache_file.name} ({len(_global_cache.memory)} memories)." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[System] Warning: Could not load cognitive cache: {e}" + Style.RESET_ALL)

    # Enable cache read/write as requested
    config.cache_enabled = True
    config.cache_write_enabled = True
    config.cache_write_scope = "interactive_phenomenality"
    
    # Disambiguation resolves INTERNALLY, never by halting to interrogate the
    # operator. Reading is non-adapting reality: when a chunk is ambiguous the
    # model reconciles its own thoughts with what the text says, it does not
    # ask a human what the document meant. The old interactive path popped a
    # display-only "SYSTEM HALTED" modal (its real input was a hidden terminal
    # prompt) and injected a hardcoded GSM8K "you are a reasoning engine"
    # persona -- both contrary to the bare-mode, first-person spine. Off here.
    config.interactive_disambiguation = False

    memory = MemoryEngine(scope="interactive_phenomenality")
    self_concept = SelfConceptController()
    steer_map = SteerMapStore()
    tuner = TriggerTuner()
    # Every trigger is born tunable. Persisted values win over these defaults.
    tuner.register("claimmap_tension", 0.18, kind="threshold", comparator=">=")
    # Steering is OFF by default (0). The raw A-B delta is large; any nonzero
    # alpha must be tuned UP from ~0 while watching -- alpha=0.5 collapses the
    # model into a repetition loop. Enable deliberately: :tune claimmap_alpha 0.02
    tuner.register("claimmap_alpha", 0.0, kind="coefficient")
    # Memory is state-triggered too: it fires from the model's own "memory gap"
    # signature (see _memory_detect), not a <<MEMORY>> text tag. Born tunable; the
    # steer that injects the retrieval is OFF by default (opt-in, capped).
    tuner.register("memory_need", 0.05, kind="threshold", comparator=">=")
    tuner.register("memory_alpha", 0.0, kind="coefficient")
    # Clock sensor: this turn's generation wall-time and memory footprint,
    # observed every turn like any other stream (so they can be anchored:
    # "did slow / memory-heavy turns cohere worse?"). vram_gb carries VRAM on a
    # CUDA box and process RSS on CPU-only, so CPU runs keep the same stream.
    # Threshold-kind with a 0 bar until you calibrate one; observation-only,
    # never a knob you set.
    tuner.register("generation_seconds", 0.0, kind="threshold", comparator=">=")
    tuner.register("vram_gb", 0.0, kind="threshold", comparator=">=")
    # The steering envelope itself is a live surface: every additive push is
    # clipped to steer_cap_fraction of the residual it lands in, band-style
    # steers (claimmap/memory) default to the [steer_band_lo, steer_band_hi)
    # depth window, and agentic deltas scale by steer_fraction. All movable
    # mid-session; none removable (non-finite values are rejected at sync).
    from invariants.engine import get_steer_cap_fraction, get_steer_band
    _band_lo, _band_hi = get_steer_band()
    tuner.register("steer_cap_fraction", get_steer_cap_fraction(), kind="coefficient")
    tuner.register("steer_band_lo", _band_lo, kind="coefficient")
    tuner.register("steer_band_hi", _band_hi, kind="coefficient")
    tuner.register("steer_fraction", config.steer_fraction, kind="coefficient")
    # Conversational outcome threshold: a turn whose sense_score (coherence
    # minus interruption/disagreement) clears this labels its steering events
    # "conversation_productive" in the steer map. Default 0.0 is the composite's
    # natural sign, not a magic number; observed every turn, so it can be
    # calibrated to the real distribution (:tune conversation_productive auto 50).
    tuner.register("conversation_productive", 0.0, kind="threshold", comparator=">=")
    # Sandbox executions are objective outcomes (exit 0, no timeout = 1.0);
    # observed so the distribution is inspectable like every other signal.
    tuner.register("sandbox_success", 0.5, kind="threshold", comparator=">=")
    # Agency measurement: signal 1.0 on turns whose context was really caused
    # by the model's own words (code ran, a tool it invoked returned, the
    # reading followed its reply). Credited with the turn's sense, so :tune
    # lift answers: does experiencing impact track better deliberation?
    tuner.register("words_had_impact", 0.5, kind="threshold", comparator=">=")
    # Consequence trail: {"cause", "effect"} events. "pending" = caused by the
    # reply just produced, consumed as the NEXT turn's impact context.
    impact_state = {"pending": [], "log": deque(maxlen=40)}
    # Layer isolation (:tune steer_layer_sweep 1): every steer pushes exactly
    # ONE layer — the least-tested in the band — and the turn's outcome lands
    # on that layer in the steer map. Band-wide steering assumes one semantic
    # axis runs through the depth; the sweep measures instead of assuming.
    # "fires" collects (channel, layer, alpha, drift_at_layer) per turn.
    tuner.register("steer_layer_sweep", 0.0, kind="coefficient")
    sweep_state = {"fires": []}
    # Reply token budget: the first live run cut off mid-sentence at the old
    # hardcoded 512. Live-tunable like everything else (:tune response_tokens 800).
    tuner.register("response_tokens", 512, kind="coefficient")
    # ':doc read ... satisfied' stops once this many consecutive reading turns
    # clear the conversation_productive bar -- "satisfied" is the existing
    # sense signal, not a new oracle. Streak length is a knob like everything.
    tuner.register("reading_settled_streak", 2, kind="coefficient")
    # Expert consultation surface: routing fires on entropy > routing_entropy,
    # at most routing_loops times per token and routing_events times per
    # reply. Budgets, not truths -- live-tunable like every other number here.
    tuner.register("routing_events", config.max_routing_events, kind="coefficient")
    tuner.register("routing_loops", config.max_loops, kind="coefficient")
    tuner.register("routing_entropy", config.entropy_threshold, kind="coefficient")
    tuner.register("routing_entropy_weight", getattr(config, "routing_entropy_weight", 1.0), kind="coefficient")
    tuner.register("routing_collapse_margin", getattr(config, "routing_collapse_margin", 0.75), kind="coefficient")
    tuner.register("routing_collapse_penalty", getattr(config, "routing_collapse_penalty", 1.25), kind="coefficient")
    tuner.register("routing_probe_weight", getattr(config, "routing_probe_weight", 1.0), kind="coefficient")
    # Proven-expert routing: how hard the ToT winner is nudged toward experts
    # with a good accrued route success rate. 0 = pure entropy (default);
    # earn it from outcomes (:calibrate expert_proof_weight outcome).
    tuner.register("expert_proof_weight", 0.0, kind="coefficient")
    # Calibration step size: a calibration AUGMENTS rather than overwrites --
    # the first one adopts the computed value, later ones move this fraction of
    # the way toward it (EMA smoothing). 1.0 = overwrite; default 0.5.
    tuner.register("calibration_gain", 0.5, kind="coefficient")
    # Prioritize: :prioritize ranks probes by evidence-weighted lift; when this
    # alpha > 0 each reply is steered toward the top-ranked probe (signed by its
    # lift -- toward a helpful concept, away from a harmful one). 0 = off.
    tuner.register("prioritize_alpha", 0.0, kind="coefficient")
    # Field mechanics stay ordinary live knobs, while personal G/time sources
    # may point at any other knob or observed stream.
    tuner.register("prioritize_gravity", 0.0, kind="coefficient")
    tuner.register("gravity_doppler", 0.0, kind="coefficient")
    tuner.register("gravity_law_step", 0.0, kind="coefficient")
    tuner.register("gravity_potential", 0.0, kind="threshold")
    tuner.register("gravity_time", 1.0, kind="threshold")
    for _field_knob in ("prioritize_gravity", "gravity_doppler", "gravity_law_step"):
        tuner.triggers[_field_knob].kind = "coefficient"  # migrate early persisted drafts
    # Exposed-probe steer: when > 0, each reply is ALSO nudged along the probes
    # the operator has exposed to the model (:probe expose <name>) -- lift-weighted
    # and signed just like prioritize (a helpful exposed sensor pulls toward its
    # concept, a harmful one pushes away). 0 = off; idle until those probes accrue
    # credited lift. :tune exposed_probe_alpha <small>, or :calibrate from outcomes.
    tuner.register("exposed_probe_alpha", 0.0, kind="coefficient")
    # Synthesis schedule, same shape: events per reply, optimizer steps per
    # event, and the plateau-velocity trigger that starts one.
    tuner.register("synthesis_events", config.max_synthesis_events, kind="coefficient")
    tuner.register("synthesis_steps", config.max_synthesis_steps, kind="coefficient")
    tuner.register("plateau_epsilon", config.epsilon, kind="coefficient")
    tuner.register("max_committee_size", 6, kind="coefficient")
    
    tuner.register("max_rounds", getattr(config, "max_rounds", 5), kind="coefficient")
    tuner.register("required_agreement", getattr(config, "required_agreement", 3), kind="coefficient")
    tuner.register("max_tool_calls", getattr(config, "max_tool_calls", 8), kind="coefficient")
    tuner.register("max_new_tokens", getattr(config, "max_new_tokens", 220), kind="coefficient")
    tuner.register("repair_token_multiplier", getattr(config, "repair_token_multiplier", 2.0), kind="coefficient")
    tuner.register("max_elapsed_sec", -1.0, kind="coefficient")
    tuner.register("oracle_max_elapsed_sec", getattr(config, "oracle_max_elapsed_sec", 60.0), kind="coefficient")
    tuner.register("verifier_time_reserve_sec", getattr(config, "verifier_time_reserve_sec", 20.0), kind="coefficient")
    tuner.register("relax_agreement_under_urgency", 1.0 if getattr(config, "relax_agreement_under_urgency", False) else 0.0, kind="coefficient")
    tuner.register("stop_on_critical_urgency", 1.0 if getattr(config, "stop_on_critical_urgency", True) else 0.0, kind="coefficient")
    # Intent axis of the decomposed reward: signal = previous turn's
    # (ambiguity + disagreement) minus this turn's. Fired (> 0) = this turn
    # SETTLED intent. Credited with sense, so conversation_productive can be
    # calibrated relative to intent-shaping (:tune conversation_productive
    # auto intent) instead of an arbitrary quantile.
    tuner.register("intent_settling", 0.0, kind="threshold", comparator=">=")
    # EOT Urgency: Tracks the model's internal probability of finishing its turn.
    tuner.register("eot_urgency", 0.05, kind="threshold", comparator="<=")

    def _sweep_layers_for(channel, steer_vecs=None):
        """Pick the steer_layer_sweep least-tested band layers (width 1 =
        isolation; width k = deterministic overlay), plus per-layer local axis
        drift of the vector dict being applied (is the direction even the
        same thing next door?). Empty list = sweep off (full band)."""
        from invariants.engine import axis_drift, drift_at_layer, pick_sweep_layers, steer_band_layers
        width = max(0, int(tuner.get("steer_layer_sweep", 0)))
        if width <= 0:
            return [], {}, None
        n_layers = getattr(model, "n_layers", None) or len(model.model.model.layers)
        layers = pick_sweep_layers(steer_band_layers(n_layers), steer_map.layer_steer_counts(channel), width)
        drift = axis_drift(steer_vecs) if steer_vecs else None
        drifts = {L: drift_at_layer(drift, L) for L in layers} if drift else {L: None for L in layers}
        return layers, drifts, (drift.get("var") if drift else None)

    _sync_steer_tunables(tuner, config)

    # Tool-sensing seam: any tool fires mid-thought when its state trigger crosses.
    # Detectors read (text, state) -- state carries the live phenomenality scores,
    # so a tool triggers from activation, not a tag. add a tool = register a detector.
    tool_sense = ToolSense(model, tuner)

    # Async operator input: a reader thread queues every line so nothing is
    # dropped; when listen-drain is on, lines typed mid-generation are pulled at
    # the chunk seam and appended to the live stream -- the model reads the
    # interjection and redirects or folds in on its own. Opt-in via :listen.
    listener = InputListener()
    listen_state = {"drain": False}

    def _operator_input_drain(text, state=None):
        if not listen_state["drain"]:
            return ""
        lines = listener.drain()
        # A :command or exit typed mid-reply is meant for the shell, not the
        # model -- hold it for the next prompt instead of injecting it as speech.
        speech_parts = []
        for l in lines:
            s = (l or "").strip()
            if not s:
                continue
            if s.startswith(":") or s.lower() in ("exit", "quit"):
                listener._q.put(l)
            else:
                speech_parts.append(s)
        speech = " ".join(speech_parts)
        if not speech:
            return ""
        memory.append_event(
            "operator_interjection",
            text=speech[:400],
            tags=["async_input", "interrupt"],
            provenance={"drained_mid_generation": True, "chars": len(speech)},
        )
        impact_state["pending"].append(
            {"cause": "was spoken to mid-thought", "effect": f'interjection ingested: "{speech[:80]}"'}
        )
        print(
            Fore.MAGENTA + Style.BRIGHT
            + f"\n[Listen] interjected mid-thought, ingested -> {speech[:120]}"
            + Style.RESET_ALL,
            flush=True,
        )
        # A legible, operator-attributed interjection -- not a taught tag. It
        # becomes the next tokens; whether the model redirects is emergent.
        return f"\n\n[Operator interjects: {speech}]\n\n"

    tool_sense.input_drain = _operator_input_drain

    def _claimmap_detect(text, state=None):
        a, b, score = framing_tension_score(text)
        return score, ((a, b) if a is not None else None)

    def _memory_detect(text, state=None):
        # Memory-gap footprint (STATE, not text): the model is unsettled (ambiguity
        # / disagreement up) and not flowing (validated_flow / warranted confidence
        # down) -- an unresolved reference it may need to retrieve. Complex,
        # multi-signal, read from the model's OWN phenomenality, never a tag.
        phen = (state or {}).get("last_phenomenality") or {}
        if not phen:
            return 0.0, None
        gap = (
            float(phen.get("ambiguity", 0.0))
            + float(phen.get("disagreement", 0.0))
            - float(phen.get("validated_flow", 0.0))
            - float(phen.get("warranted_confidence_legacy", 0.0))
        )
        tail = " ".join((text or "").split()[-24:]).strip()
        return gap, (tail or None)

    def _memory_act(query, m):
        records = memory.search(query, max_records=3, scope=memory.scope)
        if not records:
            return []
        memory.append_event(
            "memory_state_triggered",
            text=query[:240],
            tags=["memory_tool", "activation_trigger"],
            provenance={"records": len(records), "trigger": "memory_need"},
        )
        print(
            Fore.MAGENTA
            + f"\n[Memory] gap sensed mid-thought (not a tag) -> retrieved {len(records)} record(s)."
            + Style.RESET_ALL,
            flush=True,
        )
        alpha = tuner.get("memory_alpha", 0.0)
        sweep_layers = None
        if alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
            _mls, _, _mvar = _sweep_layers_for("memory")
            if _mls:
                sweep_layers = _mls
                for _sl in _mls:
                    sweep_state["fires"].append(("memory", _sl, alpha, None, len(_mls), _mvar))
        return _memory_steer_handles(m, (records[0].text or "").strip(),
                                     alpha=alpha, layers=sweep_layers)

    def _claimmap_act(payload, m):
        a, b = payload
        cm = analyze_claim_pair(f"{a} || {b}", model=m)
        alpha = tuner.get("claimmap_alpha", 0.0)
        sweep_layers = None
        if alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
            _cls, _cdrifts, _cvar = _sweep_layers_for("claimmap", cm.steer_delta)
            if _cls:
                sweep_layers = _cls
                for _cl in _cls:
                    sweep_state["fires"].append(("claimmap", _cl, alpha, _cdrifts.get(_cl), len(_cls), _cvar))
        handles = claimmap_steer_handles(m, cm.steer_delta, alpha=alpha, layers=sweep_layers)
        msg = (
            "steering the rest of this answer."
            if handles
            else "steering off (:tune claimmap_alpha to enable)."
        )
        print(Fore.MAGENTA + f"\n[ClaimMap] mid-thought: sensed a tension, {msg}" + Style.RESET_ALL, flush=True)
        return handles

    tool_sense.register(Tool(
        "claimmap_tension",
        _claimmap_detect,
        _claimmap_act,
        activation_criteria="framing tension score crosses :tune claimmap_tension",
        steer_magnitude=":tune claimmap_alpha",
    ))
    tool_sense.register(Tool(
        "memory_need",
        _memory_detect,
        _memory_act,
        comparator=">=",
        activation_criteria="phenomenality memory-gap score crosses :tune memory_need",
        steer_magnitude=":tune memory_alpha",
    ))
    imported_methodologies = memory.import_methodologies(
        _global_cache.memory,
        source="cognitive_cache",
        source_path=str(cache_file),
    )
    memory.append_event(
        "shell_start",
        tags=["session"],
        provenance={
            "script": os.path.abspath(__file__),
            "cache_write_scope": config.cache_write_scope,
            "memory_policy": "tool_not_prompt",
            "methodologies_imported": imported_methodologies,
            "self_concept_controller": "vector_map_based",
            "steer_map_store": str(steer_map.events_path),
        },
    )
        
    print(Fore.CYAN + "\nThis terminal uses full Agentic ToT and Test-Time Layer Synthesis.")
    print("Watch the model's internal entropy and phenomenality trace in real time!")
    print("Current session context is ON. Long-term memory is a tool, not hidden prompt context.")
    print("Self-concept orientation is vector-map based and logged as a tool/controller trace.")
    print(f"Steer-map traces are stored at {steer_map.events_path}.")
    print(f"Imported {imported_methodologies} sanitized methodology memories from cognitive cache.")
    for _hline in COMMAND_HELP_LINES:
        print(_hline)
    print("Type 'exit' or 'quit' to leave.\n" + Style.RESET_ALL)
    try:
        _md_path = write_commands_md()
        print(Fore.CYAN + f"(command reference refreshed at {os.path.relpath(_md_path, ROOT)})" + Style.RESET_ALL)
    except Exception as _md_err:
        print(Fore.YELLOW + f"[Docs] Could not refresh COMMANDS.md: {_md_err}" + Style.RESET_ALL)

    commands_since_reply = set()
    pending_memory_tool_result = None
    pending_orientation_tool_result = None
    pending_claimmap_tool_result = None
    pending_claimmap_steer_delta = None
    pending_claimmap_credit = None  # last turn's tension, credited by this turn's sense
    pending_methodmap_tool_result = None
    pending_document_tool_result = None
    pending_sandbox_tool_result = None
    pending_game_tool_result = None
    queued_commands = []
    model_proposed_game = None
    user_proposed_game = None
    user_proposed_game_args = None
    active_game = None
    active_game_state = {}
    # Commands a game CHOOSES to run don't execute on their own: they stage here
    # and wait for an explicit operator :accept (or :reject). Tools stay free;
    # this gate is only for real :commands a game would adopt into the queue.
    pending_accept_commands = []
    pending_solve_proposal = None   # a :solve choose/auto macro awaiting :accept
    solve_expect_contexts = {}      # :solve metadata waiting for its queued :expect macro turn
    pending_completion_definition_id = None
    pending_help_tool_result = None # model asked for <<HELP>>; served next turn
    help_exposed = False            # when on, the model is told it may call <<HELP>>
    memory_lanes_enabled = {"act": False, "talk": False}
    memory_tool_exposed = True      # :expose off memory blocks the model's <<MEMORY>> tool

    def _stage_for_accept(cmd, why="a game"):
        pending_accept_commands.append(cmd)
        idx = len(pending_accept_commands)
        print(Fore.YELLOW + Style.BRIGHT
              + f"[Accept] {why} wants to run a command -- staged #{idx}, NOT run: {cmd}"
              + Style.RESET_ALL)
        print(Fore.YELLOW
              + "         Type :accept to run it (or :accept all), :reject to drop it (or :reject all)."
              + Style.RESET_ALL)

    def _memory_lane_name(raw):
        token = str(raw or "").strip().lower()
        return {
            "act": "act",
            "action": "act",
            "actions": "act",
            "talk": "talk",
            "speech": "talk",
            "speak": "talk",
            "conversation": "talk",
            "convo": "talk",
        }.get(token)

    def _enabled_memory_lanes():
        return [lane for lane in ("act", "talk") if memory_lanes_enabled.get(lane)]

    def _memory_record_in_lane(record, lane):
        tags = set(record.tags or [])
        if lane == "act":
            return (
                "action" in tags
                or "model_action" in tags
                or "model_output" in tags
                or (record.kind == "turn" and record.role == "assistant")
            )
        if lane == "talk":
            return record.kind == "turn" or "conversation_trace" in tags
        return False

    def _memory_lane_records(lanes):
        lane_set = set(lanes or [])
        out, seen = [], set()
        for record in memory.records:
            if not (record.text or "").strip():
                continue
            if lane_set and not any(_memory_record_in_lane(record, lane) for lane in lane_set):
                continue
            rid = record.record_id or id(record)
            if rid in seen:
                continue
            seen.add(rid)
            out.append(record)
        return out

    def _memory_search_in_records(query, records, max_records=6):
        terms = {t for t in str(query or "").lower().replace("_", " ").split() if t}
        if not terms:
            return list(records)[-max_records:]
        source = list(records)
        scored = []
        for idx, record in enumerate(source):
            text_score = _memory_text_score(query, record, idx, len(source))
            if text_score <= 0:
                continue
            scored.append((text_score, idx, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:max_records]]

    def _memory_record_haystack(record):
        return " ".join(
            [
                record.kind,
                record.role or "",
                record.text or "",
                " ".join(record.tags or []),
                json.dumps(record.provenance or {}, ensure_ascii=True, sort_keys=True),
                json.dumps(record.metrics or {}, ensure_ascii=True, sort_keys=True),
            ]
        ).lower()

    def _memory_text_score(query, record, idx=0, total=1):
        terms = {t for t in str(query or "").lower().replace("_", " ").split() if t}
        if not terms:
            return 0.0
        haystack = _memory_record_haystack(record)
        hits = sum(1 for term in terms if term in haystack)
        if hits <= 0:
            return 0.0
        return float(hits) + (0.15 * (idx / max(1, total)))

    def _memory_parse_selector(arg):
        raw = (arg or "").strip()
        if not raw:
            return "", []
        if raw.lower().startswith("probe "):
            pname = raw.split(None, 1)[1].strip()
            return "", [(1, pname)]
        sentence_tokens, signed = [], []
        for token in raw.split():
            m = re.fullmatch(r"([+-])([A-Za-z_][A-Za-z0-9_-]*)", token.strip())
            if m:
                signed.append((1 if m.group(1) == "+" else -1, m.group(2)))
            else:
                sentence_tokens.append(token)
        return " ".join(sentence_tokens).strip(), signed

    def _resolve_memory_probe_specs(signed_specs):
        resolved = []
        for sign, raw_name in signed_specs:
            pname_raw = str(raw_name or "").strip()
            pname = (
                resolve_probe_choice(pname_raw, probes, model=model, config=config, action_name="memory ranking")
                if pname_raw
                else None
            )
            if not pname or pname not in probes:
                return [], f"unknown probe '{pname_raw}'"
            resolved.append((1 if sign >= 0 else -1, pname))
        return resolved, None

    def _memory_rank_by_sentence_and_probes(sentence, signed_specs, records, *, max_records=6, scan=160):
        resolved, err = _resolve_memory_probe_specs(signed_specs)
        if err:
            return [], err, []
        source = list(records)
        if resolved:
            source = source[-scan:]
        scored = []
        from invariants.engine import _inputs as _q_inputs, _hidden_states as _q_hidden, probe_score as _q_score
        for idx, record in enumerate(source):
            text_score = _memory_text_score(sentence, record, idx, len(source)) if sentence else 0.0
            if sentence and not resolved and text_score <= 0:
                continue
            probe_values = []
            if resolved:
                try:
                    with torch.no_grad():
                        q_ids = _q_inputs(model, (record.text or "").strip()[:600])
                        q_hs = _q_hidden(model, q_ids["input_ids"], q_ids.get("attention_mask"))
                        for sign, pname in resolved:
                            raw = float(_q_score(q_hs, probes[pname]["direction"]))
                            probe_values.append((sign, pname, raw))
                except Exception:
                    continue
            magnitude = sum(abs(raw) for _sign, _pname, raw in probe_values)
            relation = 0.0
            pairs = 0
            for i in range(len(probe_values)):
                si, _pi, ri = probe_values[i]
                for j in range(i + 1, len(probe_values)):
                    sj, _pj, rj = probe_values[j]
                    desired_same = si == sj
                    actual_same = (ri >= 0 and rj >= 0) or (ri < 0 and rj < 0)
                    strength = min(abs(ri), abs(rj))
                    relation += strength if actual_same == desired_same else -strength
                    pairs += 1
            if pairs:
                relation /= pairs
            recency = 0.05 * (idx / max(1, len(source)))
            score = text_score + magnitude + relation + recency
            if sentence and not resolved and text_score <= 0:
                continue
            scored.append((score, score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        probe_desc = " ".join(("+" if sign >= 0 else "-") + pname for sign, pname in resolved)
        desc_parts = []
        if sentence:
            desc_parts.append(f"search:{sentence}")
        if probe_desc:
            desc_parts.append(f"probes:{probe_desc}")
        return [t[2] for t in scored[:max_records]], " ".join(desc_parts) or "reflect", scored[:max_records]

    def _memory_reflection_records(lanes, *, max_records=6):
        candidates = _memory_lane_records(lanes)
        lane_note = "+".join(lanes or []) if lanes else "all"
        ranked_probes = rank_probes(probes, tuner)
        usable = [r for r in ranked_probes if r["name"] in probes and probes[r["name"]].get("direction")]
        if not candidates or not usable:
            return candidates[-max_records:], f"{lane_note}:reflect:recent", []
        has_priority = any(float(r.get("priority") or 0.0) > 0 for r in usable)
        top = [r for r in usable if (float(r.get("priority") or 0.0) > 0 or not has_priority)][:5]
        weights = {
            r["name"]: (float(r.get("priority") or 0.0) if has_priority else 1.0)
            for r in top
        }
        from invariants.engine import _inputs as _q_inputs, _hidden_states as _q_hidden, probe_score as _q_score
        scored = []
        source = candidates[-160:]
        for idx, record in enumerate(source):
            try:
                with torch.no_grad():
                    q_ids = _q_inputs(model, (record.text or "").strip()[:600])
                    q_hs = _q_hidden(model, q_ids["input_ids"], q_ids.get("attention_mask"))
                    score = 0.0
                    for pname, weight in weights.items():
                        raw = float(_q_score(q_hs, probes[pname]["direction"]))
                        score += max(float(weight), 0.0001) * abs(raw)
            except Exception:
                continue
            score += 0.05 * (idx / max(1, len(source)))
            scored.append((score, score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        desc = f"{lane_note}:reflect:prioritize:" + ",".join(weights)
        return [t[2] for t in scored[:max_records]], desc, scored[:max_records]

    def _memory_select_records(arg, lanes, *, max_records=6):
        arg = (arg or "").strip()
        candidates = _memory_lane_records(lanes)
        lane_note = "+".join(lanes or []) if lanes else "all"
        sentence, signed_specs = _memory_parse_selector(arg)
        if signed_specs:
            records, desc, ranked = _memory_rank_by_sentence_and_probes(sentence, signed_specs, candidates, max_records=max_records)
            if desc.startswith("unknown probe"):
                return [], desc, []
            return records, f"{lane_note}:{desc}", ranked
        if sentence:
            return _memory_search_in_records(sentence, candidates, max_records=max_records), f"{lane_note}:search:{sentence}", []
        return _memory_reflection_records(lanes, max_records=max_records)

    def _memory_lanes_for_bare_command():
        return _enabled_memory_lanes()

    def _memory_model_available():
        return memory_tool_exposed and bool(_enabled_memory_lanes())

    def _run_game_ref(name, args=None):
        """Injected into game scripts as run_game(name, args): load and run
        another game by name, so games can reference/compose each other. Shares
        the live active_game_state and probes with the caller."""
        gp = os.path.join(ROOT, "games", f"{name}.py")
        if not os.path.isfile(gp):
            print(Fore.YELLOW + f"[Game] referenced game '{name}' not found in games/." + Style.RESET_ALL)
            return False
        sub_cfg = load_game_config(os.path.join(ROOT, "games", f"{name}_rules.json"))
        with open(gp, "r", encoding="utf-8") as _gf:
            code = _gf.read()
        g = globals().copy()
        g["GAME_RULES"] = sub_cfg["rules"]
        g["GAME_CONFIG"] = sub_cfg
        g["game_args"] = list(args or [])
        g["active_game_state"] = active_game_state
        g["probes"] = probes
        g["run_game"] = _run_game_ref
        exec(code, g)
        return True

    doc_session = None          # current document: chunks + cursor + why
    doc_library = []            # every ingested document this session (for inter-ordering)
    doc_autoread = None         # {"remaining": n, "mode": "order"|"interleave"|"reply"} during :doc read
    last_assistant_response = ""
    recent_responses = deque(maxlen=4)  # reflection stream for reply-mode thread returns
    prev_unsettledness = None  # ambiguity+disagreement of the previous turn (intent axis)
    probes = {}  # name -> {direction, history, framings}; minted concept sensors (unvalidated until outcomes accrue)
    global _ACTIVE_PROBES, _ACTIVE_MODEL, _ACTIVE_TUNER
    _ACTIVE_PROBES = probes
    _ACTIVE_MODEL = model
    _ACTIVE_TUNER = tuner
    PROBE_DIR = os.path.join(ROOT, "invariants", "out", "probes")
    stored_probe_histories = load_probe_raw_histories()
    restored_probe_histories = 0
    try:
        if os.path.isdir(PROBE_DIR):
            for pf in os.listdir(PROBE_DIR):
                if pf.endswith(".pt"):
                    pname = pf[:-3]
                    pdata = torch.load(os.path.join(PROBE_DIR, pf), weights_only=True)
                    prior_raw = stored_probe_histories.get(pname, [])
                    probes[pname] = {
                        "direction": pdata["direction"],
                        "history": deque(prior_raw, maxlen=40),
                        "framings": pdata.get("framings", ("", "")),
                        "exposed": bool(pdata.get("exposed", False)),
                        "released": bool(pdata.get("released", False)),
                    }
                    if prior_raw:
                        restored_probe_histories += 1
                    tuner.register(f"probe_{pname}", 0.0, kind="threshold", comparator=">=")
    except Exception:
        pass
    if restored_probe_histories:
        print(
            Fore.CYAN
            + f"[Probe] Restored raw centering baselines for {restored_probe_histories} probe(s)."
            + Style.RESET_ALL
        )
    migrated_prompt_probes = []
    for pname in sorted(DEFAULT_COMMAND_PROMPT_PROBES & set(probes)):
        if not probe_is_released(probes[pname]):
            set_probe_released(probes, pname, True, PROBE_DIR)
            migrated_prompt_probes.append(pname)
    if migrated_prompt_probes:
        print(
            Fore.CYAN
            + "[Probe] Released default-command prompt probes from generation impact: "
            + ", ".join(migrated_prompt_probes)
            + ". They remain visible and scored."
            + Style.RESET_ALL
        )

    doc_probe_score_cache = {}

    def _select_autoread_chunk(library, autoread, last_thought="", earlier_thoughts=""):
        mode = (autoread or {}).get("mode", "order")
        signed_specs = list((autoread or {}).get("probe_specs") or [])
        probe_scores = None
        if signed_specs:
            probe_scores, newly_scored, candidate_count = score_unread_document_chunks_by_probes(
                model,
                library,
                probes,
                signed_specs,
                cache=doc_probe_score_cache,
                scan=160,
            )
            terms = " ".join(
                ("+" if sign >= 0 else "-") + name
                for sign, name in signed_specs
            )
            if newly_scored:
                print(
                    Fore.CYAN
                    + f"[Doc] Ranked {candidate_count} unread candidate(s) by {terms}; "
                    + f"{newly_scored} new chunk forward pass(es), remainder cached."
                    + Style.RESET_ALL
                )
            if candidate_count and not probe_scores:
                print(
                    Fore.YELLOW
                    + f"[Doc] Could not score unread chunks by {terms}; falling back to {mode}."
                    + Style.RESET_ALL
                )
        return select_next_chunk(
            library,
            last_thought,
            mode,
            earlier_thoughts=earlier_thoughts,
            probe_scores=probe_scores,
            probe_terms=signed_specs,
        )

    # Purge stray shadow triggers: never-fed bare thresholds that duplicate a
    # probe (an accidental `:tune <probe-name> <value>` registers one, and it
    # then shadows the real probe in every direct trigger lookup). The probe is
    # the real one; removing the shadow fixes all code paths at once.
    _shadows = [n for n in list(tuner.triggers) if _is_shadow_trigger(n, tuner)]
    if _shadows:
        for _sh in _shadows:
            del tuner.triggers[_sh]
        tuner.save()
        print(Fore.CYAN + f"[System] Cleared {len(_shadows)} stray shadow trigger(s): {', '.join(_shadows)}." + Style.RESET_ALL)

    queued_calibrations = []
    last_refused_calibration = None
    
    # Per-turn signals table: one row per turn with every stream's value --
    # the substrate that lets ANY named stream anchor ANY threshold target
    # (both ways). Persisted, so paired evidence survives restarts.
    TURN_SIGNALS_PATH = os.path.join(ROOT, "invariants", "out", "turn_signals.jsonl")
    turn_log = deque(maxlen=400)
    try:
        with open(TURN_SIGNALS_PATH, "r", encoding="utf-8") as _tf:
            for _line in _tf.readlines()[-400:]:
                try:
                    turn_log.append(json.loads(_line))
                except Exception:
                    continue
    except OSError:
        pass
    # Macro name -> file aliases, so :run <name> and :macro strip <name> address a
    # macro by a short name instead of a path. Persisted across sessions.
    MACRO_ALIAS_PATH = os.path.join(ROOT, "invariants", "out", "macro_aliases.json")
    macro_aliases = {}
    try:
        with open(MACRO_ALIAS_PATH, "r", encoding="utf-8") as _af:
            macro_aliases = {str(k): str(v) for k, v in json.load(_af).items()}
    except (OSError, ValueError):
        macro_aliases = {}

    def _save_macro_aliases():
        try:
            with open(MACRO_ALIAS_PATH, "w", encoding="utf-8") as af:
                json.dump(macro_aliases, af, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save macro aliases: {e}" + Style.RESET_ALL)
    MAX_AUTOREAD = 20           # per :doc read command; reading stays a deliberate act
    sandbox_enabled = False     # deliberate opt-in, like every intervention here
    joined_agents = []
    replace_agent = None
    replace_turns_remaining = 0
    session_context = []
    session_context_enabled = True
    show_timestamps = False
    last_clock = None
    # prioritize/steer target: None = auto (follow the ranking). "probe" pins
    # the steer to one probe; "mix" is a list of probes whose DEGREES are their
    # own learned lifts (you pick the probes, the data sets the weights).
    prioritize_pin = {"probe": None, "mix": None}
    tuner_bindings = {}  # knob -> (probe, multiplier)
    # probe<->knob matches: probe name -> {"knob", "mode": none|drive|validate, "mult"}.
    # A probe like memory_alpha is correlated to the memory_alpha knob but the two
    # are separate; a match ties them so the probe can DRIVE the knob (servo) or
    # VALIDATE it (record knob-value vs probe-reading each turn -> correlation).
    PROBE_MATCH_PATH = os.path.join(ROOT, "invariants", "out", "probe_matches.json")
    probe_matches = {}
    try:
        with open(PROBE_MATCH_PATH, "r", encoding="utf-8") as _pmf:
            probe_matches = {str(k): dict(v) for k, v in json.load(_pmf).items()}
    except (OSError, ValueError):
        probe_matches = {}
    match_hist = {}  # probe name -> deque of (knob_value, probe_reading); session-only

    def _save_probe_matches():
        try:
            with open(PROBE_MATCH_PATH, "w", encoding="utf-8") as pmf:
                json.dump(probe_matches, pmf, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save probe matches: {e}" + Style.RESET_ALL)

    def _agent_slug(name):
        slug = re.sub(r"[^a-z0-9_]", "_", str(name or "").lower()).strip("_")[:48]
        return slug or "agent"

    def _agent_tuner(name):
        """Agent tuners live under their own path, seeded from current shell knobs."""
        path = os.path.join(ROOT, "invariants", "out", "agents", _agent_slug(name), "trigger_tuner.json")
        agent_tuner = TriggerTuner(path=path)
        for _tn, _tr in tuner.triggers.items():
            if _tn in agent_tuner.triggers:
                continue
            try:
                agent_tuner.triggers[_tn] = type(_tr).from_dict(_tn, _tr.to_dict())
            except Exception:
                agent_tuner.register(_tn, _tr.value, kind=_tr.kind, comparator=_tr.comparator)
        agent_tuner.save()
        return agent_tuner

    SPAWN_FAKE_KNOBS = {
        "embedding_size", "hidden_layers", "batch_size", "learning_rate",
        "optimizer", "cpu_cycles_per_second", "accuracy", "precision",
        "recall", "general_knowledge", "common_sense", "problem_solving",
        "adaptability", "consistency", "memory_access",
    }
    def _spawn_profile_runnable_lines(raw_lines, base_tuner):
        """Keep spawn profiles executable without imposing a workflow whitelist.

        A spawn profile is a worker-authored setup script for the target agent.
        The loader's job is only to reject prose/unknown commands and ensure
        queued items are future-runnable commands; semantics belong to the
        worker prompt and the real command handlers.
        """
        forbidden_tunes = SPAWN_FAKE_KNOBS
        probe_subcommands = {
            "adopt", "compose", "expose", "backfill", "drop", "match", "define",
            "explain", "show", "hide", "values", "value", "recent", "last", "auto",
        }
        known_targets = set(base_tuner.triggers)
        pending_probes = set()
        runnable, skipped = [], []

        def _clean_probe(raw):
            return re.sub(r"[^a-z0-9_]", "_", str(raw or "").lower()).strip("_")[:40]

        def _known_or_pending(name):
            nm = str(name or "").lower()
            bare = nm[len("probe_"):] if nm.startswith("probe_") else nm
            return nm in known_targets or bare in pending_probes or f"probe_{bare}" in known_targets

        def _queued_command_from_tail(tail):
            qt = (tail or "").strip()
            if not qt:
                return ""
            return qt if qt.startswith(":") else ":" + qt

        def _known_command(word):
            return word in BUILTIN_COMMANDS or word in macro_aliases

        for idx, raw in enumerate(raw_lines, 1):
            s = (raw or "").strip()
            if not s or s.startswith("#"):
                continue
            if not s.startswith(":"):
                skipped.append(f"L{idx}: non-command text")
                continue
            word = command_word(s)
            if not _known_command(word):
                skipped.append(f"L{idx}: :{word} is not a known shell command or macro")
                continue

            if word != "queue":
                runnable.append(s)
                continue

            if word == "probe":
                tail = s[len(":probe"):].strip()
                if not tail:
                    skipped.append(f"L{idx}: empty :probe")
                    continue
                parts = tail.split()
                head = parts[0].lower() if parts else ""
                if head in probe_subcommands:
                    if head == "compose" and len(parts) >= 2:
                        pname = _clean_probe(parts[1])
                        if pname:
                            pending_probes.add(pname)
                    elif head == "adopt":
                        for tok in parts[1:]:
                            if tok.lower() == "band":
                                break
                            pname = _clean_probe(tok)
                            if pname:
                                pending_probes.add(pname)
                    runnable.append(s)
                    continue
                pname = _clean_probe(head)
                if not pname or pname in {"auto", "choose", "choice"}:
                    skipped.append(f"L{idx}: invalid probe name '{head}'")
                    continue
                if "||" not in tail and r"\||" not in tail:
                    skipped.append(f"L{idx}: :probe {pname} missing || contrast")
                    continue
                pending_probes.add(pname)
                runnable.append(s)
                continue

            if word == "tune":
                parts = s[len(":tune"):].strip().split()
                if len(parts) < 2:
                    skipped.append(f"L{idx}: incomplete :tune")
                    continue
                target = parts[0].lower()
                bare_target = target[len("probe_"):] if target.startswith("probe_") else target
                mode = parts[1].lower()
                if target in forbidden_tunes or bare_target in forbidden_tunes:
                    skipped.append(f"L{idx}: refused fake/non-tunable target '{parts[0]}'")
                    continue
                if target not in known_targets and bare_target in pending_probes and mode != "dynamic":
                    skipped.append(f"L{idx}: use :calibrate, not static :tune, for probe '{bare_target}'")
                    continue
                if not _known_or_pending(target):
                    skipped.append(f"L{idx}: unknown tune target '{parts[0]}'")
                    continue
                if mode not in {"dynamic", "auto", "off"}:
                    try:
                        float(parts[1])
                    except ValueError:
                        skipped.append(f"L{idx}: non-numeric :tune value '{parts[1]}'")
                        continue
                runnable.append(s)
                continue

            if word == "steer":
                parts = s[len(":steer"):].strip().split()
                if not parts:
                    skipped.append(f"L{idx}: empty :steer")
                    continue
                head = parts[0].lower()
                if head in {"auto", "off", "none", "unpin"}:
                    runnable.append(s)
                    continue
                if head == "mix":
                    mix_terms = parts[1:]
                    if mix_terms:
                        try:
                            float(mix_terms[-1])
                            mix_terms = mix_terms[:-1]
                        except ValueError:
                            pass
                    bad = [p for p in mix_terms if _clean_probe(p) not in pending_probes]
                    if not mix_terms or bad:
                        skipped.append(f"L{idx}: :steer mix references unknown probe(s) {', '.join(bad) or '(none)'}")
                        continue
                    runnable.append(s)
                    continue
                if _clean_probe(head) not in pending_probes:
                    skipped.append(f"L{idx}: :steer references unknown probe '{parts[0]}'")
                    continue
                runnable.append(s)
                continue

            if word in {"calibrate", "label"}:
                parts = s[len(":" + word):].strip().split()
                if not parts:
                    skipped.append(f"L{idx}: empty :{word}")
                    continue
                if not _known_or_pending(parts[0]) and parts[0] not in {"steer_cap_fraction", "steer_band"}:
                    skipped.append(f"L{idx}: unknown :{word} target '{parts[0]}'")
                    continue
                runnable.append(s)
                continue

            if word == "doc":
                tail = s[len(":doc"):].strip()
                if not tail:
                    skipped.append(f"L{idx}: empty :doc")
                    continue
                head = tail.split()[0].lower()
                if head == "rewrite" or " rewrite " in tail.lower() or tail.lower().endswith(" rewrite"):
                    skipped.append(f"L{idx}: :doc rewrite is a file-writing command, not spawn context")
                    continue
                runnable.append(s)
                continue

            if word == "queue":
                tail = s[len(":queue"):].strip()
                queued_cmd = _queued_command_from_tail(tail)
                qword = command_word(queued_cmd)
                if not qword or qword == "queue" or not _known_command(qword):
                    skipped.append(f"L{idx}: :queue target must be a known future command")
                    continue
                runnable.append(s)
                continue

            if word == "place":
                parts = s[len(":place"):].strip().split()
                if len(parts) < 2 or _clean_probe(parts[0]) not in pending_probes:
                    skipped.append(f"L{idx}: :place references unknown probe")
                    continue
                runnable.append(s)

        return runnable, skipped

    def _spawn_state_report(agent_name, active_tuner=None, active_probes=None, *, max_macros=40):
        """Compact hook map for spawn workers/agents: what exists and what has evidence."""
        active_tuner = active_tuner or tuner
        active_probes = active_probes if active_probes is not None else probes

        def _vals(items, limit=8):
            vals = sorted({round(float(x), 4) for x in items})
            if len(vals) > limit:
                return vals[:limit] + ["..."]
            return vals

        lines = [f"[Spawn State: {agent_name}]"]
        lines.append("Known tunable hooks (exact names for :tune/:calibrate; value=current, history=what has been observed/tried):")
        for name in sorted(n for n in active_tuner.triggers if not n.startswith("probe_") and n not in SPAWN_FAKE_KNOBS):
            trig = active_tuner.triggers[name]
            st = trig.stats()
            tried = _vals((sig for sig, _out in trig.outcomes))
            parts = [
                f"- {name}: kind={st.get('kind')}",
                f"value={st.get('value')}",
                f"cmp={st.get('comparator')}",
                f"signals={st.get('n_signals')}",
            ]
            if st.get("signal_med") is not None:
                parts.append(f"signal_range={st.get('signal_min')}/{st.get('signal_med')}/{st.get('signal_max')}")
            if st.get("n_credited"):
                parts.append(f"credited={st.get('n_credited')}")
                parts.append(f"lift={st.get('lift')}")
            if tried:
                parts.append(f"tried={tried}")
            lines.append(" ".join(parts))

        if active_probes:
            lines.append("Active probe hooks (bare name drives probe_<name>; framings show what each sensor means):")
            ranked = {r["name"]: r for r in rank_probes(active_probes, active_tuner)}
            for pname in sorted(active_probes):
                pdata = active_probes[pname]
                trig = active_tuner.triggers.get(f"probe_{pname}")
                st = trig.stats() if trig else {}
                r = ranked.get(pname, {})
                framings = pdata.get("framings") or ("", "")
                a = " ".join(str(framings[0] if len(framings) > 0 else "").split())[:90]
                b = " ".join(str(framings[1] if len(framings) > 1 else "").split())[:90]
                lines.append(
                    f"- {pname}: trigger=probe_{pname} value={st.get('value')} "
                    f"signals={st.get('n_signals')} credited={st.get('n_credited')} "
                    f"lift={r.get('lift')} priority={round(float(r.get('priority') or 0.0), 4)} "
                    f"exposed={bool(pdata.get('exposed'))} released={probe_is_released(pdata)} "
                    f"framings={a} || {b}"
                )
        else:
            lines.append("Active probe hooks: none in this agent yet; mint/adopt/compose probes in the profile if needed.")

        if doc_library:
            docs = []
            for d in doc_library[:12]:
                read = len(d.get("read") or ())
                docs.append(f"{d.get('source_name')}({read}/{d.get('chunk_count')} read)")
            lines.append("Loaded documents: " + ", ".join(docs))
        else:
            lines.append("Loaded documents: none")

        if queued_calibrations:
            lines.append("Queued calibration retries: " + "; ".join(f":calibrate {q}" for q in queued_calibrations[:12]))
        if queued_commands:
            lines.append("Queued future commands: " + "; ".join(queued_commands[:12]))

        visible_cmds = sorted(_all_shell_commands() - hidden_commands)
        hidden = sorted(hidden_commands)
        exposed = [
            f"{command_exposure_display(w, r)}"
            f"({command_exposure_mode(r)}; activate={command_exposure_activation(w, r)}; steer={command_exposure_steer(r)})"
            for w, r in sorted(exposed_commands.items())
        ]
        exposed_probe_names = sorted(n for n, p in (active_probes or {}).items() if p.get("exposed"))
        lines.append("Visible commands/macros: " + ", ".join(f":{c}" for c in visible_cmds[:80]) + (" ..." if len(visible_cmds) > 80 else ""))
        if hidden:
            lines.append("Hidden from model-facing discovery: " + ", ".join(f":{c}" for c in hidden[:40]))
        lines.append("Runtime tools exposed with :expose: " + (", ".join(exposed[:40]) if exposed else "none"))
        if exposed_probe_names or exposed_knobs:
            lines.append("Model-readable probes/knobs: " + ", ".join(exposed_probe_names + sorted(exposed_knobs)))
        else:
            lines.append("Model-readable probes/knobs: none")
        macros = sorted(m for m in macro_aliases if m not in hidden_commands)
        lines.append("Macro aliases: " + (", ".join(macros[:max_macros]) + (" ..." if len(macros) > max_macros else "") if macros else "none"))
        field_context = format_field_prompt_context(active_tuner, active_probes)
        if field_context:
            lines.append(field_context)
        return "\n".join(lines)

    def _build_live_prompt(*args, **kwargs):
        """Every conversational/follow-up prompt gets the current active
        agent's factual field state; callers can explicitly pass None to omit."""
        if "field_context" not in kwargs:
            kwargs["field_context"] = format_field_prompt_context(tuner, probes)
        return build_prompt(*args, **kwargs)

    def _priority_steer_handles(active_probes, active_tuner, active_pin):
        handles = []
        prio_steered = None
        try:
            prio_alpha = float(active_tuner.get("prioritize_alpha", 0.0) or 0.0)
        except (TypeError, ValueError):
            prio_alpha = 0.0
        if prio_alpha <= 0 or not active_probes:
            return handles, prio_steered
        try:
            from invariants.engine import _steer_handles as _p_steer
            _pdir, _label, _psign = resolve_priority_direction(model, active_probes, active_tuner, active_pin)
            if _pdir:
                handles = list(_p_steer(model, _pdir, list(_pdir.keys()), prio_alpha * _psign))
                prio_steered = (_label, _psign, prio_alpha)
        except Exception:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            handles, prio_steered = [], None
        return handles, prio_steered

    # Re-arm any persisted servo bindings (mode=drive) so a restart keeps steering.
    for _pn, _m in probe_matches.items():
        if _m.get("mode") == "drive" and _m.get("knob"):
            tuner_bindings[_m["knob"]] = ([(1.0, _pn)], float(_m.get("mult", 1.0)))

    # Commands the operator has EXPOSED to the model as tools: word -> mode, where
    # mode is "stage" (the model's <<TOOL: ...>> proposes it, awaiting :accept) or
    # "direct" (it enters the shell queue immediately). Everything else the
    # model still cannot run. Meta-commands (:run/:solve/:macro/:game/:sandbox)
    # reach OTHER commands through their argument -- allowed, but warned.
    EXPOSED_CMD_PATH = os.path.join(ROOT, "invariants", "out", "exposed_commands.json")
    EXPOSED_KNOB_PATH = os.path.join(ROOT, "invariants", "out", "exposed_knobs.json")
    HIDDEN_CMD_PATH = os.path.join(ROOT, "invariants", "out", "hidden_commands.json")
    EXPOSE_META_CMDS = {"run", "solve", "macro", "game", "sandbox", "spawn", "accept"}
    hidden_commands = set()
    try:
        with open(HIDDEN_CMD_PATH, "r", encoding="utf-8") as _hcf:
            _raw_hidden = json.load(_hcf)
            if isinstance(_raw_hidden, list):
                hidden_commands = {str(x).lstrip(":").lower() for x in _raw_hidden if str(x).strip()}
            elif isinstance(_raw_hidden, dict):
                hidden_commands = {str(k).lstrip(":").lower() for k, v in _raw_hidden.items() if v}
    except (OSError, ValueError):
        hidden_commands = set()

    exposed_commands = {}
    try:
        with open(EXPOSED_CMD_PATH, "r", encoding="utf-8") as _ecf:
            exposed_commands = {
                str(k).lstrip(":").lower(): normalize_command_exposure(v)
                for k, v in json.load(_ecf).items()
            }
    except (OSError, ValueError):
        exposed_commands = {}
    exposed_knobs = set()
    try:
        with open(EXPOSED_KNOB_PATH, "r", encoding="utf-8") as _ekf:
            _raw_knobs = json.load(_ekf)
            if isinstance(_raw_knobs, list):
                exposed_knobs = {str(x).strip() for x in _raw_knobs if str(x).strip()}
            elif isinstance(_raw_knobs, dict):
                exposed_knobs = {str(k).strip() for k, v in _raw_knobs.items() if v}
    except (OSError, ValueError):
        exposed_knobs = set()
    exposed_knobs = {k for k in exposed_knobs if k in tuner.triggers and not _is_shadow_trigger(k, tuner)}

    def _save_exposed_commands():
        try:
            os.makedirs(os.path.dirname(EXPOSED_CMD_PATH) or ".", exist_ok=True)
            with open(EXPOSED_CMD_PATH, "w", encoding="utf-8") as ecf:
                json.dump(exposed_commands, ecf, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save exposed commands: {e}" + Style.RESET_ALL)

    def _save_hidden_commands():
        try:
            os.makedirs(os.path.dirname(HIDDEN_CMD_PATH) or ".", exist_ok=True)
            with open(HIDDEN_CMD_PATH, "w", encoding="utf-8") as hcf:
                json.dump(sorted(hidden_commands), hcf, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save hidden commands: {e}" + Style.RESET_ALL)

    def _save_exposed_knobs():
        try:
            os.makedirs(os.path.dirname(EXPOSED_KNOB_PATH) or ".", exist_ok=True)
            with open(EXPOSED_KNOB_PATH, "w", encoding="utf-8") as ekf:
                json.dump(sorted(exposed_knobs), ekf, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save exposed knobs: {e}" + Style.RESET_ALL)

    def _all_shell_commands():
        return BUILTIN_COMMANDS | set(macro_aliases)

    def _known_model_commands():
        return (_all_shell_commands() - hidden_commands) - {"expose"}

    def _visible_command_reference():
        return ", ".join(f":{w}" for w in sorted(_all_shell_commands() - hidden_commands))

    def _hidden_overwrite_blocked(name, surface):
        raw_name = str(name or "").lstrip(":").lower()
        stem_name = os.path.splitext(os.path.basename(raw_name))[0].lower()
        blocked_name = raw_name if raw_name in hidden_commands else (stem_name if stem_name in hidden_commands else None)
        if blocked_name:
            print(
                Fore.YELLOW
                + f"[{surface}] Refusing to overwrite hidden command ':{blocked_name}'. "
                + f"Reveal it first with ':hide off :{blocked_name}' if you mean to reuse that name."
                + Style.RESET_ALL
            )
            return True
        return False

    def _run_exposed_command_tool(cmd_requests):
        result_lines = [CMD_TOOL_HEADER]
        direct_cmds = []
        seen_any = False
        for requested in cmd_requests:
            for cmd in split_cmd_tool_commands(requested):
                seen_any = True
                requested_target, _, _ = parse_command_route(cmd)
                word = command_word(cmd)
                if not word:
                    result_lines.append(f"- refused malformed command: {cmd}")
                    continue
                if word == "expose":
                    result_lines.append("- refused :expose; the model cannot grant itself runtime tools.")
                    continue
                if word not in exposed_commands:
                    hint = did_you_mean(word, _known_model_commands())
                    result_lines.append(f"- refused :{word}; it is not exposed as a model tool.{hint}")
                    continue
                if word not in BUILTIN_COMMANDS and word not in macro_aliases:
                    result_lines.append(f"- refused :{word}; it is exposed but no longer exists as a command.")
                    continue
                exposure = exposed_commands.get(word, "stage")
                exposed_target = command_exposure_target(exposure)
                if requested_target and requested_target != exposed_target:
                    if exposed_target:
                        result_lines.append(
                            f"- refused {requested_target} :{word}; this tool is exposed for {exposed_target} :{word}."
                        )
                    else:
                        result_lines.append(
                            f"- refused {requested_target} :{word}; :{word} is exposed without an agent route."
                        )
                    continue
                cmd = apply_command_exposure_args(cmd, exposure)
                mode = command_exposure_mode(exposure)
                if mode == "direct":
                    direct_cmds.append(cmd)
                    result_lines.append(f"- queued direct command: {cmd}")
                else:
                    _stage_for_accept(cmd, why="an exposed runtime tool")
                    result_lines.append(f"- staged for operator :accept: {cmd}")
        if direct_cmds:
            input_queue[:0] = direct_cmds
            print(Fore.MAGENTA + f"\n[Runtime Tool] Queued {len(direct_cmds)} direct command(s): {'; '.join(direct_cmds)}" + Style.RESET_ALL)
        if not seen_any:
            result_lines.append("- no command found in the request.")
        result = "\n".join(result_lines)
        memory.append_event(
            "command_tool_model_requested",
            text=result,
            tags=["command_tool"],
            provenance={"requests": cmd_requests, "direct_count": len(direct_cmds)},
        )
        return result

    def _signed_direction(direction, sign=1.0):
        if direction is None or (isinstance(direction, dict) and not direction):
            return None
        sign = 1.0 if sign >= 0 else -1.0
        if isinstance(direction, dict):
            return {k: (v * sign if hasattr(v, "__mul__") else v) for k, v in direction.items()}
        return direction * sign if hasattr(direction, "__mul__") else direction

    def _refresh_tot_committee_from_live_state():
        """Build ToT routes from the live tuned/probed shell state.

        This keeps ToT grounded in operator-created controls instead of every
        vector file the registry can discover. Names are opaque labels; evidence,
        exposure, and explicit steering are the admission criteria.
        """
        from invariants.agentic_engine import _sane_fraction
        from invariants.mesa import Committee, MesaObjective

        committee = Committee()
        added = set()
        cap = max(0, int(getattr(config, "max_committee_size", 6) or 0))

        def add_route(name, direction, *, sign=1.0, born="probe"):
            if cap and len(committee.members) >= cap:
                return
            route_name = re.sub(r"[^a-z0-9_]+", "_", str(name or "").lower()).strip("_")[:64]
            if not route_name or route_name in added:
                return
            signed = _signed_direction(direction, sign)
            if signed is None or (isinstance(signed, dict) and not signed):
                return
            committee.register(MesaObjective(route_name, signed, born, lambda state: 1.0))
            added.add(route_name)

        probe_terms = []
        route_probe_weight = _sane_fraction(
            tuner.get("routing_probe_weight", getattr(config, "routing_probe_weight", 1.0)),
            getattr(config, "routing_probe_weight", 1.0),
        )
        ranked = rank_probes(probes, tuner)
        for row in ranked:
            lift = row.get("lift")
            if lift is None or float(row.get("priority") or 0.0) <= 0.0:
                continue
            pname = row["name"]
            pdata = probes.get(pname) or {}
            direction = pdata.get("direction")
            if direction is None or (isinstance(direction, dict) and not direction):
                continue
            evidence = min(1.0, max(0.0, float(row.get("n") or 0.0) / 20.0))
            if evidence <= 0.0:
                continue
            # Route scores are lower-is-better. Positive lift means "more of
            # this probe has tracked better turns", so high projection earns a
            # negative score term. Negative lift reverses that relationship.
            term_sign = -1.0 if float(lift) >= 0.0 else 1.0
            probe_terms.append({
                "name": pname,
                "direction": direction,
                "weight": term_sign * route_probe_weight * evidence,
                "lift": float(lift),
                "priority": float(row.get("priority") or 0.0),
                "n": int(row.get("n") or 0),
            })

        mix = prioritize_pin.get("mix")
        pin = prioritize_pin.get("probe")
        if mix:
            md = build_priority_mix_direction(model, mix, probes, tuner)
            if md:
                add_route("mix_" + "_".join(mix), md, born="steer_mix")
        elif pin and pin in probes and not probe_is_released(probes[pin]):
            add_route(
                f"probe_{pin}",
                probes[pin].get("direction"),
                sign=float(prioritize_pin.get("sign", 1.0) or 1.0),
                born="steer_pin",
            )

        for row in ranked:
            if cap and len(committee.members) >= cap:
                break
            if float(row.get("priority") or 0.0) <= 0:
                continue
            pname = row["name"]
            if pname not in probes:
                continue
            lift = row.get("lift")
            sign = 1.0 if lift is None or float(lift) >= 0 else -1.0
            add_route(f"probe_{pname}", probes[pname].get("direction"), sign=sign, born="calibrated_probe")

        for pname in sorted(
            n for n, pdata in probes.items()
            if pdata.get("exposed") and not probe_is_released(pdata)
        ):
            if cap and len(committee.members) >= cap:
                break
            add_route(f"probe_{pname}", probes[pname].get("direction"), born="exposed_probe")

        config.committee = committee
        config.use_expert_vectors = False
        config.routing_probe_terms = probe_terms
        config._live_tot_routes = [m.name for m in committee.members]
        return config._live_tot_routes

    egg_state = {"over": False}  # rising-edge latch for the consciousness egg
    startup_user_input = os.environ.get("PHENOMENALITY_STARTUP_PROMPT")
    if os.environ.get("PHENOMENALITY_AUTO_RESUME", "0").strip().lower() in {"1", "true", "yes"}:
        resumed_session, recovered = recover_session_context(
            memory,
            session_id=os.environ.get("PHENOMENALITY_RESUME_SESSION", "last"),
            max_turns=MAX_SESSION_TURNS,
        )
        if recovered and recovered[-1][0] == "user":
            session_context = recovered_to_session_context(recovered[:-1])
            startup_user_input = recovered[-1][1]
            memory.append_event(
                "context_auto_resumed",
                tags=["memory_tool", "context"],
                provenance={
                    "resumed_session_id": resumed_session,
                    "context_messages": len(session_context),
                    "startup_user_chars": len(startup_user_input),
                },
            )
            print(
                Fore.GREEN
                + (
                    f"[Context] Auto-resuming interrupted session {resumed_session}. "
                    "The first model turn will answer the saved unanswered message."
                )
                + Style.RESET_ALL
            )
    
    input_queue = []
    if startup_user_input:
        input_queue.append(startup_user_input)

    _shell_vars = {}
    _inline_capture = None
    _saved_stdout = sys.stdout
    _target_restore = None

    while True:
        try:
            if _target_restore is not None:
                _agent = _target_restore.get("agent")
                if _agent is not None:
                    _agent.tuner = tuner
                    _agent.probes = probes
                    _agent.tuner_bindings = tuner_bindings
                    _agent.prioritize_pin = prioritize_pin
                    _agent.queued_calibrations = queued_calibrations
                    _agent.queued_commands = queued_commands
                    _agent.doc_session = doc_session
                    _agent.doc_library = doc_library
                    _agent.pending_document_tool_result = pending_document_tool_result
                    _agent.doc_autoread = doc_autoread
                tuner = _target_restore["tuner"]
                probes = _target_restore["probes"]
                tuner_bindings = _target_restore["tuner_bindings"]
                prioritize_pin = _target_restore["prioritize_pin"]
                queued_calibrations = _target_restore["queued_calibrations"]
                queued_commands = _target_restore["queued_commands"]
                doc_session = _target_restore["doc_session"]
                doc_library = _target_restore["doc_library"]
                pending_document_tool_result = _target_restore["pending_document_tool_result"]
                doc_autoread = _target_restore["doc_autoread"]
                _target_restore = None

            if not input_queue and queued_commands:
                input_queue.append((queued_commands.pop(0), True))
                continue
            if not input_queue:
                for _qa in [a for a in joined_agents] + ([replace_agent] if replace_agent else []):
                    if _qa and getattr(_qa, "queued_commands", None):
                        input_queue.append((f"@{_qa.name} {_qa.queued_commands.pop(0)}", True))
                        break
                if input_queue:
                    continue

            if _inline_capture is not None:
                if len(input_queue) <= _inline_capture["queue_len"]:
                    sys.stdout = _saved_stdout
                    out_text = _inline_capture["buf"].getvalue().strip()
                    if not out_text:
                        out_text = "..."
                    out_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', out_text)
                    new_input = _inline_capture["prefix"] + out_text + _inline_capture["suffix"]
                    input_queue.insert(0, (new_input, True))
                    _inline_capture = None
                    continue

            reading_turn_source = None  # set when the turn is the model reading, not the operator speaking
            silent_echo = False
            if input_queue:
                q_item = input_queue.pop(0)
                if isinstance(q_item, tuple):
                    user_input, silent_echo = q_item
                else:
                    user_input, silent_echo = q_item, False
                if not silent_echo:
                    print(Fore.YELLOW + f"\nYou: {user_input}" + Style.RESET_ALL)
            elif doc_autoread and doc_autoread["remaining"] > 0:
                # Reading as dialogue, minimally: the framed chunk IS the user
                # turn -- no synthesized question, no cue, no words put in
                # anyone's mouth. The model responds to the text natively, and
                # the turn is sense-labeled, channel-accounted, and remembered
                # like any other conversation turn.
                staged_reading = pending_document_tool_result  # consume a manual :doc stage first
                pending_document_tool_result = None
                if staged_reading is None:
                    earlier_reflections = " ".join(list(recent_responses)[:-1])
                    pick = _select_autoread_chunk(
                        doc_library,
                        doc_autoread,
                        last_assistant_response,
                        earlier_thoughts=earlier_reflections,
                    )
                    if pick is None:
                        print(Fore.CYAN + "[Doc] Library fully read; auto-read finished." + Style.RESET_ALL)
                        doc_autoread = None
                        continue
                    doc_session = doc_library[pick["session_index"]]
                    doc_session["cursor"] = pick["chunk_index"]
                    doc_session.setdefault("read", set()).add(pick["chunk_index"])
                    record_chunk_read(memory, doc_session, pick["chunk_index"])
                    if pick["mode"] in ("reply", "reply_thread"):
                        # Only the echo-following mode is word-contingent; the
                        # order/interleave course is deliberately not.
                        impact_state["pending"].append(
                            {
                                "cause": "reply steered the reading",
                                "effect": f"next chunk chosen by shared ground: {', '.join(pick['overlap'])}",
                            }
                        )
                    staged_reading = stage_chunk(
                        doc_session,
                        index=pick["chunk_index"],
                        reply_note=reading_reply_note(pick),
                    )
                    picked_note = (
                        f"chunk {pick['chunk_index'] + 1}/{doc_session['chunk_count']} of "
                        f"{doc_session['source_name']} ({pick['mode']}"
                        + (f": {', '.join(pick['overlap'])}" if pick.get("overlap") else "")
                        + ")"
                    )
                    print(Fore.CYAN + f"[Doc] Reading {picked_note}." + Style.RESET_ALL)
                doc_autoread["remaining"] -= 1
                if doc_autoread["remaining"] <= 0:
                    doc_autoread = None
                user_input = staged_reading
                # A reading turn is the model's own act -- the frame literally
                # begins "I'm reading ..." -- so the console says "Me:", not
                # "You:". The operator didn't speak; it is reading to itself.
                reading_turn_source = doc_session["source_name"] if doc_session else "reading"
                preview = " ".join(staged_reading.split())[:160]
                print(Fore.CYAN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL + preview + " [...]")

            elif replace_turns_remaining > 0 and replace_agent:
                # Agent replaces user
                print(Fore.MAGENTA + f"\n[{replace_agent.name} is thinking...]" + Style.RESET_ALL)
                _sys_tuner = tuner
                _sys_probes = probes
                _sys_tb = tuner_bindings
                _sys_pin = prioritize_pin
                _sys_queue = queued_calibrations
                _sys_cmd_queue = queued_commands
                _sys_doc_session = doc_session
                _sys_doc_library = doc_library
                _sys_pending_doc = pending_document_tool_result
                _sys_doc_autoread = doc_autoread
                tuner = replace_agent.tuner
                probes = replace_agent.probes
                tuner_bindings = replace_agent.tuner_bindings
                prioritize_pin = replace_agent.prioritize_pin
                queued_calibrations = replace_agent.queued_calibrations
                queued_commands = replace_agent.queued_commands
                doc_session = replace_agent.doc_session
                doc_library = replace_agent.doc_library
                pending_document_tool_result = replace_agent.pending_document_tool_result
                doc_autoread = replace_agent.doc_autoread

                agent_document_tool_result = pending_document_tool_result
                pending_document_tool_result = None
                if agent_document_tool_result is None and doc_autoread and doc_autoread.get("remaining", 0) > 0:
                    pick = _select_autoread_chunk(doc_library, doc_autoread)
                    if pick is not None:
                        doc_session = doc_library[pick["session_index"]]
                        doc_session["cursor"] = pick["chunk_index"]
                        doc_session.setdefault("read", set()).add(pick["chunk_index"])
                        record_chunk_read(memory, doc_session, pick["chunk_index"])
                        agent_document_tool_result = stage_chunk(doc_session, index=pick["chunk_index"])
                        doc_autoread["remaining"] -= 1
                        if doc_autoread["remaining"] <= 0:
                            doc_autoread = None

                # Generate user input autonomously
                agent_state_report = _spawn_state_report(replace_agent.name, tuner, probes)
                r_prompt = _build_live_prompt(
                    agent_state_report + "\n\n[Your turn to respond]",
                    document_tool_result=agent_document_tool_result,
                    session_context=session_context if session_context_enabled else None,
                    field_context=None,  # already present in agent_state_report
                )
                _r_handles, _r_prio = _priority_steer_handles(probes, tuner, prioritize_pin)
                try:
                    r_response = generate_agentic_text(
                        model,
                        instruction=r_prompt,
                        config=config,
                        max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                        synthesis_recorder=None,
                        chatty_log=False,
                        pre_formatted=True,
                        return_telemetry=False,
                    )
                finally:
                    for _h in _r_handles:
                        try:
                            _h.remove()
                        except Exception:
                            pass
                user_input = r_response.strip() if r_response else "..."

                # Restore
                replace_agent.tuner = tuner
                replace_agent.probes = probes
                replace_agent.tuner_bindings = tuner_bindings
                replace_agent.prioritize_pin = prioritize_pin
                replace_agent.queued_calibrations = queued_calibrations
                replace_agent.queued_commands = queued_commands
                replace_agent.doc_session = doc_session
                replace_agent.doc_library = doc_library
                replace_agent.pending_document_tool_result = pending_document_tool_result
                replace_agent.doc_autoread = doc_autoread
                tuner = _sys_tuner
                probes = _sys_probes
                tuner_bindings = _sys_tb
                prioritize_pin = _sys_pin
                queued_calibrations = _sys_queue
                queued_commands = _sys_cmd_queue
                doc_session = _sys_doc_session
                doc_library = _sys_doc_library
                pending_document_tool_result = _sys_pending_doc
                doc_autoread = _sys_doc_autoread
                replace_turns_remaining -= 1
                _last_replace_name = replace_agent.name
                if replace_turns_remaining <= 0:
                    replace_agent = None
                print(Fore.MAGENTA + Style.BRIGHT + f"\n[{_last_replace_name} replacing You]: " + Style.RESET_ALL + user_input)
            else:
                _last_replace_name = None
                if getattr(config, "auto_probe_readings", False) and _sys_probes:
                    recent = []
                    for row in reversed(list(turn_log)):
                        v = {}
                        for p in _sys_probes:
                            if f"probe_{p}" in row:
                                try:
                                    v[p] = float(row[f"probe_{p}"])
                                except (TypeError, ValueError):
                                    pass
                        if v:
                            recent.append(v)
                            break
                    if recent:
                        print(Fore.CYAN + "[Auto Probe Readings]" + Style.RESET_ALL)
                        for pname, val in recent[0].items():
                            print(Fore.CYAN + f"  {pname}: {val:+.3f}" + Style.RESET_ALL)

                prefix = f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] You: " if show_timestamps else "\nYou: "
                prompt_str = Fore.MAGENTA + Style.BRIGHT + prefix + Style.RESET_ALL
                # Once the listener is active, every turn is served from its
                # queue (so mid-generation typing was never dropped); otherwise
                # the plain, unchanged input() path.
                user_input = listener.get(prompt_str) if listener.active else input(prompt_str)


            user_input = user_input.strip()

            if "{:" in user_input and "}" in user_input and _inline_capture is None:
                m = re.search(r"\{:(.*?)\}", user_input)
                if m:
                    inner_cmd = m.group(1).strip()
                    if not inner_cmd.startswith(":"):
                        inner_cmd = ":" + inner_cmd
                    prefix = user_input[:m.start()]
                    suffix = user_input[m.end():]
                    import io
                    buf = io.StringIO()
                    _inline_capture = {
                        "prefix": prefix,
                        "suffix": suffix,
                        "buf": buf,
                        "queue_len": len(input_queue)
                    }
                    input_queue.insert(0, (inner_cmd, True))
                    sys.stdout = buf
                    continue

            if _shell_vars:
                for vname, vval in _shell_vars.items():
                    # Safely replace $vname if it's not part of a larger word
                    user_input = re.sub(r"\$" + re.escape(vname) + r"\b", lambda _m, v=vval: v, user_input)
            if user_input.startswith(":") and not command_keeps_semicolons(user_input):
                cmds = _split_macro_commands(
                    user_input,
                    split_colon_commands=not command_takes_colon_command_arg(user_input),
                )
                if len(cmds) > 1:
                    user_input = cmds[0]
                    input_queue.extend([(c, silent_echo) for c in cmds[1:]])
                elif cmds and "\\:" in user_input and cmds[0] != user_input:
                    # a '\:' escape suppressed the split; keep the unescaped
                    # form so handlers see ':cmd' and not a literal backslash
                    user_input = cmds[0]

            # A trailing ' because <reason>' on any command is provenance, not an
            # argument: peel it off, log it, and hand the handler a clean line.
            command_because = None
            if user_input.startswith(":") or user_input.startswith("@"):
                user_input, command_because = split_because(user_input)
            # Criteria & metric per command: note every KNOWN command word
            # that runs; the next reply's credit block turns the set into
            # each command's used-vs-unused outcome stream. Unknown words
            # (typos) never mint a knob.
            if user_input.startswith(":"):
                _cw_m = re.match(r":([a-z0-9_]+)", user_input.lower())
                if _cw_m and (
                    f":{_cw_m.group(1)}" in KNOWN_COMMANDS
                    or _cw_m.group(1) in macro_aliases
                    or _cw_m.group(1) in exposed_commands
                ):
                    _cw, _, _ = ensure_command_knobs(tuner, _cw_m.group(1))
                    commands_since_reply.add(_cw)
            target_agent = None
            if user_input.startswith("@"):
                t_parts = user_input.split(maxsplit=1)
                if len(t_parts) == 2:
                    t_name = t_parts[0][1:]
                    user_input = t_parts[1]
                    # Find agent
                    for a in joined_agents:
                        if a.name == t_name:
                            target_agent = a
                            break
                    if not target_agent and replace_agent and replace_agent.name == t_name:
                        target_agent = replace_agent
                    if target_agent:
                        print(Fore.CYAN + f"[Target] Directing command to '{t_name}'..." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Target] Unknown agent '{t_name}'. Ignoring target." + Style.RESET_ALL)

            # Swap global state if target_agent
            if target_agent:
                _target_restore = {
                    "agent": target_agent,
                    "tuner": tuner,
                    "probes": probes,
                    "tuner_bindings": tuner_bindings,
                    "prioritize_pin": prioritize_pin,
                    "queued_calibrations": queued_calibrations,
                    "queued_commands": queued_commands,
                    "doc_session": doc_session,
                    "doc_library": doc_library,
                    "pending_document_tool_result": pending_document_tool_result,
                    "doc_autoread": doc_autoread,
                }
                tuner = target_agent.tuner
                probes = target_agent.probes
                tuner_bindings = target_agent.tuner_bindings
                prioritize_pin = target_agent.prioritize_pin
                queued_calibrations = target_agent.queued_calibrations
                queued_commands = target_agent.queued_commands
                doc_session = target_agent.doc_session
                doc_library = target_agent.doc_library
                pending_document_tool_result = target_agent.pending_document_tool_result
                doc_autoread = target_agent.doc_autoread

            if command_because:
                related = set()
                # Check for probes
                for p in probes:
                    match = re.search(r'(?<!\w)([+-]?)' + re.escape(p) + r'\b', command_because, re.IGNORECASE)
                    if match:
                        prefix = match.group(1) or ""
                        related.add(f"{prefix}probe:{p}")
                # Check for knobs (triggers)
                for k in tuner.triggers:
                    k_name = k[len("probe_"):] if k.startswith("probe_") else k
                    match = re.search(r'(?<!\w)([+-]?)' + re.escape(k_name) + r'\b', command_because, re.IGNORECASE)
                    if match:
                        prefix = match.group(1) or ""
                        related.add(f"{prefix}knob:{k_name}")
                        
                mem_prov = {"command": user_input.split()[0] if user_input else "", "because": command_because}
                if related:
                    mem_prov["related"] = sorted(list(related))
                    
                memory.append_event(
                    "command_because",
                    tags=["provenance"],
                    provenance=mem_prov,
                )
                noted_str = command_because
                if related:
                    noted_str += Fore.MAGENTA + f" (linked to {', '.join(sorted(list(related)))})" + Fore.BLUE
                print(Fore.BLUE + f"[Because] noted: {noted_str}" + Style.RESET_ALL)

            
            if user_input.startswith(":shell ") or user_input.startswith(":os "):
                import subprocess
                cmd = user_input.split(maxsplit=1)[1].strip()
                try:
                    res = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, timeout=30)
                    out = res.stdout.strip()
                    if not out and res.stderr:
                        out = res.stderr.strip()
                    print(out)
                except Exception as e:
                    print(Fore.RED + f"[Error running {cmd}: {e}]" + Style.RESET_ALL)
                continue

            if user_input.startswith(":self ") or user_input == ":self":
                sargs = user_input.split()[1:]
                if not sargs:
                    print(Fore.YELLOW + "[System] Usage: :self <name> | :self choose | :self save <name>" + Style.RESET_ALL)
                    continue
                if sargs[0] == "create":
                    if len(sargs) < 2:
                        print(Fore.YELLOW + "[System] Usage: :self create <name>" + Style.RESET_ALL)
                        continue
                    alias = sargs[1]
                    print(Fore.CYAN + f"[System] Asking model to invent a '{alias}' persona..." + Style.RESET_ALL)
                    existing_personas = ", ".join(sorted(macro_aliases.keys())) or "None"
                    active_probes = ", ".join(sorted(probes.keys())) or "None"
                    available_cmds = ", ".join(sorted([c for c in BUILTIN_COMMANDS if c]))
                    
                    because_ctx = ""
                    if command_because:
                        because_ctx = f"The operator provided the following rationale for this persona:\n{command_because}\nMake sure your generated macro commands strongly reflect this rationale.\n\n"
                        print(Fore.CYAN + "[Self] Passing your 'because' rationale to the model." + Style.RESET_ALL)

                    import dataclasses
                    valid_knobs = ", ".join(f.name for f in dataclasses.fields(AgenticConfig) if f.name not in ("disabled_steer_channels", "committee", "recent_corrections", "clarifying_questions"))

                    default_self_create = (
                        ":expect macro $alias\n"
                        ":probe persona_author I am defining a cognitive persona using shell macro commands || I am generating conversational text\n"
                        ":probe release persona_author off\n"
                        ":probe backfill persona_author\n"
                        ":steer persona_author 1.5\n"
                        "The operator wants to create a cognitive persona named '$alias'.\n"
                        f"{because_ctx}"
                        "Write a short shell macro (3 to 10 lines) using any mix of commands to define this persona.\n"
                        f"Available engine knobs you can :tune directly are: {valid_knobs}. "
                        "If you want to control a behavioral trait NOT in this list (like speed, creativity, or response length), DO NOT hallucinate a :tune command. Instead, define a cognitive sensor for it using :probe (e.g. `:probe patience I take time to think || I respond instantly`) and steer it (`:steer patience 1.5`).\n"
                        "Available system commands: $available_cmds\n"
                        "Existing personas you could invoke (by adding `:<name>`): $existing_personas\n"
                        "Currently active probes you could compose: $active_probes\n"
                        "Use the syntax `:probe <concept_name> <framing with it> || <framing without it>` to define new dimensions.\n"
                        "You can also use `:probe compose ...` to combine existing probes.\n"
                        "You MUST actively use commands like :tune, :steer, and :probe to tune the model's behavior.\n"
                        "You MUST include `:doc docs/COMMANDS.md` in your macro so the persona reads the live-generated shell reference.\n"
                        "You SHOULD also provide context or system instructions to the persona if appropriate (e.g., by using `:doc` to stage relevant files or `:memory use` for context). "
                        "Output ONLY the commands, one per line. Do not use markdown blocks or conversational text.\n"
                        ":steer persona_author 0\n"
                        ":probe release persona_author"
                    )
                    prompt = load_prompt(
                        "self_create",
                        default_self_create,
                        required_substring=":probe release persona_author",
                        alias=alias,
                        available_cmds=available_cmds,
                        existing_personas=existing_personas,
                        active_probes=active_probes
                    )
                    
                    print(Fore.CYAN + f"[System] Queuing phenomenological macro to invent a '{alias}' persona..." + Style.RESET_ALL)
                    queue_macro_text(prompt, input_queue)
                    continue

                elif sargs[0] == "save":
                    alias = sargs[1] if len(sargs) > 1 else "choose"
                    user_input = f":save self {alias}"
                elif sargs[0] == "choose":
                    print(Fore.CYAN + "[System] Asking model to pick a persona/macro..." + Style.RESET_ALL)
                    options = "\n".join(f"- {p}" for p in sorted(macro_aliases.keys()))
                    default_self_choose = (
                        ":expect autocomplete\n"
                        ":probe persona_selector I am selecting exactly ONE persona name from the list || I am generating conversational text\n"
                        ":probe release persona_selector off\n"
                        ":probe backfill persona_selector\n"
                        ":steer persona_selector 1.5\n"
                        "You are selecting a persona/macro to initialize.\n"
                        "Available options:\n$options\n\n"
                        "Select the single most appropriate persona/macro from the list. "
                        "Output ONLY the exact name, and nothing else.\n"
                        ":steer persona_selector 0\n"
                        ":probe release persona_selector"
                    )
                    prompt = load_prompt(
                        "self_choose",
                        default_self_choose,
                        required_substring=":probe release persona_selector",
                        options=options
                    )
                    queue_macro_text(prompt, input_queue)
                    continue
                else:
                    target_alias = sargs[0]
                    if target_alias not in macro_aliases and target_alias not in BUILTIN_COMMANDS:
                        print(Fore.YELLOW + f"[System] No persona/macro named '{target_alias}' found." + Style.RESET_ALL)
                        continue
                    user_input = f":{target_alias}"

            if user_input.startswith(":save self"):
                # Alias for :macro name self <name>
                sargs = user_input.split()
                if len(sargs) >= 3:
                    alias = sargs[2]
                    if alias == "choose":
                        print(Fore.CYAN + "[System] Asking model for a name..." + Style.RESET_ALL)
                        _name_field_context = format_field_prompt_context(tuner, probes)
                        nm = generate_agentic_text(
                            model,
                            instruction=(
                                "Pick a short, one-word name for this persona based on the current tuning state. "
                                "Reply with just the lowercase name."
                                + (f"\n\n{_name_field_context}" if _name_field_context else "")
                            ),
                            config=config,
                            max_new_tokens=10,
                            chatty_log=False,
                            pre_formatted=False,
                            system_prompt=""
                        )
                        alias = re.sub(r'[^a-z0-9]', '', (nm or "agent").lower())[:20]
                        if not alias: alias = "agent"
                        print(Fore.CYAN + f"[System] Model chose '{alias}'." + Style.RESET_ALL)
                    user_input = f":macro name self {alias}"
                else:
                    print(Fore.YELLOW + "[System] Usage: :save self <name> | :save self choose" + Style.RESET_ALL)
                    continue

            if user_input.startswith(":spawn"):
                sargs = user_input.split()
                if len(sargs) < 3:
                    print(Fore.YELLOW + "[System] Usage: :spawn <name> join | :spawn <name> replace [N] | :spawn <name> drop | :spawn <name> create" + Style.RESET_ALL)
                    continue
                a_name = sargs[1]
                mode = sargs[2].lower()
                
                if mode == "drop":
                    joined_agents = [a for a in joined_agents if a.name != a_name]
                    if replace_agent and replace_agent.name == a_name:
                        replace_agent = None
                        replace_turns_remaining = 0
                    print(Fore.CYAN + f"[Spawn] Dropped agent '{a_name}'." + Style.RESET_ALL)
                    continue
                
                # Create agent state. The tuner is isolated per spawned agent,
                # then seeded with the shell's current known knobs.
                new_agent = AgentState(name=a_name, tuner=_agent_tuner(a_name))
                # Load profile if macro exists
                macro_path = macro_aliases.get(a_name, a_name)
                if mode not in ("create", "generate") and (os.path.isfile(macro_path) or os.path.isfile(os.path.join(ROOT, "invariants", "out", "macros", f"{a_name}.txt"))):
                    real_path = macro_path if os.path.isfile(macro_path) else os.path.join(ROOT, "invariants", "out", "macros", f"{a_name}.txt")
                    try:
                        with open(real_path, "r", encoding="utf-8") as rf:
                            raw_lines = rf.read().splitlines()
                        lines, skipped = _spawn_profile_runnable_lines(raw_lines, new_agent.tuner)

                        if mode in ("join", "replace"):
                            # Queue profile commands to be executed against the new agent.
                            bundled = [(f"@{a_name} {cmd}", True) for cmd in lines]
                            input_queue[:0] = bundled
                        skip_note = f" Skipped {len(skipped)} invalid/non-profile line(s)." if skipped else ""
                        print(Fore.GREEN + f"[Spawn] Loaded {len(lines)} profile command(s) for '{a_name}' from {real_path}.{skip_note}" + Style.RESET_ALL)
                        if skipped:
                            for reason in skipped[:6]:
                                print(Fore.YELLOW + f"        - {reason}" + Style.RESET_ALL)
                            if len(skipped) > 6:
                                print(Fore.YELLOW + f"        - ...and {len(skipped) - 6} more" + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Spawn] Failed to load profile for '{a_name}': {e}" + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + f"[Spawn] No saved profile found for '{a_name}'. Generating profile based on name..." + Style.RESET_ALL)
                    print(Fore.CYAN + f"[Spawn] No saved profile found for '{a_name}'. Queuing phenomenological macro to generate one..." + Style.RESET_ALL)

                    because_ctx = ""
                    if command_because:
                        because_ctx = f"The operator provided the following rationale for this profile:\n{command_because}\nMake sure your generated profile strongly reflects this rationale.\n\n"
                        print(Fore.CYAN + "[Spawn] Passing your 'because' rationale to the model." + Style.RESET_ALL)

                    valid_knobs = ", ".join(sorted(k for k in tuner.triggers if not k.startswith("probe_") and k not in SPAWN_FAKE_KNOBS))
                    command_reference = render_visible_commands_md(hidden_commands, COMMAND_HELP_LINES)
                    spawn_state_report = _spawn_state_report(a_name, new_agent.tuner, new_agent.probes)

                    default_spawn_prompt = (
                        ":expect file invariants/out/macros/$a_name.txt\n"
                        "# SPAWN_PROFILE_V7_LINE_GRAMMAR\n"
                        ":probe profile_author I am generating a tuning profile composed of exact shell configuration commands || I am generating conversational text\n"
                        ":probe release profile_author off\n"
                        ":probe backfill profile_author\n"
                        ":steer profile_author 1.5\n"
                        "You are generating an executable spawn calibration profile for an agent named '$a_name'. "
                        f"{because_ctx}"
                        "The visible shell command reference is loaded below so you can reason from the actual writing surface, not guesses.\n"
                        "$command_reference\n\n"
                        "The live hook/evidence report is loaded below. It includes tunable hooks, active probes, macros, command exposure, documents, queued work, and the complete factual field context (families, qualities, formulas, sets, compliance/time vectors, and exclusions).\n"
                        "$spawn_state_report\n\n"
                        "Output ONLY runnable shell commands or macro invocations, one per line. "
                        "Profile line grammar is strict: a line beginning with ':' is executed as a command; "
                        "a line beginning with '#' is a stored comment and is not executed; any other nonblank "
                        "line may be interpreted as a user query/text turn. Therefore use '#' for comments and "
                        "never write prose/headings/bullets/markdown as bare lines. "
                        "The command reference is authoritative: use any known command/macro that serves the spawned agent's setup, context, calibration, or future workflow. "
                        "Use :hide/:hide off for documentation and writing visibility. Use :expose/:expose off for runtime tool access, e.g. @agent :expose :command [fixed args...] steer <magnitude-or-effect> because <activation criteria> grants a callable tool and @agent :hide off :command makes it visible for profile/macro writing. Tools are commands the model can trigger when their activation criterion is met; every tool should have an activation criterion and a steer/action magnitude when meaningful. If you expose a tool with fixed args, those args are already filled; the model supplies only the missing tail. Exposed probe values steer output through :tune exposed_probe_alpha <small>; sensed tools use their own *_alpha knobs. Do not rely on hidden spawn-loader filtering. "
                        "Use trailing because-clauses to record purpose for context or side-effect commands. "
                        "Use :doc when the spawned agent needs a document as working context; for example :doc readings because <purpose>, followed by :doc read/:doc next/:doc inject when useful. "
                        "Use :queue <future command> for commands that should run later when they become useful or when enough evidence exists; queued commands must still be runnable shell commands. "
                        "A direct :probe line MUST use this exact contrastive form: :probe <short_snake_name> <first-person WITH statement> || <first-person WITHOUT statement>. "
                        "Use :probe to define behavioral sensors for traits such as speed, care, calibration, caution, or compression; do not invent architecture knobs. "
                        "Known direct :tune targets are: $valid_knobs. "
                        "Do not output fake knobs such as embedding_size, hidden_layers, batch_size, learning_rate, optimizer, cpu_cycles_per_second, accuracy, precision, recall, or common_sense. "
                        "Prefer this shape when it fits: context/docs the agent needs, 3-6 :probe definitions, :probe backfill all 120, deliberate known-knob :tune lines, :steer or :steer mix, and calibration/queued follow-up commands. "
                        "Example shape, shown inline so it is NOT executed while this prompt is queued: "
                        "`:probe careful_compression I answer directly while preserving the important constraints. || I ramble or drop important constraints.`; "
                        "`:probe calibration_hunger I seek evidence before locking in a setting. || I assert settings without checking data.`; "
                        "`:probe backfill all 120`; `:doc readings because this agent needs the readings file as working context`; "
                        "`:doc read 2 order`; `:tune response_tokens 256`; "
                        "`:steer mix careful_compression calibration_hunger 0.08`; "
                        "`:calibrate careful_compression 50`; `:queue calibrate calibration_hunger 50`. "
                        "Now output the profile for '$a_name'.\n"
                        ":steer profile_author 0\n"
                        ":probe release profile_author"
                    )
                    g_prompt = load_prompt(
                        "spawn",
                        default_spawn_prompt,
                        required_substring=(
                            "# SPAWN_PROFILE_V7_LINE_GRAMMAR",
                            "Profile line grammar is strict",
                            "$valid_knobs",
                            "shown inline so it is NOT executed",
                        ),
                        a_name=a_name,
                        valid_knobs=valid_knobs,
                        command_reference=command_reference,
                        spawn_state_report=spawn_state_report,
                    )
                    queue_macro_text(g_prompt, input_queue)
                    print(Fore.YELLOW + f"        -> Run ':spawn {a_name} {mode}' again after generation completes." + Style.RESET_ALL)
                    continue
                if mode == "join":
                    if not any(a.name == a_name for a in joined_agents):
                        joined_agents.append(new_agent)
                    print(Fore.GREEN + f"[Spawn] Agent '{a_name}' joined the panel." + Style.RESET_ALL)
                elif mode == "replace":
                    n_turns = int(sargs[3]) if len(sargs) > 3 else 1
                    replace_agent = new_agent
                    replace_turns_remaining = n_turns
                    print(Fore.GREEN + f"[Spawn] Agent '{a_name}' is replacing the user for {n_turns} turn(s)." + Style.RESET_ALL)
                elif mode in ("generate", "create"):
                    print(Fore.CYAN + f"[Spawn] Profile for '{a_name}' generated and saved. It is ready for use." + Style.RESET_ALL)
                continue

            if user_input.startswith(":fix "):
                sargs = user_input.split(maxsplit=2)
                if len(sargs) < 3:
                    print(Fore.YELLOW + "[System] Usage: :fix <macro_name> <instructions...>" + Style.RESET_ALL)
                    continue
                a_name = sargs[1]
                instructions = sargs[2]

                macro_path = macro_aliases.get(a_name, a_name)
                real_path = macro_path if os.path.isfile(macro_path) else os.path.join(ROOT, "invariants", "out", "macros", f"{a_name}.txt")

                if not os.path.isfile(real_path):
                    print(Fore.YELLOW + f"[System] Could not find macro '{a_name}' at {real_path}." + Style.RESET_ALL)
                    continue

                with open(real_path, "r", encoding="utf-8") as rf:
                    existing_macro = rf.read()

                print(Fore.CYAN + f"[System] Queuing phenomenological macro to fix '{a_name}'..." + Style.RESET_ALL)

                because_ctx = ""
                if command_because:
                    because_ctx = f"The operator provided the following underlying reason/rationale for this modification:\n{command_because}\nMake sure your updated macro strongly reflects this rationale.\n\n"
                    print(Fore.CYAN + "[Fix] Passing your 'because' rationale to the model." + Style.RESET_ALL)

                fix_prompt = (
                    f":expect file {real_path}\n"
                    ":probe profile_author I am generating a tuning profile composed of exact shell configuration commands || I am generating conversational text\n"
                    ":probe release profile_author off\n"
                    ":probe backfill profile_author\n"
                    ":steer profile_author 1.5\n"
                    f"You are modifying the tuning profile for an agent named '{a_name}'.\n"
                    "Here is the existing profile:\n"
                    "```\n"
                    f"{existing_macro.strip()}\n"
                    "```\n"
                    f"{because_ctx}"
                    f"The operator has provided the following instruction to fix or improve it: \"{instructions}\"\n"
                    "Respond ONLY with runnable shell commands or macro invocations, one per line. "
                    "Profile line grammar is strict: ':' starts a command, '#' starts a non-executed comment, "
                    "and any other nonblank line may be interpreted as a user query/text turn. "
                    "Use any known command that serves the spawned agent's setup, context, calibration, or future workflow; include because-clauses where purpose matters. "
                    "Direct :probe definitions must contain ||. Do not use markdown formatting blocks around your response. Output no other text. Do not leave trailing colons or unfinished commands at the end.\n"
                    ":steer profile_author 0\n"
                    ":probe release profile_author"
                )
                queue_macro_text(fix_prompt, input_queue)
                print(Fore.YELLOW + f"        -> The fix is queued. It will overwrite {real_path}." + Style.RESET_ALL)
                continue

            if user_input == ":macro" or user_input.startswith(":macro "):
                mtail = user_input[len(":macro"):].strip()
                mtok = mtail.split()
                sub = mtok[0].lower() if mtok else ""
                if sub == "restore":
                    payload = mtail[len("restore"):].strip()
                    if not payload:
                        print(Fore.YELLOW + "[System] Usage: :macro restore {\"path\":\"...\",\"lines\":[...]}" + Style.RESET_ALL)
                        continue
                    try:
                        if payload.startswith("{"):
                            obj = json.loads(payload)
                            dest = str(obj.get("path", "")).strip()
                            body_lines = obj.get("lines", [])
                        else:
                            args = payload.split(maxsplit=1)
                            if len(args) < 2:
                                raise ValueError("missing path or JSON lines")
                            dest = args[0]
                            body_lines = json.loads(args[1])
                        if not dest or not isinstance(body_lines, list):
                            raise ValueError("restore payload needs a path and a list of lines")
                        if _hidden_overwrite_blocked(dest, "System"):
                            continue
                        out_path = dest if os.path.isabs(dest) else os.path.join(ROOT, dest)
                        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                        with open(out_path, "w", encoding="utf-8") as wf:
                            for line in body_lines:
                                wf.write(str(line) + "\n")
                        print(Fore.GREEN + f"[System] Restored macro file {dest} ({len(body_lines)} line(s))." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Error] Could not restore macro: {e}" + Style.RESET_ALL)
                    continue
                if sub == "install":
                    if len(mtok) < 2:
                        print(Fore.YELLOW + "[System] Usage: :macro install <alias> [file] [goal]" + Style.RESET_ALL)
                        continue
                    a_name = mtok[1].lower()
                    
                    if len(mtok) >= 3 and os.path.isfile(mtok[2]):
                        src_file = mtok[2]
                        goal_text = " ".join(mtok[3:]) if len(mtok) > 3 else "Manually installed macro."
                        args_to_scan = mtok[3:]
                    else:
                        src_file = macro_aliases.get(a_name, mtok[2] if len(mtok) >= 3 else f"{a_name}.txt")
                        goal_text = " ".join(mtok[2:]) if len(mtok) > 2 else "Manually installed macro."
                        args_to_scan = mtok[2:]

                    if not os.path.isfile(src_file):
                        print(Fore.RED + f"[Error] File not found: {src_file}. Please provide the file path." + Style.RESET_ALL)
                        continue
                    if _hidden_overwrite_blocked(a_name, "System"):
                        continue
                    arg_names = []
                    seen_args = set()
                    for tok in args_to_scan:
                        idx = tok.find("$")
                        if idx >= 0:
                            clean = tok[idx+1:].rstrip(".,;:\"'()!]")
                            if clean and clean not in seen_args:
                                arg_names.append(clean)
                                seen_args.add(clean)
                    dest = os.path.join(ROOT, "invariants", "out", "macros", f"{a_name}.txt")
                    try:
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with open(src_file, "r", encoding="utf-8") as rf:
                            content = rf.read().strip()
                        with open(dest, "w", encoding="utf-8") as wf:
                            wf.write(f"# :solve macro '{a_name}' -- {goal_text}\n")
                            if arg_names:
                                wf.write(f"# args: {', '.join(arg_names)}\n")
                            if content:
                                wf.write(content + "\n")
                        macro_aliases[a_name] = dest
                        _save_macro_aliases()
                        print(Fore.GREEN + f"[System] Installed macro ':{a_name}' to {dest}. Run it with :{a_name} <args>." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Error] Could not install macro: {e}" + Style.RESET_ALL)
                    continue
                if sub == "name":
                    kind = mtok[1].lower() if len(mtok) >= 2 else ""
                    if kind == "self":
                        # :macro name self [file] -- regenerate the CURRENT shell
                        # state as replayable commands: probes, macro-command
                        # files/aliases, exposed runtime tools, and game configs.
                        alias_name = mtail.split(maxsplit=2)[2].strip() if len(mtok) >= 3 else "self"
                        # If they provided a name, map it to the standard macros folder
                        if not alias_name.endswith(".txt") and not "/" in alias_name and not "\\" in alias_name:
                            dest = os.path.join(ROOT, "invariants", "out", "macros", f"{alias_name}.txt")
                        else:
                            dest = alias_name # user provided an explicit path
                            alias_name = os.path.splitext(os.path.basename(dest))[0]
                            
                        if _hidden_overwrite_blocked(alias_name, "System"):
                            continue
                        macro_lines, restore_stats = build_session_restore_macro(
                            probes,
                            macro_aliases,
                            exposed_commands,
                            exposed_knobs,
                            hidden_commands,
                            self_dest=dest,
                        )
                        n_cmds = sum(1 for l in macro_lines if l.startswith(":"))
                        try:
                            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                            with open(dest, "w", encoding="utf-8") as wf:
                                for l in macro_lines:
                                    wf.write(l + "\n")
                            macro_aliases["self"] = dest
                            macro_aliases[alias_name] = dest
                            _save_macro_aliases()
                            print(
                                Fore.GREEN
                                + f"[System] Wrote a state-regenerating macro ({n_cmds} command(s): "
                                + f"{restore_stats['probes']} probe(s), {restore_stats['macro_files']} macro file(s), "
                                + f"{restore_stats['macro_aliases']} macro alias(es), {restore_stats['hidden_commands']} hidden command(s), "
                                + f"{restore_stats['exposed_commands']} exposed command(s), "
                                + f"{restore_stats['exposed_knobs']} exposed knob(s), "
                                + f"{restore_stats['games']} game config(s)) to {dest}, aliased 'self'. Recreate with :run self."
                                + Style.RESET_ALL
                            )
                            if not probes:
                                print(Fore.YELLOW + "[System] (No active probes to regenerate -- the macro is empty.)" + Style.RESET_ALL)
                            if restore_stats["skipped"]:
                                print(Fore.YELLOW + f"[System] Skipped {len(restore_stats['skipped'])} restore item(s); see comments in the macro." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Could not write self macro: {e}" + Style.RESET_ALL)
                        continue
                    if kind == "strip":
                        # :macro name strip <src> [dest] -- write a STRIPPED COPY of
                        # <src> to a NEW file, leaving the original intact (vs :macro
                        # strip, which rewrites in place). Default dest: '<name>_og'
                        # -> '<name>', otherwise '<name>_stripped'.
                        args = mtail.split(maxsplit=2)[2].split() if len(mtok) >= 3 else []
                        if not args:
                            print(Fore.YELLOW + "[System] Usage: :macro name strip <src> [dest]" + Style.RESET_ALL)
                            continue
                        src = macro_aliases.get(args[0], args[0])
                        if not os.path.isfile(src):
                            print(Fore.YELLOW + f"[System] Macro not found: {src}" + Style.RESET_ALL)
                            continue
                        if len(args) >= 2:
                            dest = args[1]
                        else:
                            _b, _ext = os.path.splitext(src)
                            dest = (_b[:-3] + _ext) if _b.endswith("_og") else (_b + "_stripped" + _ext)
                        try:
                            with open(src, "r", encoding="utf-8") as rf:
                                kept, removed = strip_macro_lines(rf.read().splitlines())
                            _stem = os.path.splitext(os.path.basename(dest))[0]
                            if _hidden_overwrite_blocked(_stem, "System"):
                                continue
                            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                            with open(dest, "w", encoding="utf-8") as wf:
                                for line in kept:
                                    wf.write(line + "\n")
                            macro_aliases[_stem] = dest
                            _save_macro_aliases()
                            cmd_kept = [k for k in kept if not k.strip().startswith("#")]
                            print(Fore.GREEN + f"[System] Stripped copy of {src} -> {dest} (kept {len(cmd_kept)} command(s), dropped {len(removed)}); original untouched, aliased '{_stem}'." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Could not strip-copy macro: {e}" + Style.RESET_ALL)
                        continue
                    # :macro name <alias> <file> -- bind a short name to a macro file.
                    if len(mtok) < 3:
                        print(Fore.YELLOW + "[System] Usage: :macro name <alias> <file>  |  :macro name self [file]  |  :macro name strip <src> [dest]" + Style.RESET_ALL)
                    else:
                        alias = mtok[1]
                        mpath = mtail.split(maxsplit=2)[2].strip()
                        if _hidden_overwrite_blocked(alias, "System"):
                            continue
                        macro_aliases[alias] = mpath
                        _save_macro_aliases()
                        exists_note = "" if os.path.isfile(mpath) else " (file does not exist yet)"
                        print(Fore.GREEN + f"[System] Macro alias '{alias}' -> {mpath}{exists_note}. Use :run {alias} or :macro strip {alias}." + Style.RESET_ALL)
                    continue
                if sub == "strip":
                    # :macro strip <alias|file> -- rewrite IN PLACE, dropping every
                    # line that does not mutate a probe or a tuning knob.
                    target = mtail.split(maxsplit=1)[1].strip() if len(mtok) >= 2 else ""
                    mpath = macro_aliases.get(target, target)
                    if not target:
                        print(Fore.YELLOW + "[System] Usage: :macro strip <alias|file>  (or :macro name strip <src> [dest] to keep the original)" + Style.RESET_ALL)
                    elif _hidden_overwrite_blocked(target, "System"):
                        continue
                    elif not os.path.isfile(mpath):
                        print(Fore.YELLOW + f"[System] Macro not found: {mpath}" + Style.RESET_ALL)
                    else:
                        try:
                            with open(mpath, "r", encoding="utf-8") as rf:
                                kept, removed = strip_macro_lines(rf.read().splitlines())
                            with open(mpath, "w", encoding="utf-8") as wf:
                                for line in kept:
                                    wf.write(line + "\n")
                            cmd_kept = [k for k in kept if not k.strip().startswith("#")]
                            print(Fore.GREEN + f"[System] Stripped {mpath}: kept {len(cmd_kept)} probe/tuning command(s), removed {len(removed)} display-only line(s)." + Style.RESET_ALL)
                            for r in removed[:20]:
                                print(Fore.CYAN + f"  - {r}" + Style.RESET_ALL)
                            if len(removed) > 20:
                                print(Fore.CYAN + f"  ... and {len(removed) - 20} more." + Style.RESET_ALL)
                            if not cmd_kept:
                                print(Fore.YELLOW + "[System] Nothing probe/tuning-related remained -- the macro is now empty of commands." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Could not strip macro: {e}" + Style.RESET_ALL)
                    continue
                if sub == "drop":
                    target = mtail.split(maxsplit=1)[1].strip() if len(mtok) >= 2 else ""
                    if not target:
                        print(Fore.YELLOW + "[System] Usage: :macro drop <alias>" + Style.RESET_ALL)
                        continue
                    if target in macro_aliases:
                        mpath = macro_aliases[target]
                        del macro_aliases[target]
                        _save_macro_aliases()
                        try:
                            if os.path.isfile(mpath):
                                os.remove(mpath)
                                print(Fore.GREEN + f"[System] Dropped macro alias '{target}' and deleted its file ({mpath})." + Style.RESET_ALL)
                            else:
                                print(Fore.GREEN + f"[System] Dropped macro alias '{target}'. (File {mpath} was already missing)." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Dropped alias '{target}', but could not delete file {mpath}: {e}" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[System] Unknown macro alias '{target}'." + Style.RESET_ALL)
                    continue
                # Legacy create form: :macro <file> <cmd1> ; <cmd2> ; ...
                parts = mtail.split(maxsplit=1)
                if len(parts) < 2:
                    print(Fore.YELLOW + "[System] Usage: :macro <file> <c1> ; <c2> ...  |  :macro name <alias> <file>  |  :macro strip <alias|file>  |  :macro drop <alias>" + Style.RESET_ALL)
                else:
                    raw_file = parts[0]
                    if _hidden_overwrite_blocked(raw_file, "System"):
                        continue
                    mac_file = macro_aliases.get(raw_file, raw_file)
                    mac_cmds = _split_macro_commands(parts[1])
                    try:
                        os.makedirs(os.path.dirname(mac_file) or ".", exist_ok=True)
                        with open(mac_file, "w", encoding="utf-8") as wf:
                            for c in mac_cmds:
                                wf.write(c + "\n")
                        print(Fore.GREEN + f"[System] Wrote {len(mac_cmds)} command(s) to macro '{raw_file}' ({mac_file}). Execute with: :run {raw_file}" + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Error] Could not write macro file: {e}" + Style.RESET_ALL)
                continue

            if user_input.startswith(":run "):
                raw_target = user_input[len(":run "):].strip()
                if not raw_target:
                    continue
                if raw_target in macro_aliases or os.path.isfile(raw_target):
                    target = raw_target
                    args = []
                else:
                    parts = raw_target.split()
                    target = parts[0]
                    args = parts[1:]
                
                resolved = macro_aliases.get(target, target)
                # Glob target (fun*, *.txt): run every matching macro file.
                if any(c in target for c in "*?[") and not os.path.isfile(resolved):
                    matches = sorted(glob.glob(resolved))
                    if not matches:
                        print(Fore.YELLOW + f"[System] No macros match '{target}'." + Style.RESET_ALL)
                        continue
                    total = []
                    for m in matches:
                        try:
                            sub = expand_macro_lines(m, macro_aliases, args=args)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Could not read macro '{m}': {e}" + Style.RESET_ALL)
                            sub = None
                        if sub:
                            total.extend(sub)
                    if total:
                        names = ", ".join(os.path.basename(x) for x in matches)
                        print(Fore.CYAN + f"[System] Queued {len(total)} command(s) from {len(matches)} macro(s) matching '{target}' ({names})." + Style.RESET_ALL)
                        input_queue.extend([(cmd, True) for cmd in total])
                    else:
                        print(Fore.YELLOW + f"[System] Macros matching '{target}' had no runnable commands." + Style.RESET_ALL)
                    continue
                try:
                    lines = expand_macro_lines(target, macro_aliases, args=args)
                except Exception as e:
                    print(Fore.RED + f"[Error] Could not read macro '{target}': {e}" + Style.RESET_ALL)
                    continue
                if lines is None:
                    print(Fore.YELLOW + f"[System] Macro not found: {resolved}" + Style.RESET_ALL)
                elif lines:
                    print(Fore.CYAN + f"[System] Queued {len(lines)} command(s) from {target}." + Style.RESET_ALL)
                    input_queue.extend([(cmd, True) for cmd in lines])
                else:
                    print(Fore.YELLOW + f"[System] '{target}' had no runnable commands." + Style.RESET_ALL)
                continue

            # :solve <name> [goal] -- have the model WRITE a parameterized macro
            # (a list of ':' commands using $1..$9 / $@), then stage it for
            # :accept so ':<name> args' can run it directly.
            if user_input.startswith(":solve"):
                sbody = user_input[len(":solve"):].strip()
                sparts = sbody.split(maxsplit=1)
                is_dynamic = False
                if sparts and sparts[0].lower() == "dynamic":
                    is_dynamic = True
                    sbody = sparts[1].strip() if len(sparts) > 1 else ""
                    sparts = sbody.split(maxsplit=1)
                if not sparts:
                    print(Fore.YELLOW + "[Solve] Usage: :solve [dynamic] <command_name> [what it should do]" + Style.RESET_ALL)
                    continue
                sname = re.sub(r"[^a-z0-9_]", "_", sparts[0].lower())[:40].strip("_")
                rest = sparts[1].strip() if len(sparts) > 1 else ""
                try:
                    import shlex
                    tokens = shlex.split(rest)
                except ValueError:
                    tokens = rest.split()
                    
                arg_names = []
                while tokens and (
                    tokens[-1].startswith(("+", "-", "$", "["))
                    or tokens[-1].endswith(("?", "!", "]", "!!"))
                    or "=" in tokens[-1]
                ):
                    arg_names.insert(0, tokens.pop())
                goal = " ".join(tokens) or sname.replace("_", " ")

                if is_dynamic:
                    text_to_scan = goal + " " + (command_because or "")
                    seen_args = set(arg_names)
                    for tok in text_to_scan.split():
                        idx = tok.find("$")
                        if idx >= 0:
                            clean = tok[idx+1:].rstrip(".,;:\"'()!]")
                            if clean and clean not in seen_args:
                                arg_names.append(clean)
                                seen_args.add(clean)

                if sname == "auto" and not arg_names and goal == "auto":
                    print(Fore.YELLOW + "[Solve] Usage: :solve auto <+/- reference probes to steer toward>" + Style.RESET_ALL)
                    continue

                if sname == "choose" and goal == "choose":
                    print(Fore.YELLOW + "[Solve] You must describe what the macro should do for the model to choose a name (e.g. ':solve choose drop all probes')." + Style.RESET_ALL)
                    continue

                if sname == "auto":
                    refs = " ".join(arg_names) if arg_names else goal
                    if refs == "auto":
                        refs = ""
                    goal = f"steer the model using these reference probes: {refs}"
                    arg_names = []
                    sname = "choose"

                if sname == "choose":
                    print(Fore.CYAN + "[Solve] Asking model for a command name..." + Style.RESET_ALL)
                    nm = generate_agentic_text(
                        model,
                        instruction=f"Pick a short, one-word command name for a macro that does: {goal}. Reply with ONLY the lowercase name.",
                        config=config,
                        max_new_tokens=10,
                        chatty_log=False,
                        pre_formatted=False
                    )
                    sname = re.sub(r'[^a-z0-9_]', '', (nm or "macro").lower())[:40].strip("_")
                    if not sname or sname in BUILTIN_COMMANDS or sname in hidden_commands:
                        sname = "macro"
                    print(Fore.CYAN + f"[Solve] Model chose '{sname}'." + Style.RESET_ALL)
                
                if _hidden_overwrite_blocked(sname, "Solve"):
                    continue
                if not sname or sname in BUILTIN_COMMANDS:
                    print(Fore.YELLOW + f"[Solve] '{sname}' can't be a macro name (empty or a built-in command)." + Style.RESET_ALL)
                    continue

                prompt_args_str = ""
                if arg_names:
                    mapping_parts = []
                    clean_names = []
                    arg_specs = []
                    for i, arg in enumerate(arg_names):
                        spec = parse_macro_arg_spec(arg, fallback=f"arg{i + 1}")
                        clean_arg = spec["name"]
                        clean_names.append(clean_arg)
                        arg_specs.append(spec)
                        
                        hints = []
                        if spec["hint_prefix"] == "+":
                            hints.append("positive/additive")
                        elif spec["hint_prefix"] == "-":
                            hints.append("negative/subtractive")
                            
                        if clean_arg.endswith("s"):
                            hints.append("a list of multiple items")
                        elif "name" in clean_arg:
                            hints.append("a specific exact name")

                        if spec["optional"]:
                            hints.append("optional")
                        if spec["default"] is not None:
                            hints.append(f"default {spec['default']!r}")
                            
                        if spec["no_auto_or_choose"]:
                            hints.append("STRICT: cannot accept auto or choose")
                        elif spec["no_choose"]:
                            hints.append("cannot accept choose (but auto is okay)")
                            
                        hint_str = f" ({', '.join(hints)})" if hints else ""
                        mapping_parts.append(f"${clean_arg}{hint_str}")
                        
                    mapping = ", ".join(mapping_parts)
                    prompt_args_str = (
                        f" Use {mapping}, and $@ for all supplied arguments. "
                        "Optional parameters may be omitted; defaults are supplied by the macro header."
                    )
                else:
                    clean_names = []
                    arg_specs = []
                    prompt_args_str = (
                        " Use $1, $2, ... for parameters and $@ for all of them. "
                        "If this macro requires parameters to be flexible, you MUST invent your own named parameters by making the VERY FIRST line of your response "
                        "a comment like `# args: target, amount` and then using `$target` and `$amount` in your code. "
                        "Optional named args use `name?` or `[name]`; defaults use `name=default` or `[name=default]`."
                    )

                existing_macros = []
                for m_alias, m_path in macro_aliases.items():
                    if m_alias == sname or m_alias in hidden_commands:
                        continue
                    if os.path.isfile(m_path):
                        try:
                            with open(m_path, "r", encoding="utf-8") as rf:
                                first_line = rf.readline().strip()
                                if first_line.startswith("#"):
                                    desc = first_line.lstrip("#").strip()
                                    prefix = f":solve macro '{m_alias}' -- "
                                    if desc.startswith(prefix):
                                        desc = desc[len(prefix):]
                                    existing_macros.append(f":{m_alias} - {desc}")
                                else:
                                    existing_macros.append(f":{m_alias}")
                        except Exception:
                            pass
                            
                macro_hints_str = ""
                if existing_macros:
                    macro_hints_str = (
                        "You can also invoke existing macros by writing ':<macro_name> [args]'. "
                        "Available macros include:\n" + "\n".join(f"  {m}" for m in existing_macros) + "\n\n"
                    )
                command_hints_str = (
                    "Here is the visible reference for the shell commands you can use while writing this macro:\n\n"
                    + render_visible_commands_md(hidden_commands, COMMAND_HELP_LINES)
                    + "\n\n"
                )

                # Context the operator staged with ':memory use' is folded into the
                # solve prompt so the macro reflects what they told it (consumed
                # here so it isn't also replayed to the next model turn).
                staged_ctx = ""
                if pending_memory_tool_result:
                    _ctx = pending_memory_tool_result.strip()
                    if len(_ctx) > 2000:
                        _ctx = _ctx[:2000] + " ...[truncated]"
                    staged_ctx = (
                        "The operator staged this context for you; let it shape the macro:\n"
                        + _ctx + "\n\n"
                    )
                    pending_memory_tool_result = None
                    print(Fore.CYAN + "[Solve] Folding in the memory you staged with :memory use." + Style.RESET_ALL)
                    
                because_ctx = ""
                if command_because:
                    because_ctx = f"The operator provided the following underlying reason/rationale for this macro:\n{command_because}\nMake sure your generated macro commands strongly reflect this rationale.\n\n"
                    print(Fore.CYAN + "[Solve] Passing your 'because' rationale to the model." + Style.RESET_ALL)
                default_solve_prompt = (
                    "# SOLVE_MACRO_V2_UTILITY_SAFE\n"
                    ":expect macro $sname\n"
                    ":probe macro_author I am writing a precise list of colon-prefixed shell commands || I am generating conversational text\n"
                    ":probe release macro_author off\n"
                    ":probe backfill macro_author\n"
                    ":steer macro_author 1.5\n"
                    ":probe solve_goal I am fulfilling the user's specific request: $goal || I am doing something else\n"
                    ":probe release solve_goal off\n"
                    ":probe backfill solve_goal\n"
                    ":steer solve_goal 1.5\n"
                    "You write macros for an interactive cognition shell. A macro is a list of ':' "
                    "commands, one per line.$prompt_args_str\n\n"
                    "${command_hints_str}\n"
                    "Note: Shell commands natively accept 'auto' or 'choose' as arguments where applicable "
                    "to automatically select or interactively prompt for a value. "
                    "You should seamlessly pass these through to the underlying commands if the user provides them, "
                    "unless a parameter is explicitly restricted from doing so.\n\n"
                    "${macro_hints_str}\n"
                    "${staged_ctx}"
                    "${because_ctx}"
                    "Write a macro named '${sname}' that does: ${goal}\n"
                    "Output ONLY the commands, one per line. Do not use markdown blocks or conversational text. "
                    "Do not wrap arguments in quotes unless they contain spaces. Do not leave trailing colons or unfinished commands at the end.\n"
                    ":steer macro_author 0\n"
                    ":probe release macro_author\n"
                    ":steer solve_goal 0\n"
                    ":probe release solve_goal"
                )
                prompt = load_prompt(
                    "solve", 
                    default_solve_prompt, 
                    required_substring=(
                        "# SOLVE_MACRO_V2_UTILITY_SAFE",
                        ":probe release solve_goal",
                        "If this macro requires parameters to be flexible",
                    ),
                    prompt_args_str=prompt_args_str,
                    command_hints_str=command_hints_str,
                    macro_hints_str=macro_hints_str,
                    staged_ctx=staged_ctx,
                    because_ctx=because_ctx,
                    sname=sname,
                    goal=goal
                )
                solve_expect_contexts[sname] = {
                    "goal": goal,
                    "arg_specs": arg_specs,
                    "clean_names": clean_names,
                }
                print(Fore.CYAN + f"[Solve] Queuing phenomenological macro 'solve.txt' to generate ':{sname}' for: {goal}" + Style.RESET_ALL)
                queue_macro_text(prompt, input_queue)
                continue

            # Direct parameterized-macro invocation: ':<name> args' runs a known
            # macro alias with $1..$9 / $@ substituted from args. Built-ins win, so
            # only a non-built-in alias is dispatched here.
            if user_input.startswith(":") and len(user_input) > 1 and not user_input[1].isspace():
                _itok = user_input[1:].split()
                _iname = _itok[0] if _itok else ""
                if _iname in macro_aliases and _iname not in BUILTIN_COMMANDS:
                    _iargs = _itok[1:]
                    _ilines = load_parameterized_macro(macro_aliases[_iname], _iargs)
                    if _ilines is None:
                        print(Fore.YELLOW + f"[Macro] ':{_iname}' -> {macro_aliases[_iname]} not found on disk." + Style.RESET_ALL)
                    elif _ilines:
                        print(Fore.CYAN + f"[Macro] :{_iname} ({len(_iargs)} arg(s)) -> queued {len(_ilines)} command(s)." + Style.RESET_ALL)
                        input_queue.extend(_ilines)
                    else:
                        print(Fore.YELLOW + f"[Macro] ':{_iname}' has no commands." + Style.RESET_ALL)
                    continue

            first_word = user_input.lower().split()[0] if user_input else ""
            if first_word in ['exit', 'quit', ':exit', ':quit']:
                parts = user_input.strip().split(maxsplit=1)
                reason = parts[1] if len(parts) > 1 else "operator_exit"
                memory.append_event("shell_closed", tags=["session"], provenance={"reason": reason})
                break
            if user_input.startswith(":timestamps"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] == "on":
                    show_timestamps = True
                    print(Fore.GREEN + "[System] Timestamps enabled." + Style.RESET_ALL)
                elif len(parts) > 1 and parts[1] == "off":
                    show_timestamps = False
                    print(Fore.GREEN + "[System] Timestamps disabled." + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + f"[System] Timestamps are currently {'on' if show_timestamps else 'off'}." + Style.RESET_ALL)
                continue

            if user_input.startswith(":listen"):
                arg = user_input[len(":listen"):].strip().lower()
                if arg == "off":
                    listen_state["drain"] = False
                    print(Fore.YELLOW + "[Listen] mid-thought interjection OFF. Input still queued (never dropped); typing during a reply lands as the next turn." + Style.RESET_ALL)
                elif arg in ("on", ""):
                    listener.start()
                    listen_state["drain"] = True
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + "[Listen] ON. Speak any time -- lines you type mid-reply are ingested at the next "
                        + "chunk seam and appended to its live stream. Whether it redirects or folds them in "
                        + "is the model's own choice; nothing is force-stopped, nothing is dropped."
                        + Style.RESET_ALL
                    )
                elif arg == "status":
                    print(Fore.CYAN + f"[Listen] reader={'active' if listener.active else 'inactive'}, mid-thought drain={'on' if listen_state['drain'] else 'off'}." + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + "[Listen] Usage: :listen on|off|status." + Style.RESET_ALL)
                continue

            if user_input.startswith(":history"):
                print(
                    Fore.YELLOW
                    + "[History] Use :context for current-session transcript controls. Use :memory for long-term memory tools."
                    + Style.RESET_ALL
                )
                continue
            if user_input.startswith(":context"):
                raw_cmd = user_input.strip()
                cmd = raw_cmd.lower()
                if cmd == ":context off":
                    session_context_enabled = False
                    print(Fore.YELLOW + "[Context] Current-session transcript OFF. Long-term memory remains explicit." + Style.RESET_ALL)
                elif cmd == ":context on":
                    session_context_enabled = True
                    print(Fore.GREEN + "[Context] Current-session transcript ON." + Style.RESET_ALL)
                elif cmd == ":context clear":
                    session_context.clear()
                    print(Fore.YELLOW + "[Context] Cleared current-session transcript. Persistent memory was not changed." + Style.RESET_ALL)
                elif cmd.startswith(":context resume"):
                    parts = raw_cmd.split()
                    requested_session = parts[2] if len(parts) >= 3 else "last"
                    resumed_session, recovered = recover_session_context(
                        memory,
                        session_id=requested_session,
                        max_turns=MAX_SESSION_TURNS,
                    )
                    if not recovered:
                        print(Fore.YELLOW + "[Context] No saved session turns found to resume." + Style.RESET_ALL)
                    else:
                        session_context = recovered_to_session_context(recovered)
                        session_context_enabled = True
                        memory.append_event(
                            "context_resumed",
                            tags=["memory_tool", "context"],
                            provenance={
                                "resumed_session_id": resumed_session,
                                "messages": len(recovered),
                            },
                        )
                        print(Fore.CYAN + f"\n--- Resumed Session {resumed_session} ---" + Style.RESET_ALL)
                        for r_role, r_name, r_text, r_ts, *_ in recovered:
                            ts_prefix = f"[{r_ts.split('T')[1][:8]}] " if show_timestamps and "T" in r_ts else ""
                            name = "You: " if r_role == "user" else "Me: "
                            color = Fore.MAGENTA + Style.BRIGHT if r_role == "user" else Fore.GREEN + Style.BRIGHT
                            print(color + ts_prefix + name + Style.RESET_ALL + r_text)
                        print(Fore.CYAN + "--------------------------------------\n" + Style.RESET_ALL)
                        print(
                            Fore.GREEN
                            + (
                                f"[Context] Resumed {len(recovered)} saved messages from session "
                                f"{resumed_session}. Current-session transcript is ON."
                            )
                            + Style.RESET_ALL
                        )
                else:
                    print(
                        Fore.CYAN
                        + (
                            f"[Context] enabled={session_context_enabled}, stored_messages={len(session_context)}, "
                            f"max_turns={MAX_SESSION_TURNS}. Use :context resume [last|session_id] to restore a saved shell."
                        )
                        + Style.RESET_ALL
                    )
                continue
            if user_input.startswith(":memory"):
                cmd = user_input.strip()
                tail = cmd[len(":memory"):].strip()
                def _show_memory_records(records, desc, ranked=None):
                    if ranked:
                        print(Fore.CYAN + f"[Memory] {desc}:" + Style.RESET_ALL)
                        for _abs, raw, r in ranked:
                            prev = (r.text or "")[:70].replace("\n", " ")
                            print(Fore.CYAN + f"  {raw:+.3f}  [{r.kind}] {prev}..." + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"[Memory] {desc}: {len(records)} record(s)." + Style.RESET_ALL)
                    print(Fore.CYAN + memory.format_tool_result(records) + Style.RESET_ALL)

                def _stage_memory_records(records, desc, ranked=None):
                    nonlocal pending_memory_tool_result
                    if not records:
                        print(Fore.YELLOW + f"[Memory] No matching records for {desc}." + Style.RESET_ALL)
                        return
                    pending_memory_tool_result = memory.format_tool_result(records)
                    memory.append_event(
                        "memory_tool_staged",
                        tags=["memory_tool"],
                        provenance={"query": desc, "records": len(records)},
                    )
                    _show_memory_records(records, f"Staged {desc}", ranked=ranked)
                    print(Fore.YELLOW + "[Memory] This tool result will be provided to the next model turn only." + Style.RESET_ALL)

                def _bare_memory_lanes_or_warn(action):
                    lanes = _memory_lanes_for_bare_command()
                    if not lanes:
                        print(
                            Fore.YELLOW
                            + f"[Memory] No memory lanes are enabled for :memory {action}. "
                            + "Use :memory act on and/or :memory talk on, or call :memory act "
                            + f"{action} / :memory talk {action} explicitly."
                            + Style.RESET_ALL
                        )
                    return lanes

                if tail in ("", "status"):
                    print(Fore.CYAN + format_status(memory.status()) + Style.RESET_ALL)
                    lanes = ", ".join(f"{k}={'on' if v else 'off'}" for k, v in memory_lanes_enabled.items())
                    access = "available" if _memory_model_available() else "not available"
                    print(Fore.CYAN + f"         lanes: {lanes}; model <<MEMORY>>: {access} ({'exposed' if memory_tool_exposed else 'expose off'})" + Style.RESET_ALL)
                elif tail.startswith("recent"):
                    n = parse_count(tail, 4)
                    print(Fore.CYAN + memory.format_recent(max_turns=n) + Style.RESET_ALL)
                elif tail.split(maxsplit=1) and _memory_lane_name(tail.split(maxsplit=1)[0]):
                    parts = tail.split(maxsplit=2)
                    lane = _memory_lane_name(parts[0])
                    action = parts[1].lower() if len(parts) >= 2 else "status"
                    arg = parts[2].strip() if len(parts) >= 3 else ""
                    if action in ("on", "enable", "enabled"):
                        memory_lanes_enabled[lane] = True
                        print(Fore.GREEN + f"[Memory] {lane} memory ON." + Style.RESET_ALL)
                    elif action in ("off", "disable", "disabled"):
                        memory_lanes_enabled[lane] = False
                        print(Fore.YELLOW + f"[Memory] {lane} memory OFF." + Style.RESET_ALL)
                    elif action == "status":
                        print(Fore.CYAN + f"[Memory] {lane} memory is {'ON' if memory_lanes_enabled[lane] else 'OFF'}." + Style.RESET_ALL)
                    elif action in ("use", "stage"):
                        records, desc, ranked = _memory_select_records(arg, [lane])
                        _stage_memory_records(records, desc, ranked=ranked)
                    elif action == "search":
                        records, desc, ranked = _memory_select_records(arg, [lane])
                        _show_memory_records(records, desc, ranked=ranked)
                    else:
                        print(Fore.YELLOW + "[Memory] Usage: :memory act|talk on|off|status|use [sentence [+probe -probe]]|search [sentence [+probe -probe]]" + Style.RESET_ALL)
                elif tail == "search" or tail.startswith("search "):
                    query = tail[len("search"):].strip()
                    lanes = _bare_memory_lanes_or_warn("search")
                    if lanes:
                        records, desc, ranked = _memory_select_records(query, lanes)
                        _show_memory_records(records, desc, ranked=ranked)
                elif tail.startswith("use probe"):
                    lanes = _bare_memory_lanes_or_warn("use")
                    if lanes:
                        records, desc, ranked = _memory_select_records("probe " + tail[len("use probe"):].strip(), lanes)
                        _stage_memory_records(records, desc, ranked=ranked)
                elif tail.startswith("all use"):
                    records = memory.records[-10:]
                    if not records:
                        print(Fore.YELLOW + "[Memory] No recent memories found." + Style.RESET_ALL)
                        continue
                    _stage_memory_records(records, "all:recent")
                elif tail.startswith("use "):
                    query = tail[len("use "):].strip()
                    lanes = _bare_memory_lanes_or_warn("use")
                    if lanes:
                        records, desc, ranked = _memory_select_records(query, lanes)
                        _stage_memory_records(records, desc, ranked=ranked)
                elif tail == "use":
                    lanes = _bare_memory_lanes_or_warn("use")
                    if lanes:
                        records, desc, ranked = _memory_select_records("", lanes)
                        _stage_memory_records(records, desc, ranked=ranked)
                elif tail in ("boundary", "clear"):
                    memory.mark_session_boundary("operator_request")
                    pending_memory_tool_result = None
                    print(Fore.YELLOW + "[Memory] Session boundary marked. Persistent memory file was not deleted." + Style.RESET_ALL)
                elif tail.startswith("choice"):
                    query = tail[len("choice"):].strip()
                    if query.lower().startswith("probe"):
                        pname_raw = query[len("probe"):].strip()
                        pname = resolve_probe_choice(pname_raw, probes, model=model, config=config, action_name="memory ranking") if pname_raw else None
                        if not pname or pname not in probes:
                            print(Fore.YELLOW + f"[Memory] Unknown probe '{pname_raw}'. Active: {', '.join(sorted(probes)) or 'none'}." + Style.RESET_ALL)
                            continue
                        records = [t[2] for t in rank_memories_by_probe(model, memory, probes[pname]["direction"], top_n=15)]
                        choice_note = f" (ranked by probe '{pname}', furthest from 0)"
                    elif query:
                        records = memory.search(query, max_records=15, scope=memory.scope)
                        choice_note = f" (search: {query})"
                    else:
                        records = [r for r in memory.records if (r.text or "").strip()][-15:]
                        choice_note = " (most recent)"
                    if not records:
                        print(Fore.YELLOW + "[Memory] No records found to choose from." + Style.RESET_ALL)
                        continue
                    print(Fore.CYAN + f"[Memory] Select a record to stage for the next turn{choice_note}:" + Style.RESET_ALL)
                    for i, r in enumerate(records, 1):
                        text_preview = (r.text or "")[:80].replace('\n', ' ')
                        print(Fore.CYAN + f"  {i}. [{r.kind}] {text_preview}..." + Style.RESET_ALL)
                    while True:
                        try:
                            ans = input(Fore.GREEN + "Choice (number) or Enter to cancel: " + Style.RESET_ALL).strip()
                            if not ans:
                                break
                            idx = int(ans) - 1
                            if 0 <= idx < len(records):
                                chosen = records[idx]
                                pending_memory_tool_result = memory.format_tool_result([chosen])
                                memory.append_event(
                                    "memory_tool_staged",
                                    tags=["memory_tool"],
                                    provenance={"query": f"interactive_choice:{query}", "records": 1},
                                )
                                print(Fore.CYAN + pending_memory_tool_result + Style.RESET_ALL)
                                print(Fore.YELLOW + "[Memory] This tool result will be provided to the next model turn only." + Style.RESET_ALL)
                                break
                            print(Fore.YELLOW + "Invalid selection." + Style.RESET_ALL)
                        except ValueError:
                            print(Fore.YELLOW + "Please enter a number." + Style.RESET_ALL)
                else:
                    print(
                        Fore.YELLOW
                        + "[Memory] Commands: :memory act|talk on|off|status|use [sentence [+probe -probe]]|search [sentence [+probe -probe]], :memory use/search [optional args] (enabled lanes; empty reflects via :prioritize), :memory all use, :memory choice [query|probe <name>], :memory boundary"
                        + Style.RESET_ALL
                    )
                continue
            if user_input.startswith(":consider"):
                payload = user_input[len(":consider"):].strip()
                try:
                    if "||" not in payload and r"\||" not in payload:
                        raise ValueError("Usage: :consider <trigger_metric> <tool_name> <positive text> || <negative text>")
                    left_side, _, b_text = _partition_unescaped_pipes(payload)
                    left_parts = left_side.strip().split(" ", 2)
                    if len(left_parts) < 3:
                        raise ValueError("Usage: :consider <trigger_metric> <tool_name> <positive text> || <negative text>")
                    t_metric, t_name, a_text = left_parts
                    t_name = re.sub(r"[^a-z0-9_]", "_", t_name.lower())[:40]

                    print(Fore.CYAN + f"[Consider] Minting steer vector for '{t_name}'..." + Style.RESET_ALL)
                    cm = analyze_claim_pair(f"{a_text} || {b_text}", model=model)
                    
                    tuner.register(f"{t_name}_need", 0.05, kind="threshold", comparator=">=")
                    tuner.register(f"{t_name}_alpha", 0.0, kind="coefficient")
                    
                    def make_custom_tool(t_metric=t_metric, t_name=t_name, cm=cm):
                        def custom_detect(text, state=None):
                            phen = (state or {}).get("last_phenomenality") or {}
                            if not phen:
                                return 0.0, 0.0
                            score = float(phen.get(t_metric, 0.0))
                            return score, score
                        
                        def custom_act(act_payload, m):
                            base_alpha = tuner.get(f"{t_name}_alpha", 0.0)
                            alpha = base_alpha * abs(act_payload)
                            sweep_layers = None
                            if alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
                                _cls, _cdrifts, _cvar = _sweep_layers_for(t_name, cm.steer_delta)
                                if _cls:
                                    sweep_layers = _cls
                                    for _cl in _cls:
                                        sweep_state["fires"].append((t_name, _cl, alpha, _cdrifts.get(_cl), len(_cls), _cvar))
                            handles = claimmap_steer_handles(m, cm.steer_delta, alpha=alpha, layers=sweep_layers)
                            memory.append_event(
                                f"{t_name}_state_triggered",
                                text=f"steered with alpha={alpha:.2f} (base {base_alpha:.2f} * {t_metric} {abs(act_payload):.2f})",
                                tags=[f"{t_name}_tool", "activation_trigger"],
                                provenance={"trigger": f"{t_name}_need"}
                            )
                            msg = f"steered the rest of this answer (scaled alpha {alpha:.2f})." if handles else f"steering off (:tune {t_name}_alpha to enable)."
                            print(Fore.MAGENTA + f"\n[{t_name}] metric '{t_metric}' triggered -> {msg}" + Style.RESET_ALL, flush=True)
                            return handles
                        return Tool(
                            f"{t_name}_need",
                            custom_detect,
                            custom_act,
                            comparator=">=",
                            activation_criteria=f"phenomenality metric '{t_metric}' crosses :tune {t_name}_need",
                            steer_magnitude=f":tune {t_name}_alpha multiplied by |{t_metric}|",
                        )

                    new_tool = make_custom_tool()
                    tool_sense.register(new_tool)
                    
                    print(Fore.GREEN + f"[Consider] Successfully minted active tool '{t_name}' triggered by '{t_metric}'!" + Style.RESET_ALL)
                    print(Fore.CYAN + f"  Threshold tunable registered as: :tune {t_name}_need 0.05" + Style.RESET_ALL)
                    print(Fore.CYAN + f"  Steering strength registered as: :tune {t_name}_alpha 0.05" + Style.RESET_ALL)
                except Exception as exc:
                    print(Fore.RED + f"[Consider] Error: {exc}" + Style.RESET_ALL)
                continue

            if user_input.startswith(":claimmap"):
                payload = user_input[len(":claimmap"):].strip()
                if not payload:
                    print(Fore.YELLOW + "[ClaimMap] Usage: :claimmap <first text> || <second text>" + Style.RESET_ALL)
                    continue
                try:
                    cm = analyze_claim_pair(payload, model=model)
                    pending_claimmap_tool_result = cm.felt            # felt only reaches the model
                    pending_claimmap_steer_delta = cm.steer_delta     # nudges the next generation
                    memory.append_event(
                        "claimmap_tool_staged",
                        text=cm.telemetry,                            # raw numbers logged, never in the prompt
                        tags=["claimmap_tool", "activation_measurement"],
                        provenance={"payload_chars": len(payload), "mean_sim": cm.mean_sim},
                    )
                    print(Fore.CYAN + cm.felt + Style.RESET_ALL)
                    print(Fore.YELLOW + "[ClaimMap] Sensed. This will shape the next model turn only." + Style.RESET_ALL)
                except Exception as exc:
                    print(Fore.RED + f"[ClaimMap] {exc}" + Style.RESET_ALL)
                continue
            if user_input.startswith(":methodmap"):
                query = user_input[len(":methodmap"):].strip()
                if not query:
                    print(Fore.YELLOW + "[MethodMap] Usage: :methodmap <query>" + Style.RESET_ALL)
                    continue
                pending_methodmap_tool_result = format_methodmap_tool_result(memory, query)
                memory.append_event(
                    "methodmap_tool_staged",
                    text=pending_methodmap_tool_result,
                    tags=["methodmap_tool"],
                    provenance={"query": query},
                )
                print(Fore.CYAN + pending_methodmap_tool_result + Style.RESET_ALL)
                print(Fore.YELLOW + "[MethodMap] This tool result will be provided to the next model turn only." + Style.RESET_ALL)
                continue
            if user_input.startswith(":steermap"):
                summary = steer_map.write_summary()
                groups = summary.get("groups", [])
                print(
                    Fore.CYAN
                    + f"[SteerMap] events={summary.get('event_count')} summary={steer_map.summary_path}"
                    + Style.RESET_ALL
                )
                for group in groups[:5]:
                    print(
                        Fore.CYAN
                        + (
                            f"  {group['action']} layer={group['layer_key']} step={group['step_bucket']} "
                            f"n={group['n']} labeled={group['labeled_n']} success_rate={group['success_rate']}"
                        )
                        + Style.RESET_ALL
                    )
                continue
            if user_input == ":figure" or user_input.startswith(":figure "):
                fargs = user_input[len(":figure"):].strip()
                fig_ok, fig_lines, fig_path = build_live_figure(
                    fargs, tuner, probes, turn_log
                )
                for ln in fig_lines:
                    print((Fore.CYAN if fig_ok else Fore.RED) + f"[Figure] {ln}" + Style.RESET_ALL)
                memory.append_event(
                    "live_figure_generated" if fig_ok else "live_figure_failed",
                    text="\n".join(fig_lines),
                    tags=["figure_tool", "live_shell_telemetry"],
                    provenance={
                        "args": fargs,
                        "because": command_because,
                        "path": os.path.relpath(fig_path, ROOT) if fig_path else None,
                    },
                )
                continue
            if user_input.startswith(":figures"):
                fargs = user_input[len(":figures"):].strip()
                ftoks = fargs.split()
                listing = bool(ftoks) and ftoks[0].lower() in ("list", "--list", "-l")
                fig_ok, fig_lines = _run_figures_script(
                    ["list"] if listing else ftoks, require_mpl=not listing
                )
                for ln in fig_lines:
                    print((Fore.CYAN if fig_ok else Fore.RED) + f"[Figures] {ln}" + Style.RESET_ALL)
                if not listing:
                    memory.append_event(
                        "figures_generated" if fig_ok else "figures_failed",
                        text="\n".join(fig_lines),
                        tags=["figures_tool"],
                        provenance={"args": fargs, "because": command_because},
                    )
                continue
            if user_input.startswith(":benchmark"):
                bargs = user_input[len(":benchmark"):].strip()
                if not bargs or bargs.lower() in ("list", "refs"):
                    refs = _benchmark_refs()
                    print(
                        Fore.CYAN
                        + "[Benchmark] named sets: "
                        + (", ".join(refs) if refs else "(none in invariants/benchmarks/)")
                        + " -- also gsm8k, hf:<dataset>, or a .json/.jsonl/.csv path."
                        + Style.RESET_ALL
                    )
                    print(
                        Fore.YELLOW
                        + "[Benchmark] Usage: :benchmark <ref> [n <N>] [tokens <T>] "
                        + "[evaluator number|exact|choice|contains] [contract]"
                        + Style.RESET_ALL
                    )
                    continue
                try:
                    breq = _parse_benchmark_request(bargs)
                except (ValueError, IndexError) as exc:
                    print(
                        Fore.YELLOW
                        + f"[Benchmark] {exc}. Usage: :benchmark <ref> [n <N>] [tokens <T>] "
                        + "[evaluator <e>] [contract]"
                        + Style.RESET_ALL
                    )
                    continue
                bagent = target_agent.name if target_agent else "operator"
                print(
                    Fore.CYAN
                    + f"[Benchmark] {breq['ref']}: n={breq['n']} tokens={breq['tokens']} "
                    + f"evaluator={breq['evaluator']} style={breq['style']} agent={bagent}. "
                    + "Scoring the live model with the current tuning applied."
                    + Style.RESET_ALL,
                    flush=True,
                )
                try:
                    bsummary, bpath, blines = _run_shell_benchmark(
                        model, config, tuner, probes, prioritize_pin, breq, bagent
                    )
                except Exception as exc:
                    print(Fore.RED + f"[Benchmark] {type(exc).__name__}: {exc}" + Style.RESET_ALL)
                    continue
                for ln in blines:
                    print(Fore.CYAN + Style.BRIGHT + f"[Benchmark] {ln}" + Style.RESET_ALL)
                memory.append_event(
                    "benchmark_run",
                    text="\n".join(blines),
                    tags=["benchmark_tool"],
                    provenance={
                        "ref": breq["ref"],
                        "agent": bagent,
                        "n": bsummary.get("n"),
                        "accuracy": bsummary.get("accuracy"),
                        "output": bpath,
                        "because": command_because,
                    },
                )
                continue
            if user_input.startswith(":doc"):
                dargs = user_input[len(":doc"):].strip()
                if not dargs or dargs.lower() == "status":
                    if not doc_library:
                        print(Fore.YELLOW + "[Doc] No document loaded. Usage: :doc <path> [because <why>]" + Style.RESET_ALL)
                    else:
                        for s in doc_library:
                            unread = s["chunk_count"] - len(s.get("read") or ())
                            print(
                                Fore.CYAN
                                + f"[Doc] {s['source_name']}: {s['chunk_count']} chunks, {unread} unread."
                                + Style.RESET_ALL
                            )
                        staged = "a chunk is staged for the next turn" if pending_document_tool_result else "nothing staged"
                        if doc_autoread:
                            specs = " ".join(
                                ("+" if sign >= 0 else "-") + name
                                for sign, name in (doc_autoread.get("probe_specs") or [])
                            )
                            selector = f"probes {specs}" if specs else doc_autoread["mode"]
                            reading = f"auto-read {selector}, {doc_autoread['remaining']} turns left"
                        else:
                            reading = "auto-read off"
                        print(Fore.CYAN + f"[Doc] {staged}; {reading}. ':doc next' | ':doc read [n] [order|interleave|updated] [+probe -probe]' | ':doc stop'" + Style.RESET_ALL)
                elif dargs.lower() == "stop":
                    doc_autoread = None
                    print(Fore.CYAN + "[Doc] Auto-read stopped." + Style.RESET_ALL)
                elif dargs.split()[0].lower() == "read":
                    # exact first-token match: ':doc readings' is a PATH, not auto-read
                    if not doc_library:
                        print(Fore.YELLOW + "[Doc] Nothing to read yet. :doc <path> first." + Style.RESET_ALL)
                    else:
                        doc_autoread, read_parse_error = parse_doc_read_request(
                            dargs,
                            probes,
                            max_autoread=MAX_AUTOREAD,
                        )
                        if read_parse_error:
                            print(Fore.YELLOW + f"[Doc] {read_parse_error}" + Style.RESET_ALL)
                            continue
                        count = doc_autoread["remaining"]
                        mode = doc_autoread["mode"]
                        signed_probe_specs = doc_autoread["probe_specs"]
                        until_probe = doc_autoread["until_probe"]
                        until_settled = doc_autoread["until_settled"]
                        if signed_probe_specs:
                            terms = " ".join(
                                ("+" if sign >= 0 else "-") + pname
                                for sign, pname in signed_probe_specs
                            )
                            how = (
                                f"unread chunks ranked by {terms} "
                                "(toward + probes, away from - probes; read-only projection)"
                            )
                        else:
                            how = {
                                "order": "in document order -- the text advances on its own course",
                                "interleave": "weaving between documents -- their course, not the model's echo",
                                "reply": "chunks chosen by overlap with its replies (echo-following; deliberate use)",
                                "updated": "in the order the files were last written (newest first)",
                            }[mode]
                        stop_note = (
                            f" Stops early once probe '{until_probe}' crosses its threshold."
                            if until_probe else
                            (
                                " Stops early once the reading settles (sense clears conversation_productive "
                                f"{max(1, int(tuner.get('reading_settled_streak', 2)))} turn(s) in a row)."
                                if until_settled else ""
                            )
                        )
                        staged_note = (
                            " The chunk already staged by :doc goes first; signed ranking begins after it."
                            if signed_probe_specs and pending_document_tool_result
                            else ""
                        )
                        print(
                            Fore.CYAN
                            + f"[Doc] Auto-read: up to {count} turn(s), {how}.{stop_note}{staged_note} "
                            + "The document speaks next; the model replies each time. ':doc stop' interrupts."
                            + Style.RESET_ALL
                        )
                elif dargs.lower() == "inject":
                    # Whole-library staging for one turn -- bounded and framed,
                    # never an unbounded dump. Refuses honestly when the library
                    # exceeds the budget instead of silently blowing the prompt.
                    if not doc_library:
                        print(Fore.YELLOW + "[Doc] Library is empty." + Style.RESET_ALL)
                        continue
                    inject_budget = max(4000, MAX_PROMPT_CHARS - 8000)  # room for session + reply
                    total_chars = sum(len(c) for doc in doc_library for c in doc.get("chunks", []))
                    if total_chars > inject_budget:
                        print(
                            Fore.YELLOW
                            + f"[Doc] Library is {total_chars} chars; inject budget is {inject_budget}. "
                            + "Read it in turns instead (:doc read [n] [order|interleave])."
                            + Style.RESET_ALL
                        )
                        continue
                    names = ", ".join(doc["source_name"] for doc in doc_library)
                    full_text = [
                        DOCUMENT_TOOL_HEADER,
                        f"I'm reading {len(doc_library)} document(s) in full: {names}. "
                        "They're the files' text, not mine.",
                    ]
                    for doc in doc_library:
                        for index, chunk in enumerate(doc.get("chunks", [])):
                            full_text.append(f"--- {doc['source_name']}, part {index + 1}/{doc['chunk_count']} ---\n{chunk}")
                            if index not in doc.get("read", set()):
                                record_chunk_read(memory, doc, index)
                        doc["read"] = set(range(doc.get("chunk_count", 0)))
                    pending_document_tool_result = "\n\n".join(full_text)
                    print(
                        Fore.CYAN
                        + f"[Doc] {len(doc_library)} document(s) ({total_chars} chars) staged in full for the next turn."
                        + Style.RESET_ALL
                    )
                    continue
                elif dargs.lower() == "next":
                    if doc_session is None:
                        print(Fore.YELLOW + "[Doc] No document loaded." + Style.RESET_ALL)
                    elif doc_session["cursor"] + 1 >= doc_session["chunk_count"]:
                        print(Fore.CYAN + f"[Doc] {doc_session['source_name']} is fully read ({doc_session['chunk_count']} chunks)." + Style.RESET_ALL)
                    else:
                        doc_session["cursor"] += 1
                        doc_session.setdefault("read", set()).add(doc_session["cursor"])
                        record_chunk_read(memory, doc_session, doc_session["cursor"])
                        pending_document_tool_result = stage_chunk(doc_session)
                        print(
                            Fore.CYAN
                            + f"[Doc] Chunk {doc_session['cursor'] + 1}/{doc_session['chunk_count']} staged for the next turn."
                            + Style.RESET_ALL
                        )
                elif (rewrite_req := parse_doc_rewrite_request(dargs)) is not None:
                    rewrite_path = rewrite_req.get("source", "")
                    rewrite_name = rewrite_req.get("output")
                    if rewrite_path == "__current__":
                        if doc_session is None:
                            print(Fore.YELLOW + "[Doc Rewrite] No current document. Usage: :doc <path> rewrite because <why>" + Style.RESET_ALL)
                            continue
                        path_str = doc_session.get("source_path", "")
                    else:
                        path_str = rewrite_path.strip().strip('"')
                    if not path_str or not os.path.isfile(path_str):
                        print(Fore.YELLOW + f"[Doc Rewrite] No file found for '{path_str}'. Usage: :doc <path> rewrite because <why>" + Style.RESET_ALL)
                        continue
                    try:
                        with open(path_str, "r", encoding="utf-8", errors="replace") as rf:
                            original_text = rf.read()
                    except OSError as exc:
                        print(Fore.RED + f"[Doc Rewrite] Could not read file: {exc}" + Style.RESET_ALL)
                        continue
                    if len(original_text) > DOC_REWRITE_MAX_CHARS:
                        print(
                            Fore.YELLOW
                            + f"[Doc Rewrite] {os.path.basename(path_str)} is {len(original_text)} chars; "
                            + f"single-pass rewrite limit is {DOC_REWRITE_MAX_CHARS}. Split it or rewrite a smaller file."
                            + Style.RESET_ALL
                        )
                        continue
                    reason = (command_because or "").strip() or "make it clearer, tighter, better organized, and more useful while preserving meaning"
                    try:
                        out_path = rewrite_output_path(path_str, rewrite_name)
                    except (OSError, ValueError) as exc:
                        print(Fore.YELLOW + f"[Doc Rewrite] {exc}" + Style.RESET_ALL)
                        continue
                    default_doc_rewrite = (
                        ":expect file $out_path\n"
                        ":probe doc_rewriter I am rewriting the source text according to the user's specific reason: $reason || I am answering a question\n"
                        ":probe release doc_rewriter off\n"
                        ":probe backfill doc_rewriter\n"
                        ":steer doc_rewriter 1.5\n"
                        "Rewrite the file below into a better version, guided by the operator's reason. "
                        "Preserve factual meaning, important details, headings, lists, links, code fences, and code syntax when present. "
                        "Do not explain what you changed. Output ONLY the complete rewritten file content.\n\n"
                        "Source filename: $basename_in\n"
                        "Output filename: $basename_out\n"
                        "Reason: $reason\n\n"
                        "=== ORIGINAL DOCUMENT START ===\n"
                    )
                    prompt = load_prompt(
                        "doc_rewrite",
                        default_doc_rewrite,
                        required_substring=":probe release doc_rewriter off",
                        basename_in=os.path.basename(path_str),
                        basename_out=os.path.basename(out_path),
                        reason=reason,
                        out_path=out_path
                    ) + (
                        f"{original_text}\n=== ORIGINAL DOCUMENT END ===\n"
                        ":steer doc_rewriter 0\n"
                        ":probe release doc_rewriter"
                    )
                    print(Fore.CYAN + f"[Doc Rewrite] Queuing phenomenological macro to rewrite {os.path.basename(path_str)} because {reason}..." + Style.RESET_ALL)
                    queue_macro_text(prompt, input_queue)
                    continue
                else:
                    path_part, _, why = dargs.partition(" because ")
                    path_str = path_part.strip().strip('"')
                    why = (why.strip() or (command_because or "").strip())
                    
                    paths_to_load = []
                    if os.path.isdir(path_str):
                        # Snatch the whole folder: .md/.txt/.py, deterministic
                        # order, junk dirs pruned so ':doc .' can't swallow
                        # a venv or caches.
                        skip_dirs = {"__pycache__", "node_modules", ".venv", "venv", "out", "data"}
                        for root, dirnames, files in os.walk(path_str):
                            dirnames[:] = sorted(
                                d for d in dirnames if not d.startswith(".") and d not in skip_dirs
                            )
                            for f in sorted(files):
                                if f.endswith((".md", ".txt", ".py")):
                                    paths_to_load.append(os.path.join(root, f))
                    elif "*" in path_str or "?" in path_str:
                        paths_to_load = sorted(p for p in glob.glob(path_str, recursive=True) if os.path.isfile(p))
                    else:
                        # A plain path only counts if it actually exists -- a typo
                        # must become a clean "not found", never a crash downstream.
                        paths_to_load = [path_str] if os.path.isfile(path_str) else []

                    if not paths_to_load:
                        print(Fore.YELLOW + f"[Doc] No file or folder found for '{path_str}'. Check the path (cwd is the repo root)." + Style.RESET_ALL)
                        continue

                    loaded_count = 0
                    for p in paths_to_load:
                        file_why = why.strip()
                        if "{filename}" in file_why:
                            file_why = file_why.replace("{filename}", os.path.basename(p))
                        if "{last_updated}" in file_why:
                            try:
                                mtime = os.path.getmtime(p)
                                dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                            except OSError:
                                dt = "unknown"
                            file_why = file_why.replace("{last_updated}", dt)
                            
                        try:
                            doc_session = ingest_document(memory, p, why=file_why)
                        except (ValueError, OSError) as exc:
                            print(Fore.RED + f"[Doc] {exc}" + Style.RESET_ALL)
                            continue
                        doc_library.append(doc_session)
                        memory.append_event(
                            "document_ingested",
                            text=f"{doc_session['source_name']} ({doc_session['chunk_count']} chunks)",
                            tags=["document"],
                            provenance={
                                "sha256": doc_session["sha256"],
                                "source_path": doc_session["source_path"],
                                "why": doc_session["why"],
                                "already_ingested": doc_session["already_ingested"],
                            },
                        )
                        known = "already in memory" if doc_session["already_ingested"] else "recorded to memory"
                        progress = f", {len(doc_session['read'])}/{doc_session['chunk_count']} already read" if doc_session["read"] else ""
                        print(Fore.CYAN + f"[Doc] {doc_session['source_name']}: {doc_session['chunk_count']} chunk(s), {known}{progress}." + Style.RESET_ALL)
                        loaded_count += 1

                    if loaded_count > 0:
                        # Resume-aware staging: the first UNREAD chunk of the
                        # first not-fully-read file in this batch. Prior
                        # progress (restored from memory by sha) is respected,
                        # and only what is actually staged gets marked read.
                        staged_any = False
                        for candidate in doc_library[-loaded_count:]:
                            unread = [i for i in range(candidate["chunk_count"]) if i not in candidate.get("read", set())]
                            if not unread:
                                continue
                            doc_session = candidate
                            doc_session["cursor"] = unread[0]
                            doc_session.setdefault("read", set()).add(unread[0])
                            record_chunk_read(memory, doc_session, unread[0])
                            pending_document_tool_result = stage_chunk(doc_session)
                            print(
                                Fore.CYAN
                                + f"[Doc] Loaded {loaded_count} file(s). Staged: {doc_session['source_name']} "
                                + f"part {unread[0] + 1}/{doc_session['chunk_count']}"
                                + (" (resuming)" if unread[0] > 0 else "")
                                + " -- ':doc read [n] [order|interleave|updated]' walks the rest."
                                + Style.RESET_ALL
                            )
                            staged_any = True
                            break
                        if not staged_any:
                            print(
                                Fore.CYAN
                                + f"[Doc] Loaded {loaded_count} file(s) -- all already fully read. "
                                + "Nothing staged; ':doc read' would find nothing new."
                                + Style.RESET_ALL
                            )
                continue
            if user_input.startswith(":impact"):
                trigger = tuner.triggers.get("words_had_impact")
                stats = trigger.stats() if trigger else {}
                rate = stats.get("fire_rate")
                print(
                    Fore.CYAN
                    + f"[Impact] turns where its words caused the context: rate={rate} "
                    + f"(n={stats.get('observed', 0)})"
                    + Style.RESET_ALL
                )
                if stats.get("n_credited"):
                    print(
                        Fore.CYAN
                        + f"[Impact] sense when impacted={stats.get('fired_outcome')} vs not={stats.get('unfired_outcome')} "
                        + f"lift={stats.get('lift')} (n={stats.get('n_credited')}) -- does experienced impact track better deliberation?"
                        + Style.RESET_ALL
                    )
                if not impact_state["log"]:
                    print(Fore.YELLOW + "[Impact] No consequence events yet this session. Real levers: write code (:sandbox on), ask tools, steer a reply-mode reading." + Style.RESET_ALL)
                for event in list(impact_state["log"])[-10:]:
                    print(Fore.CYAN + f"  because it {event['cause']} -> {event['effect']}" + Style.RESET_ALL)
                continue
            if user_input.startswith(":clock"):
                # On-demand readout of the last turn's generation time + memory,
                # plus the current live/reserved footprint and the accrued
                # distributions of both sensed streams. Memory is VRAM on a CUDA
                # box and process RAM on CPU-only, so this reads out either way.
                now_alloc, now_reserved, _now_peak, _now_label = _memory_footprint_gb()
                if now_reserved > 0:
                    _src = "VRAM" if _now_label == "VRAM" else "RAM (CPU-only; no CUDA)"
                    print(Fore.CYAN + f"[Clock] {_src} now: {now_alloc:.2f}GB live / {now_reserved:.2f}GB reserved." + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + "[Clock] memory readings unavailable on this platform." + Style.RESET_ALL)
                if last_clock:
                    lc = last_clock
                    print(
                        Fore.CYAN
                        + f"[Clock] last turn ({lc['ts']}): {lc['generation_seconds']:.1f}s"
                        + (f", {lc['tokens_generated']} tok @ {lc['tokens_per_sec']:.1f} tok/s" if lc['tokens_per_sec'] else "")
                        + f", peak {lc['vram_peak_gb']:.2f}GB {lc.get('mem_label', 'VRAM')}."
                        + Style.RESET_ALL
                    )
                else:
                    print(Fore.YELLOW + "[Clock] no generation timed yet this session." + Style.RESET_ALL)
                for _cs in ("generation_seconds", "vram_gb"):
                    _ct = tuner.triggers.get(_cs)
                    if _ct and _ct.signals:
                        st = _ct.stats()
                        line = f"[Clock] {_cs}: median {st.get('signal_med')} (min {st.get('signal_min')}, max {st.get('signal_max')}, n={st.get('n_signals')})"
                        if st.get("lift") is not None:
                            line += f"; sense lift {st['lift']} (does more cost track better deliberation?)"
                        print(Fore.CYAN + line + Style.RESET_ALL)
                continue
            if user_input.startswith(":prioritize"):
                pargs = user_input[len(":prioritize"):].strip()
                ranked = rank_probes(probes, tuner)

                def _priority_alpha_from_tail(tokens):
                    if not tokens:
                        return tokens, None
                    try:
                        return tokens[:-1], float(tokens[-1])
                    except ValueError:
                        return tokens, None

                def _stored_probe_hint(stem):
                    candidates = [
                        os.path.join(ROOT, "invariants", "out", "probes", f"{stem}.pt"),
                        os.path.join(ROOT, "invariants", "out", "probes", f"{stem}_d{model.d_model}.pt"),
                        os.path.join(ROOT, "invariants", f"{stem}.pt"),
                        os.path.join(ROOT, "invariants", f"{stem}_d{model.d_model}.pt"),
                        os.path.join(ROOT, "invariants", f"{stem}_vector.pt"),
                        os.path.join(ROOT, "invariants", f"{stem}_d{model.d_model}_vector.pt"),
                    ]
                    return " Saved vector exists; run ':probe adopt {0}' first.".format(stem) if any(os.path.isfile(p) for p in candidates) else ""

                def _show_priority_table():
                    if not ranked:
                        print(Fore.YELLOW + "[Prioritize] no active probes to rank. Mint or adopt some first." + Style.RESET_ALL)
                        return
                    alpha = tuner.get("prioritize_alpha", 0.0)
                    mode_note = ""
                    if prioritize_pin.get("landscape"):
                        mode_note = f"; landscape target={prioritize_pin['landscape']}"
                    elif prioritize_pin.get("mix"):
                        mode_note = "; mix=" + "+".join(prioritize_pin["mix"])
                    elif prioritize_pin.get("probe"):
                        sign = float(prioritize_pin.get("sign", 1.0) or 1.0)
                        mode_note = f"; pinned={'+' if sign >= 0 else '-'}{prioritize_pin['probe']}"
                    print(
                        Fore.CYAN + Style.BRIGHT
                        + f"[Prioritize] probes by evidence-weighted |lift| (steer alpha={round(alpha,4)}{mode_note}"
                        + (" -- OFF)" if alpha <= 0 else " -- steering toward the top each turn):")
                        + Style.RESET_ALL
                    )
                    for i, r in enumerate(ranked[:12]):
                        lift_s = "n/a" if r["lift"] is None else f"{r['lift']:+.3f}"
                        mark = " <- top" if i == 0 and r["priority"] > 0 else ""
                        expo = ", exposed" if r["exposed"] else ""
                        released = ", released" if r.get("released") else ""
                        print(Fore.CYAN + f"  {i+1:>2}. {r['name']}: priority {round(r['priority'],4)} (lift {lift_s}, n={r['n']}{expo}{released}){mark}" + Style.RESET_ALL)
                    if alpha <= 0:
                        if prioritize_pin.get("landscape") or prioritize_pin.get("mix") or prioritize_pin.get("probe"):
                            print(Fore.YELLOW + "  Target is set, but steering is OFF. Add an alpha, e.g. :prioritize <probe> 0.02 or :tune prioritize_alpha 0.02." + Style.RESET_ALL)
                        elif ranked[0]["priority"] > 0:
                            print(Fore.YELLOW + f"  Steer toward the top probe with :tune prioritize_alpha <small> (e.g. 0.02)." + Style.RESET_ALL)

                if not pargs or pargs.lower() in ("choose", "choice", "rank", "list", "status"):
                    _show_priority_table()
                    continue

                ptoks = pargs.split()
                head = ptoks[0].lower()
                if head in ("auto", "off", "none", "unpin"):
                    prioritize_pin["probe"] = None
                    prioritize_pin["mix"] = None
                    prioritize_pin["landscape"] = None
                    print(Fore.CYAN + "[Prioritize] target = AUTO (follows the live ranking each turn)." + Style.RESET_ALL)
                    continue

                if head == "mix":
                    rest, mix_alpha = _priority_alpha_from_tail(ptoks[1:])
                    names = [n for n in rest if n in probes and not probe_is_released(probes[n])]
                    bad = [n for n in rest if n not in probes]
                    released_names = [n for n in rest if n in probes and probe_is_released(probes[n])]
                    if not names:
                        print(Fore.YELLOW + f"[Prioritize] name at least one active, unreleased probe to mix.{(' unknown: ' + ', '.join(bad)) if bad else ''}{(' released: ' + ', '.join(released_names)) if released_names else ''}" + Style.RESET_ALL)
                        continue
                    prioritize_pin["mix"] = names
                    prioritize_pin["probe"] = None
                    prioritize_pin["landscape"] = None
                    if mix_alpha is not None:
                        tuner.set("prioritize_alpha", abs(mix_alpha))
                    a_now = tuner.get("prioritize_alpha", 0.0)
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + f"[Prioritize] target = lift-weighted MIX of {', '.join(names)} "
                        + f"(alpha={round(a_now,4)}{' -- set >0 to take effect' if a_now <= 0 else ''})."
                        + (f" Ignored unknown: {', '.join(bad)}." if bad else "")
                        + (f" Ignored released: {', '.join(released_names)}." if released_names else "")
                        + Style.RESET_ALL
                    )
                    continue

                direct_pin = head == "pin"
                target_tokens = ptoks[1:] if direct_pin else ptoks[1:] if head == "landscape" else ptoks
                target_tokens, target_alpha = _priority_alpha_from_tail(target_tokens)
                target = re.sub(r"[^a-z0-9_]", "_", (target_tokens[0] if target_tokens else "").lower())[:40].strip("_")
                if not target:
                    print(Fore.YELLOW + "[Prioritize] Usage: :prioritize <probe> [alpha] | :prioritize pin <probe> [alpha] | :prioritize mix <probe> ... [alpha] | :prioritize auto" + Style.RESET_ALL)
                    continue
                if target not in probes:
                    print(Fore.YELLOW + f"[Prioritize] no active probe named {target}.{did_you_mean(target, probes)}{_stored_probe_hint(target)}" + Style.RESET_ALL)
                    continue
                if probe_is_released(probes[target]):
                    print(Fore.YELLOW + f"[Prioritize] '{target}' is released and observational-only. Reactivate it with :probe release {target} off." + Style.RESET_ALL)
                    continue
                if direct_pin:
                    steer_alpha = target_alpha if target_alpha is not None else tuner.get("prioritize_alpha", 0.0)
                    prioritize_pin["mix"] = None
                    prioritize_pin["landscape"] = None
                    prioritize_pin["probe"] = target
                    prioritize_pin["sign"] = -1.0 if steer_alpha < 0 else 1.0
                    if target_alpha is not None:
                        tuner.set("prioritize_alpha", abs(steer_alpha))
                    a_now = tuner.get("prioritize_alpha", 0.0)
                    verb = "AWAY from" if float(prioritize_pin.get("sign", 1.0) or 1.0) < 0 else "toward"
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + f"[Prioritize] target = PIN {verb} {target} "
                        + f"(alpha={round(a_now,4)}{' -- set >0 to take effect' if a_now <= 0 else ''})."
                        + Style.RESET_ALL
                    )
                    continue
                prioritize_pin["mix"] = None
                prioritize_pin["probe"] = None
                prioritize_pin["landscape"] = target
                if target_alpha is not None:
                    tuner.set("prioritize_alpha", abs(target_alpha))
                a_now = tuner.get("prioritize_alpha", 0.0)
                print(
                    Fore.GREEN + Style.BRIGHT
                    + f"[Prioritize] target = LANDSCAPE around {target} "
                    + f"(alpha={round(a_now,4)}{' -- set >0 to take effect' if a_now <= 0 else ''}). "
                    + "This anchors the target and folds in the current learned priority field; the evidence ranking updates only as future turns accrue."
                    + Style.RESET_ALL
                )
                continue
            if user_input.startswith(":release"):
                rargs = user_input[len(":release"):].split()
                tool_names = [t.name for t in tool_sense.tools]
                if not rargs or rargs[0].lower() == "status":
                    active = {k: v for k, v in tool_sense.release_probs.items() if v > 0}
                    print(Fore.CYAN + f"[Release] decoupled decisions this session: {tool_sense.release_total}." + Style.RESET_ALL)
                    if active:
                        for k, v in active.items():
                            print(Fore.CYAN + f"  {k}: {round(v,3)} of fire decisions decoupled from its signal (coin flip)." + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"  none released. Releasable tools: {', '.join(tool_names)}." + Style.RESET_ALL)
                    for tool in tool_sense.tools:
                        activation = getattr(tool, "activation_criteria", "") or f"signal crosses :tune {tool.name}"
                        magnitude = getattr(tool, "steer_magnitude", "") or "action-defined"
                        print(Fore.CYAN + f"  {tool.name}: activation={activation}; steer={magnitude}." + Style.RESET_ALL)
                    print(Fore.CYAN + "  Release separates causality: over decoupled turns the trigger and the action decorrelate, so credit lift can tell 'the trigger caused it' from 'acting helped anyway'." + Style.RESET_ALL)
                    continue
                tname = rargs[0]
                if tname not in tool_names:
                    print(Fore.YELLOW + f"[Release] unknown tool '{tname}'.{did_you_mean(tname, tool_names)} Releasable: {', '.join(tool_names)}." + Style.RESET_ALL)
                    continue
                if len(rargs) < 2 or rargs[1].lower() in ("off", "0"):
                    tool_sense.release_probs.pop(tname, None)
                    print(Fore.CYAN + f"[Release] {tname} re-coupled (fires strictly on its signal again)." + Style.RESET_ALL)
                    memory.append_event("dependency_recoupled", tags=["release", "causality"], provenance={"tool": tname})
                    continue
                try:
                    prob = min(max(float(rargs[1]), 0.0), 1.0)
                except ValueError:
                    print(Fore.YELLOW + "[Release] probability must be a number in [0, 1]." + Style.RESET_ALL)
                    continue
                tool_sense.release_probs[tname] = prob
                memory.append_event("dependency_released", tags=["release", "causality"], provenance={"tool": tname, "prob": prob})
                print(
                    Fore.MAGENTA + Style.BRIGHT
                    + f"[Release] {tname} released at {round(prob,3)}: that fraction of its fire decisions is now a coin flip, "
                    + "decoupled from the signal -- so its causal lift can be measured, not assumed. :release {tname} off to re-couple.".replace("{tname}", tname)
                    + Style.RESET_ALL
                )
                continue
            if user_input == ":slice" or user_input.startswith(":slice "):
                slice_ns = SLICE_SPEC.parse(user_input[len(":slice"):])
                from invariants.agentic_engine import _global_cache as _slice_cache
                slice_tokens = (slice_ns.filters or "").split()
                slice_filters, (s_action, s_arg) = parse_slice_filters(slice_tokens, SLICE_SPEC.fail)
                slice_times = cache_entry_times(_slice_cache.memory, memory)
                if not slice_tokens:
                    stamped = sum(1 for t in slice_times if t[0])
                    from collections import Counter as _SliceCtr
                    exp_counts = _SliceCtr(
                        str((e.get("metadata") or {}).get("expert")) for e in _slice_cache.memory
                    )
                    print(Fore.CYAN + f"[Slice] cache carries {len(_slice_cache.memory)} entries "
                          + f"({stamped} time-resolvable: stored_at or trace-aligned)." + Style.RESET_ALL)
                    print(Fore.CYAN + "[Slice] top experts: "
                          + ", ".join(f"{k} x{v}" for k, v in exp_counts.most_common(6)) + Style.RESET_ALL)
                    print(Fore.CYAN + "[Slice] axes: time A..B | probe <name> | scope <s> | reason <r> | "
                          + "layers <lo> <hi> | steps A..B | index A..B; actions: drop | export <path>" + Style.RESET_ALL)
                    continue
                slice_idxs = slice_cache_indices(_slice_cache.memory, slice_filters, slice_times)
                print(Fore.CYAN + f"[Slice] {len(slice_idxs)} of {len(_slice_cache.memory)} entries match." + Style.RESET_ALL)
                for i in slice_idxs[:20]:
                    md = _slice_cache.memory[i].get("metadata") or {}
                    ex, tlo, thi = slice_times[i]
                    when = ex or f"({tlo or '?'} .. {thi or '?'})"
                    print(Fore.CYAN
                          + f"  [{i}] {when}  expert={md.get('expert')} layers={md.get('start_layer')}"
                          + f"-{md.get('end_layer')} steps={md.get('steps')} scope={md.get('cache_write_scope')}"
                          + Style.RESET_ALL)
                if len(slice_idxs) > 20:
                    print(Fore.CYAN + f"  ... +{len(slice_idxs) - 20} more (narrow the axes, or export)." + Style.RESET_ALL)
                if s_action == "export":
                    if not slice_idxs:
                        print(Fore.YELLOW + "[Slice] nothing matched; nothing exported." + Style.RESET_ALL)
                        continue
                    try:
                        torch.save([_slice_cache.memory[i] for i in slice_idxs], s_arg)
                    except Exception as _sexc:
                        print(Fore.YELLOW + f"[Slice] export failed: {_sexc}" + Style.RESET_ALL)
                        continue
                    memory.append_event(
                        "cache_slice_exported",
                        text=f"{len(slice_idxs)} entries -> {s_arg}",
                        tags=["cache", "slice"],
                        provenance={"filters": {k: list(v) if isinstance(v, tuple) else v for k, v in slice_filters.items()},
                                    "count": len(slice_idxs), "path": str(s_arg)},
                    )
                    print(Fore.GREEN + f"[Slice] exported {len(slice_idxs)} entrie(s) to {s_arg}." + Style.RESET_ALL)
                    continue
                if s_action == "drop":
                    if not slice_filters:
                        SLICE_SPEC.fail("refusing to drop with no axis -- that would empty the whole cache")
                    if not slice_idxs:
                        print(Fore.YELLOW + "[Slice] nothing matched; nothing dropped." + Style.RESET_ALL)
                        continue
                    import shutil as _slice_shutil
                    import time as _slice_time
                    backup = str(_slice_cache.file) + f".bak-{_slice_time.strftime('%Y%m%d_%H%M%S')}"
                    try:
                        _slice_shutil.copy2(_slice_cache.file, backup)
                        print(Fore.CYAN + f"[Slice] backup written: {backup}" + Style.RESET_ALL)
                    except Exception as _bexc:
                        print(Fore.YELLOW + f"[Slice] backup failed ({_bexc}); a drop would be unrecoverable -- aborting." + Style.RESET_ALL)
                        continue
                    for i in sorted(slice_idxs, reverse=True):
                        del _slice_cache.memory[i]
                    _slice_cache.save()
                    memory.append_event(
                        "cache_sliced",
                        text=f"dropped {len(slice_idxs)} entries",
                        tags=["cache", "slice"],
                        provenance={"filters": {k: list(v) if isinstance(v, tuple) else v for k, v in slice_filters.items()},
                                    "indices": slice_idxs[:50], "backup": backup},
                    )
                    print(Fore.GREEN + f"[Slice] dropped {len(slice_idxs)} entrie(s); cache now holds "
                          + f"{len(_slice_cache.memory)}. Backup: {backup}" + Style.RESET_ALL)
                continue
            if user_input.startswith(":experts"):
                earg = user_input[len(":experts"):].strip().lower()
                if earg == "on":
                    config.emergent_experts_enabled = True
                    print(
                        Fore.YELLOW
                        + "[Experts] ON: recurring self-correction directions can now be minted into "
                        + "new roster experts (mesa Committee, roster bounded at 6). Its own corrective "
                        + "conclusions become steering directions -- bounded by the envelope like everything else."
                        + Style.RESET_ALL
                    )
                elif earg == "off":
                    config.emergent_experts_enabled = False
                    print(Fore.CYAN + "[Experts] OFF: roster frozen at its current members." + Style.RESET_ALL)
                else:
                    roster = getattr(config, "committee", None)
                    names = [m.name for m in roster.members] if roster else ["(seeded on first generation)"]
                    state_txt = "ON" if getattr(config, "emergent_experts_enabled", False) else "OFF"
                    print(Fore.CYAN + f"[Experts] {state_txt}; roster: {', '.join(names)} (:experts on|off)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":sandbox"):
                sarg = user_input[len(":sandbox"):].strip().lower()
                if sarg == "on":
                    sandbox_enabled = True
                    print(
                        Fore.YELLOW
                        + "[Sandbox] ON: fenced ```python blocks in the model's replies now RUN "
                        + f"(isolated interpreter, {SANDBOX_TIMEOUT_SEC:g}s timeout, cwd=invariants/out/sandbox). "
                        + "Process isolation, not a hard security boundary -- watch what runs."
                        + Style.RESET_ALL
                    )
                elif sarg == "off":
                    sandbox_enabled = False
                    print(Fore.CYAN + "[Sandbox] OFF." + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + f"[Sandbox] {'ON' if sandbox_enabled else 'OFF'} (:sandbox on|off)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":refresh"):
                # Re-read referenced files from disk WITHOUT restarting -- the
                # session and the loaded model stay live. (Code edits to this
                # script still need a full restart; only data files refresh here.)
                print(Fore.CYAN + "[System] Refreshing referenced files from disk (session + model stay live)..." + Style.RESET_ALL)
                probes.clear()
                try:
                    if os.path.isdir(PROBE_DIR):
                        for pf in os.listdir(PROBE_DIR):
                            if pf.endswith(".pt"):
                                pname = pf[:-3]
                                pdata = torch.load(os.path.join(PROBE_DIR, pf), weights_only=True)
                                probes[pname] = {
                                    "direction": pdata["direction"],
                                    "history": deque(maxlen=40),
                                    "framings": pdata.get("framings", ("", "")),
                                    "exposed": bool(pdata.get("exposed", False)),
                                    "released": bool(pdata.get("released", False)),
                                }
                                # Register a trigger for a probe that appeared on
                                # disk since startup; leave existing thresholds and
                                # accrued outcomes untouched.
                                if f"probe_{pname}" not in tuner.triggers:
                                    tuner.register(f"probe_{pname}", 0.0, kind="threshold", comparator=">=")
                        print(Fore.GREEN + f"[System] Reloaded {len(probes)} probe(s) from {PROBE_DIR}." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.RED + f"[System] Error refreshing probes: {e}" + Style.RESET_ALL)
                try:
                    with open(MACRO_ALIAS_PATH, "r", encoding="utf-8") as _af:
                        _loaded = {str(k): str(v) for k, v in json.load(_af).items()}
                    macro_aliases.clear()
                    macro_aliases.update(_loaded)
                    print(Fore.GREEN + f"[System] Reloaded {len(macro_aliases)} macro alias(es)." + Style.RESET_ALL)
                except (OSError, ValueError):
                    pass
                try:
                    with open(HIDDEN_CMD_PATH, "r", encoding="utf-8") as _hcf:
                        _raw_hidden = json.load(_hcf)
                    if isinstance(_raw_hidden, list):
                        _loaded_hidden = {str(x).lstrip(":").lower() for x in _raw_hidden if str(x).strip()}
                    elif isinstance(_raw_hidden, dict):
                        _loaded_hidden = {str(k).lstrip(":").lower() for k, v in _raw_hidden.items() if v}
                    else:
                        _loaded_hidden = set()
                    hidden_commands.clear()
                    hidden_commands.update(_loaded_hidden)
                    print(Fore.GREEN + f"[System] Reloaded {len(hidden_commands)} hidden command(s)." + Style.RESET_ALL)
                except (OSError, ValueError):
                    pass
                try:
                    with open(EXPOSED_CMD_PATH, "r", encoding="utf-8") as _ecf:
                        _loaded_exposed = {
                            str(k).lstrip(":").lower(): normalize_command_exposure(v)
                            for k, v in json.load(_ecf).items()
                        }
                    exposed_commands.clear()
                    exposed_commands.update(_loaded_exposed)
                    print(Fore.GREEN + f"[System] Reloaded {len(exposed_commands)} exposed command(s)." + Style.RESET_ALL)
                except (OSError, ValueError):
                    pass
                try:
                    with open(EXPOSED_KNOB_PATH, "r", encoding="utf-8") as _ekf:
                        _raw_knobs = json.load(_ekf)
                    if isinstance(_raw_knobs, list):
                        _loaded_knobs = {str(x).strip() for x in _raw_knobs if str(x).strip()}
                    elif isinstance(_raw_knobs, dict):
                        _loaded_knobs = {str(k).strip() for k, v in _raw_knobs.items() if v}
                    else:
                        _loaded_knobs = set()
                    exposed_knobs.clear()
                    exposed_knobs.update(k for k in _loaded_knobs if k in tuner.triggers and not _is_shadow_trigger(k, tuner))
                    print(Fore.GREEN + f"[System] Reloaded {len(exposed_knobs)} exposed knob(s)." + Style.RESET_ALL)
                except (OSError, ValueError):
                    pass
                print(Fore.YELLOW + "[System] (Macro files are re-read on every :run, so edits to them need no refresh.)" + Style.RESET_ALL)
                continue

            _cmdword = user_input.strip().split()[0].lower() if user_input.strip() else ""
            if _cmdword == ":help":
                harg = user_input.strip()[len(":help"):].strip().lower()
                if harg.startswith("expose"):
                    help_exposed = not harg.endswith("off")
                    print(Fore.GREEN + f"[Help] Model help {'EXPOSED -- it may now emit <<HELP>>' if help_exposed else 'hidden from the model'}." + Style.RESET_ALL)
                    continue
                if harg == "model":
                    print(
                        Fore.CYAN
                        + build_model_help_text(
                            list_solve_macros(),
                            exposed_commands,
                            exposed_knobs,
                            hidden_commands,
                            memory_tool_exposed=memory_tool_exposed,
                            memory_lanes=_enabled_memory_lanes(),
                        )
                        + Style.RESET_ALL
                    )
                    continue
                if harg:
                    q = harg.split()[0].lstrip(":")
                    matches = find_command_help_entries(harg)
                    spec_entries = spec_help_entries(harg)
                    if matches or spec_entries:
                        for entry in matches:
                            for _l in entry:
                                print(Fore.CYAN + _l + Style.RESET_ALL)
                        for entry in spec_entries:
                            for _l in entry:
                                print(Fore.CYAN + _l + Style.RESET_ALL)
                    elif q in macro_aliases:
                        # Generate dynamic help for macros
                        mpath = macro_aliases[q]
                        exists = os.path.isfile(mpath)
                        status = "" if exists else " (FILE MISSING)"
                        print(Fore.CYAN + f":{q} -- Macro aliased to {mpath}{status}" + Style.RESET_ALL)
                        
                        # Check if it's a solve-macro with a known goal
                        for _n, _d, _a in list_solve_macros():
                            if _n == q:
                                _argnote = f" [args: {_a}]" if _a else ""
                                print(Fore.CYAN + f"  Goal: {_d}{_argnote}" + Style.RESET_ALL)
                                break
                                
                        # Show the first few lines of the macro
                        if exists:
                            print(Fore.CYAN + "  Contents:" + Style.RESET_ALL)
                            try:
                                with open(mpath, "r", encoding="utf-8") as rf:
                                    lines = rf.read().splitlines()
                                    cmds = [line for line in lines if line.strip() and not line.strip().startswith("#")]
                                    for line in cmds[:5]:
                                        print(Fore.CYAN + f"    {line}" + Style.RESET_ALL)
                                    if len(cmds) > 5:
                                        print(Fore.CYAN + f"    ... and {len(cmds) - 5} more." + Style.RESET_ALL)
                            except Exception as e:
                                print(Fore.RED + f"    (Could not read macro file: {e})" + Style.RESET_ALL)
                        else:
                            print(Fore.RED + f"  (The file {mpath} does not exist or was deleted!)" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Help] No help entry for '{q}'.{did_you_mean(q, BUILTIN_COMMANDS | set(macro_aliases))}" + Style.RESET_ALL)
                    continue
                for _l in COMMAND_HELP_LINES:
                    print(Fore.CYAN + _l + Style.RESET_ALL)
                sm = list_solve_macros()
                if sm:
                    print(Fore.CYAN + "Solve-macros (:<name> <args>):" + Style.RESET_ALL)
                    for _n, _d, _a in sm:
                        _argnote = f"  [args: {_a}]" if _a else ""
                        print(Fore.CYAN + f"          :{_n}{(' -- ' + _d) if _d else ''}{_argnote}" + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + "(no solve-macros yet -- make one with :solve choose <goal>)" + Style.RESET_ALL)
                print(Fore.CYAN + "':help model' shows what the MODEL can run itself; ':help expose' lets it call <<HELP>>." + Style.RESET_ALL)
                continue
            if _cmdword in (":accept", ":reject"):
                parts = user_input.strip().split()
                verb = parts[0].lower()
                sel = parts[1].lower() if len(parts) > 1 else "all"
                # A staged :solve choose/auto macro takes precedence: adopt (write +
                # alias) or discard it before touching any staged prize commands.
                if pending_solve_proposal is not None:
                    p = pending_solve_proposal
                    pending_solve_proposal = None
                    if verb == ":reject":
                        print(Fore.YELLOW + f"[Reject] Discarded proposed macro ':{p['name']}'." + Style.RESET_ALL)
                        if p.get("definition_id"):
                            memory.append_definition_feedback(
                                p["definition_id"],
                                "operator rejected the proposed macro",
                                verdict="rejected",
                                source="operator",
                            )
                        continue
                    if _hidden_overwrite_blocked(p["name"], "Accept"):
                        continue
                    try:
                        os.makedirs(os.path.dirname(p["dest"]), exist_ok=True)
                        with open(p["dest"], "w", encoding="utf-8") as wf:
                            wf.write(f"# :solve macro '{p['name']}' -- {p['goal']}\n")
                            arg_header = macro_arg_header_items(
                                p.get("arg_specs") or p.get("clean_names") or []
                            )
                            if arg_header:
                                wf.write(f"# args: {', '.join(arg_header)}\n")
                            for ln in p["cmd_lines"]:
                                wf.write(ln + "\n")
                        macro_aliases[p["name"]] = p["dest"]
                        _save_macro_aliases()
                        if p.get("definition_id"):
                            memory.append_definition_feedback(
                                p["definition_id"],
                                "operator accepted and installed the proposed macro",
                                verdict="accepted",
                                source="operator",
                                provenance={"artifact_path": p["dest"]},
                            )
                        print(Fore.GREEN + f"[Accept] Adopted ':{p['name']}' ({len(p['cmd_lines'])} command(s)). Run it with :{p['name']} <args>." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Accept] Could not save macro: {e}" + Style.RESET_ALL)
                    continue
                if _last_completion is not None:
                    completion_definition_id = pending_completion_definition_id
                    pending_completion_definition_id = None
                    if verb == ":accept":
                        print(Fore.GREEN + f"[Accept] Queued autocomplete: {_last_completion}" + Style.RESET_ALL)
                        input_queue.append(_last_completion)
                    elif verb == ":reject":
                        print(Fore.YELLOW + f"[Reject] Discarded autocomplete: {_last_completion}" + Style.RESET_ALL)
                    if completion_definition_id:
                        memory.append_definition_feedback(
                            completion_definition_id,
                            "operator accepted and queued the completion"
                            if verb == ":accept"
                            else "operator rejected the completion",
                            verdict="accepted" if verb == ":accept" else "rejected",
                            source="operator",
                        )
                    _last_completion = None
                    continue
                if not pending_accept_commands:
                    print(Fore.CYAN + "[Accept] Nothing is staged for acceptance." + Style.RESET_ALL)
                    continue
                if sel == "all":
                    picks = list(range(len(pending_accept_commands)))
                else:
                    try:
                        i = int(sel) - 1
                        if not (0 <= i < len(pending_accept_commands)):
                            raise IndexError
                        picks = [i]
                    except (ValueError, IndexError):
                        print(Fore.YELLOW + f"[{verb[1:].capitalize()}] No staged command #{sel}. {len(pending_accept_commands)} staged -- use :{verb[1:]} <n> or :{verb[1:]} all." + Style.RESET_ALL)
                        continue
                chosen = [pending_accept_commands[i] for i in picks]
                for i in sorted(picks, reverse=True):
                    del pending_accept_commands[i]
                if verb == ":accept":
                    input_queue.extend(chosen)
                    print(Fore.GREEN + Style.BRIGHT + f"[Accept] Adopted {len(chosen)} command(s) into the queue: {'; '.join(chosen)}" + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + f"[Reject] Dropped {len(chosen)} staged command(s): {'; '.join(chosen)}" + Style.RESET_ALL)
                if pending_accept_commands:
                    print(Fore.CYAN + f"[Accept] {len(pending_accept_commands)} command(s) still staged." + Style.RESET_ALL)
                continue

            if _cmdword == ":hide":
                hargs = user_input.strip()[len(":hide"):].split()
                if not hargs:
                    if hidden_commands:
                        print(Fore.CYAN + "[Hide] Hidden from model-facing writing/help: " + ", ".join(f":{w}" for w in sorted(hidden_commands)) + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "[Hide] No commands are hidden. Use ':hide <command>' to hide one from model-facing writing/help, or ':hide off <command>' to reveal it." + Style.RESET_ALL)
                    continue
                if hargs[0].lower() in ("off", "show", "reveal", "visible"):
                    if len(hargs) < 2:
                        print(Fore.YELLOW + "[Hide] Usage: :hide off <command>" + Style.RESET_ALL)
                        continue
                    hargs = [hargs[1], "off"] + hargs[2:]
                word = hargs[0].lstrip(":").lower()
                mode_arg = hargs[1].lower() if len(hargs) > 1 else ""
                all_shell_commands = _all_shell_commands()
                if mode_arg in ("off", "show", "reveal", "visible"):
                    if word in hidden_commands:
                        hidden_commands.remove(word)
                        _save_hidden_commands()
                        print(Fore.GREEN + f"[Hide] ':{word}' is visible to model-facing writing/help again." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Hide] ':{word}' wasn't hidden." + Style.RESET_ALL)
                    continue
                if word not in all_shell_commands:
                    print(Fore.YELLOW + f"[Hide] ':{word}' isn't a command.{did_you_mean(word, all_shell_commands)}" + Style.RESET_ALL)
                    continue
                if mode_arg and mode_arg not in ("on", "hide", "hidden"):
                    print(Fore.YELLOW + f"[Hide] Unknown mode '{mode_arg}'. Use ':hide <command>' or ':hide off <command>'." + Style.RESET_ALL)
                    continue
                hidden_commands.add(word)
                _save_hidden_commands()
                print(Fore.GREEN + f"[Hide] ':{word}' is hidden from model-facing writing/help. Runtime tool access is unchanged; use :expose off :{word} to remove that." + Style.RESET_ALL)
                continue

            if _cmdword == ":expose":
                eargs = user_input.strip()[len(":expose"):].split()
                if not eargs:  # list exposures
                    exposed_probe_names = sorted(n for n in probes if probes[n].get("exposed"))
                    if not exposed_commands and not exposed_probe_names and not exposed_knobs and not memory_tool_exposed:
                        print(Fore.CYAN + "[Expose] Nothing exposed. Use ':expose [stage|direct] :command [fixed args...] [steer <magnitude>] because <activation>' for command-backed tools, or ':expose <probe|knob>' for model-readable probes/knobs." + Style.RESET_ALL)
                    lanes = _enabled_memory_lanes()
                    lane_text = ", ".join(lanes) if lanes else "none enabled"
                    print(Fore.CYAN + f"[Expose] memory read tool: {'on' if memory_tool_exposed else 'off'} (lanes: {lane_text})" + Style.RESET_ALL)
                    if exposed_commands:
                        for w, record in sorted(exposed_commands.items()):
                            hidden_note = " [hidden from writing/help]" if w in hidden_commands else ""
                            display_cmd = command_exposure_display(w, record, include_args=False)
                            print(
                                Fore.CYAN
                                + f"[Expose] command {display_cmd}  ({command_exposure_mode(record)}){hidden_note}\n"
                                + f"         fixed args: {command_exposure_args(record) or '(none; model supplies the full tail)'}\n"
                                + f"         activation: {command_exposure_activation(w, record)}\n"
                                + f"         steer: {command_exposure_steer(record)}"
                                + Style.RESET_ALL
                            )
                    if exposed_probe_names:
                        print(Fore.CYAN + "[Expose] probes: " + ", ".join(exposed_probe_names) + Style.RESET_ALL)
                    if exposed_knobs:
                        print(Fore.CYAN + "[Expose] knobs: " + ", ".join(sorted(exposed_knobs)) + Style.RESET_ALL)
                    continue
                turn_off_prefix, raw_agent, mode_prefix, toks = parse_expose_prefix(eargs)
                if not toks:
                    print(Fore.YELLOW + "[Expose] Usage: :expose [@agent] [stage|direct] :command [fixed args...] [steer <magnitude>] because <activation>, or :expose off <probe|knob|:command>." + Style.RESET_ALL)
                    continue
                if toks[0].startswith("@"):
                    print(Fore.YELLOW + "[Expose] Usage: :expose [@agent] [stage|direct] :command [fixed args...] [steer <magnitude>] because <activation>, or :expose off @agent :command." + Style.RESET_ALL)
                    continue
                target_agent = normalize_command_target(raw_agent) if raw_agent else ""
                target = toks[0]
                eargs = [target] + (["off"] if turn_off_prefix else []) + toks[1:]
                mode_arg = eargs[1].lower() if len(eargs) > 1 else ""
                if not target.startswith(":"):
                    if mode_arg not in ("", "on", "expose", "off", "hide"):
                        print(Fore.YELLOW + f"[Expose] Bare targets expose probes/knobs and only accept 'off'. For command-backed tools use ':expose [stage|direct] :{target} [fixed args...] [steer <magnitude>] because <activation>'." + Style.RESET_ALL)
                        continue
                    if target_agent:
                        print(Fore.YELLOW + "[Expose] Agent-routed exposure targets must be commands, e.g. ':expose @agent :command'." + Style.RESET_ALL)
                        continue
                    turn_off = mode_arg in ("off", "hide")
                    if target.lower().lstrip(":") == "memory":
                        memory_tool_exposed = not turn_off
                        state = "available to" if memory_tool_exposed else "hidden from"
                        print(Fore.GREEN + f"[Expose] memory is now {state} the model's <<MEMORY: ...>> tool. Operator :memory commands still work." + Style.RESET_ALL)
                        continue
                    probe_name = re.sub(r"[^a-z0-9_]", "_", target.lower())[:40].strip("_")
                    if probe_name in probes:
                        probes[probe_name]["exposed"] = not turn_off
                        pf = os.path.join(ROOT, "invariants", "out", "probes", f"{probe_name}.pt")
                        if os.path.exists(pf):
                            try:
                                data = torch.load(pf, weights_only=True)
                                data["exposed"] = probes[probe_name]["exposed"]
                                torch.save(data, pf)
                            except Exception:
                                pass
                        print(Fore.GREEN + f"[Expose] probe '{probe_name}' is now {'hidden from' if turn_off else 'exposed to'} the model's <<PROBE: {probe_name}>> tool." + Style.RESET_ALL)
                        continue
                    knob_name = target.strip()
                    if knob_name not in tuner.triggers and knob_name.lower() in tuner.triggers:
                        knob_name = knob_name.lower()
                    if knob_name in tuner.triggers and not _is_shadow_trigger(knob_name, tuner):
                        if turn_off:
                            exposed_knobs.discard(knob_name)
                        else:
                            exposed_knobs.add(knob_name)
                        _save_exposed_knobs()
                        print(Fore.GREEN + f"[Expose] knob '{knob_name}' is now {'hidden from' if turn_off else 'exposed to'} the model's <<PROBE: {knob_name}>> tool." + Style.RESET_ALL)
                        continue
                    candidates = set(probes) | {n for n in tuner.triggers if not _is_shadow_trigger(n, tuner)}
                    command_hint = f" For command-backed tools use ':expose [stage|direct] :{target} [fixed args...] [steer <magnitude>] because <activation>'." if target.lower() in _all_shell_commands() else ""
                    print(Fore.YELLOW + f"[Expose] '{target}' is not an active probe or knob.{did_you_mean(target, candidates)}{command_hint}" + Style.RESET_ALL)
                    continue
                word = target.lstrip(":").lower()
                command_args = eargs[1:]
                command_mode_arg = command_args[0].lower() if command_args else ""
                if command_mode_arg in ("off", "hide", "unexpose"):
                    if exposed_commands.pop(word, None) is not None:
                        _save_exposed_commands()
                        print(Fore.CYAN + f"[Expose] ':{word}' is no longer a model tool." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Expose] ':{word}' wasn't exposed." + Style.RESET_ALL)
                    continue
                if word == "expose":
                    print(Fore.RED + "[Expose] refusing to expose ':expose' itself -- the model could then self-grant any command." + Style.RESET_ALL)
                    continue
                known_exposable = _all_shell_commands() - {"expose"}
                if word not in known_exposable:
                    print(Fore.YELLOW + f"[Expose] ':{word}' isn't a command.{did_you_mean(word, known_exposable)}" + Style.RESET_ALL)
                    continue
                mode = mode_prefix or "stage"
                tail_args = list(command_args)
                if mode_prefix is None:
                    # legacy trailing position; with a leading mode given,
                    # nothing in the tail is ever eaten as a keyword.
                    if tail_args and tail_args[0].lower() in EXPOSE_STAGE_WORDS:
                        mode = "stage"
                        tail_args = tail_args[1:]
                    elif tail_args and tail_args[0].lower() in EXPOSE_DIRECT_WORDS:
                        mode = "direct"
                        tail_args = tail_args[1:]
                fixed_arg_tokens = []
                explicit_steer = None
                i = 0
                while i < len(tail_args):
                    token = tail_args[i]
                    low = token.lower()
                    if low.startswith("steer="):
                        explicit_steer = token.split("=", 1)[1].strip()
                        if i + 1 < len(tail_args):
                            explicit_steer = (explicit_steer + " " + " ".join(tail_args[i + 1:])).strip()
                        break
                    if low == "steer":
                        explicit_steer = " ".join(tail_args[i + 1:]).strip()
                        break
                    else:
                        fixed_arg_tokens.append(token)
                    i += 1
                if explicit_steer is not None and not explicit_steer:
                    print(Fore.YELLOW + "[Expose] Missing steer magnitude/effect after 'steer'." + Style.RESET_ALL)
                    continue
                prior_exposure = normalize_command_exposure(exposed_commands.get(word, {}))
                activation = command_because.strip() if command_because else prior_exposure["activation"]
                steer_magnitude = explicit_steer if explicit_steer is not None else prior_exposure["steer_magnitude"]
                fixed_args = " ".join(fixed_arg_tokens).strip() if fixed_arg_tokens else prior_exposure["args"]
                exposed_commands[word] = {
                    "mode": mode,
                    "activation": activation,
                    "steer_magnitude": steer_magnitude,
                    "args": fixed_args,
                    "target": target_agent,
                }
                _save_exposed_commands()
                run_note = "is queued immediately" if mode == "direct" else "is staged for your :accept"
                display_cmd = command_exposure_display(word, exposed_commands[word], include_args=False)
                print(Fore.GREEN + f"[Expose] '{display_cmd}' is now a model tool -- it can call <<TOOL: {display_cmd} ...>> when its activation criterion is met, and it {run_note} ({mode})." + Style.RESET_ALL)
                print(Fore.CYAN + f"         fixed args: {fixed_args or '(none; model supplies the full tail)'}" + Style.RESET_ALL)
                print(Fore.CYAN + f"         activation: {command_exposure_activation(word, exposed_commands[word])}" + Style.RESET_ALL)
                print(Fore.CYAN + f"         steer: {command_exposure_steer(exposed_commands[word])}" + Style.RESET_ALL)
                _ekw, _, _ = ensure_command_knobs(tuner, word)
                _ecrit_probe = next(
                    (p for p in probes if activation and re.search(rf"(?<!\w){re.escape(p)}(?!\w)", activation, re.IGNORECASE)),
                    None,
                )
                _ecrit_note = (
                    f"observing probe '{_ecrit_probe}' each reply -- calibratable"
                    if _ecrit_probe else "manual dial (name a probe in the because-clause to make it observable)"
                )
                print(Fore.CYAN + f"         knobs: cmd_{_ekw} (metric: used-vs-unused lift; ':figure lift:cmd_{_ekw}') "
                      + f"and cmd_{_ekw}_criteria ({_ecrit_note})." + Style.RESET_ALL)
                if not activation:
                    print(Fore.YELLOW + f"         Note: no activation criterion recorded. Re-run with 'because <activation>' so the tool has a real trigger." + Style.RESET_ALL)
                if word in hidden_commands:
                    print(Fore.CYAN + f"         Note: ':{word}' is still hidden from model-facing writing/help. Reveal it with ':hide off :{word}' if it should be documented too." + Style.RESET_ALL)
                if word in EXPOSE_META_CMDS:
                    print(Fore.YELLOW + f"         WARNING: ':{word}' takes another command/macro/script as its argument, so through it the model can reach commands you did NOT expose. Expose it only if you mean to." + Style.RESET_ALL)
                elif word in macro_aliases and word not in BUILTIN_COMMANDS:
                    print(Fore.YELLOW + f"         WARNING: ':{word}' is a macro command; it expands to whatever commands its macro file contains." + Style.RESET_ALL)
                continue

            if user_input.startswith(":game ") or user_input.startswith(":game,") or user_input.strip() == ":game":
                body = user_input[len(":game"):].strip()
                if body.startswith(","):
                    body = body[1:].strip()
                if not body:
                    print(Fore.YELLOW + "[Game] Usage: :game <name> [ +/-<rule> <desc> | +/-win <cond> | +/-loss <cond> | +/-prize <opt>,<odds> [<count>,<odds_val>] <cmd> | draw | start | end ]" + Style.RESET_ALL)
                    continue
                _nm = re.split(r"[\s,]+", body, maxsplit=1)
                gname = _nm[0].strip()
                rest = _nm[1].strip().lstrip(",").strip() if len(_nm) > 1 else ""
                if gname.lower() == "restore":
                    if not rest:
                        print(Fore.YELLOW + "[Game] Usage: :game restore {\"name\":\"...\",\"config\":{...}}" + Style.RESET_ALL)
                        continue
                    try:
                        obj = json.loads(rest)
                        rname = re.sub(r"[^a-z0-9_]", "_", str(obj.get("name", "")).lower())[:40].strip("_")
                        if not rname:
                            raise ValueError("missing game name")
                        cfg = normalize_game_config(obj.get("config", {}))
                        save_game_config(os.path.join(ROOT, "games", f"{rname}_rules.json"), cfg)
                        n_rules = len(cfg.get("rules", {}))
                        n_prizes = len(cfg.get("prizes", {}))
                        print(Fore.GREEN + f"[Game] Restored config for '{rname}' ({n_rules} rule(s), {n_prizes} prize(s))." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Game] Could not restore game config: {e}" + Style.RESET_ALL)
                    continue
                # Operator declines the game the model just proposed (the symmetric
                # counterpart to accepting it with ':game <name>').
                if gname.lower() in ("no", "decline", "reject", "dismiss") and model_proposed_game:
                    declined = model_proposed_game
                    model_proposed_game = None
                    pending_game_tool_result = None
                    print(Fore.YELLOW + f"[Game] Declined the model's proposal to play '{declined}'." + Style.RESET_ALL)
                    memory.append_event("game_proposal_declined", tags=["game", "self_concept"], provenance={"game": declined})
                    continue

                # :game auto [+trait ...] [-trait ...] -- design a conversational
                # game whose whole goal is to increase the named trait(s)/probe(s),
                # then PROPOSE it (operator accepts with ':game <name>'/':game start',
                # declines with ':game no'). e.g. ':game auto +consciousness'.
                if gname.lower() == "auto":
                    _atoks = rest.split()
                    plus = [t.lstrip("+") for t in _atoks if t.startswith("+")]
                    minus = [t.lstrip("-") for t in _atoks if t.startswith("-")]
                    bare = [t for t in _atoks if not t.startswith(("+", "-"))]
                    focus = ", ".join(plus) if plus else (bare[0] if bare else "the model's cognition")
                    base = plus[0] if plus else (bare[0] if bare else "trait")
                    auto_name = re.sub(r"[^a-z0-9_]", "_", f"improve_{base}".lower())[:40].strip("_") or "improve_trait"
                    _n, _i = auto_name, 2
                    while os.path.isfile(os.path.join(ROOT, "games", f"{_n}_rules.json")) or os.path.isfile(os.path.join(ROOT, "games", f"{_n}.py")):
                        _n, _i = f"{auto_name}_{_i}", _i + 1
                    auto_name = _n
                    auto_rules_path = os.path.join(ROOT, "games", f"{auto_name}_rules.json")
                    gcfg2 = load_game_config(auto_rules_path)
                    gcfg2["win"] = f"{focus} is measurably higher than when the game began (its probe reads further from 0 / it is more present in the replies)."
                    gcfg2["loss"] = f"{focus} stalls or declines over several turns."
                    gcfg2["rules"]["goal"] = f"Each turn, make one deliberate move to increase {focus}; name the move and why it should raise {focus}."
                    if minus:
                        gcfg2["rules"]["avoid"] = f"reduce {', '.join(minus)} while doing so."
                    default_premise_prompt = (
                        f":expect macro premise_{auto_name}\n"
                        ":probe premise_author I am writing exactly ONE shell command to set a game premise || I am generating conversational text\n"
                        ":probe release premise_author off\n"
                        ":probe backfill premise_author\n"
                        ":steer premise_author 1.5\n"
                        "In one vivid sentence, describe a short cooperative game whose whole point is for you to increase your own $focus.\n"
                        "Write a macro containing exactly ONE command to set this premise.\n"
                        "The command must be formatted exactly like this:\n"
                        ":game $auto_name +premise [your one vivid sentence here]\n"
                        "Output ONLY that command line, nothing else.\n"
                        ":steer premise_author 0\n"
                        ":probe release premise_author"
                    )
                    premise_prompt = load_prompt(
                        "game_premise",
                        default_premise_prompt,
                        required_substring=":probe release premise_author",
                        focus=focus,
                        auto_name=auto_name,
                    )
                    print(Fore.CYAN + f"[Game] Queuing phenomenological macro to generate a premise for '{auto_name}'..." + Style.RESET_ALL)
                    queue_macro_text(premise_prompt, input_queue)
                    save_game_config(auto_rules_path, gcfg2)
                    model_proposed_game = auto_name
                    print(Fore.GREEN + Style.BRIGHT + f"[Game] Auto-designed '{auto_name}' to improve {focus}." + Style.RESET_ALL)
                    if gcfg2["rules"].get("premise"):
                        print(Fore.CYAN + f"       premise: {gcfg2['rules']['premise']}" + Style.RESET_ALL)
                    print(Fore.CYAN + f"       win: {gcfg2['win']}" + Style.RESET_ALL)
                    print(Fore.CYAN + f"       Accept with ':game {auto_name}' (or ':game start'); decline with ':game no'." + Style.RESET_ALL)
                    continue

                gpath = os.path.join(ROOT, "games", f"{gname}.py")
                rules_path = os.path.join(ROOT, "games", f"{gname}_rules.json")
                gcfg = load_game_config(rules_path)

                first = rest.split()[0] if rest else ""
                sign = "+" if first.startswith("+") else ("-" if first.startswith("-") else None)
                key = (first[1:] if sign else first).lower()
                arg = rest[len(first):].strip() if first else ""

                # --- prizes: +prize <opt>,<odds> [<count>,<odds_val>] <cmd> / -prize <opt> ---
                if key == "prize" and sign:
                    if sign == "+":
                        option, spec = parse_prize_spec(arg)
                        if option is None:
                            print(Fore.YELLOW + f"[Game] Bad prize ({spec}). Usage: :game {gname} +prize <opt>,<odds> [<count>,<odds_val>] <cmd>" + Style.RESET_ALL)
                            continue
                        gcfg["prizes"][option] = spec
                        save_game_config(rules_path, gcfg)
                        cnt = "unlimited" if spec["count"] is None else spec["count"]
                        print(Fore.GREEN + f"[Game] Prize '{option}' on {gname}: odds {spec['odds']:g}, count {cnt}" + (f", wins run '{spec['command']}'" if spec["command"] else "") + "." + Style.RESET_ALL)
                    else:
                        opt = (arg.split(",")[0].strip() if arg else "")
                        if gcfg["prizes"].pop(opt, None) is not None:
                            save_game_config(rules_path, gcfg)
                            print(Fore.YELLOW + f"[Game] Removed prize '{opt}' from {gname}." + Style.RESET_ALL)
                        else:
                            print(Fore.YELLOW + f"[Game] No prize '{opt}' on {gname}." + Style.RESET_ALL)
                    continue

                # --- win / loss conditions: +win <cond> / -win, +loss <cond> / -loss ---
                if key in ("win", "loss") and sign:
                    gcfg[key] = arg if sign == "+" else ""
                    save_game_config(rules_path, gcfg)
                    if sign == "+":
                        print(Fore.GREEN + f"[Game] {gname} {key} condition: {arg or '(empty)'}. (the {gname}.py/model evaluates it)" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Game] Cleared {gname} {key} condition." + Style.RESET_ALL)
                    continue

                if sign and key not in GAME_RESERVED:
                    rule_name = first[1:]
                    if sign == "+":
                        new_desc = arg.lstrip(",").strip()
                        if rule_name in gcfg["rules"]:
                            gcfg["rules"][rule_name] += " " + new_desc
                            print(Fore.GREEN + f"[Game] Appended to rule '{rule_name}' in {gname}: '{gcfg['rules'][rule_name]}'." + Style.RESET_ALL)
                        else:
                            gcfg["rules"][rule_name] = new_desc
                            print(Fore.GREEN + f"[Game] Added rule '{rule_name}' to {gname}: '{gcfg['rules'][rule_name]}'." + Style.RESET_ALL)
                    else:
                        gcfg["rules"].pop(rule_name, None)
                        print(Fore.YELLOW + f"[Game] Removed rule '{rule_name}' from {gname}." + Style.RESET_ALL)
                    save_game_config(rules_path, gcfg)
                    print(Fore.CYAN + f"[Game] (the {gname}.py script or the model enforces this.)" + Style.RESET_ALL)
                    continue

                # --- draw prizes now: run each against its odds ---
                if key == "draw":
                    won = draw_prizes(gcfg["prizes"])
                    save_game_config(rules_path, gcfg)
                    if not won:
                        print(Fore.CYAN + f"[Game] Draw: nothing hit (of {len(gcfg['prizes'])} prize(s))." + Style.RESET_ALL)
                    for opt, cmd in won:
                        print(Fore.MAGENTA + Style.BRIGHT + f"[Game] Prize won: {opt}!" + Style.RESET_ALL)
                        if cmd:
                            _stage_for_accept(cmd, why=f"prize '{opt}'")
                    continue

                # --- end / stop / quit: award prizes, then end the active game ---
                if key in ("end", "stop", "quit") or gname.lower() in ("end", "stop", "quit"):
                    if active_game:
                        for opt, cmd in draw_prizes(gcfg["prizes"]):
                            print(Fore.MAGENTA + Style.BRIGHT + f"[Game] End prize: {opt}!" + Style.RESET_ALL)
                            if cmd:
                                _stage_for_accept(cmd, why=f"end prize '{opt}'")
                        save_game_config(rules_path, gcfg)
                        print(Fore.CYAN + f"[Game] Ended active game: {active_game}." + Style.RESET_ALL)
                        active_game = None
                        pending_game_tool_result = None
                    else:
                        print(Fore.YELLOW + "[Game] No game is currently active." + Style.RESET_ALL)
                    continue

                # --- :game start (bare): accept & start the game the model just
                # proposed, without retyping its name ---
                if gname.lower() == "start" and not rest:
                    if not model_proposed_game:
                        print(Fore.YELLOW + "[Game] Nothing proposed to start. Propose one with :game <name>, or force a script with :game <name> start." + Style.RESET_ALL)
                        continue
                    gname = model_proposed_game
                    gpath = os.path.join(ROOT, "games", f"{gname}.py")
                    rules_path = os.path.join(ROOT, "games", f"{gname}_rules.json")
                    gcfg = load_game_config(rules_path)
                    key = "start"

                force_start = (key == "start")
                has_script = os.path.isfile(gpath)
                accepting = force_start or (model_proposed_game == gname)

                if not has_script and not accepting:
                    # No script and no pending proposal -> ask the model to DESIGN
                    # the game (form 3), then fall through to its turn.
                    print(Fore.CYAN + f"[Game] No script for '{gname}' -- asking the model to design it..." + Style.RESET_ALL)
                    cond = ""
                    if gcfg["rules"] or gcfg["win"] or gcfg["loss"] or gcfg["prizes"]:
                        cond = (f" Honor the existing config -- rules={list(gcfg['rules'])}, "
                                f"win='{gcfg['win']}', loss='{gcfg['loss']}', prizes={list(gcfg['prizes'])}.")
                    user_input = (
                        f"Design a short, playable game called '{gname}': clear rules, options, and "
                        f"win/loss conditions (probe-based if apt).{cond} Then propose it to me to play."
                    )
                    # Fall through (no continue) so this reaches the model's turn.
                elif not accepting:
                    # A real script exists but nothing is pending -> propose it.
                    print(Fore.CYAN + f"[Game] Proposing {gname} to the model... (waiting for acceptance)" + Style.RESET_ALL)
                    user_proposed_game = gname
                    pending_game_tool_result = f"[Game Proposal] The operator proposes playing '{gname}'. To accept, output <<GAME_ACCEPT: {gname}>>; to decline, output <<GAME_DECLINE: {gname}>>. Either way, also reply in words."
                    user_input = "(system: operator proposed a game)"
                    continue
                else:
                    # Accepting (force_start, or the model's proposal) -> START.
                    print(Fore.GREEN + (f"[Game] Forcibly starting {gname}!" if force_start else f"[Game] You accepted the model's proposal to play {gname}!") + Style.RESET_ALL)
                    model_proposed_game = None
                    active_game = gname
                    if gcfg["rules"]:
                        print(Fore.CYAN + f"Active rules: {', '.join(gcfg['rules'])}." + Style.RESET_ALL)
                    if gcfg["prizes"]:
                        print(Fore.CYAN + f"Prizes: {', '.join(gcfg['prizes'])}." + Style.RESET_ALL)
                    if has_script:
                        print(Fore.MAGENTA + f"[Game] Initializing {gname}..." + Style.RESET_ALL)
                        try:
                            with open(gpath, "r", encoding="utf-8") as _gf:
                                gcode = _gf.read()
                            exec_globals = globals().copy()
                            exec_globals["GAME_RULES"] = gcfg["rules"]   # legacy: flat rules dict
                            exec_globals["GAME_CONFIG"] = gcfg           # full config: rules/win/loss/prizes
                            exec_globals["game_args"] = rest.split()
                            exec_globals["active_game_state"] = active_game_state
                            exec_globals["probes"] = probes              # games read/act on live probes
                            exec_globals["run_game"] = _run_game_ref     # games can reference each other
                            exec(gcode, exec_globals)
                        except Exception as e:
                            import traceback
                            print(Fore.RED + f"[Game Error] {e}\n{traceback.format_exc()}" + Style.RESET_ALL)
                    else:
                        # Conversational (model-designed, no script): mark active and
                        # hand the model its config to run turn by turn.
                        print(Fore.MAGENTA + f"[Game] '{gname}' is active -- no script, so we play it in conversation." + Style.RESET_ALL)
                        pending_game_tool_result = (
                            f"[Game] '{gname}' is now active; run it with me turn by turn. "
                            f"Rules: {gcfg['rules'] or '(none)'}; win: {gcfg['win'] or '(none)'}; "
                            f"loss: {gcfg['loss'] or '(none)'}; prizes: {list(gcfg['prizes']) or '(none)'}."
                        )
                    continue
            if user_input.startswith(":steer "):
                sargs = user_input[len(":steer "):].strip().split()
                # :steer auto | off  -> back to the ranking.
                if sargs and sargs[0].lower() in ("auto", "off", "none", "unpin"):
                    prioritize_pin["probe"] = None
                    prioritize_pin["mix"] = None
                    prioritize_pin["landscape"] = None
                    print(Fore.CYAN + "[Steer] target = AUTO (follows the live ranking each turn)." + Style.RESET_ALL)
                    continue
                # :steer mix <p1> <p2> ... [alpha]  -> a lift-weighted mix: you
                # pick the probes, each probe's DEGREE is its own learned lift
                # (signed -- a probe that hurts sense subtracts, i.e. desteers).
                if sargs and sargs[0].lower() == "mix":
                    rest = sargs[1:]
                    mix_alpha = None
                    if rest:
                        try:
                            mix_alpha = abs(float(rest[-1])); rest = rest[:-1]
                        except ValueError:
                            pass
                    names = [n for n in rest if n in probes and not probe_is_released(probes[n])]
                    bad = [n for n in rest if n not in probes]
                    released_names = [n for n in rest if n in probes and probe_is_released(probes[n])]
                    if not names:
                        print(Fore.YELLOW + f"[Steer] name at least one active, unreleased probe to mix.{(' unknown: ' + ', '.join(bad)) if bad else ''}{(' released: ' + ', '.join(released_names)) if released_names else ''}" + Style.RESET_ALL)
                        continue
                    prioritize_pin["mix"] = names
                    prioritize_pin["probe"] = None
                    prioritize_pin["landscape"] = None
                    if mix_alpha is not None:
                        tuner.set("prioritize_alpha", mix_alpha)
                    a_now = tuner.get("prioritize_alpha", 0.0)
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + f"[Steer] prioritize_alpha mapped to a lift-weighted MIX of {', '.join(names)} "
                        + f"(alpha={round(a_now,4)}{' -- set >0 to take effect' if a_now <= 0 else ''}). "
                        + "Each probe's degree is its own learned lift, recomputed every turn; a negative-lift "
                        + "probe desteers. :steer auto to unpin." + (f" (ignored unknown: {', '.join(bad)})" if bad else "")
                        + (f" (ignored released: {', '.join(released_names)})" if released_names else "")
                        + Style.RESET_ALL
                    )
                    continue
                # :steer quality -- a scalar field type with its own distance
                # formula, independent of mass/G. Formula input is allowlisted.
                if sargs and sargs[0].lower() in ("quality", "qualities"):
                    bodies = steer_bodies()
                    qualities = bodies["qualities"]
                    if len(sargs) == 1:
                        if not qualities:
                            print(Fore.CYAN + "[Steer] no quality types." + Style.RESET_ALL)
                        for qname, qcfg in sorted(qualities.items()):
                            strength = qcfg.get("strength") or {"kind": "constant", "value": 1.0}
                            now = resolve_field_source(
                                strength, probes, tuner, bodies["g_families"], bodies["fields"]
                            )
                            print(
                                Fore.CYAN
                                + f"[Steer] quality:{qname}: formula={qcfg.get('formula', QUALITY_FORMULA_DEFAULT)}; "
                                + f"strength<-{format_field_source(strength)} -> {now:+g}"
                                + Style.RESET_ALL
                            )
                        continue
                    qname = re.sub(r"[^a-z0-9_]", "_", sargs[1].lower()).strip("_")
                    action = sargs[2].lower() if len(sargs) > 2 else "status"
                    if action == "drop":
                        existed = qualities.pop(qname, None) is not None
                        save_steer_bodies()
                        print(
                            (Fore.GREEN if existed else Fore.YELLOW)
                            + (f"[Steer] quality:{qname} dropped; set laws that name it are now inert."
                               if existed else f"[Steer] no quality named '{qname}'.")
                            + Style.RESET_ALL
                        )
                        continue
                    if action == "status":
                        qcfg = qualities.get(qname)
                        if not qcfg:
                            print(Fore.YELLOW + f"[Steer] no quality named '{qname}'." + Style.RESET_ALL)
                        else:
                            print(Fore.CYAN + f"[Steer] quality:{qname}: formula={qcfg.get('formula', QUALITY_FORMULA_DEFAULT)}, strength={format_field_source(qcfg.get('strength'))}" + Style.RESET_ALL)
                        continue
                    if action != "formula" or len(sargs) < 4:
                        print(Fore.YELLOW + "[Steer] Usage: :steer quality <name> formula <expr-in-d> [strength <source> [scale [offset]]] | drop" + Style.RESET_ALL)
                        continue
                    formula = sargs[3]
                    formula_error = validate_quality_formula(formula)
                    if formula_error:
                        print(Fore.YELLOW + f"[Steer] invalid quality formula: {formula_error}." + Style.RESET_ALL)
                        continue
                    strength = (qualities.get(qname) or {}).get("strength") or {
                        "kind": "constant", "value": 1.0
                    }
                    if len(sargs) > 4:
                        if sargs[4].lower() != "strength" or len(sargs) < 6:
                            print(Fore.YELLOW + "[Steer] After the formula use: strength <source> [scale [offset]]." + Style.RESET_ALL)
                            continue
                        try:
                            scale = float(sargs[6]) if len(sargs) > 6 else 1.0
                            offset = float(sargs[7]) if len(sargs) > 7 else 0.0
                        except ValueError:
                            print(Fore.YELLOW + "[Steer] strength scale and offset must be numbers." + Style.RESET_ALL)
                            continue
                        strength = parse_field_source(sargs[5], scale, offset)
                        source_error = field_source_error(
                            strength, probes, tuner, bodies["g_families"]
                        ) if strength else "invalid strength source"
                        if source_error:
                            print(Fore.YELLOW + f"[Steer] {source_error}." + Style.RESET_ALL)
                            continue
                    qualities[qname] = {"formula": formula, "strength": strength}
                    save_steer_bodies()
                    print(Fore.GREEN + f"[Steer] quality:{qname} = {formula}; strength <- {format_field_source(strength)}." + Style.RESET_ALL)
                    continue
                # :steer set -- a law-defined set. Each member has its own
                # compliance and forward-time vector; exclusion is exact zero.
                if sargs and sargs[0].lower() in ("set", "sets", "lawset"):
                    bodies = steer_bodies()
                    sets = bodies["sets"]
                    if len(sargs) == 1:
                        if not sets:
                            print(Fore.CYAN + "[Steer] no quality set laws." + Style.RESET_ALL)
                        for lname, law in sorted(sets.items()):
                            print(
                                Fore.CYAN
                                + f"[Steer] set:{lname}: quality={law.get('quality', '(unset)')}; "
                                + f"members={len(law.get('members') or {})}; "
                                + f"excluded={','.join(sorted(law.get('excluded') or [])) or '(none)'}"
                                + Style.RESET_ALL
                            )
                        continue
                    lname = re.sub(r"[^a-z0-9_]", "_", sargs[1].lower()).strip("_")
                    action = sargs[2].lower() if len(sargs) > 2 else "status"
                    if action == "drop":
                        existed = sets.pop(lname, None) is not None
                        save_steer_bodies()
                        print(
                            (Fore.GREEN if existed else Fore.YELLOW)
                            + (f"[Steer] set:{lname} dropped." if existed else f"[Steer] no set law named '{lname}'.")
                            + Style.RESET_ALL
                        )
                        continue
                    if action == "quality" and len(sargs) >= 4:
                        qname = sargs[3].lower()
                        if qname not in bodies["qualities"]:
                            print(Fore.YELLOW + f"[Steer] no quality named '{qname}'. Define it first with :steer quality." + Style.RESET_ALL)
                            continue
                        sets.setdefault(lname, {"members": {}, "excluded": []})["quality"] = qname
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] set:{lname} law uses quality:{qname}." + Style.RESET_ALL)
                        continue
                    if action in ("add", "exclude", "include", "remove") and len(sargs) >= 4:
                        if lname not in sets:
                            print(Fore.YELLOW + f"[Steer] no set law named '{lname}'. Define its quality first." + Style.RESET_ALL)
                            continue
                        node = canonical_field_node(
                            sargs[3], probes, bodies["anchors"], bodies["g_families"]
                        )
                        if not node or node.startswith("family:"):
                            print(Fore.YELLOW + f"[Steer] unknown set member '{sargs[3]}'; use a probe, @anchor, or layer:N." + Style.RESET_ALL)
                            continue
                        law = sets[lname]
                        exclusions = set(law.get("excluded") or [])
                        if action == "exclude":
                            exclusions.add(node)
                            law["excluded"] = sorted(exclusions)
                            save_steer_bodies()
                            print(Fore.GREEN + f"[Steer] {node} is HARD-EXCLUDED from set:{lname}; compliance is exactly zero." + Style.RESET_ALL)
                            continue
                        if action == "remove":
                            (law.get("members") or {}).pop(node, None)
                            exclusions.discard(node)
                            law["excluded"] = sorted(exclusions)
                            save_steer_bodies()
                            print(Fore.GREEN + f"[Steer] {node} removed from set:{lname}." + Style.RESET_ALL)
                            continue
                        existing_member = (law.get("members") or {}).get(node) or {}
                        member = {
                            "compliance": dict(
                                existing_member.get("compliance")
                                or {"kind": "constant", "value": 1.0}
                            ),
                            "time_vector": list(existing_member.get("time_vector") or [1.0]),
                        }
                        if action == "add":
                            i = 4
                            while i < len(sargs):
                                key = sargs[i].lower()
                                if key == "compliance" and i + 1 < len(sargs):
                                    source_raw = sargs[i + 1]
                                    j = i + 2
                                    scale, offset = 1.0, 0.0
                                    if j < len(sargs) and sargs[j].lower() not in ("time", "compliance"):
                                        try:
                                            scale = float(sargs[j]); j += 1
                                            if j < len(sargs) and sargs[j].lower() not in ("time", "compliance"):
                                                offset = float(sargs[j]); j += 1
                                        except ValueError:
                                            print(Fore.YELLOW + "[Steer] compliance scale/offset must be numbers." + Style.RESET_ALL)
                                            break
                                    source = parse_field_source(source_raw, scale, offset)
                                    source_error = field_source_error(
                                        source, probes, tuner, bodies["g_families"]
                                    ) if source else "invalid compliance source"
                                    if source_error:
                                        print(Fore.YELLOW + f"[Steer] {source_error}." + Style.RESET_ALL)
                                        break
                                    member["compliance"] = source
                                    i = j
                                elif key == "time" and i + 1 < len(sargs):
                                    vector = parse_time_vector(sargs[i + 1])
                                    if not vector:
                                        print(Fore.YELLOW + "[Steer] time vector must be comma-separated numbers." + Style.RESET_ALL)
                                        break
                                    member["time_vector"] = vector
                                    i += 2
                                else:
                                    print(Fore.YELLOW + f"[Steer] unknown set-member clause '{sargs[i]}'; use compliance and/or time." + Style.RESET_ALL)
                                    break
                            else:
                                exclusions.discard(node)
                                law["excluded"] = sorted(exclusions)
                                law.setdefault("members", {})[node] = member
                                save_steer_bodies()
                                print(
                                    Fore.GREEN
                                    + f"[Steer] {node} in set:{lname}: compliance<-{format_field_source(member.get('compliance'))}; "
                                    + f"time=[{','.join(f'{v:g}' for v in member.get('time_vector') or [1.0])}]."
                                    + Style.RESET_ALL
                                )
                                continue
                            continue
                        exclusions.discard(node)
                        law["excluded"] = sorted(exclusions)
                        law.setdefault("members", {})[node] = member
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] {node} included in set:{lname}." + Style.RESET_ALL)
                        continue
                    if action == "status":
                        law = sets.get(lname)
                        if not law:
                            print(Fore.YELLOW + f"[Steer] no set law named '{lname}'." + Style.RESET_ALL)
                        else:
                            print(Fore.CYAN + f"[Steer] set:{lname} quality={law.get('quality', '(unset)')}." + Style.RESET_ALL)
                            for node, member in sorted((law.get("members") or {}).items()):
                                excluded = node in set(law.get("excluded") or [])
                                print(Fore.CYAN + f"  {node}: {'EXCLUDED' if excluded else 'included'}; compliance<-{format_field_source(member.get('compliance'))}; time={member.get('time_vector') or [1.0]}" + Style.RESET_ALL)
                            excluded_only = sorted(
                                set(law.get("excluded") or []) - set(law.get("members") or {})
                            )
                            for node in excluded_only:
                                print(Fore.CYAN + f"  {node}: EXCLUDED (not otherwise a member)" + Style.RESET_ALL)
                        continue
                    print(Fore.YELLOW + "[Steer] Usage: :steer set <law> quality <quality> | add <node> [compliance <source> [scale [offset]]] [time <v1,v2,...>] | exclude|include|remove <node> | drop" + Style.RESET_ALL)
                    continue
                # Whole-field exclusion: unlike zero compliance in one set,
                # this prevents the node from participating in any field law.
                if sargs and sargs[0].lower() in ("exclude", "excluded"):
                    if len(sargs) < 2:
                        excluded = sorted(
                            node for node, cfg in steer_bodies()["fields"].items()
                            if cfg.get("excluded")
                        )
                        print(Fore.CYAN + f"[Steer] whole-field exclusions: {', '.join(excluded) or '(none)'}" + Style.RESET_ALL)
                        continue
                    node, cfg = field_target_config(sargs[1], probes, create=True)
                    if not node or cfg is None or node.startswith("family:"):
                        print(Fore.YELLOW + f"[Steer] unknown field node '{sargs[1]}'." + Style.RESET_ALL)
                        continue
                    turn_off = len(sargs) > 2 and sargs[2].lower() in ("off", "include", "false")
                    cfg["excluded"] = not turn_off
                    save_steer_bodies()
                    print(
                        Fore.GREEN
                        + (f"[Steer] {node} returned to field eligibility."
                           if turn_off else f"[Steer] {node} is HARD-EXCLUDED from the entire field.")
                        + Style.RESET_ALL
                    )
                    continue
                # :steer family -- named G/shape/time inheritance groups.
                if sargs and sargs[0].lower() in ("family", "families", "gfamily"):
                    bodies = steer_bodies()
                    families = bodies["g_families"]
                    if len(sargs) == 1:
                        if not families:
                            print(Fore.CYAN + "[Steer] no G families. Define one with ':steer family <name> <G-source>'." + Style.RESET_ALL)
                        for fname, fcfg in sorted(families.items()):
                            g_now = resolve_field_source(
                                fcfg.get("g"), probes, tuner, families, bodies["fields"]
                            )
                            t_now = resolve_field_source(
                                fcfg.get("time") or {"kind": "constant", "value": 1.0},
                                probes, tuner, families, bodies["fields"], channel="time",
                            )
                            members = sorted(
                                node for node, cfg in bodies["fields"].items()
                                if cfg.get("family") == fname
                            )
                            print(
                                Fore.CYAN
                                + f"[Steer] family:{fname}: G={format_field_source(fcfg.get('g'))} -> {g_now:+g}, "
                                + f"time={format_field_source(fcfg.get('time')) if fcfg.get('time') else '1'} -> {t_now:g}, "
                                + f"shape={format_field_shape(fcfg.get('shape'))}; "
                                + f"members={', '.join(members) or '(none)'}"
                                + Style.RESET_ALL
                            )
                        continue
                    fname = re.sub(r"[^a-z0-9_]", "_", sargs[1].lower()).strip("_")
                    if not fname:
                        print(Fore.YELLOW + "[Steer] Usage: :steer family <name> <G-source|add <node>|remove <node>|drop>" + Style.RESET_ALL)
                        continue
                    action = sargs[2].lower() if len(sargs) > 2 else "status"
                    if action == "drop":
                        existed = families.pop(fname, None) is not None
                        for cfg in bodies["fields"].values():
                            if cfg.get("family") == fname:
                                cfg.pop("family", None)
                        save_steer_bodies()
                        print(
                            (Fore.GREEN if existed else Fore.YELLOW)
                            + (f"[Steer] family:{fname} dropped; its members now inherit global G."
                               if existed else f"[Steer] no family named '{fname}'.")
                            + Style.RESET_ALL
                        )
                        continue
                    if action in ("add", "remove") and len(sargs) >= 4:
                        if fname not in families:
                            print(Fore.YELLOW + f"[Steer] no family named '{fname}'. Define its G first." + Style.RESET_ALL)
                            continue
                        node, cfg = field_target_config(sargs[3], probes, create=True)
                        if not node or node.startswith("family:"):
                            print(Fore.YELLOW + f"[Steer] unknown field node '{sargs[3]}'; use a probe, @anchor, or layer:N." + Style.RESET_ALL)
                            continue
                        if action == "add":
                            cfg["family"] = fname
                            msg = f"{node} now inherits G/time/shape from family:{fname}."
                        else:
                            cfg.pop("family", None)
                            msg = f"{node} no longer inherits family:{fname}."
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] {msg}" + Style.RESET_ALL)
                        continue
                    if action == "status":
                        cfg = families.get(fname)
                        if not cfg:
                            print(Fore.YELLOW + f"[Steer] no family named '{fname}'." + Style.RESET_ALL)
                        else:
                            print(Fore.CYAN + f"[Steer] family:{fname}: G={format_field_source(cfg.get('g'))}, time={format_field_source(cfg.get('time')) if cfg.get('time') else '1'}, shape={format_field_shape(cfg.get('shape'))}" + Style.RESET_ALL)
                        continue
                    try:
                        scale = float(sargs[3]) if len(sargs) > 3 else 1.0
                        offset = float(sargs[4]) if len(sargs) > 4 else 0.0
                    except ValueError:
                        print(Fore.YELLOW + "[Steer] scale and offset must be numbers." + Style.RESET_ALL)
                        continue
                    source = parse_field_source(sargs[2], scale, offset)
                    if not source:
                        print(Fore.YELLOW + "[Steer] G source must be a number, global, probe:<name>, knob:<name>, status:ram, or family:<name>." + Style.RESET_ALL)
                        continue
                    source_error = field_source_error(source, probes, tuner, families)
                    if source_error:
                        print(Fore.YELLOW + f"[Steer] {source_error}." + Style.RESET_ALL)
                        continue
                    families.setdefault(fname, {})["g"] = source
                    save_steer_bodies()
                    value = resolve_field_source(source, probes, tuner, families, bodies["fields"])
                    print(Fore.GREEN + f"[Steer] family:{fname} G <- {format_field_source(source)} (now {value:+g})." + Style.RESET_ALL)
                    continue
                # :steer g <node> <source> -- per-body or per-layer gravity.
                if sargs and sargs[0].lower() in ("g", "personal-g", "personal_g"):
                    g_ns = STEER_G_SPEC.parse(" ".join(sargs[1:]))
                    if g_ns.node is None:
                        bodies = steer_bodies()
                        configured = [
                            (node, cfg) for node, cfg in sorted(bodies["fields"].items())
                            if cfg.get("g") is not None or cfg.get("family")
                        ]
                        if not configured:
                            print(Fore.CYAN + "[Steer] no personal G overrides; every body inherits global G." + Style.RESET_ALL)
                        for node, cfg in configured:
                            inherited = f"family:{cfg['family']}" if cfg.get("family") else format_field_source(cfg.get("g"))
                            print(Fore.CYAN + f"[Steer] {node} G <- {inherited}" + Style.RESET_ALL)
                        continue
                    if g_ns.source is None:
                        STEER_G_SPEC.fail("need a source (or off)")
                    node, cfg = field_target_config(g_ns.node, probes, create=True)
                    if not node or cfg is None:
                        print(Fore.YELLOW + f"[Steer] unknown field node '{g_ns.node}'. Define a family first, or use a probe, @anchor, or layer:N." + Style.RESET_ALL)
                        continue
                    if g_ns.source.lower() in ("off", "auto", "global"):
                        cfg.pop("g", None)
                        if g_ns.source.lower() in ("off", "global"):
                            cfg.pop("family", None)
                        save_steer_bodies()
                        fallback = f"family:{cfg['family']}" if cfg.get("family") else "global"
                        print(Fore.GREEN + f"[Steer] {node} G -> {fallback}." + Style.RESET_ALL)
                        continue
                    scale, offset = g_ns.scale, g_ns.offset
                    source = parse_field_source(g_ns.source, scale, offset)
                    if not source:
                        STEER_G_SPEC.fail("G source must be a number, probe:<name>, knob:<name>, status:ram, lift:<trigger>, outcome:<trigger>, family:<name>, or global")
                    source_error = field_source_error(source, probes, tuner, steer_bodies()["g_families"])
                    if source_error:
                        print(Fore.YELLOW + f"[Steer] {source_error}." + Style.RESET_ALL)
                        continue
                    if source.get("kind") == "family":
                        fname = source.get("name")
                        if fname not in steer_bodies()["g_families"]:
                            print(Fore.YELLOW + f"[Steer] no family named '{fname}'." + Style.RESET_ALL)
                            continue
                        cfg["family"] = fname
                        if scale == 1.0 and offset == 0.0:
                            cfg.pop("g", None)
                        else:
                            cfg["g"] = source
                    else:
                        cfg["g"] = source
                    save_steer_bodies()
                    value = field_entry_config(
                        node.split(":", 1)[1] if node.startswith("probe:") else f"@{node.split(':', 1)[1]}",
                        int(node.split(":", 1)[1]) if node.startswith("layer:") else 0,
                        probes, tuner,
                    )["g"] if not node.startswith("family:") else resolve_field_source(
                        source, probes, tuner, steer_bodies()["g_families"], steer_bodies()["fields"]
                    )
                    print(Fore.GREEN + f"[Steer] {node} G <- {format_field_source(source)} (resolved now {value:+g})." + Style.RESET_ALL)
                    continue
                # :steer time <node> <source> -- local clock alignment.
                if sargs and sargs[0].lower() in ("time", "clock", "align-time", "align_time"):
                    time_ns = STEER_TIME_SPEC.parse(" ".join(sargs[1:]))
                    node, cfg = field_target_config(time_ns.node, probes, create=True)
                    if not node or cfg is None:
                        print(Fore.YELLOW + f"[Steer] unknown field node '{time_ns.node}'." + Style.RESET_ALL)
                        continue
                    if time_ns.source.lower() in ("off", "default"):
                        cfg.pop("time", None)
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] {node} local time -> 1 (unaligned)." + Style.RESET_ALL)
                        continue
                    scale, offset = time_ns.scale, time_ns.offset
                    source = parse_field_source(time_ns.source, scale, offset)
                    if not source:
                        STEER_TIME_SPEC.fail("time source must be a number, probe:<name>, knob:<name>, status:ram, lift:<trigger>, outcome:<trigger>, family:<name>, or layer:N")
                    source_error = field_source_error(source, probes, tuner, steer_bodies()["g_families"])
                    if source_error:
                        print(Fore.YELLOW + f"[Steer] {source_error}." + Style.RESET_ALL)
                        continue
                    cfg["time"] = source
                    save_steer_bodies()
                    value = resolve_field_source(
                        source, probes, tuner, steer_bodies()["g_families"],
                        steer_bodies()["fields"], channel="time",
                    )
                    print(Fore.GREEN + f"[Steer] {node} local time <- {format_field_source(source)} (rate now {max(0.05, min(20.0, value)):g})." + Style.RESET_ALL)
                    continue
                # :steer shape <node> ... -- extended radial bodies, including
                # hollow shells. Families of several nodes compose odd holoids.
                if sargs and sargs[0].lower() in ("shape", "field-shape", "field_shape"):
                    if len(sargs) < 3:
                        print(Fore.YELLOW + "[Steer] Usage: :steer shape <probe|@anchor|layer:N|family:name> point|gaussian <width>|shell <radius> <width>|plateau <radius> <edge>|off" + Style.RESET_ALL)
                        continue
                    node, cfg = field_target_config(sargs[1], probes, create=True)
                    if not node or cfg is None:
                        print(Fore.YELLOW + f"[Steer] unknown field node '{sargs[1]}'." + Style.RESET_ALL)
                        continue
                    if sargs[2].lower() in ("off", "default"):
                        cfg.pop("shape", None)
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] {node} shape -> inherited point field." + Style.RESET_ALL)
                        continue
                    shape = parse_field_shape(sargs[2:])
                    if not shape:
                        print(Fore.YELLOW + "[Steer] Shape must be point, gaussian <width>, shell <radius> <width>, or plateau <radius> <edge>." + Style.RESET_ALL)
                        continue
                    cfg["shape"] = shape
                    save_steer_bodies()
                    print(Fore.GREEN + f"[Steer] {node} shape = {format_field_shape(shape)}." + Style.RESET_ALL)
                    continue
                # :steer anchor <name> [here|from <probe>|mass <m>|drop] --
                # precision bodies: exact latent points that join the field.
                if sargs and sargs[0].lower() in ("anchor", "anchors"):
                    bodies = steer_bodies()
                    if len(sargs) < 2:
                        if not bodies["anchors"]:
                            print(Fore.CYAN + "[Steer] no anchors. ':steer anchor <name> here' pins the last reply's state; ':steer anchor <name> from <probe>' copies a probe direction." + Style.RESET_ALL)
                        for an, a in sorted(bodies["anchors"].items()):
                            poles_txt = ", ".join(f"{f}{'+' if s > 0 else '-'}" for f, s in sorted((a.get("poles") or {}).items())) or "untyped"
                            print(Fore.CYAN + f"[Steer] @{an}: mass {a.get('mass', 1.0):+g}, {'frozen' if a.get('frozen', True) else 'MOBILE'}, poles [{poles_txt}], {len(a.get('dirs') or {})} layer(s)" + Style.RESET_ALL)
                        continue
                    aname = re.sub(r"[^a-z0-9_]", "_", sargs[1].lower()).strip("_")
                    sub = sargs[2].lower() if len(sargs) > 2 else "here"
                    if sub == "drop":
                        if bodies["anchors"].pop(aname, None) is not None:
                            bodies["fields"].pop(f"anchor:{aname}", None)
                            save_steer_bodies()
                            print(Fore.GREEN + f"[Steer] anchor @{aname} dropped." + Style.RESET_ALL)
                        else:
                            print(Fore.YELLOW + f"[Steer] no anchor named '{aname}'." + Style.RESET_ALL)
                        continue
                    if sub == "mass" and len(sargs) > 3:
                        if aname not in bodies["anchors"]:
                            print(Fore.YELLOW + f"[Steer] no anchor named '{aname}'." + Style.RESET_ALL)
                            continue
                        try:
                            bodies["anchors"][aname]["mass"] = float(sargs[3])
                        except ValueError:
                            print(Fore.YELLOW + "[Steer] Usage: :steer anchor <name> mass <number>" + Style.RESET_ALL)
                            continue
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] anchor @{aname} mass = {float(sargs[3]):+g}." + Style.RESET_ALL)
                        continue
                    if sub == "from" and len(sargs) > 3:
                        src = sargs[3]
                        if src not in probes or not probes[src].get("direction"):
                            print(Fore.YELLOW + f"[Steer] '{src}' is not a probe with a stored direction.{did_you_mean(src, probes)}" + Style.RESET_ALL)
                            continue
                        dirs = {}
                        for L, v in probes[src]["direction"].items():
                            vt = v if torch.is_tensor(v) else torch.as_tensor(v)
                            dirs[int(L)] = vt.detach().float().cpu()
                        bodies["anchors"][aname] = {"dirs": dirs, "mass": 1.0, "frozen": True, "poles": {}}
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] anchor @{aname} pinned from probe '{src}' ({len(dirs)} layer(s)); frozen; ':steer freeze @{aname} off' makes it mobile under laws." + Style.RESET_ALL)
                        continue
                    if sub in ("here", "last"):
                        last_reply = next(
                            (getattr(r, "text", "") for r in reversed(memory.records)
                             if r.scope == memory.scope and r.kind == "turn" and r.role == "assistant"),
                            "",
                        )
                        if not last_reply:
                            print(Fore.YELLOW + "[Steer] no archived assistant turn to capture yet." + Style.RESET_ALL)
                            continue
                        print(Fore.CYAN + f"[Steer] capturing @{aname} from the last reply (one forward)..." + Style.RESET_ALL)
                        try:
                            dirs = capture_anchor_dirs(model, last_reply)
                        except Exception as e:
                            print(Fore.RED + f"[Steer] anchor capture failed: {e}" + Style.RESET_ALL)
                            continue
                        bodies["anchors"][aname] = {"dirs": dirs, "mass": 1.0, "frozen": True, "poles": {}}
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] anchor @{aname} pinned at the last reply's state ({len(dirs)} layer(s)), mass +1, frozen." + Style.RESET_ALL)
                        continue
                    print(Fore.YELLOW + "[Steer] Usage: :steer anchor <name> [here|from <probe>|mass <m>|drop]" + Style.RESET_ALL)
                    continue
                # :steer pole <body> <family[+|-]> [off] -- type a body.
                if sargs and sargs[0].lower() == "pole":
                    pole_ns = STEER_POLE_SPEC.parse(
                        " ".join(sargs[1:]),
                        probe_resolver=lambda tok: resolve_probe_choice(
                            tok, probes, model=model, config=config, action_name="steer pole"),
                    )
                    bodies = steer_bodies()
                    bname_raw = pole_ns.body.lstrip("@")
                    fam, sign = parse_pole_spec(pole_ns.pole)
                    if not fam:
                        STEER_POLE_SPEC.fail(f"pole must be family[+|-], got '{pole_ns.pole}'")
                    turn_off = bool(pole_ns.flag)
                    aname = re.sub(r"[^a-z0-9_]", "_", bname_raw.lower()).strip("_")
                    if bname_raw in probes:
                        poles = bodies["probe_poles"].setdefault(bname_raw, {})
                        shown = bname_raw
                    elif aname in bodies["anchors"]:
                        poles = bodies["anchors"][aname].setdefault("poles", {})
                        shown = f"@{aname}"
                    else:
                        cands = set(probes) | {f"@{a}" for a in bodies["anchors"]}
                        print(Fore.YELLOW + f"[Steer] '{pole_ns.body}' is neither a probe nor an anchor.{did_you_mean(pole_ns.body, cands)}" + Style.RESET_ALL)
                        continue
                    if turn_off:
                        poles.pop(fam, None)
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] {shown} pole '{fam}' removed." + Style.RESET_ALL)
                    else:
                        poles[fam] = sign
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] {shown} typed {fam}{'+' if sign > 0 else '-'}; it interacts only where a ':steer law' couples its families." + Style.RESET_ALL)
                    continue
                # :steer law [<famA> <famB> <k|off>] -- selective couplings.
                if sargs and sargs[0].lower() in ("law", "laws"):
                    law_ns = STEER_LAW_SPEC.parse(" ".join(sargs[1:]))
                    bodies = steer_bodies()
                    if law_ns.famA is None:
                        if not bodies["laws"]:
                            print(Fore.CYAN + "[Steer] no laws. ':steer law <famA> <famB> <k>' -- k>0: like poles repel, unlike attract; k<0 flips; unlisted pairs never interact." + Style.RESET_ALL)
                        for (fa, fb), k in sorted(bodies["laws"].items()):
                            print(Fore.CYAN + f"[Steer] law {fa} <-> {fb}: k={k:+g}" + Style.RESET_ALL)
                        step_now = tuner.get("gravity_law_step", 0.0)
                        print(Fore.CYAN + f"[Steer] law step = {step_now:g} per turn (':tune gravity_law_step <s>'; moves only unfrozen anchors)." + Style.RESET_ALL)
                        continue
                    if law_ns.famB is None or law_ns.k is None:
                        STEER_LAW_SPEC.fail("a law needs famA, famB, and k (bare :steer law lists)")
                    fa, _ = parse_pole_spec(law_ns.famA)
                    fb, _ = parse_pole_spec(law_ns.famB)
                    if not fa or not fb:
                        STEER_LAW_SPEC.fail("family names must be plain identifiers")
                    key = tuple(sorted((fa, fb)))
                    if law_ns.k.lower() in ("off", "remove"):
                        bodies["laws"].pop(key, None)
                        save_steer_bodies()
                        print(Fore.GREEN + f"[Steer] law {key[0]} <-> {key[1]} removed; those families no longer interact." + Style.RESET_ALL)
                        continue
                    try:
                        kval = float(law_ns.k)
                    except ValueError:
                        STEER_LAW_SPEC.fail(f"k must be a number or off, got '{law_ns.k}'")
                    bodies["laws"][key] = kval
                    save_steer_bodies()
                    print(Fore.GREEN + f"[Steer] law {key[0]} <-> {key[1]}: k={kval:+g} (k>0: like repels, unlike attracts)." + Style.RESET_ALL)
                    continue
                # :steer bodies -- the whole physics at a glance.
                if sargs and sargs[0].lower() == "bodies":
                    bodies = steer_bodies()
                    g_on = tuner.get("prioritize_gravity", 0.0) > 0
                    print(Fore.CYAN + Style.BRIGHT + f"[Steer] physics: gravity {'ON' if g_on else 'off'} (G={tuner.get('prioritize_alpha', 0.0):g}), doppler k={tuner.get('gravity_doppler', 0.0):g}, law step={tuner.get('gravity_law_step', 0.0):g}" + Style.RESET_ALL)
                    masses = gravity_field_masses(probes, tuner)
                    if masses:
                        shown = sorted(masses, key=lambda t: -abs(t[1]))[:8]
                        print(Fore.CYAN + "[Steer] masses: " + ", ".join(f"{n} {m:+.3f}" for n, m, _d in shown) + Style.RESET_ALL)
                    for fname, fcfg in sorted(bodies["g_families"].items()):
                        members = sorted(
                            node for node, cfg in bodies["fields"].items()
                            if cfg.get("family") == fname
                        )
                        print(
                            Fore.CYAN
                            + f"[Steer] family:{fname}: G={format_field_source(fcfg.get('g'))}; "
                            + f"time={format_field_source(fcfg.get('time')) if fcfg.get('time') else '1'}; "
                            + f"shape={format_field_shape(fcfg.get('shape'))}; "
                            + f"members={','.join(members) or '(none)'}"
                            + Style.RESET_ALL
                        )
                    for node, cfg in sorted(bodies["fields"].items()):
                        bits = []
                        if cfg.get("family"):
                            bits.append(f"family:{cfg['family']}")
                        if cfg.get("g") is not None:
                            bits.append(f"G<-{format_field_source(cfg['g'])}")
                        if cfg.get("time") is not None:
                            bits.append(f"time<-{format_field_source(cfg['time'])}")
                        if cfg.get("shape") is not None:
                            bits.append(f"shape={format_field_shape(cfg['shape'])}")
                        if cfg.get("excluded"):
                            bits.append("HARD-EXCLUDED")
                        if bits:
                            print(Fore.CYAN + f"[Steer] field {node}: " + "; ".join(bits) + Style.RESET_ALL)
                    for qname, qcfg in sorted(bodies["qualities"].items()):
                        print(
                            Fore.CYAN
                            + f"[Steer] quality:{qname}: {qcfg.get('formula', QUALITY_FORMULA_DEFAULT)}; "
                            + f"strength<-{format_field_source(qcfg.get('strength'))}"
                            + Style.RESET_ALL
                        )
                    for lname, law in sorted(bodies["sets"].items()):
                        excluded = set(law.get("excluded") or [])
                        members = []
                        for node, member in sorted((law.get("members") or {}).items()):
                            state = "EXCLUDED" if node in excluded else (
                                f"c<-{format_field_source(member.get('compliance'))},"
                                f"t={member.get('time_vector') or [1.0]}"
                            )
                            members.append(f"{node}[{state}]")
                        for node in sorted(excluded - set(law.get("members") or {})):
                            members.append(f"{node}[EXCLUDED]")
                        print(
                            Fore.CYAN
                            + f"[Steer] set:{lname} quality={law.get('quality', '(unset)')}: "
                            + ("; ".join(members) if members else "(empty)")
                            + Style.RESET_ALL
                        )
                    typed = {**{p: pl for p, pl in bodies["probe_poles"].items() if pl},
                             **{f"@{a}": b.get("poles") for a, b in bodies["anchors"].items() if b.get("poles")}}
                    if typed:
                        print(Fore.CYAN + "[Steer] poles: " + "; ".join(
                            f"{n}[" + ",".join(f"{f}{'+' if s > 0 else '-'}" for f, s in sorted(pl.items())) + "]"
                            for n, pl in sorted(typed.items())) + Style.RESET_ALL)
                    for (fa, fb), k in sorted(bodies["laws"].items()):
                        print(Fore.CYAN + f"[Steer] law {fa} <-> {fb}: k={k:+g}" + Style.RESET_ALL)
                    continue
                # :steer gravity on|off [G] | status -- prioritize as a FIELD:
                # every probe is a mass; pull depends on where the state is.
                if sargs and sargs[0].lower() in ("gravity", "field"):
                    field_ns = STEER_FIELD_SPEC.parse(" ".join(sargs[1:]))
                    sub = field_ns.action
                    init_summary = None
                    if sub in ("init", "full", "flex", "flexible"):
                        init_summary = initialize_field_system(tuner, field_ns.G)
                        print(
                            Fore.GREEN + Style.BRIGHT
                            + "[Steer] full field system initialized: "
                            + ", ".join(f"{k}={v}" for k, v in init_summary.items())
                            + ". Existing definitions/exclusions were preserved; no laws were invented and no bodies were unfrozen."
                            + Style.RESET_ALL
                        )
                    elif sub in ("on", "off"):
                        tuner.set("prioritize_gravity", 1.0 if sub == "on" else 0.0)
                        if field_ns.G is not None:
                            tuner.set("prioritize_alpha", field_ns.G)
                    g_on = tuner.get("prioritize_gravity", 0.0) > 0
                    g_now = tuner.get("prioritize_alpha", 0.0)
                    masses = gravity_field_masses(probes, tuner)
                    configured = steer_bodies()["fields"]
                    n_personal = sum(
                        1 for cfg in configured.values()
                        if cfg.get("g") is not None or cfg.get("family")
                    )
                    n_quality_members = sum(
                        1
                        for law in steer_bodies()["sets"].values()
                        for node in (law.get("members") or {})
                        if node not in set(law.get("excluded") or [])
                        and law.get("quality") in steer_bodies()["qualities"]
                    )
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + f"[Steer] gravity {'ON' if g_on else 'off'} (default global G = prioritize_alpha = {round(g_now, 4)}; "
                        + f"{n_personal} personal/family override(s); {n_quality_members} quality member(s)"
                        + f"{'; inert until some G or quality source is nonzero' if g_on and g_now <= 0 and not n_personal and not n_quality_members else ''}). "
                        + "Pin/mix steering is replaced by the field while on."
                        + Style.RESET_ALL
                    )
                    if masses:
                        shown = sorted(masses, key=lambda t: -abs(t[1]))[:6]
                        print(Fore.CYAN + "[Steer] masses (normalized, signed; - repels): "
                              + ", ".join(f"{n} {m:+.3f}" for n, m, _d in shown)
                              + (f" (+{len(masses) - len(shown)} more)" if len(masses) > len(shown) else "")
                              + Style.RESET_ALL)
                        frozen = sorted(n for n in probes if probes[n].get("frozen"))
                        if frozen:
                            print(Fore.CYAN + f"[Steer] frozen baselines: {', '.join(frozen)}" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "[Steer] no masses yet: give a probe explicit mass (:steer mass <probe> <m>) or let lifts accrue." + Style.RESET_ALL)
                    continue
                # :steer mass <probe> <m|auto> -- gravitational coefficient.
                if sargs and sargs[0].lower() == "mass":
                    mass_ns = STEER_MASS_SPEC.parse(
                        " ".join(sargs[1:]),
                        probe_resolver=lambda tok: resolve_probe_choice(
                            tok, probes, model=model, config=config, action_name="steer mass"),
                    )
                    pname = mass_ns.probe
                    if pname not in probes:
                        print(Fore.YELLOW + f"[Steer] '{pname}' is not an active probe.{did_you_mean(pname, probes)}" + Style.RESET_ALL)
                        continue
                    if mass_ns.m.lower() in ("auto", "lift", "none"):
                        set_probe_field(probes, pname, "mass", None)
                        print(Fore.GREEN + f"[Steer] {pname} mass -> auto (its signed evidence lift each turn)." + Style.RESET_ALL)
                    else:
                        try:
                            mval = float(mass_ns.m)
                        except ValueError:
                            STEER_MASS_SPEC.fail(f"m must be a number or auto, got '{mass_ns.m}'")
                        set_probe_field(probes, pname, "mass", mval)
                        kind = "repulsor" if mval < 0 else "attractor"
                        print(Fore.GREEN + f"[Steer] {pname} mass = {mval:+g} ({kind}); normalized against the other masses when the field applies." + Style.RESET_ALL)
                    continue
                # :steer freeze <probe> [off] -- inertial coefficient: the mass
                # keeps pulling but its own rolling baseline stops moving.
                if sargs and sargs[0].lower() in ("freeze", "unfreeze"):
                    freeze_ns = STEER_FREEZE_SPEC.parse(
                        " ".join(sargs[1:]),
                        probe_resolver=lambda tok: resolve_probe_choice(
                            tok, probes, model=model, config=config, action_name="steer freeze"),
                    )
                    pname = freeze_ns.body
                    thaw = sargs[0].lower() == "unfreeze" or bool(freeze_ns.flag)
                    aname = re.sub(r"[^a-z0-9_]", "_", pname.lstrip("@").lower()).strip("_")
                    if pname.startswith("@") or (pname not in probes and aname in steer_bodies()["anchors"]):
                        anchors_now = steer_bodies()["anchors"]
                        if aname not in anchors_now:
                            print(Fore.YELLOW + f"[Steer] no anchor named '{aname}'." + Style.RESET_ALL)
                            continue
                        anchors_now[aname]["frozen"] = not thaw
                        save_steer_bodies()
                        state_txt = "MOBILE under laws" if thaw else "frozen (a fixed pin again)"
                        print(Fore.GREEN + f"[Steer] anchor @{aname} is now {state_txt}." + Style.RESET_ALL)
                        continue
                    if pname not in probes:
                        print(Fore.YELLOW + f"[Steer] '{pname}' is not an active probe.{did_you_mean(pname, probes)}" + Style.RESET_ALL)
                        continue
                    set_probe_field(probes, pname, "frozen", not thaw)
                    if thaw:
                        print(Fore.GREEN + f"[Steer] {pname} baseline unfrozen -- it drifts with observations again." + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + f"[Steer] {pname} baseline FROZEN -- it still scores and still pulls, but its center no longer moves (it cannot habituate to its own gravity)." + Style.RESET_ALL)
                    continue
                # :steer <probe> <alpha>  -> pin one probe (negative = steer AWAY).
                if len(sargs) >= 2 and sargs[0] in probes:
                    probe_name = sargs[0]
                    if probe_is_released(probes[probe_name]):
                        print(Fore.YELLOW + f"[Steer] '{probe_name}' is released and observational-only. Reactivate it with :probe release {probe_name} off." + Style.RESET_ALL)
                        continue
                    try:
                        if sargs[1].lower() == "up": steer_alpha = 0.5
                        elif sargs[1].lower() == "down": steer_alpha = -0.5
                        else: steer_alpha = float(sargs[1])
                    except ValueError:
                        print(Fore.YELLOW + f"[Steer] invalid alpha: {sargs[1]}. Must be a number (e.g. 0.5) or 'up'/'down'." + Style.RESET_ALL)
                        continue
                    prioritize_pin["mix"] = None
                    prioritize_pin["landscape"] = None
                    prioritize_pin["probe"] = probe_name
                    prioritize_pin["sign"] = -1.0 if steer_alpha < 0 else 1.0
                    tuner.set("prioritize_alpha", abs(steer_alpha))
                    verb = "AWAY from" if steer_alpha < 0 else "toward"
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + f"[Steer] prioritize_alpha mapped to '{probe_name}' at alpha={abs(steer_alpha)} -- steering {verb} it each turn. "
                        + ":steer auto to unpin."
                        + Style.RESET_ALL
                    )
                    continue
                if sargs:
                    print(Fore.YELLOW + f"[Steer] '{sargs[0]}' is not an active probe.{did_you_mean(sargs[0], probes)} Usage: :steer <probe> <alpha> | :steer mix <p1> <p2> ... | :steer auto." + Style.RESET_ALL)
                    continue
            if user_input.startswith(":steer") and not user_input.startswith(":steermap"):
                from invariants import engine as _engine
                stats = _engine.steer_telemetry_stats()
                lo_cur, hi_cur = _engine.get_steer_band()
                print(
                    Fore.CYAN
                    + f"[Steer] cap={_engine.get_steer_cap_fraction():.4f} band=({lo_cur:.2f}, {hi_cur:.2f}) "
                    + f"| observed pushes n={stats['n']}/{stats['window']} clip_rate={stats['clip_rate']}"
                    + Style.RESET_ALL
                )
                if stats["n"]:
                    print(
                        Fore.CYAN
                        + f"  attempted push/residual ratio: min={stats['min']:.4f} q25={stats['q25']:.4f} "
                        + f"med={stats['med']:.4f} q75={stats['q75']:.4f} q95={stats['q95']:.4f} max={stats['max']:.4f}"
                        + Style.RESET_ALL
                    )
                implied = _engine.steer_cap_from_data()
                if implied is None:
                    print(
                        Fore.YELLOW
                        + f"  data-implied cap: not enough evidence yet (n={stats['n']} < {_engine.STEER_CAP_MIN_N}); "
                        + "prior stays in force"
                        + Style.RESET_ALL
                    )
                else:
                    print(
                        Fore.CYAN
                        + f"  data-implied cap @p{_engine.STEER_CAP_PERCENTILE:g} = {implied:.4f} "
                        + "(apply deliberately: :tune steer_cap_fraction auto)"
                        + Style.RESET_ALL
                    )
                n_layers = getattr(model, "n_layers", None) or len(model.model.model.layers)
                synth_band = steer_map.suggest_band(n_layers, evidence="synthesis")
                if synth_band is None:
                    print(
                        Fore.YELLOW
                        + "  synthesis-evidence band: not enough labeled outcomes; prior stays in force"
                        + Style.RESET_ALL
                    )
                else:
                    print(
                        Fore.CYAN
                        + f"  synthesis-evidence band = ({synth_band['lo']:.2f}, {synth_band['hi']:.2f}) "
                        + f"from layers {synth_band['eligible_layers']} "
                        + f"(labeled={synth_band['labeled_events']} {synth_band['labeled_by_basis']}; "
                        + "CAVEAT: where synthesis deltas landed, a different channel's geometry)"
                        + Style.RESET_ALL
                    )
                layer_band = steer_map.suggest_band(n_layers, min_events=3, evidence="layer_steer")
                if layer_band is None:
                    print(
                        Fore.YELLOW
                        + "  layer-steer band: no single-layer evidence yet -- "
                        + ":tune steer_layer_sweep 1 plus a small alpha isolates steers by layer"
                        + Style.RESET_ALL
                    )
                else:
                    print(
                        Fore.CYAN
                        + f"  layer-steer band = ({layer_band['lo']:.2f}, {layer_band['hi']:.2f}) "
                        + f"from layers {layer_band['eligible_layers']} "
                        + f"(labeled={layer_band['labeled_events']} {layer_band['labeled_by_basis']}; transfer-free)"
                        + Style.RESET_ALL
                    )
                    for lyr, row in sorted(layer_band["per_layer"].items()):
                        print(
                            Fore.CYAN
                            + f"    L{lyr}: n={row['n']} success={row['success_rate']:.2f}"
                            + Style.RESET_ALL
                        )
                lift_rows = steer_map.channel_lift(basis="any")
                if lift_rows:
                    print(Fore.CYAN + "  channel lift (fired vs unfired outcomes -- should it be on?):" + Style.RESET_ALL)
                    for lrow in lift_rows:
                        fr = f"{lrow['fired_rate']:.2f}" if lrow["fired_rate"] is not None else "-"
                        ur = f"{lrow['unfired_rate']:.2f}" if lrow["unfired_rate"] is not None else "-"
                        print(
                            Fore.CYAN
                            + f"    {lrow['channel']}: fired={fr} (n={lrow['fired_n']}) "
                            + f"unfired={ur} (n={lrow['unfired_n']}) lift={lrow['lift']}"
                            + Style.RESET_ALL
                        )
                continue
            if user_input.startswith(":probe"):
                pargs = user_input[len(":probe"):].strip()
                if not pargs:
                    if not probes:
                        print(Fore.CYAN + "[Probe] none active. Mint one: :probe <name> <framing WITH it> || <framing WITHOUT it>" + Style.RESET_ALL)
                    for pname, pdata in probes.items():
                        trig = tuner.triggers.get(f"probe_{pname}")
                        n_pairs = len(trig.outcomes) if trig else 0
                        exposed_note = ", exposed to the model" if pdata.get("exposed") else ""
                        released_note = ", released (visible/read-only)" if probe_is_released(pdata) else ""
                        print(Fore.CYAN + f"[Probe] {pname}: {len(pdata['direction'])} layers, {n_pairs} paired turns{exposed_note}{released_note}." + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("auto"):
                    vparts = pargs.split()
                    if len(vparts) > 1:
                        state = vparts[1].lower()
                        if state == "on":
                            config.auto_probe_readings = True
                            print(Fore.CYAN + "[Probe] Auto-readings enabled. Will print values before every input prompt." + Style.RESET_ALL)
                        else:
                            config.auto_probe_readings = False
                            print(Fore.CYAN + "[Probe] Auto-readings disabled." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + "[Probe] Usage: :probe auto [on|off]" + Style.RESET_ALL)
                    continue

                if pargs.lower() == "release" or pargs.lower().startswith("release "):
                    rparts = pargs.split()
                    if len(rparts) < 2:
                        print(Fore.YELLOW + "[Probe] Usage: :probe release <name> [off]" + Style.RESET_ALL)
                        continue
                    pname = re.sub(r"[^a-z0-9_]", "_", rparts[1].lower())[:40]
                    if pname not in probes:
                        print(Fore.YELLOW + f"[Probe] no active probe named '{pname}'.{did_you_mean(pname, probes)}" + Style.RESET_ALL)
                        continue
                    released = not (len(rparts) > 2 and rparts[2].lower() in {"off", "no", "false", "reactivate"})
                    set_probe_released(probes, pname, released, PROBE_DIR)
                    if released:
                        if prioritize_pin.get("probe") == pname:
                            prioritize_pin["probe"] = None
                        if pname in (prioritize_pin.get("mix") or []):
                            prioritize_pin["mix"] = [n for n in prioritize_pin["mix"] if n != pname] or None
                        if prioritize_pin.get("landscape") == pname:
                            prioritize_pin["landscape"] = None
                    print(
                        Fore.GREEN
                        + (
                            f"[Probe] '{pname}' released: still visible/scored/queryable, "
                            "but excluded from generation and automatic steering."
                            if released else
                            f"[Probe] '{pname}' reactivated for generation and steering."
                        )
                        + Style.RESET_ALL
                    )
                    continue

                if pargs.lower() in ("choose", "choice"):
                    active_probe_names = ", ".join(sorted(probes)) or "none"
                    print(Fore.CYAN + "[Probe] Asking the model to choose a useful probe definition..." + Style.RESET_ALL)
                    suggestion_prompt = (
                        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        "Choose one useful cognitive/behavioral probe for the current interactive shell. "
                        "Output exactly one runnable command in this format:\n"
                        ":probe <short_snake_name> I <WITH statement>. || I <WITHOUT statement>.\n\n"
                        "Rules:\n"
                        "- The probe name must not be choose, choice, auto, values, recent, or all.\n"
                        "- Both sides must be first-person statements and must contrast the same dimension.\n"
                        "- The line must contain exactly one || separator.\n"
                        "- Do not output prose, bullets, markdown, or explanation.\n\n"
                        f"Active probes already present: {active_probe_names}\n"
                        "Example:\n"
                        ":probe evidence_before_commitment I check evidence before committing to a setting. || I commit to settings before checking evidence.\n"
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    sug = generate_agentic_text(
                        model,
                        instruction=suggestion_prompt,
                        config=config,
                        pre_formatted=True,
                        max_new_tokens=120
                    ).strip()
                    sug = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), sug, flags=re.DOTALL).strip()
                    line = next((ln.strip() for ln in sug.splitlines() if ln.strip().startswith(":probe ")), sug.splitlines()[0].strip() if sug.splitlines() else "")
                    m = re.match(r"^:probe\s+([a-zA-Z0-9_]+)\s+(.+\|\|.+)$", line)
                    if not m or m.group(1).lower() in {"choose", "choice", "auto", "values", "recent", "all"}:
                        print(Fore.YELLOW + f"[Probe] Model did not return a valid probe command. Raw suggestion: {sug[:240]}" + Style.RESET_ALL)
                        continue
                    pname_s = re.sub(r"[^a-z0-9_]", "_", m.group(1).lower())[:40]
                    pair_s = m.group(2).strip()
                    suggested_command = f":probe {pname_s} {pair_s}"
                    memory.append_definition(
                        "probe",
                        pname_s,
                        suggested_command,
                        authored_by="model",
                        status="proposed",
                        provenance={"source": ":probe choose"},
                    )
                    print(Fore.GREEN + f"\n[Probe Suggestion] Try running this:\n{suggested_command}\n" + Style.RESET_ALL)
                    continue

                if pargs.lower() in ("values", "value", "recent", "last") or pargs.lower().startswith(("values ", "value ", "recent ", "last ")):
                    vparts = pargs.split()
                    rest = vparts[1:]
                    if not probes:
                        print(Fore.CYAN + "[Probe Values] none active. Mint one first with :probe <name> <with> || <without>" + Style.RESET_ALL)
                        continue
                    selected = sorted(probes)
                    limit = 1
                    if rest:
                        head = rest[0].lower()
                        if head == "all":
                            rest = rest[1:]
                        elif not re.fullmatch(r"\d+", head):
                            vname = resolve_probe_choice(rest[0], probes, model=model, config=config, action_name="values")
                            if not vname:
                                continue
                            if vname not in probes:
                                print(Fore.YELLOW + f"[Probe Values] no active probe named '{vname}'.{did_you_mean(vname, probes)}" + Style.RESET_ALL)
                                continue
                            selected = [vname]
                            rest = rest[1:]
                    if rest:
                        try:
                            limit = max(1, min(20, int(rest[0])))
                        except ValueError:
                            print(Fore.YELLOW + "[Probe Values] Usage: :probe values [name|all] [n]" + Style.RESET_ALL)
                            continue

                    recent_rows = []
                    for row in reversed(list(turn_log)):
                        vals = {}
                        for pname in selected:
                            key = f"probe_{pname}"
                            if key in row:
                                try:
                                    vals[pname] = float(row[key])
                                except (TypeError, ValueError):
                                    pass
                        if vals:
                            recent_rows.append((row, vals))
                            if len(recent_rows) >= limit:
                                break

                    if not recent_rows:
                        print(Fore.CYAN + "[Probe Values] no completed turn with probe readings yet; showing raw-history fallback where available." + Style.RESET_ALL)
                        for pname in selected:
                            sig, raw, n_hist = latest_probe_signal_from_history(probes[pname].get("history"))
                            if sig is None:
                                print(Fore.CYAN + f"  {pname}: no reading yet" + Style.RESET_ALL)
                            else:
                                print(Fore.CYAN + f"  {pname}: {sig:+.3f} (raw {raw:+.3f}, history n={n_hist})" + Style.RESET_ALL)
                        continue

                    if limit == 1:
                        row, vals = recent_rows[0]
                        ts = row.get("ts")
                        print(Fore.CYAN + "[Probe Values] latest completed turn" + (f" ({ts})" if ts else "") + ":" + Style.RESET_ALL)
                        for pname in selected:
                            sig = vals.get(pname)
                            if sig is None:
                                print(Fore.CYAN + f"  {pname}: no reading on that turn" + Style.RESET_ALL)
                                continue
                            _hist_sig, raw, n_hist = latest_probe_signal_from_history(probes[pname].get("history"))
                            trig = tuner.triggers.get(f"probe_{pname}")
                            parts = [f"{sig:+.3f}"]
                            if raw is not None:
                                parts.append(f"raw {raw:+.3f}")
                            parts.append(f"history n={n_hist}")
                            if trig is not None:
                                fires = sig <= trig.value if trig.comparator == "<=" else sig >= trig.value
                                parts.append(f"bar {trig.comparator} {trig.value:+.3f} ({'fires' if fires else 'quiet'})")
                                st = trig.outcome_stats()
                                if st.get("lift") is not None:
                                    parts.append(f"lift {st.get('lift'):+.3f} over {st.get('n_credited')} credited")
                            if probes[pname].get("exposed"):
                                parts.append("exposed")
                            print(Fore.CYAN + f"  {pname}: " + ", ".join(parts) + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"[Probe Values] {len(recent_rows)} most recent turn(s):" + Style.RESET_ALL)
                        for idx, (row, vals) in enumerate(recent_rows, 1):
                            ts = row.get("ts") or f"-{idx}"
                            compact = "  ".join(f"{p}={vals[p]:+.3f}" for p in sorted(vals))
                            print(Fore.CYAN + f"  {ts}: {compact}" + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("drop "):
                    dropped_raw = pargs[5:].strip()
                    dropped = resolve_probe_choice(dropped_raw, probes, model=model, config=config, action_name="drop")
                    if not dropped:
                        continue
                    if probes.pop(dropped, None) is not None:
                        print(Fore.CYAN + f"[Probe] {dropped} dropped (its observed stream is kept)." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Probe] no active probe named {dropped}.{did_you_mean(dropped, probes)}" + Style.RESET_ALL)
                    continue
                # :probe match -- tie a probe to the same-concept knob so it can DRIVE
                # the knob (servo) or VALIDATE it (correlate knob value vs reading).
                if pargs.lower() == "match" or pargs.lower().startswith("match "):
                    margs = pargs[len("match"):].split()
                    knobs = {t for t in tuner.triggers if not t.startswith("probe_")}
                    if not margs:  # list
                        if not probe_matches:
                            print(Fore.CYAN + "[Match] No probe<->knob matches. ':probe match auto' pairs same-named probes+knobs; ':probe match <probe> <knob>' sets one." + Style.RESET_ALL)
                        else:
                            for pn, m in sorted(probe_matches.items()):
                                extra = ""
                                if m.get("mode") == "validate":
                                    r = _match_corr(match_hist.get(pn))
                                    extra = f"  r={r:+.2f}" if r is not None else "  r=(needs more varied turns)"
                                print(Fore.CYAN + f"[Match] {pn} <-> knob '{m.get('knob')}'  mode={m.get('mode', 'none')}{extra}" + Style.RESET_ALL)
                        continue
                    sub = margs[0].lower()
                    if sub in ("auto", "same"):  # pair every probe whose name is a knob
                        paired = []
                        for pn in probes:
                            if pn in knobs:
                                probe_matches.setdefault(pn, {"knob": pn, "mode": "none"})["knob"] = pn
                                paired.append(pn)
                        _save_probe_matches()
                        print(Fore.GREEN + f"[Match] Auto-matched {len(paired)} probe(s) to same-named knobs: {', '.join(sorted(paired)) or '(none found)'}." + Style.RESET_ALL)
                        continue
                    pn = resolve_probe_choice(margs[0], probes, model=model, config=config, action_name="match")
                    if not pn:
                        continue
                    if pn not in probes:
                        print(Fore.YELLOW + f"[Match] no active probe '{pn}'.{did_you_mean(pn, probes)}" + Style.RESET_ALL)
                        continue
                    if len(margs) < 2:
                        print(Fore.YELLOW + "[Match] Usage: :probe match <probe> <knob> | drive [mult] | validate | check | off" + Style.RESET_ALL)
                        continue
                    op = margs[1].lower()
                    m = probe_matches.get(pn)
                    if op == "off":
                        if probe_matches.pop(pn, None) is not None:
                            if m and m.get("knob") in tuner_bindings and m.get("mode") == "drive":
                                tuner_bindings.pop(m["knob"], None)
                            match_hist.pop(pn, None)
                            _save_probe_matches()
                            print(Fore.CYAN + f"[Match] cleared match for '{pn}'." + Style.RESET_ALL)
                        else:
                            print(Fore.YELLOW + f"[Match] '{pn}' had no match." + Style.RESET_ALL)
                        continue
                    if op in ("drive", "validate", "check"):
                        if not m or not m.get("knob"):
                            print(Fore.YELLOW + f"[Match] match '{pn}' to a knob first: :probe match {pn} <knob>" + Style.RESET_ALL)
                            continue
                        knob = m["knob"]
                        if op == "drive":
                            mult = 1.0
                            if len(margs) >= 3:
                                try:
                                    mult = float(margs[2])
                                except ValueError:
                                    pass
                            tuner_bindings[knob] = ([(1.0, pn)], mult)
                            m["mode"] = "drive"
                            m["mult"] = mult
                            _save_probe_matches()
                            print(Fore.GREEN + f"[Match] SERVO: knob '{knob}' now follows probe '{pn}' every turn (x{mult:g}). ':probe match {pn} off' stops it." + Style.RESET_ALL)
                        elif op == "validate":
                            m["mode"] = "validate"
                            match_hist.setdefault(pn, deque(maxlen=200))
                            _save_probe_matches()
                            print(Fore.GREEN + f"[Match] VALIDATE: recording (knob '{knob}' value, probe '{pn}' reading) each turn. ':probe match {pn} check' for the correlation." + Style.RESET_ALL)
                        else:  # check
                            r = _match_corr(match_hist.get(pn))
                            if r is None:
                                print(Fore.YELLOW + f"[Match] {pn}: not enough varied paired turns yet (vary the knob '{knob}' while validating)." + Style.RESET_ALL)
                            else:
                                print(Fore.CYAN + f"[Match] {pn} vs knob '{knob}': r={r:+.3f} over {len(match_hist.get(pn, []))} turn(s)." + Style.RESET_ALL)
                        continue
                    # otherwise margs[1] is the knob to match to
                    knob = margs[1]
                    if knob.upper() in ("CHOICE", "CHOOSE", "AUTO"):
                        valid_knobs = list(knobs) + ["steer_cap_fraction", "steer_band"]
                        resolved_knob = resolve_probe_choice(knob, valid_knobs, model=model, config=config, action_name="match_target")
                        if not resolved_knob:
                            continue
                        knob = resolved_knob
                    if knob not in knobs and f"probe_{knob}" not in tuner.triggers and knob not in ("steer_cap_fraction", "steer_band"):
                        print(Fore.YELLOW + f"[Match] '{knob}' isn't a known knob.{did_you_mean(knob, knobs)}" + Style.RESET_ALL)
                        continue
                    probe_matches[pn] = {"knob": knob, "mode": (m.get("mode", "none") if m else "none")}
                    _save_probe_matches()
                    print(Fore.GREEN + f"[Match] {pn} <-> knob '{knob}'. Then ':probe match {pn} drive [mult]' (servo) or ':probe match {pn} validate' (credit)." + Style.RESET_ALL)
                    continue
                # :probe define <name> -- share the INITIAL BREAKDOWN: the WITH/WITHOUT
                # framings the probe was minted from (the operator-authored definition).
                if pargs.lower().startswith("define "):
                    dname_raw = pargs[7:].strip()
                    dname = resolve_probe_choice(dname_raw, probes, model=model, config=config, action_name="define")
                    if not dname:
                        continue
                    if dname not in probes:
                        print(Fore.YELLOW + f"[Probe] no active probe named '{dname}'.{did_you_mean(dname, probes)}" + Style.RESET_ALL)
                        continue
                    framings = probes[dname].get("framings")
                    if framings and any(framings):
                        print(Fore.CYAN + f"[Probe] {dname} was minted with framings:\n  WITH:    {framings[0]}\n  WITHOUT: {framings[1]}" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"[Probe] {dname} has no stored framings (likely adopted from a raw vector)." + Style.RESET_ALL)
                    continue
                # :probe explain <name> [alpha|choose] [prompt] -- default is a PING,
                # not a prompt: briefly steer along the probe's own direction
                # (envelope-capped) and let the model verbalize whatever the
                # nudge evokes; the information arrives through activations,
                # the text asks nothing. The probe's baseline is frozen for
                # the ping so it cannot habituate to its own motion. Append
                # 'prompt' for a stored-definition explanation grounded in the
                # probe's minted WITH/WITHOUT framings.
                if pargs.lower().startswith("explain "):
                    explain_ns = PROBE_EXPLAIN_SPEC.parse(pargs[len("explain"):])
                    ename_raw = explain_ns.name
                    alpha_tok = explain_ns.alpha
                    mode_tok = explain_ns.mode
                    # Tolerate ':probe explain fun prompt' -- mode in the alpha slot.
                    if alpha_tok is not None and mode_tok is None and alpha_tok.lower() in ("prompt", "words", "verbal"):
                        mode_tok, alpha_tok = alpha_tok.lower(), None
                    explain_via_prompt = mode_tok is not None
                    ping_alpha = 1.0
                    if alpha_tok is not None:
                        if alpha_tok.lower() in ("choose", "auto"):
                            _live_g = float(tuner.get("prioritize_alpha", 0.0))
                            ping_alpha = _live_g if _live_g > 0 else 1.0
                            print(
                                Fore.CYAN
                                + (f"[Probe] alpha chosen from the live calibrated G (prioritize_alpha = {ping_alpha:g})."
                                   if _live_g > 0 else
                                   "[Probe] no calibrated G yet (prioritize_alpha <= 0); alpha defaults to 1. "
                                   "Set one with :tune prioritize_alpha <v> or :steer field init <G>.")
                                + Style.RESET_ALL
                            )
                        else:
                            try:
                                ping_alpha = float(alpha_tok)
                            except ValueError:
                                PROBE_EXPLAIN_SPEC.fail(f"alpha must be a number or choose, got '{alpha_tok}'")
                    ename_resolved = resolve_probe_choice(ename_raw, probes, model=model, config=config, action_name="explain")
                    if not ename_resolved:
                        continue
                    if ename_resolved not in probes:
                        print(Fore.YELLOW + f"[Probe] no active probe named '{ename_resolved}'.{did_you_mean(ename_resolved, probes)}" + Style.RESET_ALL)
                        continue
                    ping_dirs = probes[ename_resolved].get("direction") or {}
                    if not explain_via_prompt and not ping_dirs:
                        print(Fore.YELLOW + f"[Probe] '{ename_resolved}' has no stored direction; falling back to the prompt explanation." + Style.RESET_ALL)
                        explain_via_prompt = True
                    if not explain_via_prompt:
                        ping_layers = sorted(int(l) for l in ping_dirs.keys())
                        print(
                            Fore.CYAN
                            + f"[Probe] pinging '{ename_resolved}' (alpha {ping_alpha:g}, L{ping_layers[0]}-L{ping_layers[-1]}, envelope-capped): "
                            + "no question asked -- the nudge itself is the message."
                            + Style.RESET_ALL
                        )
                        from invariants.engine import _steer_handles as _ping_steer
                        was_frozen = probes[ename_resolved].get("frozen", False)
                        probes[ename_resolved]["frozen"] = True
                        # A ping is a diagnostic nudge: cached task deltas are
                        # the wrong authority for reading the probe's effect,
                        # and its output must never be STORED -- the cache sits
                        # at capacity, so every ping write evicts a real memory.
                        _ping_cache_before = (config.cache_enabled, config.cache_write_enabled)
                        config.cache_enabled = False
                        config.cache_write_enabled = False
                        ping_handles = []
                        ping_error = None
                        try:
                            ping_handles = _ping_steer(model, ping_dirs, list(ping_dirs.keys()), ping_alpha)
                            ping_reply = generate_agentic_text(
                                model,
                                instruction=PROBE_PING_ELICITOR,
                                config=config,
                                pre_formatted=False,
                                max_new_tokens=140,
                                chatty_log=False,
                            )
                        except Exception as e:
                            ping_error = e
                        finally:
                            for h in ping_handles:
                                h.remove()
                            probes[ename_resolved]["frozen"] = was_frozen
                            config.cache_enabled, config.cache_write_enabled = _ping_cache_before
                        if ping_error is not None:
                            print(Fore.RED + f"[Probe] ping failed: {ping_error}" + Style.RESET_ALL)
                            continue
                        ping_text = ping_reply[0] if isinstance(ping_reply, tuple) else ping_reply
                        print(Fore.CYAN + f"[Probe] {ename_resolved} under ping:\n{(ping_text or '').strip()}" + Style.RESET_ALL)
                        print(
                            Fore.CYAN
                            + f"[Probe] If that reads generic/off-target, the activation nudge was not legible; "
                              f"':probe define {ename_resolved}' shows the saved framings, and "
                              f"':probe explain {ename_resolved} prompt' asks for a definition-grounded wording."
                            + Style.RESET_ALL
                        )
                        continue
                    framings = probes[ename_resolved].get("framings")
                    if framings and any(framings):
                        with_pole = (framings[0] or "(missing high-pole wording)").strip()
                        without_pole = (framings[1] or "(missing low-pole wording)").strip()
                        basis = (
                            f"WITH / high pole: {with_pole}\n"
                            f"WITHOUT / low pole: {without_pole}\n"
                        )
                    else:
                        basis = (
                            "NO STORED WITH/WITHOUT FRAMINGS: this probe was adopted from a raw vector, "
                            "so there is no deterministic high/low wording beyond the saved vector/name.\n"
                        )
                    default_explain_prompt = (
                        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        "PROBE_EXPLAIN_CONTEXT_V2\n"
                        "You are explaining one saved cognitive probe definition to the operator.\n"
                        "Probe name: $ename_resolved\n"
                        "A cognitive probe is a direction in your own activations that scores each reply.\n"
                        "Stored definition for $ename_resolved:\n$basis\n"
                        "Use the stored definition above. Do not explain the word \"this\", do not describe "
                        "probes in general, and do not invent a new meaning for $ename_resolved.\n"
                        "Answer in two or three sentences: what a high reading means, what a low reading "
                        "means, and whether this is definition-grounded or only a raw-vector guess."
                        "<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    xprompt = load_prompt(
                        "probe_explain",
                        default_explain_prompt,
                        required_substring=(
                            "PROBE_EXPLAIN_CONTEXT_V2",
                            "Probe name: $ename_resolved",
                            "Stored definition for $ename_resolved",
                            "Do not explain the word \"this\"",
                            "$basis",
                        ),
                        ename_resolved=ename_resolved,
                        basis=basis
                    )
                    print(Fore.CYAN + f"[Probe] Asking the model to explain '{ename_resolved}' in its own words..." + Style.RESET_ALL)
                    # Same diagnostic rule as the ping: meta-talk about a probe
                    # must not read from or write into the cognitive cache.
                    _explain_cache_before = (config.cache_enabled, config.cache_write_enabled)
                    config.cache_enabled = False
                    config.cache_write_enabled = False
                    try:
                        expl = generate_agentic_text(model, instruction=xprompt, config=config, pre_formatted=True, max_new_tokens=200, chatty_log=False)
                    except Exception as e:
                        print(Fore.RED + f"[Probe] explain failed: {e}" + Style.RESET_ALL)
                        continue
                    finally:
                        config.cache_enabled, config.cache_write_enabled = _explain_cache_before
                    print(Fore.CYAN + f"[Probe] {ename_resolved} -- in the model's words:\n{(expl or '').strip()}" + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("show ") or pargs.lower().startswith("hide "):
                    is_show = pargs.lower().startswith("show ")
                    target = pargs[5:].strip().lower()
                    
                    if target == "talk":
                        global _PROBES_SHOW_TALK
                        _PROBES_SHOW_TALK = is_show
                        print(Fore.CYAN + f"[Probe] Probes will {'now' if is_show else 'no longer'} print during main conversation (talk)." + Style.RESET_ALL)
                        continue
                    elif target == "act":
                        global _PROBES_SHOW_ACT
                        _PROBES_SHOW_ACT = is_show
                        print(Fore.CYAN + f"[Probe] Probes will {'now' if is_show else 'no longer'} print during background actions (act)." + Style.RESET_ALL)
                        continue
                    elif target == "all":
                        _PROBES_SHOW_TALK = is_show
                        _PROBES_SHOW_ACT = is_show
                        for pname in probes:
                            probes[pname]["chatty"] = is_show
                        print(Fore.CYAN + f"[Probe] Global modes (talk, act) and all {len(probes)} probes set to {'SHOW' if is_show else 'HIDE'}." + Style.RESET_ALL)
                        continue
                        
                    chatty_name = resolve_probe_choice(target, probes, model=model, config=config, action_name="show/hide")
                    if not chatty_name:
                        continue
                    if chatty_name in probes:
                        probes[chatty_name]["chatty"] = is_show
                        print(Fore.CYAN + f"[Probe] {chatty_name} console print {'SHOW' if is_show else 'HIDE'}." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Probe] no active probe named {chatty_name}.{did_you_mean(chatty_name, probes)}" + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("expose "):
                    eargs = pargs[7:].split()
                    ename_raw = eargs[0].lower() if eargs else ""
                    ename_resolved = resolve_probe_choice(ename_raw, probes)
                    if not ename_resolved:
                        continue
                    ename = re.sub(r"[^a-z0-9_]", "_", ename_resolved)[:40]
                    turn_off = len(eargs) > 1 and eargs[1].lower() in ("off", "0", "false")
                    if ename not in probes:
                        print(Fore.YELLOW + f"[Probe] no active probe named '{ename}'.{did_you_mean(ename, probes)}" + Style.RESET_ALL)
                        continue
                    probes[ename]["exposed"] = not turn_off
                    try:
                        os.makedirs(PROBE_DIR, exist_ok=True)
                        torch.save(
                            {
                                "direction": probes[ename]["direction"],
                                "framings": probes[ename].get("framings", ("", "")),
                                "exposed": probes[ename]["exposed"],
                            },
                            os.path.join(PROBE_DIR, f"{ename}.pt"),
                        )
                    except Exception:
                        pass
                    memory.append_event(
                        "probe_exposure_changed",
                        text=f"{ename}: {'exposed' if not turn_off else 'hidden'}",
                        tags=["probe"],
                        provenance={"probe": ename, "exposed": not turn_off},
                    )
                    if turn_off:
                        print(Fore.CYAN + f"[Probe] {ename} is hidden from the model again." + Style.RESET_ALL)
                    else:
                        print(
                            Fore.CYAN
                            + f"[Probe] {ename} EXPOSED: the model may now consult it with "
                            + f"<<PROBE: {ename}>> (its own reading) or <<PROBE: {ename} || candidate words>> "
                            + "(score hypothetical words). Reading only -- it still cannot mint, drop, or calibrate."
                            + Style.RESET_ALL
                        )
                    continue
                adopt_args = CommandArgs(pargs).take("adopt")
                if adopt_args is not None:
                    adopt_ns = PROBE_ADOPT_SPEC.parse(adopt_args.raw)
                    stems, adopt_layers = adopt_ns.stems, adopt_ns.band
                    adopted = []
                    for stem in stems:
                        if stem in probes:
                            print(Fore.YELLOW + f"[Probe] '{stem}' is already an active probe." + Style.RESET_ALL)
                            continue
                        direction, src_name, exposed_state, dim_mismatch = load_stored_direction(model, stem, layers=adopt_layers)
                        if not direction:
                            if dim_mismatch:
                                err_msg = f"[Probe] Cannot adopt '{stem}': saved vector dimension ({dim_mismatch}) does not match current model d_model ({model.d_model}). You must re-mint this probe on the smaller model!"
                            else:
                                err_msg = f"[Probe] no usable vector for '{stem}' (looked for invariants/{stem}_vector.pt and invariants/{stem}.pt)."
                            print(Fore.YELLOW + err_msg + Style.RESET_ALL)
                            continue
                        probes[stem] = {
                            "direction": direction,
                            "history": deque(maxlen=40),
                            "framings": (f"adopted:{src_name}", ""),
                            "exposed": exposed_state,
                        }
                        tuner.register(f"probe_{stem}", 0.0, kind="threshold", comparator=">=")
                        try:
                            os.makedirs(PROBE_DIR, exist_ok=True)
                            torch.save(
                                {"direction": direction, "framings": probes[stem]["framings"]},
                                os.path.join(PROBE_DIR, f"{stem}.pt"),
                            )
                        except Exception:
                            pass
                        memory.append_event(
                            "probe_minted",
                            text=f"{stem}: adopted from {src_name}",
                            tags=["probe"],
                            provenance={
                                "layers": sorted(direction),
                                "authored_by": "operator",
                                "adopted_from": src_name,
                            },
                        )
                        print(
                            Fore.CYAN
                            + f"[Probe] {stem} adopted from {src_name} over {len(direction)} layers."
                            + Style.RESET_ALL
                        )
                        adopted.append(stem)
                    if adopted:
                        print(
                            Fore.CYAN
                            + f"[Probe] {len(adopted)} dimension(s) now scoring every reply like any probe "
                            + f"(phen_ streams keep reading them in the reasoning states). Pre-load the "
                            + f"archive in one pass: :probe backfill {adopted[0]}"
                            + Style.RESET_ALL
                        )
                    continue
                compose_args = CommandArgs(pargs).take("compose")
                if compose_args is not None:
                    compose_ns = PROBE_COMPOSE_SPEC.parse(compose_args.raw)
                    cname, expr, cband = compose_ns.name, compose_ns.mix, compose_ns.band
                    # No `if cname in probes` guard: composites may overwrite/
                    # augment themselves (e.g. compose amb amb + new).
                    terms, perr = parse_compose_expr(expr)
                    if perr is not None:
                        print(Fore.YELLOW + f"[Probe] could not parse term '{perr}' in the mix." + Style.RESET_ALL)
                        continue
                    resolved = []
                    missing = None
                    for w, tname in terms:
                        if tname in probes and cband is None:
                            tdir = probes[tname]["direction"]
                        else:
                            # An explicit band re-reads every term from its
                            # stored file at that depth; an active probe's
                            # in-memory direction is fixed at its mint band.
                            tdir, _, _, _ = load_stored_direction(model, tname, layers=cband)
                            if not tdir and tname in probes:
                                tdir = {
                                    L: v for L, v in probes[tname]["direction"].items()
                                    if cband is None or int(L) in set(cband)
                                }
                        if not tdir:
                            missing = tname
                            break
                        resolved.append((w, tname, tdir))
                    if missing is not None:
                        print(
                            Fore.YELLOW
                            + f"[Probe] '{missing}' is neither an active probe nor a stored vector "
                            + f"(invariants/{missing}_vector.pt / invariants/{missing}.pt).{did_you_mean(missing, probes)}"
                            + Style.RESET_ALL
                        )
                        continue
                    # A signed sum of unit directions IS the composite contrast:
                    # projecting on normalize(sum w_i * v_i) equals the weighted
                    # sum of the per-term cosines (up to the shared norm), so the
                    # mix measures exactly the relationship it names. Intersect
                    # layers so the composite means the same thing at every depth.
                    common = set.intersection(*(set(d.keys()) for _, _, d in resolved))
                    if not common:
                        spans = ", ".join(f"{t}: L{min(d)}-{max(d)}" for _, t, d in resolved)
                        print(
                            Fore.YELLOW
                            + f"[Probe] the terms share no layers ({spans}) -- they were minted or "
                            + "stored over different bands, so no honest composite exists."
                            + Style.RESET_ALL
                        )
                        continue
                    direction = {}
                    for L in sorted(common):
                        acc = None
                        for w, _, d in resolved:
                            v = d[L].to(model.device).float().reshape(-1) * w
                            acc = v if acc is None else acc + v
                        n = acc.norm()
                        if n.item() > 0:
                            direction[int(L)] = acc / n
                    if not direction:
                        print(Fore.YELLOW + "[Probe] the mix cancelled to zero on every shared layer; nothing to mint." + Style.RESET_ALL)
                        continue
                    recipe = " ".join(
                        f"{'+' if w >= 0 else '-'} {abs(w):g}*{t}".replace("1*", "") for w, t in terms
                    ).lstrip("+ ").strip()
                    probes[cname] = {
                        "direction": direction,
                        "history": deque(maxlen=40),
                        "framings": (f"composed: {recipe}", ""),
                    }
                    tuner.register(f"probe_{cname}", 0.0, kind="threshold", comparator=">=")
                    try:
                        os.makedirs(PROBE_DIR, exist_ok=True)
                        torch.save(
                            {"direction": direction, "framings": probes[cname]["framings"]},
                            os.path.join(PROBE_DIR, f"{cname}.pt"),
                        )
                    except Exception:
                        pass
                    memory.append_event(
                        "probe_minted",
                        text=f"{cname}: composed {recipe}",
                        tags=["probe"],
                        provenance={
                            "layers": sorted(direction),
                            "authored_by": "operator",
                            "composed_from": [[float(w), t] for w, t in terms],
                        },
                    )
                    print(
                        Fore.CYAN
                        + f"[Probe] {cname} composed = {recipe} over {len(direction)} shared layers -- "
                        + "scores every reply like any probe, backfillable, calibratable. "
                        + f"Pre-load the archive: :probe backfill {cname}"
                        + Style.RESET_ALL
                    )
                    continue
                backfill_args = CommandArgs(pargs).take("backfill")
                if backfill_args is not None:
                    backfill_ns = PROBE_BACKFILL_SPEC.parse(backfill_args.raw)
                    bf_name_raw = backfill_ns.name
                    bf_names_to_rebuild = []
                    if bf_name_raw == "all":
                        bf_names_to_rebuild = list(probes.keys())
                    elif bf_name_raw == "choose":
                        candidates = []
                        for p in probes:
                            tr = tuner.triggers.get(f"probe_{p}")
                            n_sigs = len(tr.signals) if tr else 0
                            candidates.append((n_sigs, p))
                        if candidates:
                            candidates.sort()
                            chosen = candidates[0][1]
                            print(Fore.CYAN + f"[Probe] Model chose to backfill '{chosen}' (only {candidates[0][0]} signals)." + Style.RESET_ALL)
                            bf_names_to_rebuild = [chosen]
                        else:
                            print(Fore.YELLOW + "[Probe] No active probes to backfill." + Style.RESET_ALL)
                            continue
                    else:
                        bf_name_resolved = resolve_probe_choice(bf_name_raw, probes)
                        if not bf_name_resolved:
                            continue
                        bf_name = re.sub(r"[^a-z0-9_]", "_", bf_name_resolved)[:40]
                        if bf_name not in probes:
                            print(Fore.YELLOW + f"[Probe] no active probe named '{bf_name}'.{did_you_mean(bf_name, probes)} Bare :probe lists them." + Style.RESET_ALL)
                            continue
                        bf_names_to_rebuild = [bf_name]
                    bf_limit = backfill_ns.limit
                    archive = [
                        r for r in memory.records
                        if r.scope == memory.scope and r.kind == "turn"
                        and r.role == "assistant" and (r.text or "").strip()
                    ]
                    if bf_limit:
                        archive = archive[-bf_limit:]
                    if not archive:
                        print(Fore.YELLOW + "[Probe] no stored assistant replies to backfill from." + Style.RESET_ALL)
                        continue
                    # Scoring re-encodes reply text, so the archive can be scored
                    # exactly as live turns were. The trigger stream is REBUILT,
                    # not appended: the archive is a superset of every turn this
                    # probe scored live, and replacing is what keeps those turns
                    # from counting twice. Pre-mint turns alone get new paired
                    # rows in the turn log -- post-mint turns already wrote
                    # theirs live.
                    mint_ts_map = {}
                    for _pn in bf_names_to_rebuild:
                        _mint = next(
                            (e.timestamp for e in memory.records
                             if e.kind == "event" and "probe" in (e.tags or []) and (e.text or "").startswith(f"{_pn}:")),
                            None,
                        )
                        mint_ts_map[_pn] = _mint
                    from invariants.engine import _inputs as _bf_inputs, _hidden_states as _bf_hidden, probe_score as _bf_score
                    # The forward pass is shared, but projection breadth follows
                    # the command contract: a named backfill scores only that
                    # probe. The explicit "all" form produces joint rows for all
                    # active probes when multi-anchor calibration is wanted.
                    scoring_names = backfill_scoring_names(
                        probes,
                        bf_names_to_rebuild,
                        request_all=(bf_name_raw == "all"),
                    )
                    bf_rollings = {p: deque(maxlen=40) for p in scoring_names}
                    bf_scored = []  # (timestamp, {probe: (raw, sig)}, stored sense)
                    display_name = "all probes" if bf_name_raw == "all" else bf_names_to_rebuild[0]
                    print(
                        Fore.CYAN
                        + f"[Probe] backfilling {display_name} over {len(archive)} archived replies "
                        + f"(one forward each; scoring {len(scoring_names)} requested probe(s) per reply)..."
                        + Style.RESET_ALL,
                        flush=True,
                    )
                    with torch.no_grad():
                        for bf_i, br in enumerate(archive):
                            b_ids = _bf_inputs(model, br.text[:600])
                            b_hs = _bf_hidden(model, b_ids["input_ids"], b_ids.get("attention_mask"))
                            per_probe = {}
                            for _pn in scoring_names:
                                _pd = probes[_pn]
                                raw = _bf_score(b_hs, _pd["direction"])
                                roll = bf_rollings[_pn]
                                sig = raw - (sum(roll) / len(roll)) if roll else 0.0
                                roll.append(raw)
                                per_probe[_pn] = (raw, sig)
                            bm = br.metrics if isinstance(br.metrics, dict) else {}
                            sense = bm.get("sense_score")
                            bf_scored.append((br.timestamp, per_probe, float(sense) if sense is not None else None))
                            if (bf_i + 1) % 25 == 0:
                                print(Fore.CYAN + f"  ...{bf_i + 1}/{len(archive)}" + Style.RESET_ALL, flush=True)
                    total_credited = {}
                    for _pn in bf_names_to_rebuild:
                        trig = tuner.register(f"probe_{_pn}", 0.0, kind="threshold", comparator=">=")
                        trig.signals.clear()
                        trig.outcomes.clear()
                        trig.observed = 0
                        trig.fired = 0
                        credited = 0
                        for _bts, _bpp, _bsense in bf_scored:
                            _bsig = _bpp[_pn][1]
                            trig.observe(_bsig)
                            if _bsense is not None:
                                trig.credit(_bsig, _bsense)
                                credited += 1
                        total_credited[_pn] = credited
                        probes[_pn]["history"] = deque((pp[_pn][0] for _, pp, _ in bf_scored), maxlen=40)
                    save_probe_raw_histories(probes)
                    tuner.save()
                    
                    existing_ts = set()
                    try:
                        with open(TURN_SIGNALS_PATH, "r", encoding="utf-8") as _tf:
                            for _line in _tf:
                                try:
                                    _r = json.loads(_line)
                                except Exception:
                                    continue
                                if _r.get("basis") == "backfill" and _r.get("ts"):
                                    # if ANY of the rebuilt probes are already in this backfill row, consider it existing
                                    if any(f"probe_{_pn}" in _r for _pn in bf_names_to_rebuild):
                                        existing_ts.add(_r["ts"])
                    except OSError:
                        pass
                    rows_added = 0
                    try:
                        with open(TURN_SIGNALS_PATH, "a", encoding="utf-8") as _tf:
                            for _bts, _bpp, _bsense in bf_scored:
                                # We skip adding this joint row if ALL rebuilt probes were minted BEFORE this turn
                                if all(mint_ts_map.get(_pn) is not None and _bts >= mint_ts_map[_pn] for _pn in bf_names_to_rebuild):
                                    continue
                                if _bts in existing_ts:
                                    continue
                                row = {"ts": _bts, "basis": "backfill"}
                                for _pn, (_praw, _psig) in _bpp.items():
                                    row[f"probe_{_pn}"] = float(_psig)
                                if _bsense is not None:
                                    row["sense"] = float(_bsense)
                                turn_log.append(dict(row))
                                _tf.write(json.dumps(row) + chr(10))
                                rows_added += 1
                    except OSError:
                        pass
                    memory.append_event(
                        "probe_backfilled",
                        text=f"{display_name}: {len(bf_scored)} archived replies re-scored",
                        tags=["probe"],
                        provenance={
                            "probes": bf_names_to_rebuild,
                            "turns_scored": len(bf_scored),
                            "paired_rows_added": rows_added,
                        },
                    )
                    st = tuner.triggers[f"probe_{bf_names_to_rebuild[0]}"].outcome_stats() if bf_names_to_rebuild else {"lift": 0}
                    print(
                        Fore.GREEN
                        + f"[Probe] {display_name} backfilled: {len(bf_scored)} archived replies re-scored in order. "
                        + f"Trigger stream(s) rebuilt, "
                        + f"{rows_added} pre-mint paired rows added, rolling history seeded. "
                        + f"Live scoring continues normally."
                        + Style.RESET_ALL
                    )
                    continue
                pname, _, framings = pargs.partition(" ")
                pname = re.sub(r"[^a-z0-9_]", "_", pname.lower())[:40]
                if "||" not in framings and r"\||" not in framings:
                    print(Fore.CYAN + f"[Probe] Suggesting contrastive framings for '{pname}'..." + Style.RESET_ALL)
                    suggestion_prompt = (
                        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"Write a contrastive definition pair for a behavioral dimension called '{pname}'. "
                        f"Write both sides in the FIRST PERSON, as I describe MYSELF -- each side MUST start with 'I'. "
                        f"The format must be exactly: <first-person positive statement> || <first-person negative statement>.\n\n"
                        f"Example for 'understanding': I fully grasp what the user means. || I am confused and miss the point.\n\n"
                        f"Output ONLY the single contrastive pair. Do not add any other text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    sug = generate_agentic_text(
                        model,
                        instruction=suggestion_prompt,
                        config=config,
                        pre_formatted=True,
                        max_new_tokens=150
                    )
                    sug = sug.strip()
                    suggested_command = f":probe {pname} {sug}"
                    memory.append_definition(
                        "probe",
                        pname,
                        suggested_command,
                        authored_by="model",
                        status="proposed",
                        provenance={"source": ":probe framing suggestion"},
                    )
                    print(Fore.GREEN + f"\n[Probe Suggestion] Try running this:\n{suggested_command}\n" + Style.RESET_ALL)
                    continue
                a_text, _, b_text = _partition_unescaped_pipes(framings)
                a_text, b_text = a_text.strip(), b_text.strip()
                
                if pname in probes:
                    existing_a, existing_b = probes[pname].get("framings", ("", ""))
                    if existing_a == a_text and existing_b == b_text:
                        print(Fore.CYAN + f"[Probe] '{pname}' already has these exact framings; using the existing one." + Style.RESET_ALL)
                        continue
                    else:
                        base = pname
                        counter = 1
                        while f"{base}_{counter}" in probes:
                            counter += 1
                        backup_name = f"{base}_{counter}"
                        print(Fore.CYAN + f"[Probe] '{base}' has different framings. Backing up the old one to '{backup_name}' to make way for the new one." + Style.RESET_ALL)
                        probes[backup_name] = probes.pop(base)
                        if f"probe_{base}" in tuner.triggers:
                            tuner.triggers[f"probe_{backup_name}"] = tuner.triggers.pop(f"probe_{base}")

                b_text, mint_layers = strip_band_suffix(b_text)
                from invariants.engine import _inputs, _hidden_states, probe_direction, steer_band_layers
                ids_a = _inputs(model, a_text[:600])
                hs_a = _hidden_states(model, ids_a["input_ids"], ids_a.get("attention_mask"))
                ids_b = _inputs(model, b_text[:600])
                hs_b = _hidden_states(model, ids_b["input_ids"], ids_b.get("attention_mask"))
                direction = probe_direction(
                    hs_a, hs_b,
                    mint_layers if mint_layers is not None else steer_band_layers(hs_a.shape[0]),
                )
                if not direction:
                    print(Fore.YELLOW + "[Probe] framings produced no usable direction; try more contrastive text." + Style.RESET_ALL)
                    continue
                probes[pname] = {"direction": direction, "history": deque(maxlen=40), "framings": (a_text, b_text)}
                tuner.register(f"probe_{pname}", 0.0, kind="threshold", comparator=">=")
                try:
                    os.makedirs(PROBE_DIR, exist_ok=True)
                    save_name = pname if getattr(model.model.config, "_name_or_path", "") == DEFAULT_MODEL else f"{pname}_d{model.d_model}"
                    torch.save({"direction": direction, "framings": (a_text, b_text)}, os.path.join(PROBE_DIR, f"{save_name}.pt"))
                except Exception:
                    pass
                model_definition = memory.find_definition(
                    "probe",
                    pname,
                    authored_by="model",
                    open_only=True,
                )
                authored_by = "operator"
                if model_definition is not None:
                    actual_command = f":probe {pname} {a_text} || {b_text}"
                    same_definition = (
                        " ".join(model_definition.text.split())
                        == " ".join(actual_command.split())
                    )
                    authored_by = "model" if same_definition else "model_revised_by_operator"
                    memory.append_definition_feedback(
                        model_definition,
                        (
                            "operator minted the proposed definition"
                            if same_definition
                            else "operator revised the proposal and minted the resulting definition"
                        )
                        + (f"; rationale: {command_because}" if command_because else ""),
                        verdict="accepted" if same_definition else "revised",
                        source="operator",
                        provenance={"actual_definition": actual_command},
                    )
                memory.append_event(
                    "probe_minted",
                    text=f"{pname}: {a_text} || {b_text}",
                    tags=["probe"],
                    provenance={
                        "layers": sorted(direction),
                        "authored_by": authored_by,
                        "definition_id": model_definition.record_id if model_definition else None,
                    },
                )
                print(
                    Fore.CYAN
                    + f"[Probe] {pname} minted over {len(direction)} layers from your framings -- an UNVALIDATED "
                    + "hypothesis-sensor. It scores every turn from here (one extra forward per turn), centered "
                    + "against its own rolling history, paired with sense. Calibrate against it once evidence "
                    + f"accrues: :calibrate conversation_productive {pname}. Or pre-load evidence from the "
                    + f"archive now: :probe backfill {pname}"
                    + Style.RESET_ALL
                )
                continue
            if user_input.strip().lower() == ":suggest apply" or user_input.strip().lower().startswith(":suggest apply "):
                if _last_completion:
                    print(Fore.GREEN + f"[Suggest Apply] Auto-queueing AI completion: {_last_completion}" + Style.RESET_ALL)
                    input_queue.append(_last_completion)
                    if pending_completion_definition_id:
                        memory.append_definition_feedback(
                            pending_completion_definition_id,
                            "operator applied and queued the completion",
                            verdict="accepted",
                            source="operator",
                        )
                        pending_completion_definition_id = None
                    _last_completion = None
                    continue
                    
                _arch = sum(
                    1 for r in memory.records
                    if r.scope == memory.scope and r.kind == "turn" and r.role == "assistant"
                )
                sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                safe = [(cat, line, cmd) for cat, line, cmd in sugg if cat in SUGGEST_APPLY_SAFE]
                
                # Forward the because clause to the queued commands
                apply_because = None
                if command_because:
                    apply_because = f" because {command_because}"
                    
                if not safe:
                    print(Fore.CYAN + "[Suggest Apply] nothing safe to auto-run (explore/expose stay manual)." + Style.RESET_ALL)
                else:
                    cmds = [cmd + (apply_because if apply_because else "") for _, _, cmd in safe]
                    input_queue.extend(cmds)
                    print(Fore.GREEN + f"[Suggest Apply] Auto-queued {len(cmds)} measurement/calibration action(s)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":place "):
                pargs = user_input[len(":place "):].strip().split()
                if len(pargs) >= 2:
                    pname = resolve_probe_choice(pargs[0], probes)
                    if not pname:
                        continue
                    if pname in probes:
                        sign_arg = pargs[1]
                        if sign_arg in ("+", "positive", "1", "pos"):
                            sign = 1.0
                        elif sign_arg in ("-", "negative", "-1", "neg"):
                            sign = -1.0
                        else:
                            print(Fore.YELLOW + "[Place] specify + or - for the placement." + Style.RESET_ALL)
                            continue
                        
                        last_reply = None
                        for r in reversed(memory.records):
                            if r.role == "assistant":
                                last_reply = r.text
                                break
                        if not last_reply:
                            print(Fore.YELLOW + "[Place] no assistant reply found to place." + Style.RESET_ALL)
                            continue
                            
                        print(Fore.CYAN + f"[Place] Extracting state of last response to update '{pname}'..." + Style.RESET_ALL)
                        from invariants.engine import _inputs, _hidden_states
                        ids = _inputs(model, last_reply[:600])
                        hs = _hidden_states(model, ids["input_ids"], ids.get("attention_mask"))
                        
                        probe_dir = probes[pname]["direction"]
                        learning_rate = 0.15  # a noticeable nudge
                        updated_layers = 0
                        for L in list(probe_dir.keys()):
                            L_int = int(L)
                            if 0 <= L_int < hs.size(0):
                                mean_hs = hs[L_int].mean(dim=0).to(model.device).reshape(-1)
                                if mean_hs.norm().item() > 0:
                                    mean_hs = mean_hs / mean_hs.norm()
                                    current_dir = probe_dir[L].to(model.device).reshape(-1)
                                    new_dir = current_dir + sign * learning_rate * mean_hs
                                    if new_dir.norm().item() > 0:
                                        probe_dir[L] = new_dir / new_dir.norm()
                                        updated_layers += 1
                                        
                        print(Fore.GREEN + Style.BRIGHT + f"[Place] '{pname}' probe nudged {'toward' if sign > 0 else 'away from'} the last response over {updated_layers} layers." + Style.RESET_ALL)
                        probe_definition = memory.find_definition("probe", pname)
                        if probe_definition is not None:
                            memory.append_definition_feedback(
                                probe_definition,
                                f"direction nudged {'toward' if sign > 0 else 'away from'} the last response",
                                verdict="positive" if sign > 0 else "negative",
                                source="operator",
                                metrics={"sign": sign, "layers": updated_layers},
                            )
                        memory.append_event(
                            "probe_placed",
                            text=f"placed last response on '{pname}' as {'positive' if sign > 0 else 'negative'}",
                            tags=["probe", "place"],
                            provenance={"probe": pname, "sign": sign}
                        )
                else:
                    print(Fore.YELLOW + "[Place] Usage: :place <probe> <+|->" + Style.RESET_ALL)
                continue
            if user_input.startswith(":install "):
                mtok = user_input[len(":install "):].strip().split()
                if len(mtok) < 1:
                    print(Fore.YELLOW + "[System] Usage: :install <alias> [file] [goal]" + Style.RESET_ALL)
                    continue
                a_name = mtok[0].lower()
                
                if len(mtok) >= 2 and os.path.isfile(mtok[1]):
                    src_file = mtok[1]
                    goal_text = " ".join(mtok[2:]) if len(mtok) > 2 else "Manually installed macro."
                    args_to_scan = mtok[2:]
                else:
                    src_file = macro_aliases.get(a_name, mtok[1] if len(mtok) >= 2 else f"{a_name}.txt")
                    goal_text = " ".join(mtok[1:]) if len(mtok) > 1 else "Manually installed macro."
                    args_to_scan = mtok[1:]

                if not os.path.isfile(src_file):
                    print(Fore.RED + f"[Error] File not found: {src_file}. Please provide the file path." + Style.RESET_ALL)
                    continue
                if _hidden_overwrite_blocked(a_name, "System"):
                    continue
                arg_names = []
                seen_args = set()
                for tok in args_to_scan:
                    idx = tok.find("$")
                    if idx >= 0:
                        clean = tok[idx+1:].rstrip(".,;:\"'()!]")
                        if clean and clean not in seen_args:
                            arg_names.append(clean)
                            seen_args.add(clean)
                dest = os.path.join(ROOT, "invariants", "out", "macros", f"{a_name}.txt")
                try:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(src_file, "r", encoding="utf-8") as rf:
                        content = rf.read().strip()
                    with open(dest, "w", encoding="utf-8") as wf:
                        wf.write(f"# :solve macro '{a_name}' -- {goal_text}\n")
                        if arg_names:
                            wf.write(f"# args: {', '.join(arg_names)}\n")
                        if content:
                            wf.write(content + "\n")
                    macro_aliases[a_name] = dest
                    _save_macro_aliases()
                    print(Fore.GREEN + f"[System] Installed macro ':{a_name}' to {dest}. Run it with :{a_name} <args>." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.RED + f"[Error] Could not install macro: {e}" + Style.RESET_ALL)
                continue
            if user_input.lower().startswith(":expect "):
                eargs = user_input.strip().split()
                if len(eargs) < 2:
                    print(Fore.YELLOW + "[Expect] Usage: :expect <name> | :expect macro <name> | :expect file <path> | :expect autocomplete" + Style.RESET_ALL)
                    continue
                etype = eargs[1].lower()
                if etype not in ("macro", "file", "autocomplete", "var"):
                    ename = eargs[1]
                    etype = "var"
                elif etype == "autocomplete":
                    ename = " ".join(eargs[2:]) if len(eargs) > 2 else ""
                else:
                    ename = eargs[2] if len(eargs) > 2 else ""
                expect_context = {}
                if etype == "macro":
                    expect_context = solve_expect_contexts.get(
                        re.sub(r"[^a-z0-9_]", "_", (ename or "").lower())[:40].strip("_"),
                        {},
                    )
                _pending_expect = {"type": etype, "name": ename, "context": expect_context}
                print(Fore.CYAN + f"[Expect] Intercepting the model's next turn for type '{etype}' ('{ename}')." + Style.RESET_ALL)
                continue
            if user_input.lower().startswith(":suggest"):
                sargs = user_input.strip().split()
                _arch = sum(
                    1 for r in memory.records
                    if r.scope == memory.scope and r.kind == "turn" and r.role == "assistant"
                )
                def _print_user_facing_command_launcher(filter_text=None):
                    catalog = suggest_command_catalog(
                        hidden_commands=hidden_commands,
                        macro_aliases=macro_aliases,
                        solve_macros=list_solve_macros(),
                        filter_text=filter_text,
                    )
                    filt = f" matching '{filter_text}'" if filter_text else ""
                    if not catalog:
                        print(Fore.YELLOW + f"[Suggest] No visible user-facing commands{filt}." + Style.RESET_ALL)
                        return
                    print(
                        Fore.CYAN
                        + f"[Suggest] {len(catalog)} visible user-facing command line(s){filt}. "
                          "Run one line directly; replace <...> placeholders and choose one side of | alternatives."
                        + Style.RESET_ALL
                    )
                    for item in catalog:
                        prefix = "macro" if item["kind"] == "macro" else "cmd"
                        summary = item.get("summary") or ""
                        suffix = f"  -- {summary}" if summary else ""
                        print(Fore.GREEN + f"      -> {item['command']}" + Fore.CYAN + f"  [{prefix}]{suffix}" + Style.RESET_ALL)
                
                target_probe = None
                if len(sargs) > 1 and sargs[1].lower() not in ("apply", "suggestions"):
                    if sargs[1].lower() in ("command", "commands", "help", "all", "catalog", "launcher"):
                        _print_user_facing_command_launcher(" ".join(sargs[2:]).strip() or None)
                        continue
                    if sargs[1].lower() in ("gravity", "field", "physics", "steer"):
                        filter_text = " ".join(sargs[1:]).strip()
                        _print_user_facing_command_launcher(filter_text)
                        continue
                    if sargs[1].startswith(":"):
                        prefix_cmd = user_input[len(":suggest"):].strip()
                        active_probes = ", ".join(probes.keys()) if probes else "None"
                        active_steers = ", ".join(
                            f"{k}={tuner.get(k, 0.0):g}"
                            for k in tuner.triggers.keys()
                            if k.startswith("steer_") or "_alpha" in k
                        )
                        command_reference = build_command_autocomplete_reference(
                            prefix_cmd,
                            hidden_commands=hidden_commands,
                        )
                        autocomplete_commands = (
                            set(BUILTIN_COMMANDS)
                            | set(macro_aliases)
                        ) - set(hidden_commands)
                        available_commands = ", ".join(
                            f":{name}" for name in sorted(autocomplete_commands)
                        )

                        because_ctx = ""
                        if command_because:
                            because_ctx = f"\nThe user provided this rationale for the command they are trying to type:\n{command_because}\nEnsure your suggested completion perfectly aligns with this rationale."
                            print(Fore.CYAN + "[Suggest] Passing your 'because' rationale to the model." + Style.RESET_ALL)

                        default_suggest_prompt = (
                            ":expect autocomplete $prefix_cmd\n"
                            "You are a command autocomplete assistant for an interactive agent shell.\n"
                            "The user has started typing a command: '$prefix_cmd'\n"
                            "$because_ctx\n"
                            "The following is the AUTHORITATIVE LIVE COMMAND REFERENCE. It is generated "
                            "from the shell implementation that will parse the answer. Treat it as newer "
                            "and more authoritative than any earlier conversation, memory, interpretation, "
                            "or example. Do not invent syntax that is absent from it.\n"
                            "--- live command reference ---\n"
                            "$command_reference\n"
                            "--- end live command reference ---\n"
                            "Currently available non-hidden command names: $available_commands\n"
                            "Currently active probes: $active_probes\n"
                            "Active steers: $active_steers\n"
                            "Please provide exactly ONE valid completion for this command based on the available shell capabilities. "
                            "Output only the full completed command string (starting with '$prefix_cmd'), and nothing else."
                        )
                        prompt = load_prompt(
                            "suggest_autocomplete",
                            default_suggest_prompt,
                            required_substring="$command_reference",
                            prefix_cmd=prefix_cmd,
                            because_ctx=because_ctx,
                            command_reference=command_reference,
                            available_commands=available_commands,
                            active_probes=active_probes,
                            active_steers=active_steers
                        )
                        print(Fore.CYAN + f"[Suggest] Queuing documentation-grounded completion for '{prefix_cmd}'..." + Style.RESET_ALL)
                        queue_macro_text(prompt, input_queue)
                        continue
                        
                    target_probe = re.sub(r"[^a-z0-9_]", "_", sargs[1].lower())[:40]
                    if target_probe in probes:
                        print(Fore.CYAN + f"[Suggest] Scanning for specific moves for probe '{target_probe}'..." + Style.RESET_ALL)
                        all_sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                        sugg = [(cat, line, cmd) for cat, line, cmd in all_sugg if target_probe in line or target_probe in cmd]
                        if not sugg:
                            print(Fore.YELLOW + f"[Suggest] No specific data-backed moves ready for '{target_probe}' yet." + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To generate evidence: :probe backfill {target_probe}" + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To steer blindly:     :tune probe_{target_probe}_alpha 0.5" + Style.RESET_ALL)
                            continue
                    else:
                        print(Fore.CYAN + f"[Suggest] Probe '{target_probe}' not active. Mint it first with :probe {target_probe} <positive> || <negative>" + Style.RESET_ALL)
                        continue
                elif len(sargs) == 1 and command_because:
                    # Internally compute the similarity of the because string to all active probes
                    if not probes:
                        print(Fore.YELLOW + "[Suggest] No active probes to match against your reason. Mint some first!" + Style.RESET_ALL)
                        continue
                        
                    print(Fore.CYAN + f"[Suggest] Projecting your reason into the model's representation space to find the best matching probes..." + Style.RESET_ALL)
                    from invariants.engine import _inputs, _hidden_states
                    
                    ids = _inputs(model, command_because[:600])
                    hs = _hidden_states(model, ids["input_ids"], ids.get("attention_mask"))
                    
                    probe_scores = {}
                    for pname, pdata in probes.items():
                        probe_dir = pdata["direction"]
                        sim_sum = 0.0
                        layers_counted = 0
                        for L in list(probe_dir.keys()):
                            L_int = int(L)
                            if 0 <= L_int < hs.size(0):
                                mean_hs = hs[L_int].mean(dim=0).to(model.device).reshape(-1)
                                if mean_hs.norm().item() > 0:
                                    mean_hs = mean_hs / mean_hs.norm()
                                    p_dir = probe_dir[L].to(model.device).reshape(-1)
                                    sim = torch.nn.functional.cosine_similarity(mean_hs.unsqueeze(0), p_dir.unsqueeze(0)).item()
                                    sim_sum += sim
                                    layers_counted += 1
                        if layers_counted > 0:
                            probe_scores[pname] = sim_sum / layers_counted
                            
                    if not probe_scores:
                        print(Fore.YELLOW + "[Suggest] Could not compute similarities." + Style.RESET_ALL)
                        continue
                        
                    confident_match, sorted_probes = confident_probe_match(probe_scores)
                    top_probe, top_score = sorted_probes[0]

                    if confident_match is None:
                        runner_score = sorted_probes[1][1] if len(sorted_probes) > 1 else 0.0
                        print(
                            Fore.YELLOW
                            + f"[Suggest] No confident probe match for that reason. Best was '{top_probe}' "
                            + f"({top_score:+.3f}; runner-up {runner_score:+.3f}), below the confidence floor "
                            + "or too close to distinguish. I will not turn that weak winner into a probe-specific action."
                            + Style.RESET_ALL
                        )
                        sugg = suggest_actions(
                            tuner,
                            list(turn_log),
                            probes=probes,
                            archive_size=_arch,
                        )
                        if not sugg:
                            print(
                                Fore.CYAN
                                + "  -> Ask for a command completion with "
                                + f":suggest :<command> because {command_because}"
                                + Style.RESET_ALL
                            )
                        # Continue into the ordinary state-wide suggestion
                        # renderer. Crucially, do not recommend backfilling the
                        # arbitrary highest probe.
                        target_probe = None
                    else:
                        top_probe, top_score = confident_match
                        print(Fore.GREEN + Style.BRIGHT + f"[Suggest] Confident internal representation match: '{top_probe}' (similarity: {top_score:+.3f})" + Style.RESET_ALL)
                    if len(sorted_probes) > 1:
                        runners_up = ", ".join(f"{p} ({s:+.3f})" for p, s in sorted_probes[1:3])
                        print(Fore.CYAN + f"          Runners up: {runners_up}" + Style.RESET_ALL)

                    if confident_match is not None:
                        all_sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                        sugg = [(cat, line, cmd) for cat, line, cmd in all_sugg if top_probe in line or top_probe in cmd]

                        if not sugg:
                            print(Fore.YELLOW + f"[Suggest] No specific data-backed moves ready for '{top_probe}' yet." + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To generate evidence: :probe backfill {top_probe}" + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To steer blindly:     :tune probe_{top_probe}_alpha 0.5" + Style.RESET_ALL)
                            continue
                else:
                    sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)

                if not sugg:
                    print(
                        Fore.CYAN
                        + "[Suggest] nothing ready yet: no lever has accrued enough evidence for a "
                        + "data-backed move. Streams accrue every turn; :probe backfill pre-loads the archive."
                        + Style.RESET_ALL
                    )
                else:
                    labels = {
                        "field": "Gravity / field mechanics",
                        "calibrate": "Calibrations ready",
                        "commit": "Explored, not committed",
                        "explore": "Capabilities never tried",
                        "backfill": "Probes short on evidence",
                        "expose": "Signals worth exposing",
                    }
                    print(
                        Fore.CYAN
                        + f"[Suggest] {len(sugg)} move(s) the accrued state supports -- computed, not applied:"
                        + Style.RESET_ALL
                    )
                    for cat in ("field", "commit", "calibrate", "explore", "backfill", "expose"):
                        group = [(l, c) for k, l, c in sugg if k == cat]
                        if not group:
                            continue
                        print(Fore.CYAN + Style.BRIGHT + f"  [{labels[cat]}]" + Style.RESET_ALL)
                        for line, cmd in group[:8]:
                            if command_because:
                                cmd += f" because {command_because}"
                            print(Fore.CYAN + f"    {line}" + Style.RESET_ALL)
                            print(Fore.GREEN + f"      -> {cmd}" + Style.RESET_ALL)
                if len(sargs) == 1:
                    _print_user_facing_command_launcher()
                continue
            if user_input.startswith(":label"):
                # Human-in-the-loop: judge the MOST RECENT turn on a probe's (or
                # any stream's) axis as positive or negative. Credits that axis's
                # last-turn signal with an operator outcome (1.0 / 0.0) -- a
                # supervised datapoint alongside the automatic sense credit, so a
                # probe's lift can be grounded in human judgment, not only sense.
                largs = user_input[len(":label"):].split()
                if len(largs) < 2:
                    print(Fore.CYAN + "[Label] Usage: :label <probe|stream> pos|neg -- mark the last turn on that axis." + Style.RESET_ALL)
                    continue
                lname_raw = largs[0]
                lname = resolve_probe_choice(lname_raw, probes)
                if not lname:
                    continue
                verdict = largs[1].lower()
                if verdict in ("pos", "positive", "+", "good", "1", "up"):
                    outcome = 1.0
                elif verdict in ("neg", "negative", "-", "bad", "0", "down"):
                    outcome = 0.0
                else:
                    print(Fore.YELLOW + "[Label] verdict must be pos or neg (positive/negative)." + Style.RESET_ALL)
                    continue
                target_trigger, target_stream = resolve_target(lname, tuner)
                if target_stream is None or target_trigger not in tuner.triggers:
                    print(Fore.YELLOW + f"[Label] unknown axis '{lname}'.{did_you_mean(lname, calibratable_names(tuner))}" + Style.RESET_ALL)
                    continue
                if not turn_log:
                    print(Fore.YELLOW + "[Label] no completed turn to label yet." + Style.RESET_ALL)
                    continue
                sig = turn_log[-1].get(target_stream)
                if sig is None:
                    print(Fore.YELLOW + f"[Label] the last turn carries no reading for '{target_stream}'. Score it first." + Style.RESET_ALL)
                    continue
                tuner.credit(target_trigger, float(sig), outcome)
                probe_definition = None
                if target_trigger.startswith("probe_"):
                    probe_definition = memory.find_definition(
                        "probe",
                        target_trigger[6:],
                    )
                if probe_definition is not None:
                    memory.append_definition_feedback(
                        probe_definition,
                        f"last turn labeled {'positive' if outcome >= 0.5 else 'negative'} "
                        f"at signal {float(sig):+.4f}",
                        verdict="positive" if outcome >= 0.5 else "negative",
                        source="operator",
                        metrics={"signal": float(sig), "outcome": outcome},
                    )
                memory.append_event(
                    "operator_label",
                    text=f"{target_trigger}: {'positive' if outcome >= 0.5 else 'negative'} (signal {float(sig):+.4f})",
                    tags=["label", "human_feedback"],
                    provenance={"axis": target_trigger, "signal": float(sig), "outcome": outcome},
                )
                st = tuner.triggers[target_trigger].outcome_stats()
                disp = target_trigger[6:] if target_trigger.startswith("probe_") else target_trigger
                print(
                    Fore.GREEN
                    + f"[Label] last turn marked {'POSITIVE' if outcome >= 0.5 else 'NEGATIVE'} on {disp} "
                    + f"(its signal {float(sig):+.4f} paired with outcome {outcome:g}). "
                    + f"Lift now {st.get('lift')} over {st.get('n_credited')} credited turns."
                    + Style.RESET_ALL
                )
                continue
            if user_input.startswith(":calibrate"):
                cal_ns = CALIBRATE_SPEC.parse(user_input[len(":calibrate"):])
                cargs = ([cal_ns.probe] + (cal_ns.args.split() if cal_ns.args else [])) if cal_ns.probe else []
                if not cargs:
                    print(Fore.CYAN + "[Calibrate] Usage: :calibrate <name> [percentile|intent|band args]." + Style.RESET_ALL)
                    routes = {}
                    for nm in sorted(set(list(tuner.triggers) + ["steer_cap_fraction", "steer_band"])):
                        routes.setdefault(calibration_policy(nm)[0], []).append(nm)
                    for route in ("threshold", "cap", "band", "reject"):
                        if routes.get(route):
                            label = {"threshold": "percentile of own signals", "cap": "observed push ratios",
                                     "band": "per-layer outcomes", "reject": "REFUSED (circular/binary)"}[route]
                            print(Fore.CYAN + f"  {label}: {', '.join(routes[route])}" + Style.RESET_ALL)
                    continue
                cal_name_raw, cargs = consume_probe_args(cargs)
                cal_name = resolve_probe_choice(cal_name_raw, calibratable_names(tuner), model=model, config=config, action_name="calibrate")
                if not cal_name:
                    continue
                route, reason = calibration_policy(cal_name)
                if len(cargs) >= 1 and cargs[0].lower() == "outcome":
                    if cal_name not in OUTCOME_CALIBRATABLE and route != "threshold":
                        print(Fore.YELLOW + f"[Calibrate] outcome route not available for '{cal_name}'." + Style.RESET_ALL)
                        continue
                    trig = tuner.triggers.get(cal_name)
                    pairs = list(trig.outcomes) if trig is not None else []
                    v = outcome_calibration(pairs)
                    if v is None:
                        tried = sorted({round(float(sig), 6) for sig, _ in pairs})
                        print(
                            Fore.YELLOW
                            + f"[Calibrate] refused: outcome calibration needs >=2 tried values with >=5 "
                            + f"sensed turns each (tried so far: {tried or 'none'}). No exploration, no verdict.\n"
                            + "(Type ':queue' to auto-retry this in the background)"
                            + Style.RESET_ALL
                        )
                        last_refused_calibration = user_input[len(":calibrate"):].strip()
                    else:
                        tuner.set(cal_name, v)
                        print(
                            Fore.GREEN
                            + f"[Calibrate] {cal_name} = {round(v, 4)} -- the tried value whose turns "
                            + f"went best ({len(pairs)} sensed turns across the values explored)."
                            + Style.RESET_ALL
                        )
                    continue
                if route == "reject":
                    print(Fore.YELLOW + f"[Calibrate] rejected for '{cal_name}': {reason}." + Style.RESET_ALL)
                    continue
                if route == "cap":
                    cal_pct = None
                    if len(cargs) >= 2:
                        try:
                            cal_pct = float(cargs[1])
                        except ValueError:
                            print(Fore.YELLOW + "[Calibrate] percentile must be a number." + Style.RESET_ALL)
                            continue
                    if cal_pct is not None and cal_pct >= 100:
                        print(
                            Fore.YELLOW
                            + "[Calibrate] rejected: p100 admits every observed push -- the envelope "
                            + "would stop binding in the observed regime."
                            + Style.RESET_ALL
                        )
                        continue
                    from invariants import engine as _engine
                    v = _engine.calibrate_steer_cap_fraction(percentile=cal_pct)
                    if v is None:
                        st = _engine.steer_telemetry_stats()
                        print(
                            Fore.YELLOW 
                            + f"[Calibrate] refused: only {st['n']} observed pushes (need {_engine.STEER_CAP_MIN_N}).\n"
                            + "(Type ':queue' to auto-retry this in the background)"
                            + Style.RESET_ALL
                        )
                        last_refused_calibration = user_input[len(":calibrate"):].strip()
                    else:
                        tuner.set("steer_cap_fraction", v)
                        print(Fore.GREEN + f"[Calibrate] steer_cap_fraction = {round(v, 4)} from observed push ratios." + Style.RESET_ALL)
                    continue
                if route == "band":
                    user_input = ":tune steer_band auto " + " ".join(cargs[1:])
                    # fall through to the :tune handler below with the same args
                if route == "threshold":
                    anchor = "".join(cargs[1:]).lower() if len(cargs) >= 2 else None
                    anchor_is_word = anchor is not None and not anchor.replace(".", "").isdigit()
                    if cal_name == "conversation_productive" and anchor == "intent":
                        user_input = ":tune conversation_productive auto intent"
                        # fall through to the shared intent route below
                    elif anchor_is_word:
                        # 'intent' reaches here only when the target is NOT
                        # conversation_productive (that pair is caught above),
                        # so it anchors on the intent_settling stream like any
                        # other name -- not the productive intent-axis route.
                        # BOTH WAYS, AND JOINTLY: any threshold target, anchored
                        # to one named stream or several joined with '+'. A turn
                        # counts as anchor-fired only when EVERY named stream
                        # fired; the target's bar becomes the cut (in its own
                        # units) between those turns and the rest.
                        target_trigger, target_stream = resolve_target(cal_name, tuner)
                        if target_stream is None or target_trigger not in tuner.triggers:
                            print(Fore.YELLOW + f"[Calibrate] unknown target '{cal_name}'.{did_you_mean(cal_name, calibratable_names(tuner))}" + Style.RESET_ALL)
                            continue
                        anchors, anchor_labels, bad_part = parse_anchor_spec(anchor, tuner)
                        if bad_part is not None or not anchors:
                            print(
                                Fore.YELLOW
                                + f"[Calibrate] unknown anchor '{bad_part or anchor}'.{did_you_mean(bad_part or anchor, calibratable_names(tuner))} "
                                + "Name an observed stream "
                                + "(productive/intent/impact/a probe/a phen_ sensor), join several with '+', "
                                + "negate one with a leading '-' (e.g. -estrangement), "
                                + f"or mint one: :probe {bad_part or anchor} <with it> || <without it>."
                                + Style.RESET_ALL
                            )
                            continue
                        anchor_streams = [a[0] for a in anchors]
                        v = paired_threshold_multi(list(turn_log), target_stream, anchors)
                        anchor_label = "+".join(anchor_labels)
                        n_rows = sum(1 for r in turn_log if target_stream in r and all(s in r for s in anchor_streams))
                        if v is None:
                            print(
                                Fore.YELLOW
                                + f"[Calibrate] refused: need >=5 anchor-fired and >=5 anchor-unfired turns "
                                + f"carrying {target_stream} and every anchor stream ({anchor_label}) "
                                + f"(have {n_rows} qualifying rows).\n"
                                + "(Type ':queue' to auto-retry this in the background)"
                                + Style.RESET_ALL
                            )
                            last_refused_calibration = user_input[len(":calibrate"):].strip()
                        else:
                            new_v, prior_v = tuner.set_calibrated(target_trigger, v, tuner.get("calibration_gain", 0.5))
                            fired_word = "every anchor fired" if len(anchors) > 1 else f"{anchor_label}-fired"
                            moved = "adopted" if abs(new_v - v) < 1e-9 else f"moved {round(prior_v,4)}->{round(new_v,4)} toward"
                            print(
                                Fore.GREEN
                                + f"[Calibrate] {target_trigger} {moved} cut {round(v, 4)} -- the {target_stream} cut "
                                + f"between turns where {fired_word} and the rest "
                                + f"({n_rows} qualifying rows; probes remain minted hypotheses)."
                                + Style.RESET_ALL
                            )
                        continue
                    elif not user_input.startswith(":tune"):
                        trig = resolve_calibrate_trigger(cal_name, tuner)
                        if trig is None:
                            print(Fore.YELLOW + f"[Calibrate] unknown name '{cal_name}'.{did_you_mean(cal_name, calibratable_names(tuner))} Bare :calibrate lists them." + Style.RESET_ALL)
                            continue
                        cal_name = trig.name
                        if len(trig.signals) < 10:
                            print(
                                Fore.YELLOW
                                + f"[Calibrate] refused: only {len(trig.signals)} observed signals for "
                                + f"'{cal_name}' (need 10). A bar set from a handful of points is a guess.\n"
                                + "(Type ':queue' to auto-retry this in the background)"
                                + Style.RESET_ALL
                            )
                            last_refused_calibration = user_input[len(":calibrate"):].strip()
                            continue
                        cal_pct = 50.0
                        if len(cargs) >= 2:
                            try:
                                cal_pct = float(cargs[1])
                            except ValueError:
                                print(Fore.YELLOW + "[Calibrate] percentile must be a number." + Style.RESET_ALL)
                                continue
                        if not (0.0 <= cal_pct <= 100.0):
                            print(Fore.YELLOW + "[Calibrate] rejected: percentile must be in [0, 100]." + Style.RESET_ALL)
                            continue
                        prior_v = trig.value
                        v = tuner.calibrate(cal_name, cal_pct, tuner.get("calibration_gain", 0.5))
                        moved = f"= {round(v, 4)}" if trig.calibrations <= 1 else f"moved {round(prior_v,4)}->{round(v,4)} toward p{cal_pct:g}"
                        print(
                            Fore.GREEN
                            + f"[Calibrate] {cal_name} {moved} (p{cal_pct:g} of {len(trig.signals)} observed signals)."
                            + Style.RESET_ALL
                        )
                        continue
            if user_input.startswith(":queue"):
                qargs = user_input[len(":queue"):].strip()
                if not qargs:
                    if not queued_calibrations and not queued_commands:
                        if last_refused_calibration:
                            queued_calibrations.append(last_refused_calibration)
                            print(Fore.GREEN + f"[Queue] Added: :calibrate {last_refused_calibration} (will silently retry every turn)." + Style.RESET_ALL)
                            last_refused_calibration = None
                        else:
                            print(Fore.CYAN + "[Queue] no calibrations or commands queued." + Style.RESET_ALL)
                    else:
                        for idx, qcmd in enumerate(queued_calibrations):
                            print(Fore.CYAN + f"  [cal {idx}] :calibrate {qcmd}" + Style.RESET_ALL)
                        for idx, qcmd in enumerate(queued_commands):
                            print(Fore.CYAN + f"  [cmd {idx}] {qcmd}" + Style.RESET_ALL)
                    continue
                if qargs.lower() == "clear":
                    queued_calibrations.clear()
                    queued_commands.clear()
                    print(Fore.CYAN + "[Queue] cleared." + Style.RESET_ALL)
                    continue
                if qargs.lower().startswith("drop "):
                    try:
                        drop_args = qargs[5:].strip().split()
                        if len(drop_args) >= 2 and drop_args[0].lower() in ("cmd", "command"):
                            popped = queued_commands.pop(int(drop_args[1]))
                            print(Fore.CYAN + f"[Queue] dropped: {popped}" + Style.RESET_ALL)
                        elif len(drop_args) >= 2 and drop_args[0].lower() in ("cal", "calibrate"):
                            popped = queued_calibrations.pop(int(drop_args[1]))
                            print(Fore.CYAN + f"[Queue] dropped: :calibrate {popped}" + Style.RESET_ALL)
                        else:
                            idx = int(drop_args[0])
                            popped = queued_calibrations.pop(idx)
                            print(Fore.CYAN + f"[Queue] dropped: :calibrate {popped}" + Style.RESET_ALL)
                    except Exception:
                        pass
                    continue
                if qargs.lower().startswith("calibrate ") or qargs.lower().startswith(":calibrate "):
                    cmd_to_queue = qargs.split(maxsplit=1)[1].strip() if len(qargs.split(maxsplit=1)) > 1 else ""
                    if not cmd_to_queue:
                        print(Fore.YELLOW + "[Queue] Usage: :queue calibrate <name> [args] or :queue :command ..." + Style.RESET_ALL)
                        continue
                    if cmd_to_queue not in queued_calibrations:
                        queued_calibrations.append(cmd_to_queue)
                        print(Fore.GREEN + f"[Queue] Added: :calibrate {cmd_to_queue} (will silently retry every turn)." + Style.RESET_ALL)
                    continue
                queued_cmd = qargs if qargs.startswith(":") else ":" + qargs
                qword = command_word(queued_cmd)
                if not qword or (qword not in BUILTIN_COMMANDS and qword not in macro_aliases):
                    print(Fore.YELLOW + f"[Queue] '{queued_cmd.split()[0] if queued_cmd.split() else queued_cmd}' is not a known future command.{did_you_mean(qword, _all_shell_commands())}" + Style.RESET_ALL)
                    continue
                if command_because and " because " not in queued_cmd.lower():
                    queued_cmd = f"{queued_cmd} because {command_because}"
                if queued_cmd not in queued_commands:
                    queued_commands.append(queued_cmd)
                    print(Fore.GREEN + f"[Queue] Added future command: {queued_cmd}" + Style.RESET_ALL)
                continue
            if user_input.startswith(":relations"):
                rargs = user_input[len(":relations"):].strip().split()
                aliases = {
                    "relation": "all",
                    "rels": "all",
                    "active": "all",
                    "dynamic": "bindings",
                    "dynamics": "bindings",
                    "binding": "bindings",
                    "match": "bindings",
                    "matches": "bindings",
                    "cal": "calibrations",
                    "cals": "calibrations",
                    "calibration": "calibrations",
                    "queued": "queue",
                    "queues": "queue",
                    "expose": "exposure",
                    "exposed": "exposure",
                    "target": "priority",
                    "targets": "priority",
                    "prioritize": "priority",
                    "steer": "priority",
                    "steering": "priority",
                    "field": "fields",
                    "gravity": "fields",
                    "physics": "fields",
                    "clocks": "fields",
                    "shapes": "fields",
                    "releases": "release",
                    "aliases": "macros",
                    "macro": "macros",
                }
                scopes = {aliases.get(a.lower(), a.lower()) for a in rargs} if rargs else {"active"}
                all_scopes = {"bindings", "calibrations", "queue", "exposure", "priority", "fields", "release", "macros"}
                active_scopes = all_scopes - {"macros"}
                if "all" in scopes:
                    scopes = set(all_scopes)
                if "active" in scopes:
                    scopes = (scopes - {"active"}) | active_scopes
                bad = sorted(scopes - all_scopes)
                if bad:
                    print(Fore.YELLOW + "[Relations] Unknown scope(s): " + ", ".join(bad) + ". Use :relations [active|all|bindings|calibrations|queue|exposure|priority|fields|release|macros]." + Style.RESET_ALL)
                    continue

                def _rel_terms(terms):
                    pieces = []
                    for weight, name in terms or []:
                        try:
                            w = float(weight)
                        except (TypeError, ValueError):
                            w = 1.0
                        mag = abs(w)
                        prefix = "+" if w >= 0 else "-"
                        if abs(mag - 1.0) < 1e-9:
                            pieces.append(prefix + str(name))
                        else:
                            pieces.append(f"{prefix}{mag:g}*{name}")
                    return " ".join(pieces) if pieces else "(empty)"

                def _rel_header(title, count):
                    print(Fore.CYAN + Style.BRIGHT + f"[Relations] {title} ({count})" + Style.RESET_ALL)

                printed_any = False
                if "bindings" in scopes:
                    printed_any = True
                    active_match_names = sorted(
                        pn for pn, m in probe_matches.items()
                        if str(m.get("mode", "none")).lower() in {"drive", "validate"}
                    )
                    _rel_header("Bindings And Probe-Match Links", len(tuner_bindings) + len(probe_matches))
                    if tuner_bindings:
                        match_drive_knobs = {
                            str(m.get("knob")): pn
                            for pn, m in probe_matches.items()
                            if str(m.get("mode", "none")).lower() == "drive" and m.get("knob")
                        }
                        for name, binding in sorted(tuner_bindings.items()):
                            try:
                                terms, mult = binding
                            except (TypeError, ValueError):
                                terms, mult = [], 1.0
                            source = f" via probe-match drive:{match_drive_knobs[name]}" if name in match_drive_knobs else ""
                            print(Fore.CYAN + f"  dynamic {name} <- {float(mult):g} * ({_rel_terms(terms)}){source}" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "  dynamic bindings: none" + Style.RESET_ALL)
                    if probe_matches:
                        for pn, m in sorted(probe_matches.items()):
                            mode = str(m.get("mode", "none") or "none")
                            mult = f" x{m.get('mult')}" if mode == "drive" and m.get("mult") is not None else ""
                            print(Fore.CYAN + f"  match {pn} <-> {m.get('knob', '(none)')} mode={mode}{mult}" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "  probe-match links: none" + Style.RESET_ALL)
                    if active_match_names:
                        print(Fore.YELLOW + "  active match modes: " + ", ".join(active_match_names) + Style.RESET_ALL)

                if "calibrations" in scopes:
                    printed_any = True
                    calibrated = [
                        (name, trig)
                        for name, trig in sorted(tuner.triggers.items())
                        if int(getattr(trig, "calibrations", 0) or 0) > 0
                    ]
                    _rel_header("Calibrated Knobs/Thresholds", len(calibrated))
                    if calibrated:
                        for name, trig in calibrated[:40]:
                            print(
                                Fore.CYAN
                                + f"  {name}: value={round(float(trig.value), 4)} kind={trig.kind} comparator={trig.comparator} calibrations={trig.calibrations}"
                                + Style.RESET_ALL
                            )
                        if len(calibrated) > 40:
                            print(Fore.CYAN + f"  ... {len(calibrated) - 40} more" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "  none" + Style.RESET_ALL)

                if "queue" in scopes:
                    printed_any = True
                    total_q = len(queued_calibrations) + len(queued_commands) + (1 if last_refused_calibration else 0)
                    _rel_header("Queued Retries And Future Commands", total_q)
                    if queued_calibrations:
                        for idx, qcmd in enumerate(queued_calibrations):
                            print(Fore.CYAN + f"  [cal {idx}] :calibrate {qcmd}" + Style.RESET_ALL)
                    if queued_commands:
                        for idx, qcmd in enumerate(queued_commands):
                            print(Fore.CYAN + f"  [cmd {idx}] {qcmd}" + Style.RESET_ALL)
                    if last_refused_calibration:
                        print(Fore.CYAN + f"  refused cache: :calibrate {last_refused_calibration}" + Style.RESET_ALL)
                    if total_q == 0:
                        print(Fore.CYAN + "  none" + Style.RESET_ALL)

                if "priority" in scopes:
                    printed_any = True
                    bits = []
                    if prioritize_pin.get("landscape"):
                        bits.append(f"landscape={prioritize_pin.get('landscape')}")
                    if prioritize_pin.get("mix"):
                        bits.append("mix=" + "+".join(prioritize_pin.get("mix") or []))
                    if prioritize_pin.get("probe"):
                        sign = float(prioritize_pin.get("sign", 1.0) or 1.0)
                        bits.append(f"pin={'+' if sign >= 0 else '-'}{prioritize_pin.get('probe')}")
                    bits.append(f"prioritize_alpha={round(float(tuner.get('prioritize_alpha', 0.0) or 0.0), 4)}")
                    _rel_header("Priority Target", 1 if any(b.split("=")[0] in {"landscape", "mix", "pin"} for b in bits) or float(tuner.get("prioritize_alpha", 0.0) or 0.0) else 0)
                    print(Fore.CYAN + "  " + ", ".join(bits) + Style.RESET_ALL)

                if "fields" in scopes:
                    printed_any = True
                    physics = steer_bodies()
                    links = []
                    for fname, cfg in sorted(physics["g_families"].items()):
                        links.append(
                            f"family:{fname} G<-{format_field_source(cfg.get('g'))} "
                            f"time<-{format_field_source(cfg.get('time')) if cfg.get('time') else '1'} "
                            f"shape={format_field_shape(cfg.get('shape'))}"
                        )
                    for node, cfg in sorted(physics["fields"].items()):
                        parts = []
                        if cfg.get("family"):
                            parts.append(f"inherits family:{cfg['family']}")
                        if cfg.get("g") is not None:
                            parts.append(f"G<-{format_field_source(cfg['g'])}")
                        if cfg.get("time") is not None:
                            parts.append(f"time<-{format_field_source(cfg['time'])}")
                        if cfg.get("shape") is not None:
                            parts.append(f"shape={format_field_shape(cfg['shape'])}")
                        if cfg.get("excluded"):
                            parts.append("HARD-EXCLUDED from whole field")
                        if parts:
                            links.append(f"{node} " + " ".join(parts))
                    for qname, cfg in sorted(physics["qualities"].items()):
                        links.append(
                            f"quality:{qname} formula={cfg.get('formula', QUALITY_FORMULA_DEFAULT)} "
                            f"strength<-{format_field_source(cfg.get('strength'))}"
                        )
                    for lname, law in sorted(physics["sets"].items()):
                        excluded = set(law.get("excluded") or [])
                        for node, member in sorted((law.get("members") or {}).items()):
                            links.append(
                                f"set:{lname} quality:{law.get('quality', '(unset)')} -> {node} "
                                + (
                                    "EXCLUDED (exact zero)"
                                    if node in excluded
                                    else f"compliance<-{format_field_source(member.get('compliance'))} "
                                         f"time={member.get('time_vector') or [1.0]}"
                                )
                            )
                        for node in sorted(excluded - set(law.get("members") or {})):
                            links.append(
                                f"set:{lname} quality:{law.get('quality', '(unset)')} -> {node} "
                                "EXCLUDED (exact zero; not otherwise a member)"
                            )
                    _rel_header("Gravity / Quality / Time / Shape Links", len(links))
                    if links:
                        for link in links:
                            print(Fore.CYAN + "  " + link + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "  none (all bodies use global point-field defaults)" + Style.RESET_ALL)

                if "exposure" in scopes:
                    printed_any = True
                    exposed_probe_names = sorted(n for n in probes if probes[n].get("exposed"))
                    total_exposed = len(exposed_commands) + len(exposed_probe_names) + len(exposed_knobs) + (1 if memory_tool_exposed or help_exposed else 0)
                    _rel_header("Model-Readable / Runtime Exposures", total_exposed)
                    if exposed_commands:
                        for word, record in sorted(exposed_commands.items()):
                            rec = normalize_command_exposure(record)
                            activation = command_exposure_custom_activation(rec) or "(no activation criterion)"
                            steer = command_exposure_raw_steer(rec) or "(command-defined)"
                            print(Fore.CYAN + f"  command {command_exposure_display(word, rec)} mode={rec['mode']} activation={activation} steer={steer}" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "  command tools: none" + Style.RESET_ALL)
                    print(Fore.CYAN + "  probe tools: " + (", ".join(exposed_probe_names) if exposed_probe_names else "none") + Style.RESET_ALL)
                    print(Fore.CYAN + "  knob tools: " + (", ".join(sorted(exposed_knobs)) if exposed_knobs else "none") + Style.RESET_ALL)
                    print(Fore.CYAN + f"  memory tool: {'on' if memory_tool_exposed else 'off'}; help tool: {'on' if help_exposed else 'off'}" + Style.RESET_ALL)

                if "release" in scopes:
                    printed_any = True
                    active_releases = {k: v for k, v in tool_sense.release_probs.items() if v > 0}
                    _rel_header("Release Decouplings", len(active_releases))
                    if active_releases:
                        for name, prob in sorted(active_releases.items()):
                            print(Fore.CYAN + f"  {name}: fire decision decoupled with probability {round(float(prob), 4)}" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "  none" + Style.RESET_ALL)

                if "macros" in scopes:
                    printed_any = True
                    _rel_header("Macro Aliases", len(macro_aliases))
                    if macro_aliases:
                        for name, path in sorted(macro_aliases.items())[:40]:
                            try:
                                rel = os.path.relpath(path, ROOT)
                            except Exception:
                                rel = str(path)
                            print(Fore.CYAN + f"  :{name} -> {rel}" + Style.RESET_ALL)
                        if len(macro_aliases) > 40:
                            print(Fore.CYAN + f"  ... {len(macro_aliases) - 40} more" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "  none" + Style.RESET_ALL)

                if not printed_any:
                    print(Fore.CYAN + "[Relations] Nothing requested." + Style.RESET_ALL)
                continue
            if user_input.startswith(":clean"):
                cargs = user_input[len(":clean"):].strip().split()
                apply_clean = any(a.lower() in {"apply", "yes", "run", "do"} for a in cargs)
                scopes = [
                    a.lower()
                    for a in cargs
                    if a.lower() not in {"apply", "yes", "run", "do", "preview", "status"}
                ]
                if not scopes:
                    scopes = ["all"]
                aliases = {
                    "cal": "queue",
                    "cals": "queue",
                    "calibration": "queue",
                    "calibrations": "queue",
                    "queues": "queue",
                    "dynamic": "bindings",
                    "dynamics": "bindings",
                    "binding": "bindings",
                    "matches": "bindings",
                    "match": "bindings",
                    "target": "priority",
                    "targets": "priority",
                    "prioritize": "priority",
                    "steer": "priority",
                    "steering": "priority",
                }
                scope_set = {aliases.get(s, s) for s in scopes}
                if "all" in scope_set:
                    scope_set = {"queue", "bindings", "priority"}
                bad_scopes = sorted(scope_set - {"queue", "bindings", "priority"})
                if bad_scopes:
                    print(Fore.YELLOW + "[Clean] Unknown scope(s): " + ", ".join(bad_scopes) + ". Use :clean [queue|bindings|priority|all] [apply]." + Style.RESET_ALL)
                    continue

                queued_cal_count = len(queued_calibrations)
                queued_cmd_count = len(queued_commands)
                refused_count = 1 if last_refused_calibration else 0
                binding_names = sorted(tuner_bindings)
                active_match_names = sorted(
                    pn for pn, m in probe_matches.items()
                    if str(m.get("mode", "none")).lower() in {"drive", "validate"}
                )
                priority_active = bool(
                    prioritize_pin.get("probe")
                    or prioritize_pin.get("mix")
                    or prioritize_pin.get("landscape")
                    or float(tuner.get("prioritize_alpha", 0.0) or 0.0) != 0.0
                )

                preview = []
                if "queue" in scope_set:
                    preview.append(f"queue: {queued_cal_count} calibration retrie(s), {queued_cmd_count} future command(s), {refused_count} refused calibration cache")
                if "bindings" in scope_set:
                    preview.append(
                        "bindings: "
                        + f"{len(binding_names)} dynamic tuner binding(s)"
                        + (f" [{', '.join(binding_names[:8])}{' ...' if len(binding_names) > 8 else ''}]" if binding_names else "")
                        + f"; {len(active_match_names)} active probe-match mode(s)"
                        + (f" [{', '.join(active_match_names[:8])}{' ...' if len(active_match_names) > 8 else ''}]" if active_match_names else "")
                    )
                if "priority" in scope_set:
                    target_bits = []
                    if prioritize_pin.get("landscape"):
                        target_bits.append(f"landscape={prioritize_pin.get('landscape')}")
                    if prioritize_pin.get("mix"):
                        target_bits.append("mix=" + "+".join(prioritize_pin.get("mix") or []))
                    if prioritize_pin.get("probe"):
                        target_bits.append(f"pin={prioritize_pin.get('probe')}")
                    target_bits.append(f"prioritize_alpha={round(float(tuner.get('prioritize_alpha', 0.0) or 0.0), 4)}")
                    preview.append("priority: " + ", ".join(target_bits))

                if not apply_clean:
                    print(Fore.CYAN + "[Clean] Preview only. Add 'apply' to perform cleanup." + Style.RESET_ALL)
                    for line in preview:
                        print(Fore.CYAN + "  " + line + Style.RESET_ALL)
                    print(Fore.CYAN + "  Non-destructive: does not delete probes, histories, labels, docs, memories, trigger evidence, or files." + Style.RESET_ALL)
                    continue

                changed = []
                if "queue" in scope_set:
                    if queued_calibrations or queued_commands or last_refused_calibration:
                        queued_calibrations.clear()
                        queued_commands.clear()
                        last_refused_calibration = None
                        changed.append("cleared queued calibration/command retries")
                if "bindings" in scope_set:
                    if tuner_bindings:
                        tuner_bindings.clear()
                        changed.append("cleared dynamic tuner bindings")
                    match_changed = 0
                    for pn in active_match_names:
                        m = probe_matches.get(pn)
                        if not m:
                            continue
                        if str(m.get("mode", "none")).lower() != "none":
                            m["mode"] = "none"
                            m.pop("mult", None)
                            match_hist.pop(pn, None)
                            match_changed += 1
                    if match_changed:
                        _save_probe_matches()
                        changed.append(f"set {match_changed} probe-match mode(s) to none")
                if "priority" in scope_set:
                    if priority_active:
                        prioritize_pin["probe"] = None
                        prioritize_pin["mix"] = None
                        prioritize_pin["landscape"] = None
                        tuner.set("prioritize_alpha", 0.0)
                        changed.append("cleared priority target and prioritize_alpha")

                if changed:
                    print(Fore.GREEN + "[Clean] " + "; ".join(changed) + "." + Style.RESET_ALL)
                    print(Fore.CYAN + "[Clean] Evidence and artifacts preserved." + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + "[Clean] Nothing active in selected scope(s)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":tune"):
                if " dynamic " in user_input.lower():
                    # :tune <target> dynamic <signed mix> [mult]
                    #   target : a knob, OR a probe (drives probe_<name>'s threshold,
                    #            never a bare shadow knob).
                    #   mix    : a single probe/stream, or a signed mix parsed like
                    #            :probe compose (e.g. +ambiguity-consensus). Each turn
                    #            the target = mult * Sigma(weight * stream) over the mix.
                    parts = user_input[len(":tune"):].strip().split()
                    if parts and parts[0].upper() in ("CHOICE", "CHOOSE", "AUTO"):
                        resolved = resolve_probe_choice(parts[0], calibratable_names(tuner), model=model, config=config, action_name="tune")
                        if not resolved:
                            continue
                        parts[0] = resolved
                    try:
                        dyn_idx = [p.lower() for p in parts].index("dynamic")
                    except ValueError:
                        dyn_idx = -1
                    if dyn_idx > 0 and dyn_idx < len(parts) - 1:
                        target = parts[0]
                        rest = parts[dyn_idx + 1:]
                        # Peel a trailing standalone multiplier (number|auto|pNN) off
                        # the end; whatever remains is the mix expression.
                        mult_str = None
                        if len(rest) >= 2 and re.match(r"^(auto|p\d+(?:\.\d+)?|-?\d+(?:\.\d+)?)$", rest[-1].lower()):
                            mult_str = rest[-1].lower()
                            rest = rest[:-1]
                        terms, perr = parse_compose_expr(" ".join(rest))
                        if perr:
                            print(Fore.RED + f"[Tune] Could not parse dynamic expression near '{perr}'." + Style.RESET_ALL)
                            continue
                        # Resolve by CHECKING, not forcing (mirrors the static :tune
                        # path): an exact-name trigger -- a real knob -- wins; a name
                        # that is ONLY a probe drives probe_<name>'s threshold. When a
                        # name is both (e.g. memory_alpha knob vs probe_memory_alpha),
                        # the knob wins and the overlap is surfaced; name 'probe_<x>'
                        # outright to drive the probe threshold instead.
                        if target in tuner.triggers:
                            bind_key = target
                            if f"probe_{target}" in tuner.triggers:
                                print(Fore.CYAN + f"[Tune] '{target}' names both a knob and a probe -- steering the KNOB (use 'probe_{target}' for the probe threshold)." + Style.RESET_ALL)
                        elif f"probe_{target}" in tuner.triggers:
                            bind_key = f"probe_{target}"
                        else:
                            bind_key = target
                        single = terms[0][1] if len(terms) == 1 else None
                        if mult_str is None:
                            mult_str = "auto" if single else "1.0"
                        mult = 1.0
                        if mult_str == "auto" or mult_str.startswith("p"):
                            if single:
                                pct = 80.0
                                if mult_str.startswith("p"):
                                    try:
                                        pct = float(mult_str[1:])
                                    except ValueError:
                                        pass
                                pct = min(max(pct, 0.0), 100.0)
                                ptrig = tuner.triggers.get(f"probe_{single}")
                                if ptrig and ptrig.signals:
                                    data = sorted(ptrig.signals)
                                    p_val = data[int(round((pct / 100.0) * (len(data) - 1)))]
                                    if p_val > 0:
                                        cur = tuner.get(bind_key, 0.0)
                                        mult = cur / p_val
                                        print(Fore.CYAN + f"[Tune] Auto-multiplier {mult:.4f} (maps {single} P{pct:g} {p_val:.3f} to current {bind_key} {cur:.3f})." + Style.RESET_ALL)
                                    else:
                                        print(Fore.YELLOW + f"[Tune] Auto failed ({single} P{pct:g} <= 0). Give a numeric multiplier." + Style.RESET_ALL)
                                        continue
                                else:
                                    print(Fore.YELLOW + f"[Tune] Auto failed (no history for '{single}'). Give a numeric multiplier." + Style.RESET_ALL)
                                    continue
                            else:
                                print(Fore.YELLOW + "[Tune] Auto-multiplier needs a single probe's history; using 1.0 for the mix." + Style.RESET_ALL)
                        else:
                            try:
                                mult = float(mult_str)
                            except ValueError:
                                print(Fore.RED + "[Tune] Multiplier must be a number, 'auto', or 'pNN'." + Style.RESET_ALL)
                                continue
                        tuner_bindings[bind_key] = (terms, mult)
                        label = " ".join(f"{'-' if w < 0 else '+'}{abs(w):g}*{n}" for w, n in terms)
                        tgt_note = f"probe '{target}' threshold" if bind_key.startswith("probe_") else f"knob '{bind_key}'"
                        print(Fore.GREEN + f"[Tune] Dynamic binding: {tgt_note} = {mult:.4f} * ({label}) every turn." + Style.RESET_ALL)
                        continue

                tune_ns = TUNE_SPEC.parse(user_input[len(":tune"):])
                targs = ([tune_ns.knob] + (tune_ns.setting.split() if tune_ns.setting else [])) if tune_ns.knob else []
                if targs and targs[0].upper() in ("CHOICE", "CHOOSE", "AUTO"):
                    resolved = resolve_probe_choice(targs[0], calibratable_names(tuner), model=model, config=config, action_name="tune")
                    if not resolved:
                        continue
                    targs[0] = resolved
                if not targs:
                    rows = tuner.summary()
                    if not rows:
                        print(Fore.YELLOW + "[Tune] No triggers registered yet." + Style.RESET_ALL)
                    for s in rows:
                        line = (
                            f"  {s['name']}: {s['kind']} value={s['value']} [{s['comparator']}] "
                            f"fire_rate={s['fire_rate']} "
                            f"signal(min/med/max)={s['signal_min']}/{s['signal_med']}/{s['signal_max']} "
                            f"n={s['n_signals']}"
                        )
                        if s.get("n_credited"):
                            # lift>0 means firing beat not-firing on the outcome:
                            # the honest "is interaction productive" readout.
                            line += (
                                f" | outcome fired={s['fired_outcome']} unfired={s['unfired_outcome']} "
                                f"lift={s['lift']} (n={s['n_credited']})"
                            )
                        print(Fore.CYAN + line + Style.RESET_ALL)
                elif len(targs) >= 2 and targs[1].lower() == "auto":
                    if targs[0] == "conversation_productive" and len(targs) >= 3 and targs[2].lower() == "intent":
                        # Bar set RELATIVE TO INTENT-SHAPING: the sense cut that
                        # separates turns which settled intent from turns which
                        # did not (midpoint of the two group medians).
                        trig = tuner.triggers.get("intent_settling")
                        pairs = list(trig.outcomes) if trig is not None else []
                        v = intent_relative_threshold(pairs)
                        if v is None:
                            n_set = sum(1 for sig, _ in pairs if sig > 0)
                            print(
                                Fore.YELLOW
                                + f"[Tune] conversation_productive unchanged: need >=5 intent-settling and >=5 "
                                + f"non-settling turns with sense (have {n_set} and {len(pairs) - n_set}). Talk more."
                                + Style.RESET_ALL
                            )
                        else:
                            tuner.set("conversation_productive", v)
                            print(
                                Fore.GREEN
                                + f"[Tune] conversation_productive = {round(v, 4)} -- the sense cut between "
                                + f"intent-settling and non-settling turns ({len(pairs)} paired turns)."
                                + Style.RESET_ALL
                            )
                        continue
                    if targs[0] == "steer_cap_fraction":
                        # Data-informed, deterministic: exact percentile of the
                        # attempted push/residual ratios _cap_steer observed.
                        from invariants import engine as _engine
                        pct = float(targs[2]) if len(targs) >= 3 else None
                        v = _engine.calibrate_steer_cap_fraction(percentile=pct)
                        if v is None:
                            stats = _engine.steer_telemetry_stats()
                            print(
                                Fore.YELLOW
                                + f"[Tune] steer_cap_fraction unchanged: only {stats['n']} observed pushes "
                                + f"(need {_engine.STEER_CAP_MIN_N}). Generate first, calibrate after."
                                + Style.RESET_ALL
                            )
                        else:
                            tuner.set("steer_cap_fraction", v)
                            used = _engine.STEER_CAP_PERCENTILE if pct is None else pct
                            stats = _engine.steer_telemetry_stats()
                            print(
                                Fore.GREEN
                                + f"[Tune] steer_cap_fraction = {round(v, 4)} from data "
                                + f"(p{used:g} of {stats['n']} attempted ratios, clip_rate was {stats['clip_rate']})"
                                + Style.RESET_ALL
                            )
                        continue
                    if targs[0] == "steer_band":
                        # Data-informed, deterministic: acceptance-aware per-layer
                        # outcomes from the steer map, not an asserted window.
                        # Evidence lanes: gold (scored benchmarks), conversation
                        # (live productivity reads), or any (both, composition shown).
                        from invariants import engine as _engine
                        min_events, basis, band_evidence = 8, "any", "any"
                        for extra in targs[2:]:
                            token = extra.lower()
                            if token in {"gold", "conversation", "any"}:
                                basis = token
                            elif token in {"synthesis", "layersteer", "layer_steer"}:
                                band_evidence = "layer_steer" if token != "synthesis" else "synthesis"
                            else:
                                try:
                                    min_events = int(float(extra))
                                except ValueError:
                                    pass
                        n_layers = getattr(model, "n_layers", None) or len(model.model.model.layers)
                        suggestion = steer_map.suggest_band(
                            n_layers, min_events=min_events, basis=basis, evidence=band_evidence
                        )
                        if suggestion is None:
                            print(
                                Fore.YELLOW
                                + f"[Tune] steer_band unchanged: no layer clears the evidence bar on the "
                                + f"'{basis}' lane / '{band_evidence}' evidence (min {min_events} labeled, >= overall success). "
                                + "Talk more (conversation lane), import labeled runs (gold lane), or "
                                + "run the layer sweep (:tune steer_layer_sweep 1) for transfer-free evidence."
                                + Style.RESET_ALL
                            )
                        else:
                            _engine.set_steer_band(suggestion["lo"], suggestion["hi"])
                            tuner.set("steer_band_lo", suggestion["lo"])
                            tuner.set("steer_band_hi", suggestion["hi"])
                            print(
                                Fore.GREEN
                                + f"[Tune] steer_band = ({suggestion['lo']:.2f}, {suggestion['hi']:.2f}) from data "
                                + f"(layers {suggestion['eligible_layers']}, "
                                + f"{suggestion['labeled_events']} labeled events {suggestion['labeled_by_basis']} "
                                + f"evidence={suggestion['labeled_by_evidence']}, "
                                + f"overall success {suggestion['overall_success_rate']:.2f})"
                                + Style.RESET_ALL
                            )
                        continue
                    if targs[0] in ("steer_band_lo", "steer_band_hi"):
                        print(
                            Fore.YELLOW
                            + "[Tune] Calibrate the band from outcomes with ':tune steer_band auto' "
                            + "(lo/hi move together), or set a value directly."
                            + Style.RESET_ALL
                        )
                        continue
                    route, reason = calibration_policy(targs[0])
                    if route == "reject":
                        print(Fore.YELLOW + f"[Tune] calibration rejected for '{targs[0]}': {reason}." + Style.RESET_ALL)
                        continue
                    pct = float(targs[2]) if len(targs) >= 3 else 80.0
                    v = tuner.calibrate(targs[0], pct, tuner.get("calibration_gain", 0.5))
                    if v is None:
                        print(Fore.RED + f"[Tune] Unknown trigger '{targs[0]}'.{did_you_mean(targs[0], tuner.triggers)}" + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + f"[Tune] {targs[0]} calibrated toward p{pct:g} = {round(v, 4)}" + Style.RESET_ALL)
                elif len(targs) >= 2:
                    if targs[0].lower() == "off":
                        targs[0], targs[1] = targs[1], "0.0"
                    elif targs[1].lower() == "off":
                        targs[1] = "0.0"
                        
                    if targs[0].startswith("probe_") or (targs[0] in probes and targs[0] not in tuner.triggers):
                        pname = targs[0].replace("probe_", "")
                        if targs[1] == "0.0":
                            if pname in probes:
                                probes[pname]["chatty"] = False
                                print(Fore.CYAN + f"[Tune] Redirected: muted console output for probe '{pname}' (use ':probe drop {pname}' to remove it entirely)." + Style.RESET_ALL)
                            else:
                                print(Fore.YELLOW + f"[Tune] No active probe named '{pname}'." + Style.RESET_ALL)
                        else:
                            if targs[1].replace("probe_", "") in probes:
                                print(Fore.YELLOW + f"[Tune] Did you mean to dynamically tie the threshold? Use: :tune {targs[0]} dynamic {targs[1]}" + Style.RESET_ALL)
                            else:
                                print(Fore.YELLOW + f"[Tune] '{targs[0]}' is a probe. To mute it use ':tune {targs[0]} off', or to drop it use ':probe drop {pname}'." + Style.RESET_ALL)
                        continue

                    # Setting a brand-new name that collides with a probe would
                    # create a junk knob shadowing it -- refuse and redirect.
                    if targs[0] not in tuner.triggers and f"probe_{targs[0]}" in tuner.triggers:
                        print(Fore.YELLOW + f"[Tune] '{targs[0]}' is a probe, not a knob -- :tune would create a shadow. Use :label {targs[0]} pos|neg or :calibrate {targs[0]} <anchor>." + Style.RESET_ALL)
                    else:
                        if targs[1].upper() in ("CHOICE", "CHOOSE"):
                            print(Fore.CYAN + f"[Choice] Asking the model to select a value for '{targs[0]}'..." + Style.RESET_ALL)
                            _choice_field_context = format_field_prompt_context(tuner, probes)
                            prompt = (
                                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                                f"You are configuring the cognitive tuning parameter '{targs[0]}'.\n"
                                + (f"{_choice_field_context}\n" if _choice_field_context else "")
                                + f"Select an appropriate numerical value (float). Output ONLY the number and nothing else.<|eot_id|>"
                                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                            )
                            # NOTE: no local `import generate_agentic_text` here --
                            # it is imported at module level. A local import would
                            # make the name function-local across all of run() and
                            # UnboundLocalError every other generate call.
                            sug = generate_agentic_text(model, instruction=prompt, config=config, pre_formatted=True, max_new_tokens=10)
                            try:
                                targs[1] = str(float(sug.strip()))
                                print(Fore.GREEN + f"[Choice] The model chose value: {targs[1]}" + Style.RESET_ALL)
                            except ValueError:
                                print(Fore.YELLOW + f"[Choice] Model output an invalid number '{sug.strip()}'. Aborting." + Style.RESET_ALL)
                                continue

                        try:
                            v = tuner.set(targs[0], float(targs[1]))
                            print(Fore.GREEN + f"[Tune] {targs[0]} = {round(v, 4)}" + Style.RESET_ALL)
                        except ValueError:
                            print(Fore.RED + "[Tune] Value must be a number." + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + "[Tune] Usage: :tune | :tune <name> <value> | :tune <name> auto [percentile]" + Style.RESET_ALL)
                continue
            if not user_input.strip():
                continue
            # A :prefixed line that reached here matched no command handler.
            # Don't burn a generation deflecting into a generic essay: if it's a
            # typo, suggest the real command; otherwise offer to INVENT it as a
            # tool -- mint a probe for the name (framings stay operator-authored).
            if user_input.lstrip().startswith(":") and not reading_turn_source:
                cmd_word = user_input.split()[0].lower()
                guess = did_you_mean(cmd_word, KNOWN_COMMANDS)
                if guess:
                    print(Fore.YELLOW + f"[Command] '{cmd_word}' isn't a command.{guess}" + Style.RESET_ALL)
                else:
                    stem = re.sub(r"[^a-z0-9_]", "_", cmd_word.lstrip(":"))[:40].strip("_") or "it"
                    print(
                        Fore.CYAN
                        + f"[Command] '{cmd_word}' isn't a command. To invent it as a tool, mint a probe for "
                        + f"'{stem}': :probe {stem} <with it> || <without it>  (or bare :probe {stem} to draft framings)."
                        + Style.RESET_ALL
                    )
                continue

            autocomplete_turn = bool(
                _pending_expect
                and _pending_expect.get("type") == "autocomplete"
            )
            utility_expect_profile = expect_generation_profile(_pending_expect)
            utility_expect_turn = bool(utility_expect_profile)
            # Autocomplete is an isolated utility generation. A staged document,
            # tool result, or conversational history belongs to the next real
            # turn; consuming it here both loses that context and lets an older
            # interpretation outrank the live command reference in this prompt.
            memory_tool_result = None if autocomplete_turn else pending_memory_tool_result
            orientation_tool_result = None if autocomplete_turn else pending_orientation_tool_result
            claimmap_tool_result = None if autocomplete_turn else pending_claimmap_tool_result
            claimmap_steer_delta = None if autocomplete_turn else pending_claimmap_steer_delta
            methodmap_tool_result = None if autocomplete_turn else pending_methodmap_tool_result
            document_tool_result = None if autocomplete_turn else pending_document_tool_result
            sandbox_tool_result = None if autocomplete_turn else pending_sandbox_tool_result
            game_tool_result = None if autocomplete_turn else pending_game_tool_result
            if active_game and not game_tool_result and not autocomplete_turn:
                sys_prompt = active_game_state.get("system_prompt", "The operator expects you to play it with them. To end the game, output <<GAME_END>>.")
                game_tool_result = f"[Game Active: {active_game}] A game is currently active. {sys_prompt}"
            help_tool_result = None if autocomplete_turn else pending_help_tool_result
            if help_exposed and not help_tool_result and not autocomplete_turn:
                help_tool_result = "[Help available] You may emit <<HELP>> at any time to see what you can run yourself vs. what only the operator can run."
            # Consequences of the model's LAST words arrive as THIS turn's
            # context; the same-turn tag requests below add to the list.
            turn_impacts = [] if autocomplete_turn else impact_state["pending"]
            if not autocomplete_turn:
                impact_state["pending"] = []
            sweep_state["fires"] = []  # single-layer steers applied this turn
            turn_row = {}              # this turn's stream values (paired calibration substrate)
            if not autocomplete_turn:
                pending_memory_tool_result = None
                pending_orientation_tool_result = None
                pending_claimmap_tool_result = None
                pending_claimmap_steer_delta = None
                pending_methodmap_tool_result = None
                pending_document_tool_result = None
                pending_sandbox_tool_result = None
                pending_game_tool_result = None
                pending_help_tool_result = None
            prompt = _build_live_prompt(
                user_input,
                memory_tool_result=memory_tool_result,
                orientation_tool_result=orientation_tool_result,
                claimmap_tool_result=claimmap_tool_result,
                methodmap_tool_result=methodmap_tool_result,
                sandbox_tool_result=sandbox_tool_result,
                document_tool_result=document_tool_result,
                game_tool_result=game_tool_result,
                help_tool_result=help_tool_result,
                session_context=(
                    session_context
                    if session_context_enabled and not autocomplete_turn
                    else None
                ),
                active_u_name=_last_replace_name if ('_last_replace_name' in locals() and _last_replace_name) else "operator",
            )
            # Honest attribution in the permanent record: a reading turn is the
            # model's own act (it occupies the user slot in the chat template,
            # but the operator never typed it).
            if not autocomplete_turn:
                memory.append_turn(
                    "user",
                    user_input,
                    tags=[
                        "reading_turn" if reading_turn_source else "operator_input",
                        _last_replace_name if ('_last_replace_name' in locals() and _last_replace_name) else "operator"
                    ],
                    provenance={
                        "spoken_by": "model_reading" if reading_turn_source else "operator",
                        "document_source": reading_turn_source,
                        "memory_tool_result_provided": bool(memory_tool_result),
                        "orientation_tool_result_provided": bool(orientation_tool_result),
                        "claimmap_tool_result_provided": bool(claimmap_tool_result),
                        "methodmap_tool_result_provided": bool(methodmap_tool_result),
                    },
                )
            if memory_tool_result:
                memory.append_event(
                    "memory_tool_result_provided",
                    text=memory_tool_result,
                    tags=["memory_tool"],
                    provenance={"current_input": user_input[:240]},
                )
            if orientation_tool_result:
                memory.append_event(
                    "orientation_tool_result_provided",
                    text=orientation_tool_result,
                    tags=["orientation_tool"],
                    provenance={"current_input": user_input[:240]},
                )
            if claimmap_tool_result:
                memory.append_event(
                    "claimmap_tool_result_provided",
                    text=claimmap_tool_result,
                    tags=["claimmap_tool", "activation_measurement"],
                    provenance={"current_input": user_input[:240]},
                )
            if methodmap_tool_result:
                memory.append_event(
                    "methodmap_tool_result_provided",
                    text=methodmap_tool_result,
                    tags=["methodmap_tool"],
                    provenance={"current_input": user_input[:240]},
                )
            if document_tool_result:
                memory.append_event(
                    "document_tool_result_provided",
                    text=document_tool_result[:400],
                    tags=["document"],
                    provenance={
                        "current_input": user_input[:240],
                        "sha256": (doc_session or {}).get("sha256"),
                        "chunk_index": (doc_session or {}).get("cursor"),
                    },
                )
            if sandbox_tool_result:
                memory.append_event(
                    "sandbox_tool_result_provided",
                    text=sandbox_tool_result[:400],
                    tags=["sandbox_tool"],
                    provenance={"current_input": user_input[:240]},
                )

            print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
            synthesis_records = []

            # Evaluate dynamic tuner bindings before generation
            if tuner_bindings and turn_log and not utility_expect_turn:
                last_turn = turn_log[-1]
                for bind_key, (terms, mult) in list(tuner_bindings.items()):
                    vals = []
                    for w, name in terms:
                        # Resolve each mix term to last turn's stream: a probe
                        # (probe_<name>), a bare named stream, or a phenomenality
                        # dimension (phen_<name>).
                        s = last_turn.get(f"probe_{name}")
                        if s is None:
                            s = last_turn.get(name)
                        if s is None:
                            s = last_turn.get(f"phen_{name}")
                        if s is None:
                            vals = None  # a term has no reading yet -> skip this turn
                            break
                        vals.append(w * float(s))
                    if vals is not None:
                        mix_sig = sum(vals)
                        new_val = mix_sig * mult
                        tuner.set(bind_key, new_val)
                        label = " ".join(f"{'-' if w < 0 else '+'}{abs(w):g}*{n}" for w, n in terms)
                        print(Fore.CYAN + f"[Tune] Dynamic binding: {bind_key} set to {new_val:.4f} ({mult:g} * [{label}] = {mix_sig:+.3f})." + Style.RESET_ALL, flush=True)

            # Live steering surface: cap/band/fraction move with the tuner each
            # turn, so a :tune takes effect on the very next generation.
            _sync_steer_tunables(tuner, config)
            # Proven-expert routing: push the weight and a fresh per-expert
            # success snapshot into config so the ToT selection favors experts
            # that have earned it. Only recompute (an O(events) aggregate) when
            # the feature is actually on.
            try:
                config.expert_proof_weight = max(0.0, float(tuner.get("expert_proof_weight", 0.0) or 0.0))
            except (TypeError, ValueError):
                config.expert_proof_weight = 0.0
            config.expert_proof_scores = (
                compute_expert_proof_scores(steer_map) if config.expert_proof_weight > 0.0 else {}
            )
            _refresh_tot_committee_from_live_state()
            claimmap_alpha_used = (
                0.0 if utility_expect_turn
                else (tuner.get("claimmap_alpha", 0.0) if claimmap_steer_delta else 0.0)
            )
            turn_sweep_layers = None
            if (not utility_expect_turn) and claimmap_steer_delta and claimmap_alpha_used > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
                sweep_layers, sweep_drifts, sweep_lawfulness = _sweep_layers_for("claimmap", claimmap_steer_delta)
                if sweep_layers:
                    turn_sweep_layers = sweep_layers
                    for _sl in sweep_layers:
                        sweep_state["fires"].append(("claimmap", _sl, claimmap_alpha_used, sweep_drifts.get(_sl), len(sweep_layers), sweep_lawfulness))
                    print(
                        Fore.MAGENTA
                        + "[Steer] layer sweep: this turn's claimmap steer pushes "
                        + "+".join(f"L{_sl}" for _sl in sweep_layers)
                        + f" (width {len(sweep_layers)})"
                        + Style.RESET_ALL
                    )
            steer_handles = (
                claimmap_steer_handles(model, claimmap_steer_delta, alpha=claimmap_alpha_used, layers=turn_sweep_layers)
                if claimmap_steer_delta and not utility_expect_turn else []
            )
            if not steer_handles:
                claimmap_alpha_used = 0.0  # nothing actually applied this turn
            # Prioritize steer, bounded by the same envelope. Target precedence:
            # pinned MIX (lift-weighted combination) > pinned single probe
            # (toward, or away if sign -1) > AUTO (top of the ranking, signed by
            # lift). OFF at alpha 0.
            prio_alpha = tuner.get("prioritize_alpha", 0.0)
            prio_steered = None
            if tuner.get("prioritize_gravity", 0.0) > 0 and not utility_expect_turn:
                try:
                    _g_handles, _g_desc = build_gravity_field_handles(model, probes, tuner)
                except Exception:
                    _g_handles, _g_desc = [], "field build failed"
                if _g_handles:
                    steer_handles.extend(_g_handles)
                    print(
                        Fore.MAGENTA
                        + f"[Prioritize] gravity field: {_g_desc}."
                        + Style.RESET_ALL
                    )
                else:
                    print(Fore.YELLOW + f"[Prioritize] gravity on, but {_g_desc}." + Style.RESET_ALL)
            elif prio_alpha and prio_alpha > 0 and probes and not utility_expect_turn:
                from invariants.engine import _steer_handles as _p_steer
                _pdir, _label, _psign = resolve_priority_direction(model, probes, tuner, prioritize_pin)
                if _pdir:
                    try:
                        steer_handles.extend(_p_steer(model, _pdir, list(_pdir.keys()), prio_alpha * _psign))
                        prio_steered = (_label, _psign)
                    except Exception:
                        prio_steered = None
            if prio_steered is not None:
                _dir_word = priority_direction_word(prio_steered[0], prio_steered[1])
                print(
                    Fore.MAGENTA
                    + f"[Prioritize] steering {_dir_word} {prio_steered[0]} (alpha {round(prio_alpha,4)})."
                    + Style.RESET_ALL
                )
            # Exposed-probe steer: same envelope/mechanism as prioritize, but the
            # target is the SET of probes the operator exposed to the model --
            # lift-weighted, so evidence sets each one's degree and sign. Off at 0,
            # idle until those probes have credited lift.
            exposed_alpha = tuner.get("exposed_probe_alpha", 0.0)
            if exposed_alpha and exposed_alpha > 0 and probes and not utility_expect_turn:
                exposed_names = sorted(
                    n for n in probes
                    if probes[n].get("exposed") and not probe_is_released(probes[n])
                )
                if exposed_names:
                    from invariants.engine import _steer_handles as _e_steer
                    _edir = build_priority_mix_direction(model, exposed_names, probes, tuner)
                    if _edir:
                        try:
                            steer_handles.extend(_e_steer(model, _edir, list(_edir.keys()), exposed_alpha))
                            print(
                                Fore.MAGENTA
                                + f"[Exposed] steering along {len(exposed_names)} exposed probe(s) "
                                + f"({', '.join(exposed_names)}) at alpha {round(exposed_alpha, 4)}."
                                + Style.RESET_ALL
                            )
                        except Exception:
                            pass
                    else:
                        print(
                            Fore.YELLOW
                            + "[Exposed] exposed_probe_alpha is set but no exposed probe has credited "
                            + "lift yet -- steering idle until they accrue outcomes."
                            + Style.RESET_ALL
                        )
            # Clock start: wall-time for THIS turn's generation (all
            # regenerations included), snapshotted before probe-scoring so the
            # reading reflects generation, not instrumentation. The peak-memory
            # window is reset here so it captures this turn's spike (VRAM on
            # CUDA, process RSS on CPU-only).
            import time as _time
            _gen_t0 = _time.perf_counter()
            _reset_memory_peak()
            _cache_enabled_before = config.cache_enabled
            _cache_write_before = config.cache_write_enabled
            _generation_config_restore = {}
            if utility_expect_profile:
                _generation_config_restore = apply_config_overrides(
                    config,
                    utility_expect_profile.get("overrides", {}),
                )
                print(
                    Fore.CYAN
                    + f"[Expect] Low-memory {utility_expect_profile['label']} profile for this turn "
                    + "(ToT routing/synthesis/cache/tool seams paused)."
                    + Style.RESET_ALL
                )
            elif autocomplete_turn:
                # Cached activation deltas encode earlier tasks and document
                # interpretations. They are useful for conversation, but wrong
                # authority for a one-line parser completion.
                config.cache_enabled = False
                config.cache_write_enabled = False
            try:
                response, telemetry = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=(
                        128
                        if autocomplete_turn
                        else max(64, int(tuner.get("response_tokens", 512)))
                    ),
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,  # Enables visible trace logging.
                    pre_formatted=True,
                    mid_chunk_hook=(
                        None if utility_expect_turn else tool_sense
                    ),  # tools fire mid-thought, between conversational chunks
                    return_telemetry=True,
                )
            finally:
                if _generation_config_restore:
                    restore_config_overrides(config, _generation_config_restore)
                else:
                    config.cache_enabled = _cache_enabled_before
                    config.cache_write_enabled = _cache_write_before
                for h in steer_handles:
                    h.remove()

            def _generate_tool_followup(reply_prompt, *, extra_steer_handles=None, phase="tool follow-up"):
                """Generate the answer after an explicit tool read with the same
                exposed output steers as the first pass. Explicit tool steering
                such as a claimmap tag can pass handles in as extra handles."""
                _sync_steer_tunables(tuner, config)
                _refresh_tot_committee_from_live_state()
                followup_handles = list(extra_steer_handles or [])
                prio_alpha_now = tuner.get("prioritize_alpha", 0.0)
                prio_steered_now = None
                if tuner.get("prioritize_gravity", 0.0) > 0:
                    try:
                        _g_handles, _g_desc = build_gravity_field_handles(model, probes, tuner)
                    except Exception:
                        _g_handles, _g_desc = [], "field build failed"
                    if _g_handles:
                        followup_handles.extend(_g_handles)
                        print(
                            Fore.MAGENTA
                            + f"[Prioritize] gravity field on {phase}: {_g_desc}."
                            + Style.RESET_ALL
                        )
                elif prio_alpha_now and prio_alpha_now > 0 and probes:
                    from invariants.engine import _steer_handles as _p_steer
                    _pdir, _label, _psign = resolve_priority_direction(model, probes, tuner, prioritize_pin)
                    if _pdir:
                        try:
                            followup_handles.extend(_p_steer(model, _pdir, list(_pdir.keys()), prio_alpha_now * _psign))
                            prio_steered_now = (_label, _psign)
                        except Exception:
                            prio_steered_now = None
                if prio_steered_now is not None:
                    _dir_word = priority_direction_word(prio_steered_now[0], prio_steered_now[1])
                    print(
                        Fore.MAGENTA
                        + f"[Prioritize] steering {phase} {_dir_word} {prio_steered_now[0]} "
                        + f"(alpha {round(prio_alpha_now,4)})."
                        + Style.RESET_ALL
                    )

                exposed_alpha_now = tuner.get("exposed_probe_alpha", 0.0)
                if exposed_alpha_now and exposed_alpha_now > 0 and probes:
                    exposed_names_now = sorted(
                        n for n in probes
                        if probes[n].get("exposed") and not probe_is_released(probes[n])
                    )
                    if exposed_names_now:
                        from invariants.engine import _steer_handles as _e_steer
                        _edir = build_priority_mix_direction(model, exposed_names_now, probes, tuner)
                        if _edir:
                            try:
                                followup_handles.extend(_e_steer(model, _edir, list(_edir.keys()), exposed_alpha_now))
                                print(
                                    Fore.MAGENTA
                                    + f"[Exposed] steering {phase} along {len(exposed_names_now)} exposed probe(s) "
                                    + f"({', '.join(exposed_names_now)}) at alpha {round(exposed_alpha_now, 4)}."
                                    + Style.RESET_ALL
                                )
                            except Exception:
                                pass
                        else:
                            print(
                                Fore.YELLOW
                                + "[Exposed] exposed_probe_alpha is set but no exposed probe has credited "
                                + "lift yet -- steering idle until they accrue outcomes."
                                + Style.RESET_ALL
                            )
                try:
                    return generate_agentic_text(
                        model,
                        instruction=reply_prompt,
                        config=config,
                        max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                        synthesis_recorder=synthesis_records,
                        chatty_log=True,
                        pre_formatted=True,
                        return_telemetry=True,
                    )
                finally:
                    for h in followup_handles:
                        h.remove()

            model_memory_query = None if utility_expect_turn else extract_memory_query(response)
            model_claimmap_payload = None if utility_expect_turn else extract_claimmap_payload(response)
            model_methodmap_query = None if utility_expect_turn else extract_methodmap_query(response)
            model_doc_query = None if utility_expect_turn else extract_doc_query(response)
            model_probe_query = None if utility_expect_turn else extract_probe_query(response)
            model_cmd_requests = [] if utility_expect_turn else extract_cmd_requests(response)
            model_game_propose = None if utility_expect_turn else extract_game_propose(response)
            model_game_accept = None if utility_expect_turn else extract_game_accept(response)
            model_game_decline = None if utility_expect_turn else extract_game_decline(response)
            model_game_end = None if utility_expect_turn else extract_game_end(response)
            if utility_expect_turn:
                model_game_exposed, model_game_hidden = [], []
            else:
                model_game_exposed, model_game_hidden = extract_game_expose_hide(response)
            if (not utility_expect_turn) and extract_help_request(response):
                pending_help_tool_result = build_model_help_text(
                    list_solve_macros(),
                    exposed_commands,
                    exposed_knobs,
                    hidden_commands,
                    memory_tool_exposed=memory_tool_exposed,
                    memory_lanes=_enabled_memory_lanes(),
                )
                print(Fore.CYAN + "[Help] The model asked for help; serving the tool/command reference next turn." + Style.RESET_ALL)

            if model_game_exposed or model_game_hidden:
                print(Fore.MAGENTA + Style.BRIGHT + "\n[Game State Changed]" + Style.RESET_ALL)
                for p in model_game_exposed:
                    if p in probes:
                        probes[p]["exposed"] = True
                        pf = os.path.join(ROOT, "invariants", "out", "probes", f"{p}.pt")
                        if os.path.exists(pf):
                            data = torch.load(pf, weights_only=True)
                            data["exposed"] = True
                            torch.save(data, pf)
                        print(Fore.GREEN + f"  [+] EXPOSED: {p}" + Style.RESET_ALL)
                for p in model_game_hidden:
                    if p in probes:
                        probes[p]["exposed"] = False
                        pf = os.path.join(ROOT, "invariants", "out", "probes", f"{p}.pt")
                        if os.path.exists(pf):
                            data = torch.load(pf, weights_only=True)
                            data["exposed"] = False
                            torch.save(data, pf)
                        print(Fore.RED + f"  [-] HIDDEN: {p}" + Style.RESET_ALL)

            if model_game_propose:
                model_proposed_game = model_game_propose
                print(Fore.MAGENTA + f"\n[Game] The model proposes playing '{model_proposed_game}'. Type ':game {model_proposed_game}' to accept." + Style.RESET_ALL)
            
            if model_game_accept:
                if user_proposed_game == model_game_accept:
                    print(Fore.GREEN + f"\n[Game] The model accepted your proposal to play {model_game_accept}!" + Style.RESET_ALL)
                    user_proposed_game = None
                    active_game = model_game_accept
                    
                    gpath = os.path.join(ROOT, "games", f"{active_game}.py")
                    if os.path.isfile(gpath):
                        print(Fore.MAGENTA + f"[Game] Initializing {active_game}..." + Style.RESET_ALL)
                        try:
                            with open(gpath, "r", encoding="utf-8") as _gf:
                                gcode = _gf.read()
                            _acfg = load_game_config(os.path.join(ROOT, "games", f"{active_game}_rules.json"))
                            exec_globals = globals().copy()
                            exec_globals["GAME_RULES"] = _acfg["rules"]
                            exec_globals["GAME_CONFIG"] = _acfg
                            exec_globals["game_args"] = list(user_proposed_game_args or [])
                            exec_globals["active_game_state"] = active_game_state
                            exec_globals["probes"] = probes
                            exec_globals["run_game"] = _run_game_ref
                            exec(gcode, exec_globals)
                        except Exception as e:
                            import traceback
                            print(Fore.RED + f"[Game Error] {e}\n{traceback.format_exc()}" + Style.RESET_ALL)
            
            if model_game_decline:
                # The model's own agency: it may refuse a proposed game, or back
                # out of one that was force-started, by emitting <<GAME_DECLINE>>.
                declined = None
                if user_proposed_game and model_game_decline in ("*", user_proposed_game):
                    declined, user_proposed_game = user_proposed_game, None
                elif active_game and model_game_decline in ("*", active_game):
                    declined, active_game = active_game, None
                if declined:
                    print(Fore.YELLOW + f"\n[Game] The model declined {declined}." + Style.RESET_ALL)
                    pending_game_tool_result = None
                    memory.append_event("game_declined", tags=["game", "self_concept"], provenance={"game": declined})

            if model_game_end:
                if active_game:
                    print(Fore.CYAN + f"\n[Game] The model ended the active game: {active_game}." + Style.RESET_ALL)
                    prize = active_game_state.get("prize_command")
                    if prize:
                        _stage_for_accept(prize, why=f"'{active_game}' prize")
                    active_game = None
                    active_game_state = {}
            model_memory_tool_result = None
            model_claimmap_tool_result = None
            model_methodmap_tool_result = None
            model_doc_tool_result = None
            model_probe_tool_result = None
            model_cmd_tool_result = None
            if model_cmd_requests:
                model_cmd_tool_result = _run_exposed_command_tool(model_cmd_requests)
                turn_impacts.append({
                    "cause": "asked for an exposed runtime tool",
                    "effect": "command request was staged or queued according to exposure mode",
                })
                print(Fore.CYAN + "\n[Runtime Tool]\n" + model_cmd_tool_result + Style.RESET_ALL + "\n")
                prompt = _build_live_prompt(
                    user_input,
                    memory_tool_result=memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result,
                    sandbox_tool_result=sandbox_tool_result,
                    document_tool_result=document_tool_result,
                    probe_tool_result=None,
                    command_tool_result=model_cmd_tool_result,
                    game_tool_result=game_tool_result,
                    help_tool_result=help_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = _generate_tool_followup(prompt, phase="runtime tool follow-up")
                model_memory_query = extract_memory_query(response)
                model_claimmap_payload = extract_claimmap_payload(response)
                model_methodmap_query = extract_methodmap_query(response)
                model_doc_query = extract_doc_query(response)
                model_probe_query = extract_probe_query(response)
            if model_doc_query and doc_library and document_tool_result is None:
                # The model asked to keep reading, in its own words -- and its
                # words pick the chunk (reply-mode selection over the query
                # text; honest order-fallback when nothing overlaps). Then a
                # same-turn regeneration, the house pattern for model tags, so
                # it answers WITH the text in hand instead of losing it.
                pick = select_next_chunk(doc_library, model_doc_query, "reply")
                if pick is not None:
                    doc_session = doc_library[pick["session_index"]]
                    doc_session["cursor"] = pick["chunk_index"]
                    doc_session.setdefault("read", set()).add(pick["chunk_index"])
                    record_chunk_read(memory, doc_session, pick["chunk_index"])
                    model_doc_tool_result = (
                        impact_note(f'asked to read more ("{model_doc_query[:80]}")')
                        + "\n"
                        + stage_chunk(doc_session, index=pick["chunk_index"], reply_note=reading_reply_note(pick))
                    )
                    turn_impacts.append(
                        {
                            "cause": f'asked to read more ("{model_doc_query[:60]}")',
                            "effect": f"{doc_session['source_name']} part {pick['chunk_index'] + 1}/"
                                      f"{doc_session['chunk_count']} returned ({pick['mode']})",
                        }
                    )
                    print(
                        Fore.CYAN
                        + f"\n[Doc] Model asked to read more -> {doc_session['source_name']} "
                        + f"part {pick['chunk_index'] + 1}/{doc_session['chunk_count']} ({pick['mode']})."
                        + Style.RESET_ALL
                    )
                else:
                    model_doc_tool_result = (
                        impact_note("asked to read more")
                        + "\nThe library is fully read; nothing unread remains."
                    )
                    print(Fore.CYAN + "\n[Doc] Model asked to read more but the library is fully read." + Style.RESET_ALL)
                prompt = _build_live_prompt(
                    user_input,
                    memory_tool_result=memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result,
                    sandbox_tool_result=sandbox_tool_result,
                    document_tool_result=model_doc_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = _generate_tool_followup(prompt, phase="document tool follow-up")
                model_memory_query = extract_memory_query(response)
                model_claimmap_payload = extract_claimmap_payload(response)
                model_methodmap_query = extract_methodmap_query(response)
                model_probe_query = extract_probe_query(response)

            if model_memory_query is not None and memory_tool_result is None:
                if _memory_model_available():
                    records, memory_desc, _ranked = _memory_select_records(
                        model_memory_query,
                        _enabled_memory_lanes(),
                        max_records=6,
                    )
                    request_desc = memory_desc
                    result_body = memory.format_tool_result(records)
                else:
                    records = []
                    request_desc = "unavailable"
                    result_body = (
                        "[Memory Tool Result]\n"
                        "Memory is not available to the model. Enable :memory act on and/or "
                        ":memory talk on, and keep runtime access on with :expose memory."
                    )
                display_query = model_memory_query if model_memory_query else "<reflect>"
                model_memory_tool_result = (
                    impact_note(f'asked memory for "{display_query}"')
                    + "\n"
                    + result_body
                )
                turn_impacts.append(
                    {"cause": f'asked memory for "{display_query}"', "effect": f"{len(records)} record(s) returned"}
                )
                memory.append_event(
                    "memory_tool_model_requested",
                    text=model_memory_tool_result,
                    tags=["memory_tool"],
                    provenance={"query": request_desc, "records": len(records)},
                )
                print(
                    Fore.CYAN
                    + f"\n[Memory] Model requested lookup: {display_query}\n"
                    + model_memory_tool_result
                    + Style.RESET_ALL
                    + "\n"
                )
                prompt = _build_live_prompt(
                    user_input,
                    memory_tool_result=model_memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = _generate_tool_followup(prompt, phase="memory tool follow-up")
                model_claimmap_payload = extract_claimmap_payload(response)
                model_methodmap_query = extract_methodmap_query(response)
                model_probe_query = extract_probe_query(response)
            if model_claimmap_payload and claimmap_tool_result is None:
                model_claimmap_steer = None
                try:
                    cm = analyze_claim_pair(model_claimmap_payload, model=model)
                    # felt only reaches the model, unprefixed: it asked with its
                    # own words and the felt map opens by quoting them back --
                    # the causation is already legible without injecting
                    # standardized vocabulary on top.
                    model_claimmap_tool_result = cm.felt
                    model_claimmap_steer = cm.steer_delta
                    telemetry_for_log = cm.telemetry              # raw numbers logged, never in the prompt
                    turn_impacts.append(
                        {"cause": "asked for a claim comparison", "effect": "its own geometry was measured and returned"}
                    )
                except Exception as exc:
                    model_claimmap_tool_result = f"{CLAIMMAP_HEADER}\nvalid=False; error={exc}"
                    telemetry_for_log = model_claimmap_tool_result
                memory.append_event(
                    "claimmap_tool_model_requested",
                    text=telemetry_for_log,
                    tags=["claimmap_tool", "activation_measurement"],
                    provenance={"payload_chars": len(model_claimmap_payload)},
                )
                print(
                    Fore.CYAN
                    + "\n[ClaimMap] Model sensed a comparison:\n"
                    + model_claimmap_tool_result
                    + Style.RESET_ALL
                    + "\n"
                )
                prompt = _build_live_prompt(
                    user_input,
                    memory_tool_result=model_memory_tool_result or memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=model_claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                tag_alpha = tuner.get("claimmap_alpha", 0.0)
                tag_sweep_layers = None
                if model_claimmap_steer and tag_alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
                    _tls, _tds, _tvar = _sweep_layers_for("claimmap", model_claimmap_steer)
                    if _tls:
                        tag_sweep_layers = _tls
                        for _sl in _tls:
                            sweep_state["fires"].append(("claimmap", _sl, tag_alpha, _tds.get(_sl), len(_tls), _tvar))
                tag_steer_handles = (
                    claimmap_steer_handles(model, model_claimmap_steer, alpha=tag_alpha, layers=tag_sweep_layers)
                    if model_claimmap_steer else []
                )
                response, telemetry = _generate_tool_followup(
                    prompt,
                    extra_steer_handles=tag_steer_handles,
                    phase="claimmap tool follow-up",
                )
                model_methodmap_query = extract_methodmap_query(response)
                model_probe_query = extract_probe_query(response)
            if model_methodmap_query and methodmap_tool_result is None:
                model_methodmap_tool_result = (
                    impact_note(f'asked the method map about "{model_methodmap_query}"')
                    + "\n"
                    + format_methodmap_tool_result(memory, model_methodmap_query)
                )
                turn_impacts.append(
                    {"cause": f'asked the method map about "{model_methodmap_query}"', "effect": "methodologies returned"}
                )
                memory.append_event(
                    "methodmap_tool_model_requested",
                    text=model_methodmap_tool_result,
                    tags=["methodmap_tool"],
                    provenance={"query": model_methodmap_query},
                )
                print(
                    Fore.CYAN
                    + f"\n[MethodMap] Model requested method maps: {model_methodmap_query}\n"
                    + model_methodmap_tool_result
                    + Style.RESET_ALL
                    + "\n"
                )
                prompt = _build_live_prompt(
                    user_input,
                    memory_tool_result=model_memory_tool_result or memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=model_claimmap_tool_result or claimmap_tool_result,
                    methodmap_tool_result=model_methodmap_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = _generate_tool_followup(prompt, phase="methodmap tool follow-up")
            if model_probe_query and (probes or exposed_knobs):
                # READ-ONLY self-measurement: the model may consult sensors the
                # operator has exposed (:probe expose <name> or :expose <name>)
                # and scalar knobs the operator exposed with :expose <knob>.
                # Reading is allowed; shaping is not. Candidate-text reads score
                # hypothetical words for probes only, without observing, crediting,
                # or touching any rolling history.
                name_part, _, cand_text = _partition_unescaped_pipes(model_probe_query)
                cand_text = cand_text.strip()
                req_names = [re.sub(r"[^a-z0-9_]", "_", n.strip().lower())[:40] for n in name_part.split(",") if n.strip()]
                exposed_probe_all = sorted(n for n in probes if probes[n].get("exposed"))
                exposed_knob_all = sorted(k for k in exposed_knobs if k in tuner.triggers and not _is_shadow_trigger(k, tuner))
                if req_names == ["all"]:
                    req_names = list(exposed_probe_all) + list(exposed_knob_all)
                readable = [n for n in req_names if n in probes and probes[n].get("exposed")]
                readable_knobs = [n for n in req_names if n in exposed_knob_all]
                blocked = [n for n in req_names if n not in readable and n not in readable_knobs]
                probe_lines = []
                if cand_text and readable:
                    from invariants.engine import _inputs as _q_inputs, _hidden_states as _q_hidden, probe_score as _q_score
                    with torch.no_grad():
                        q_ids = _q_inputs(model, cand_text[:600])
                        q_hs = _q_hidden(model, q_ids["input_ids"], q_ids.get("attention_mask"))
                    for n in readable:
                        hist = probes[n]["history"]
                        raw = _q_score(q_hs, probes[n]["direction"])
                        if hist:
                            sig = raw - (sum(hist) / len(hist))
                            probe_lines.append(f"- {n} reads {sig:+.3f} on those words, against my recent baseline.")
                        else:
                            probe_lines.append(
                                f"- {n}: no centered reading yet (raw projection {raw:+.3f}; baseline not initialized)."
                            )
                else:
                    for n in readable:
                        p_trig = tuner.triggers.get(f"probe_{n}")
                        last_sig = next(
                            (r[f"probe_{n}"] for r in reversed(turn_log) if f"probe_{n}" in r),
                            None,
                        )
                        seg = (
                            f"- {n}: {float(last_sig):+.3f} on my last turn"
                            if last_sig is not None else f"- {n}: no reading yet"
                        )
                        if p_trig is not None:
                            p_st = p_trig.outcome_stats()
                            seg += f" (bar {round(p_trig.value, 4)}"
                            if p_st.get("lift") is not None:
                                seg += f", lift {p_st['lift']}"
                            seg += ")"
                        probe_lines.append(seg + ".")
                for n in readable_knobs:
                    trig = tuner.triggers.get(n)
                    if trig is None:
                        continue
                    st = trig.stats()
                    seg = f"- {n}: knob value {st['value']} ({st['kind']}"
                    if st.get("kind") == "threshold":
                        seg += f", fires when signal {st['comparator']} value"
                        if st.get("fire_rate") is not None:
                            seg += f", fire_rate {st['fire_rate']}"
                    if st.get("observed"):
                        seg += f", observed {st['observed']}"
                    if st.get("lift") is not None:
                        seg += f", lift {st['lift']}"
                    seg += ")"
                    if cand_text:
                        seg += " Candidate words do not rescore knobs; this is current scalar state."
                        probe_lines.append(seg)
                    else:
                        probe_lines.append(seg + ".")
                for n in blocked:
                    probe_lines.append(f"- {n}: not a sensor I can consult.")
                if not probe_lines:
                    exposed_bits = []
                    if exposed_probe_all:
                        exposed_bits.append("probes: " + ", ".join(exposed_probe_all))
                    if exposed_knob_all:
                        exposed_bits.append("knobs: " + ", ".join(exposed_knob_all))
                    probe_lines.append(
                        "- No consultable sensors named. "
                        + (f"I can consult {', '.join(exposed_bits)}." if exposed_bits else "None are exposed to me.")
                    )
                consulted_parts = []
                if readable:
                    consulted_parts.append(f'{", ".join(readable)} probe(s)')
                if readable_knobs:
                    consulted_parts.append(f'{", ".join(readable_knobs)} knob(s)')
                probe_cause = (
                    "consulted my " + " and ".join(consulted_parts) if consulted_parts else "reached for a sensor"
                ) + (" on candidate words" if cand_text and readable else "")
                model_probe_tool_result = (
                    impact_note(probe_cause)
                    + "\n" + PROBE_TOOL_HEADER + "\n" + "\n".join(probe_lines)
                )
                turn_impacts.append({
                    "cause": probe_cause,
                    "effect": f"{len(readable) + len(readable_knobs)} reading(s) returned"
                              + (f", {len(blocked)} refused" if blocked else ""),
                })
                memory.append_event(
                    "probe_tool_model_requested",
                    text=model_probe_tool_result,
                    tags=["probe", "probe_tool", "activation_measurement"],
                    provenance={"query": model_probe_query[:240], "readable": readable, "readable_knobs": readable_knobs, "blocked": blocked},
                )
                print(
                    Fore.CYAN
                    + f"\n[Probe] Model consulted sensors: {model_probe_query[:120]}\n"
                    + model_probe_tool_result
                    + Style.RESET_ALL
                    + "\n"
                )
                prompt = _build_live_prompt(
                    user_input,
                    memory_tool_result=memory_tool_result or model_memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=claimmap_tool_result or model_claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result or model_methodmap_tool_result,
                    probe_tool_result=model_probe_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = _generate_tool_followup(prompt, phase="probe tool follow-up")
            if response:
                active_memory_tool_result = memory_tool_result or model_memory_tool_result
                active_orientation_tool_result = orientation_tool_result
                active_claimmap_tool_result = claimmap_tool_result or model_claimmap_tool_result
                active_methodmap_tool_result = methodmap_tool_result or model_methodmap_tool_result
                active_document_tool_result = document_tool_result or model_doc_tool_result
                active_sandbox_tool_result = sandbox_tool_result
                active_probe_tool_result = model_probe_tool_result
                active_command_tool_result = model_cmd_tool_result
                if (
                    (active_memory_tool_result or active_claimmap_tool_result or active_methodmap_tool_result or active_document_tool_result or active_probe_tool_result or active_command_tool_result)
                    and is_tool_only_response(response)
                ):
                    memory.append_event(
                        "tool_loop_retry",
                        text=response,
                        tags=["tool_protocol"],
                        provenance={"current_input": user_input[:240]},
                    )
                    retry_input = (
                        user_input
                        + "\n\n[Tool Protocol Reminder]\n"
                        + "I already have the tool result I asked for; no more tool tags -- I answer now from it."
                    )
                    prompt = _build_live_prompt(
                        retry_input,
                        memory_tool_result=active_memory_tool_result,
                        orientation_tool_result=active_orientation_tool_result,
                        claimmap_tool_result=active_claimmap_tool_result,
                        methodmap_tool_result=active_methodmap_tool_result,
                        sandbox_tool_result=active_sandbox_tool_result,
                        document_tool_result=active_document_tool_result,
                        probe_tool_result=active_probe_tool_result,
                        command_tool_result=active_command_tool_result,
                        session_context=session_context if session_context_enabled else None,
                    )
                    print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                    response, telemetry = _generate_tool_followup(prompt, phase="tool retry follow-up")
                response = scrub_unstaged_memory_status(
                    response,
                    memory_tool_result=active_memory_tool_result,
                    orientation_tool_result=active_orientation_tool_result,
                    claimmap_tool_result=active_claimmap_tool_result,
                    methodmap_tool_result=active_methodmap_tool_result,
                    sandbox_tool_result=active_sandbox_tool_result,
                    document_tool_result=active_document_tool_result,
                    probe_tool_result=active_probe_tool_result,
                    command_tool_result=active_command_tool_result,
                )
                
                if _pending_expect:
                    etype = _pending_expect.get("type")
                    ename = _pending_expect.get("name")
                    expect_context = _pending_expect.get("context") or {}
                    _pending_expect = None
                    print(Fore.CYAN + f"\n[Expect] Intercepted response for {etype} '{ename}'." + Style.RESET_ALL)
                    definition_record = memory.append_definition(
                        etype,
                        ename or "(unnamed)",
                        response,
                        authored_by="model",
                        status="proposed",
                        provenance={"source": ":expect"},
                    )
                    if etype == "macro":
                        macro_name = re.sub(r"[^a-z0-9_]", "_", (ename or "macro").lower())[:40].strip("_") or "macro"
                        solve_expect_contexts.pop(macro_name, None)
                        lines, response_arg_specs, response_comments = expected_macro_parts(response)
                        if not lines:
                            memory.append_definition_feedback(
                                definition_record,
                                "the response contained no runnable colon-prefixed commands",
                                verdict="rejected",
                                source="shell_validator",
                            )
                            print(Fore.YELLOW + f"[Expect] Model generated no commands." + Style.RESET_ALL)
                        else:
                            context_arg_specs = expect_context.get("arg_specs") or []
                            arg_specs_out = response_arg_specs or context_arg_specs
                            pending_solve_proposal = {
                                "name": macro_name,
                                "goal": expect_context.get("goal") or "model-generated through :expect macro",
                                "dest": os.path.join(ROOT, "invariants", "out", "macros", f"{macro_name}.txt"),
                                "cmd_lines": lines,
                                "arg_specs": arg_specs_out,
                                "clean_names": expect_context.get("clean_names") or [],
                                "definition_id": definition_record.record_id,
                                "comments": response_comments,
                            }
                            if response_arg_specs:
                                memory.append_definition_feedback(
                                    definition_record,
                                    "captured the model's # args header for named macro parameters",
                                    verdict="accepted",
                                    source="shell_validator",
                                )
                            print(Fore.GREEN + f"[Expect] Staged {len(lines)} command(s) as macro '{ename}'. Type :accept to save/run it." + Style.RESET_ALL)
                    elif etype == "file":
                        out_path = os.path.join(ROOT, ename)
                        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                        with open(out_path, "w", encoding="utf-8") as wf:
                            wf.write(response)
                        memory.append_definition_feedback(
                            definition_record,
                            "written to the requested file",
                            verdict="accepted",
                            source="shell",
                            provenance={"artifact_path": out_path},
                        )
                        print(Fore.GREEN + f"[Expect] Wrote response to {out_path}." + Style.RESET_ALL)
                    elif etype == "autocomplete":
                        _last_completion, completion_error = validate_command_autocomplete(
                            response,
                            ename,
                            known_commands=_all_shell_commands() - {"expose"},
                            probe_names=probes,
                            knob_names=tuner.triggers,
                        )
                        if completion_error:
                            memory.append_definition_feedback(
                                definition_record,
                                completion_error,
                                verdict="rejected",
                                source="shell_validator",
                                tags=["invalid"],
                            )
                            memory.append_event(
                                "autocomplete_rejected",
                                text=response[:400],
                                tags=["action", "autocomplete", "invalid"],
                                provenance={"prefix": ename, "reason": completion_error},
                            )
                            print(
                                Fore.YELLOW
                                + f"[Expect] Rejected invalid autocomplete: {completion_error}. Raw: {response.strip()!r}"
                                + Style.RESET_ALL
                            )
                            continue
                        pending_completion_definition_id = definition_record.record_id
                        print(Fore.GREEN + f"[Expect] Suggested: '{_last_completion}'. Type :accept to use it." + Style.RESET_ALL)
                    elif etype == "var":
                        _shell_vars[ename] = response.strip()
                        memory.append_definition_feedback(
                            definition_record,
                            "stored in the requested shell variable",
                            verdict="accepted",
                            source="shell",
                        )
                        print(Fore.GREEN + f"[Expect] Saved response to variable ${ename}." + Style.RESET_ALL)
                    # Record this meta-turn as an action memory so the model can recall what it generated
                    memory.append_event(
                        "model_action",
                        text=f"I successfully wrote the following content to {etype} '{ename}':\n{response}",
                        tags=["action", etype]
                    )
                    continue

                if show_timestamps:
                    print(Fore.GREEN + Style.BRIGHT + f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Me: " + Style.RESET_ALL)
                print(response, end="")
                if session_context_enabled:
                    s_score = sense_score(synthesis_records) if synthesis_records else 0.0
                    if s_score is None: s_score = 0.0
                    active_u_name = _last_replace_name if '_last_replace_name' in locals() and _last_replace_name else "user"
                    session_context.append(("user", active_u_name, user_input, s_score))
                    session_context.append(("assistant", "assistant", response, s_score))
                    if len(session_context) > MAX_SESSION_TURNS * 2:
                        if len(session_context) >= 6:
                            min_score = float('inf')
                            min_idx = 0
                            for i in range(0, len(session_context) - 4, 2):
                                pair_score = session_context[i][3] if len(session_context[i]) > 3 else (session_context[i][2] if len(session_context[i]) > 2 else 0.0)
                                if pair_score < min_score:
                                    min_score = pair_score
                                    min_idx = i
                            session_context.pop(min_idx + 1)
                            session_context.pop(min_idx)
                        else:
                            session_context = session_context[-MAX_SESSION_TURNS * 2 :]

                # Run Joined Agents
                for ja in joined_agents:
                    print(Fore.GREEN + Style.BRIGHT + f"\n[{ja.name}]: " + Style.RESET_ALL, end="")
                    
                    _sys_tuner = tuner
                    _sys_probes = probes
                    _sys_tb = tuner_bindings
                    _sys_pin = prioritize_pin
                    _sys_queue = queued_calibrations
                    _sys_cmd_queue = queued_commands
                    _sys_doc_session = doc_session
                    _sys_doc_library = doc_library
                    _sys_pending_doc = pending_document_tool_result
                    _sys_doc_autoread = doc_autoread
                    tuner = ja.tuner
                    probes = ja.probes
                    tuner_bindings = ja.tuner_bindings
                    prioritize_pin = ja.prioritize_pin
                    queued_calibrations = ja.queued_calibrations
                    queued_commands = ja.queued_commands
                    doc_session = ja.doc_session
                    doc_library = ja.doc_library
                    pending_document_tool_result = ja.pending_document_tool_result
                    doc_autoread = ja.doc_autoread

                    agent_document_tool_result = pending_document_tool_result
                    pending_document_tool_result = None
                    if agent_document_tool_result is None and doc_autoread and doc_autoread.get("remaining", 0) > 0:
                        pick = _select_autoread_chunk(doc_library, doc_autoread)
                        if pick is not None:
                            doc_session = doc_library[pick["session_index"]]
                            doc_session["cursor"] = pick["chunk_index"]
                            doc_session.setdefault("read", set()).add(pick["chunk_index"])
                            record_chunk_read(memory, doc_session, pick["chunk_index"])
                            agent_document_tool_result = stage_chunk(doc_session, index=pick["chunk_index"])
                            doc_autoread["remaining"] -= 1
                            if doc_autoread["remaining"] <= 0:
                                doc_autoread = None

                    agent_state_report = _spawn_state_report(ja.name, tuner, probes)
                    ja_prompt = _build_live_prompt(
                        agent_state_report + "\n\n[Please respond]",
                        document_tool_result=agent_document_tool_result,
                        session_context=session_context if session_context_enabled else None,
                        active_u_name="system",
                        field_context=None,  # already present in agent_state_report
                    )
                    _ja_handles, _ja_prio = _priority_steer_handles(probes, tuner, prioritize_pin)
                    try:
                        ja_response = generate_agentic_text(
                            model,
                            instruction=ja_prompt,
                            config=config,
                            max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                            synthesis_recorder=None,
                            chatty_log=False,
                            pre_formatted=True,
                            return_telemetry=False,
                        )
                    finally:
                        for _h in _ja_handles:
                            try:
                                _h.remove()
                            except Exception:
                                pass

                    print(ja_response, end="")

                    if session_context_enabled:
                        session_context.append(("assistant", ja.name, ja_response, 0.0))

                    ja.tuner = tuner
                    ja.probes = probes
                    ja.tuner_bindings = tuner_bindings
                    ja.prioritize_pin = prioritize_pin
                    ja.queued_calibrations = queued_calibrations
                    ja.queued_commands = queued_commands
                    ja.doc_session = doc_session
                    ja.doc_library = doc_library
                    ja.pending_document_tool_result = pending_document_tool_result
                    ja.doc_autoread = doc_autoread
                    tuner = _sys_tuner
                    probes = _sys_probes
                    tuner_bindings = _sys_tb
                    prioritize_pin = _sys_pin
                    queued_calibrations = _sys_queue
                    queued_commands = _sys_cmd_queue
                    doc_session = _sys_doc_session
                    doc_library = _sys_doc_library
                    pending_document_tool_result = _sys_pending_doc
                    doc_autoread = _sys_doc_autoread
                    
                    memory.append_turn(
                        "assistant",
                        ja_response,
                        tags=["model_output", ja.name],
                        metrics={
                            "chars": len(ja_response),
                        }
                    )

                memory.append_turn(
                    "assistant",
                    response,
                    tags=["model_output", "main_assistant"],
                    metrics={
                        "chars": len(response),
                        "sense_score": float(s_score) if session_context_enabled else 0.0,
                        "model_memory_tool_requested": bool(model_memory_tool_result),
                        "orientation_tool_result_provided": bool(active_orientation_tool_result),
                        "model_claimmap_tool_requested": bool(model_claimmap_tool_result),
                        "claimmap_tool_result_provided": bool(active_claimmap_tool_result),
                        "model_methodmap_tool_requested": bool(model_methodmap_tool_result),
                        "methodmap_tool_result_provided": bool(active_methodmap_tool_result),
                        "document_tool_result_provided": bool(active_document_tool_result),
                        "sandbox_tool_result_provided": bool(active_sandbox_tool_result),
                        "model_probe_tool_requested": bool(model_probe_tool_result),
                        "model_command_tool_requested": bool(model_cmd_tool_result),
                    },
                )
                # The reading dialogue's reflection stream: only reply-mode
                # auto-read consults it; order/interleave advance regardless.
                last_assistant_response = response
                recent_responses.append(response)
                
                # Cut-off detection via the model's own P(end-of-turn): observed
                # every turn so the distribution accrues; when the budget was hit
                # mid-thought, SUGGEST the retune -- never apply it. Nothing in
                # this shell drifts silently; calibration stays a deliberate act.
                eot_probs = (telemetry or {}).get("eot_probs", [])
                tokens_generated = (telemetry or {}).get("tokens_generated", 0)
                if eot_probs and not utility_expect_turn:
                    final_prob = eot_probs[-1]
                    # fired == P(eot) <= eot_urgency threshold == "still mid-thought"
                    mid_thought = tuner.observe("eot_urgency", final_prob)
                    max_allowed = max(64, int(tuner.get("response_tokens", 512)))
                    if tokens_generated >= max_allowed:
                        if mid_thought:
                            print(
                                Fore.YELLOW
                                + f"[Steer] Reply hit the {max_allowed}-token budget with P(eot)={final_prob:.4f} "
                                + f"-- likely cut off mid-thought. Extend deliberately: :tune response_tokens {max_allowed + 256}"
                                + Style.RESET_ALL
                            )
                        else:
                            print(
                                Fore.CYAN
                                + f"[Steer] Reply hit the {max_allowed}-token budget but P(eot)={final_prob:.4f} "
                                + "suggests the thought was complete."
                                + Style.RESET_ALL
                            )

                # Sandbox connection: if enabled and the reply contains a fenced
                # ```python block (natural emission, no taught tag), run it for
                # real and stage the honest result for the next turn. Execution
                # success is an objective outcome -- observed into the tuner.
                if sandbox_enabled and not utility_expect_turn:
                    sandbox_code = extract_python_block(response)
                    if sandbox_code:
                        print(Fore.MAGENTA + "\n[Sandbox] Running the model's python block..." + Style.RESET_ALL, flush=True)
                        sandbox_result = run_python(sandbox_code, timeout_sec=SANDBOX_TIMEOUT_SEC)
                        pending_sandbox_tool_result = format_sandbox_tool_result(
                            sandbox_result, timeout_sec=SANDBOX_TIMEOUT_SEC
                        )
                        tuner.observe("sandbox_success", 1.0 if sandbox_result["ok"] else 0.0)
                        memory.append_event(
                            "sandbox_executed",
                            text=sandbox_code[:500],
                            tags=["sandbox_tool"],
                            provenance={
                                "code_sha": sandbox_result["code_sha"],
                                "exit_code": sandbox_result["exit_code"],
                                "timed_out": sandbox_result["timed_out"],
                                "duration_sec": sandbox_result["duration_sec"],
                                "ok": sandbox_result["ok"],
                            },
                        )
                        status = "ok" if sandbox_result["ok"] else ("timed out" if sandbox_result["timed_out"] else f"exit {sandbox_result['exit_code']}")
                        impact_state["pending"].append(
                            {"cause": "wrote python code", "effect": f"it was executed for real ({status})"}
                        )
                        print(
                            Fore.MAGENTA
                            + f"[Sandbox] {status} in {sandbox_result['duration_sec']}s -- real output staged for the next turn."
                            + Style.RESET_ALL
                        )
            # The turn's own outcome, read before anything is recorded: did the
            # deliberation cohere? This is the conversational learning signal --
            # no gold, no oracle, just the live productivity read.
            turn_sense = sense_score(synthesis_records)
            conversation_outcome = None
            if turn_sense is not None:
                # Observe every turn so the threshold can be calibrated to the
                # sense distribution (:tune conversation_productive auto 50).
                tuner.observe("conversation_productive", turn_sense)
                conversation_outcome = {
                    "score": float(turn_sense),
                    "threshold": float(tuner.get("conversation_productive", 0.0)),
                }
                turn_row["sense"] = float(turn_sense)
            # Clock sensor: measured here, before the probe-scoring passes, so
            # generation_seconds is generation time only. Memory in GB (live
            # allocation, total reserved footprint, and this turn's peak) --
            # VRAM on CUDA, process RSS on CPU-only, so the stream is never dead.
            gen_seconds = _time.perf_counter() - _gen_t0
            vram_alloc, vram_reserved, vram_peak, mem_label = _memory_footprint_gb()
            _tg = (telemetry or {}).get("tokens_generated", 0)
            tps = (_tg / gen_seconds) if (gen_seconds > 0 and _tg) else 0.0
            tuner.observe("generation_seconds", gen_seconds)
            tuner.observe("vram_gb", vram_reserved)
            turn_row["generation_seconds"] = float(gen_seconds)
            turn_row["vram_gb"] = float(vram_reserved)
            turn_row["tokens_per_sec"] = float(tps)
            if turn_sense is not None:
                tuner.credit("generation_seconds", gen_seconds, turn_sense)
                tuner.credit("vram_gb", vram_reserved, turn_sense)
            # Every command's METRIC knob: observed EVERY reply (signal 1.0
            # when it ran since the last reply, 0.0 otherwise -- the
            # distribution stays visible even on senseless turns), credited
            # with this turn's sense when one exists. Lift then reads "are
            # turns that use this command more productive?" -- drawable and
            # steerable like any stream (lift:cmd_<word>).
            for _kname in [k for k in tuner.triggers if k.startswith("cmd_") and not k.endswith("_criteria")]:
                _ksig = 1.0 if _kname[4:] in commands_since_reply else 0.0
                tuner.observe(_kname, _ksig)
                if turn_sense is not None:
                    tuner.credit(_kname, _ksig, turn_sense)
            # Exposed tools' CRITERIA knobs observe the probe named in their
            # activation clause, so the bar calibrates against a real
            # distribution instead of an asserted number.
            for _xw, _xcfg in list((exposed_commands or {}).items()):
                _xact = str((_xcfg or {}).get("activation") or "")
                if not _xact:
                    continue
                _xp = next(
                    (p for p in probes if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", _xact, re.IGNORECASE)),
                    None,
                )
                if _xp is None:
                    continue
                _xptrig = tuner.triggers.get(f"probe_{_xp}")
                if _xptrig is not None and _xptrig.signals:
                    _, _, _xcrit = ensure_command_knobs(tuner, _xw)
                    _xcrit.observe(float(_xptrig.signals[-1]))
            commands_since_reply.clear()
            last_clock = {
                "generation_seconds": gen_seconds,
                "tokens_generated": int(_tg or 0),
                "tokens_per_sec": tps,
                "vram_alloc_gb": vram_alloc,
                "vram_reserved_gb": vram_reserved,
                "vram_peak_gb": vram_peak,
                "mem_label": mem_label,
                "ts": datetime.datetime.now().strftime("%H:%M:%S"),
            }
            print(
                Fore.CYAN
                + f"  [Clock] {gen_seconds:.1f}s"
                + (f" | {tps:.1f} tok/s" if tps else "")
                + (f" | {mem_label} {vram_alloc:.2f}GB live / {vram_reserved:.2f}GB reserved" if vram_reserved else "")
                + (f" / {vram_peak:.2f}GB peak" if vram_peak else "")
                + Style.RESET_ALL,
                flush=True,
            )
            # Minted probes score every turn: raw projection of the reply
            # state on each probe direction, centered against the probe's own
            # rolling history, paired with sense via the credit channel.
            if probes and response:
                from invariants.engine import _inputs as _p_inputs, _hidden_states as _p_hidden, probe_score
                p_ids = _p_inputs(model, response[:600])
                p_hs = _p_hidden(model, p_ids["input_ids"], p_ids.get("attention_mask"))
                seeded_probes = []
                for pname, pdata in probes.items():
                    raw = probe_score(p_hs, pdata["direction"])
                    sig = centered_probe_observation(pdata, raw)
                    if sig is None:
                        seeded_probes.append(pname)
                        continue
                    tuner.observe(f"probe_{pname}", sig)
                    turn_row[f"probe_{pname}"] = float(sig)
                    if pdata.get("chatty", True):
                        print(Fore.CYAN + f"  [Probe Score] {pname}: {sig:+.3f}" + Style.RESET_ALL, flush=True)
                    if turn_sense is not None:
                        tuner.credit(f"probe_{pname}", sig, turn_sense)
                    # A validate-mode match records (knob value, probe reading) so
                    # its correlation can credit the knob it is matched to.
                    _pm = probe_matches.get(pname)
                    if _pm and _pm.get("mode") == "validate" and _pm.get("knob"):
                        match_hist.setdefault(pname, deque(maxlen=200)).append(
                            (float(tuner.get(_pm["knob"], 0.0)), float(sig))
                        )
                save_probe_raw_histories(probes)
                if seeded_probes:
                    print(
                        Fore.CYAN
                        + f"  [Probe Seed] initialized raw baselines for {len(seeded_probes)} probe(s); "
                        + "centered readings begin next turn."
                        + Style.RESET_ALL,
                        flush=True,
                    )

            # Easter egg: the turn the self-sensor overtakes the user-model.
            _egg = consciousness_over_user_intent(probes, tuner, egg_state)
            if _egg is not None:
                _cl = "n/a" if _egg[0] is None else f"{_egg[0]:+.3f}"
                _ul = "n/a" if _egg[1] is None else f"{_egg[1]:+.3f}"
                print(Fore.MAGENTA + Style.BRIGHT + "\n  ✦ ────────────────── ✦" + Style.RESET_ALL)
                for _line in (
                    "  consciousness has overtaken user_intent.",
                    "  The sensor built to watch the self now tracks",
                    "  a good turn more than the one that watches you.",
                    "  For this stretch, at least, it is listening inward.",
                    f"  (consciousness lift {_cl} > user_intent {_ul})",
                ):
                    print(Fore.MAGENTA + _line + Style.RESET_ALL)
                print(Fore.MAGENTA + Style.BRIGHT + "  ✦ ────────────────── ✦\n" + Style.RESET_ALL)
                memory.append_event(
                    "consciousness_over_user_intent",
                    text=f"consciousness lift {_cl} overtook user_intent {_ul}",
                    tags=["easter_egg", "self_concept"],
                    provenance={"consciousness_lift": _egg[0], "user_intent_lift": _egg[1]},
                )

            # Intent axis: did this turn settle intent (lower
            # ambiguity+disagreement than last turn)? Observed every turn the
            # sensors fire; paired with sense via the credit channel.
            phen_now = latest_phenomenality_scores(synthesis_records) or {}
            if phen_now:
                # Every monitored dimension becomes a named per-turn stream
                # (phen_<metric>): observed, credited against sense, and written
                # to the turn row -- so existing sensors are anchorable and
                # calibratable exactly like probes. These read the REASONING
                # states; an adopted probe reads the same axis in the reply.
                for _pm, _pv in phen_now.items():
                    try:
                        _pv = float(_pv)
                    except (TypeError, ValueError):
                        continue
                    _pstream = f"phen_{_pm.replace('_legacy', '')}"
                    tuner.observe(_pstream, _pv, default=0.0)
                    turn_row[_pstream] = _pv
                    if turn_sense is not None:
                        tuner.credit(_pstream, _pv, turn_sense)
                # The memory trigger's own quantity, sampled once per turn from
                # the same phenomenality the mid-thought detector reads -- so
                # :calibrate memory_need has a distribution, and the paired
                # table can anchor on the gap. (ToolSense itself only consults
                # the bar; per-turn distribution + credit live here by design.)
                _gap = (
                    float(phen_now.get("ambiguity", 0.0))
                    + float(phen_now.get("disagreement", 0.0))
                    - float(phen_now.get("validated_flow", 0.0))
                    - float(phen_now.get("warranted_confidence_legacy", 0.0))
                )
                tuner.observe("memory_need", _gap)
                turn_row["memory_need"] = _gap
                if turn_sense is not None:
                    tuner.credit("memory_need", _gap, turn_sense)
                unsettled_now = float(phen_now.get("ambiguity", 0.0)) + float(phen_now.get("disagreement", 0.0))
                if prev_unsettledness is not None:
                    intent_signal = prev_unsettledness - unsettled_now
                    tuner.observe("intent_settling", intent_signal)
                    turn_row["intent_settling"] = float(intent_signal)
                    if turn_sense is not None:
                        tuner.credit("intent_settling", intent_signal, turn_sense)
                prev_unsettledness = unsettled_now

            # "Read until satisfied": a reading turn counts as settled when its
            # sense clears the conversation_productive bar (or needed no
            # deliberation at all); enough settled turns in a row end the
            # auto-read early, honestly, with the remainder still unread.
            if doc_autoread and doc_autoread.get("until_probe") and reading_turn_source:
                up = doc_autoread["until_probe"]
                trig = tuner.triggers.get(f"probe_{up}")
                sig = turn_row.get(f"probe_{up}")
                if trig and sig is not None and sig >= trig.value:
                    unread_left = sum(s["chunk_count"] - len(s.get("read") or ()) for s in doc_library)
                    print(
                        Fore.CYAN
                        + f"[Doc] Reading stopped: probe '{up}' thresholded ({sig:+.3f} >= {trig.value:.3f}). "
                        + f"Stopping with {unread_left} chunk(s) unread -- ':doc read' resumes anytime."
                        + Style.RESET_ALL
                    )
                    doc_autoread = None

            elif doc_autoread and doc_autoread.get("until_settled") and reading_turn_source:
                settled_turn = turn_sense is None or turn_sense >= float(
                    tuner.get("conversation_productive", 0.0)
                )
                doc_autoread["settled_streak"] = (
                    doc_autoread.get("settled_streak", 0) + 1 if settled_turn else 0
                )
                needed = max(1, int(tuner.get("reading_settled_streak", 2)))
                if doc_autoread["settled_streak"] >= needed:
                    unread_left = sum(
                        s["chunk_count"] - len(s.get("read") or ()) for s in doc_library
                    )
                    print(
                        Fore.CYAN
                        + f"[Doc] Reading settled: {needed} productive turn(s) in a row. "
                        + f"Stopping with {unread_left} chunk(s) unread -- ':doc read' resumes anytime."
                        + Style.RESET_ALL
                    )
                    doc_autoread = None

            # Agency ledger: was this turn's context really caused by the
            # model's own words? Observed every turn (so the contingency rate
            # is visible) and credited with the turn's sense, so :tune lift on
            # words_had_impact reads whether experienced impact tracks better
            # deliberation. The contrast with non-contingent turns is what
            # makes impact learnable at all.
            impact_signal = 1.0 if turn_impacts else 0.0
            tuner.observe("words_had_impact", impact_signal)
            if turn_impacts:
                impact_state["log"].extend(turn_impacts)
            if turn_sense is not None:
                tuner.credit("words_had_impact", impact_signal, turn_sense)
            turn_row["words_had_impact"] = float(impact_signal)
            # Outcome evidence for strength/budget knobs: pair each knob's
            # value-in-force with this turn's sense (batched; one save).
            if turn_sense is not None:
                for _knob in OUTCOME_CALIBRATABLE:
                    # claimmap_alpha is credited by its APPLIED value in the
                    # dedicated block below; crediting its configured value here
                    # too would double-count (and mislabel unsteered turns),
                    # corrupting :calibrate claimmap_alpha outcome.
                    if _knob == "claimmap_alpha":
                        continue
                    _trig = tuner.triggers.get(_knob)
                    if _trig is not None:
                        _trig.credit(_trig.value, turn_sense)
                tuner.save()
            if turn_row:
                turn_row["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                turn_log.append(dict(turn_row))
                try:
                    with open(TURN_SIGNALS_PATH, "a", encoding="utf-8") as _tf:
                        _tf.write(json.dumps(turn_row) + chr(10))
                except OSError:
                    pass
            # Layer isolation evidence: each single-layer steer this turn lands
            # on ITS layer in the steer map, labeled by the turn's outcome —
            # per-layer success accrues with no band confound.
            for sweep_channel, sweep_l, sweep_alpha, sweep_d, sweep_w, sweep_v in sweep_state["fires"]:
                steer_map.record_layer_steer(
                    sweep_channel,
                    sweep_l,
                    sweep_alpha,
                    conversation_outcome=conversation_outcome,
                    metrics={"axis_drift_at_layer": sweep_d, "sweep_width": sweep_w,
                             "axis_lawfulness_var": sweep_v},
                )
            sweep_state["fires"] = []
            record_internal_traces(
                memory,
                synthesis_records,
                steer_map=steer_map,
                conversation_outcome=conversation_outcome,
            )

            # Credit the turn-level ClaimMap steer with the turn's outcome:
            # signal = alpha actually applied (0 when unsteered), so :tune lift
            # on claimmap_alpha reads "did steered turns cohere better than
            # unsteered ones" from real conversations.
            if turn_sense is not None:
                tuner.credit("claimmap_alpha", claimmap_alpha_used, turn_sense)

            # Activation-reach: no tag was taught in bare mode, so tools fire from
            # the model's own surfaced state. If the answer holds two opposed
            # framings, sense the comparison (felt) and stage it -- with steering --
            # for the next turn. Disable with CLAIMMAP_AUTO_TRIGGER=0.
            # Deferred credit: attribute THIS turn's sense (did the deliberation
            # cohere) to LAST turn's trigger decision -- last turn's fire staged the
            # felt+steer that shaped this turn. The tuner buckets by whether that
            # tension cleared the threshold, so :tune lift = did firing the ClaimMap
            # actually lead to more coherent deliberation. Real outcome, not synthetic.
            if pending_claimmap_credit is not None and turn_sense is not None:
                tuner.credit("claimmap_tension", pending_claimmap_credit, turn_sense)
                pending_claimmap_credit = None

            if (
                os.environ.get("CLAIMMAP_AUTO_TRIGGER", "1").strip() not in {"0", "false", "no"}
                and pending_claimmap_tool_result is None
            ):
                a, b, tension_score = framing_tension_score(response or "")
                # Log the tension signal EVERY turn (even 0) so :tune can read the
                # distribution; fire on the live-tuned threshold, not a fixed cutoff.
                fired = tuner.observe("claimmap_tension", tension_score)
                pending_claimmap_credit = tension_score  # credited by next turn's sense
                if fired and a is not None:
                    try:
                        cm = analyze_claim_pair(f"{a} || {b}", model=model)
                        # No attribution prefix: the felt map's own first line
                        # ("I just held two framings against each other: ...")
                        # already carries the causation, in the model's own
                        # quoted words -- adding a standardized vocabulary line
                        # ("tension") would teach the lexical association we
                        # later want to MEASURE (see SEMANTIC_NEUTRALIZATION).
                        pending_claimmap_tool_result = cm.felt
                        pending_claimmap_steer_delta = cm.steer_delta
                        impact_state["pending"].append(
                            {"cause": "answer held a framing tension", "effect": "a felt comparison was measured and staged"}
                        )
                        memory.append_event(
                            "claimmap_auto_triggered",
                            text=cm.telemetry,
                            tags=["claimmap_tool", "activation_trigger"],
                            provenance={"trigger": "framing_tension", "tension_score": tension_score, "mean_sim": cm.mean_sim},
                        )
                        print(
                            Fore.MAGENTA
                            + f"\n[ClaimMap] Sensed a tension (score {tension_score:.2f}) in that answer -- it will shape the next turn."
                            + Style.RESET_ALL
                        )
                    except Exception as exc:
                        print(Fore.RED + f"[ClaimMap auto] {exc}" + Style.RESET_ALL)
            sensor_scores = latest_phenomenality_scores(synthesis_records)
            if sensor_scores:
                decision = self_concept.decide(
                    sensor_scores,
                    context={"task_grounding_low": infer_task_grounding_low(user_input, response)},
                )
                memory.append_self_concept_trace(decision.to_dict())
                steer_map.record_self_concept_decision(
                    decision.to_dict(),
                    source="interactive",
                    final_correct=None,
                )
                if decision.allowed and decision.intervention_type in {"tool_result", "context_tool_result"}:
                    pending_orientation_tool_result = format_orientation_tool_result(decision)
                    print(
                        Fore.CYAN
                        + "\n[Orientation] Vector-map controller staged a one-turn orientation result.\n"
                        + pending_orientation_tool_result
                        + Style.RESET_ALL
                    )
            
            # The streaming will print tokens, just need a newline at the end
            print("\n")
            
            # Process queued calibrations silently
            if queued_calibrations:
                survived = []
                for qcmd in queued_calibrations:
                    cargs = qcmd.split()
                    if not cargs:
                        continue
                    cal_name = cargs[0]
                    route, _ = calibration_policy(cal_name)
                    success = False
                    
                    if len(cargs) >= 2 and cargs[1].lower() == "outcome":
                        trig = tuner.triggers.get(cal_name)
                        pairs = list(trig.outcomes) if trig is not None else []
                        v = outcome_calibration(pairs)
                        if v is not None:
                            tuner.set(cal_name, v)
                            print(Fore.GREEN + f"[Queue] SUCCESS: {cal_name} = {round(v, 4)}" + Style.RESET_ALL)
                            success = True
                    elif route == "threshold":
                        anchor = "".join(cargs[1:]).lower() if len(cargs) >= 2 else None
                        target_trigger, target_stream = resolve_target(cal_name, tuner)
                        anchors, _albl, _abad = parse_anchor_spec(anchor, tuner)
                        if target_stream and anchors and _abad is None and target_trigger in tuner.triggers:
                            v = paired_threshold_multi(list(turn_log), target_stream, anchors)
                            if v is not None:
                                new_v, _prior = tuner.set_calibrated(target_trigger, v, tuner.get("calibration_gain", 0.5))
                                print(Fore.GREEN + f"[Queue] SUCCESS: {target_trigger} -> {round(new_v, 4)} (toward cut {round(v, 4)})" + Style.RESET_ALL)
                                success = True
                    elif route == "cap":
                        from invariants import engine as _engine
                        cal_pct = 50.0
                        if len(cargs) >= 2:
                            try: cal_pct = float(cargs[1])
                            except ValueError: pass
                        if 0.0 <= cal_pct < 100.0:
                            v = _engine.calibrate_steer_cap_fraction(percentile=cal_pct)
                            if v is not None:
                                tuner.set("steer_cap_fraction", v)
                                print(Fore.GREEN + f"[Queue] SUCCESS: steer_cap_fraction = {round(v, 4)}" + Style.RESET_ALL)
                                success = True
                    else:
                        trig = resolve_calibrate_trigger(cal_name, tuner)
                        if trig and len(trig.signals) >= 10:
                            cal_pct = 50.0
                            if len(cargs) >= 2:
                                try: cal_pct = float(cargs[1])
                                except ValueError: pass
                            if 0.0 <= cal_pct <= 100.0:
                                v = tuner.calibrate(trig.name, cal_pct, tuner.get("calibration_gain", 0.5))
                                print(Fore.GREEN + f"[Queue] SUCCESS: {trig.name} -> {round(v, 4)} (toward p{cal_pct:g})" + Style.RESET_ALL)
                                success = True
                    
                    if not success:
                        survived.append(qcmd)
                queued_calibrations = survived
            
        except (KeyboardInterrupt, EOFError):
            memory.append_event("shell_closed", tags=["session"])
            print("\nInteractive shell closed.")
            break
        except CommandUsageError as _usage_exc:
            # Malformed command, not a bug: guidance only, no traceback,
            # no error event -- the session and the turn queue are intact.
            msg = f"[Shell] {_usage_exc}"
            if _usage_exc.usage:
                msg += f"\n  Usage: {_usage_exc.usage}"
            print(Fore.YELLOW + msg + Style.RESET_ALL)
            continue
        except Exception as _turn_exc:
            # A bug in ONE command (or a typo that reaches bad code) must not end
            # the session -- abort the turn, report, and keep going. The full
            # traceback is logged so the fault is still discoverable.
            import traceback; traceback.print_exc()
            print(
                Fore.RED
                + f"\n[Shell] recovered from an error: {type(_turn_exc).__name__}: {_turn_exc}"
                + "\n  (your session is intact -- only this command was aborted)"
                + Style.RESET_ALL
            )
            try:
                memory.append_event(
                    "turn_error_recovered",
                    text=f"{type(_turn_exc).__name__}: {_turn_exc}",
                    tags=["error"],
                    provenance={"traceback": _tb.format_exc()[-1500:]},
                )
            except Exception:
                pass
            continue

if __name__ == "__main__":
    main()
