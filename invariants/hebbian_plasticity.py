"""Hebbian plasticity for the gravity field.

Standard field physics keeps probe directions fixed and varies mass. That is
correct for the static component of concept geometry, but it leaves the field
blind to *which concepts have worked together*. This module adds the missing
layer: an outcome-weighted co-activation matrix that continuously reshapes the
field's *subjective distances*, making probes that reliably fire together on
productive turns feel geometrically closer to each other (LTP), and probes
that co-fire on unproductive turns feel farther apart (LTD).

The result is a living cognitive map whose topology is earned by evidence
rather than set by hand.

                     Mathematical core
                     -----------------
After every scored turn, for each pair of probes (A, B) that both produced a
non-trivial signal, the symmetric co-activation entry is updated:

    Δ(A,B) = learning_rate * signal_A * signal_B * outcome

where outcome is the sense score for that turn (positive = productive turn).
Because probe signals are centered (zero mean), the product signal_A * signal_B
is:
  - positive when both probes fire in the same direction → LTP on good turns,
    LTD on bad ones.
  - negative when they fire in opposite directions → LTD on good turns (the
    field should separate concepts that were productive in orthogonal roles).

This is the outcome-weighted covariance rule, a signed extension of classical
Hebb. Entries decay each turn toward zero so only sustained co-activation
maintains a bond.

                     Integration surface
                     -------------------
The caller is the shell turn loop in interactive_phenomenality.py:

    # Initialization (once, near TriggerTuner):
    plasticity = HebbianPlasticityMatrix(path=..., **knobs)

    # After each scored turn:
    probe_sigs = {name: turn_row[f"probe_{name}"]
                  for name in probes if f"probe_{name}" in turn_row}
    plasticity.update(probe_sigs, outcome=sense_score_value)
    plasticity.decay()

    # Pass into the field hook builder:
    handles, desc = build_gravity_field_handles(
        model, probes, tuner, cap_fraction=cap,
        plasticity=plasticity, plasticity_signals=probe_sigs,
    )

The live field compiles local relationship stencils from this matrix and
measures their support against each current token. Distance deformation fades
with distance according to the radius knob. Legacy scalar distance_scale and
mass_multiplier methods remain available to older callers.

                     Knobs (all runtime-tunable via :plasticity set)
                     ------------------------------------------------
    learning_rate     How fast new co-activations build bonds.  Default 0.05.
    decay_rate        Per-turn exponential decay toward zero.   Default 0.005.
    strength          How much potentiation shifts distances.   Default 0.40.
    radius            Distance range of local deformation.      Default 0.35.
    mass_strength     Fraction of strength applied to mass.     Default 0.50.
    max_potentiation  Cap on LTP distance reduction (0-1).      Default 0.70.
    max_depression    Cap on LTD distance increase (as fraction). Default 0.40.
    min_signal        Probe signal magnitude to participate.    Default 0.005.
    enabled           Master on/off.                           Default True.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core matrix
# ---------------------------------------------------------------------------

class HebbianPlasticityMatrix:
    """Outcome-weighted co-activation matrix that modulates field geometry.

    The matrix is symmetric: potentiation(A, B) == potentiation(B, A).
    Positive values → LTP (probes feel closer in field space).
    Negative values → LTD (probes feel farther apart).
    """

    VERSION = 1

    def __init__(
        self,
        path: Optional[str] = None,
        learning_rate: float = 0.05,
        decay_rate: float = 0.005,
        strength: float = 0.40,
        mass_strength: float = 0.50,
        max_potentiation: float = 0.70,
        max_depression: float = 0.40,
        min_signal: float = 0.005,
        enabled: bool = True,
        radius: float = 0.35,
    ):
        self.path = path
        self.learning_rate = float(learning_rate)
        self.decay_rate = float(decay_rate)
        self.strength = float(strength)
        self.mass_strength = float(mass_strength)
        self.max_potentiation = float(max_potentiation)
        self.max_depression = float(max_depression)
        self.min_signal = float(min_signal)
        self.enabled = bool(enabled)
        self.radius = float(radius)
        for knob in ("learning_rate", "decay_rate", "strength", "mass_strength",
                     "max_potentiation", "max_depression", "min_signal", "radius"):
            self.set_knob(knob, getattr(self, knob))

        # Symmetric co-activation table.
        # Key: (min_name, max_name) tuple for canonical ordering.
        # Value: signed float in [-max_depression, max_potentiation].
        self._cofire: Dict[Tuple[str, str], float] = {}

        # Lightweight audit trail: last N update summaries.
        self._history: List[dict] = []
        self._history_max = 200

        # Monotonic turn counter for telemetry.
        self._turn: int = 0

        self._load()

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(a: str, b: str) -> Tuple[str, str]:
        """Canonical symmetric key — order-independent."""
        return (a, b) if a <= b else (b, a)

    # ------------------------------------------------------------------
    # Core update: called once per scored turn
    # ------------------------------------------------------------------

    def update(
        self,
        probe_signals: Dict[str, float],
        outcome: float,
    ) -> int:
        """Record co-activations for this turn, weighted by outcome.

        probe_signals: {probe_name: centered_score} for every scored probe.
        outcome: sense score for the turn (positive = productive).

        Returns the number of pairs updated.
        """
        if not self.enabled or not probe_signals:
            return 0
        if not math.isfinite(outcome):
            return 0

        # Filter to probes with non-trivial signal.
        active = {
            name: sig
            for name, sig in probe_signals.items()
            if math.isfinite(sig) and abs(sig) >= self.min_signal
        }
        names = sorted(active)  # sorted for determinism
        updated = 0

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sig_a = active[a]
                sig_b = active[b]

                # Outcome-weighted covariance:
                #   positive when probes agree in direction AND outcome is good
                #   negative otherwise (LTD)
                delta = self.learning_rate * sig_a * sig_b * outcome

                key = self._key(a, b)
                old = self._cofire.get(key, 0.0)
                new = max(
                    -self.max_depression,
                    min(self.max_potentiation, old + delta),
                )
                if abs(new) < 1e-9:
                    self._cofire.pop(key, None)
                else:
                    self._cofire[key] = new
                updated += 1

        self._turn += 1
        self._record_history(len(active), updated, outcome)
        return updated

    # ------------------------------------------------------------------
    # Decay: called every turn (even with no active probes)
    # ------------------------------------------------------------------

    def decay(self) -> int:
        """Exponential per-turn decay toward zero.  Returns pairs pruned."""
        if not self.enabled or self.decay_rate <= 0.0:
            return 0
        factor = 1.0 - self.decay_rate
        to_prune = []
        for key, val in self._cofire.items():
            new_val = val * factor
            if abs(new_val) < 1e-9:
                to_prune.append(key)
            else:
                self._cofire[key] = new_val
        for key in to_prune:
            del self._cofire[key]
        return len(to_prune)

    # ------------------------------------------------------------------
    # Field integration API
    # ------------------------------------------------------------------

    def potentiation(self, a: str, b: str) -> float:
        """Signed potentiation for pair (a, b).
        Positive → LTP (feel closer).  Negative → LTD (feel farther).
        """
        value = self._cofire.get(self._key(a, b), 0.0)
        controller = getattr(self, 'target_controller', None)
        if self.enabled and controller is not None:
            value += controller.adjustment(a, b)
        return max(-self.max_depression, min(self.max_potentiation, value))

    def distance_scale(
        self,
        probe_name: str,
        current_signals: Dict[str, float],
    ) -> float:
        """Compute the distance multiplier for probe_name given current signals.

        When the current generation state is near probe A (A has a high positive
        signal this chunk) and probe B has strong LTP history with A, the
        effective distance to B is *reduced* — B feels closer, so its pull
        increases.  Conversely, LTD history makes B feel farther.

        Returns a scale factor applied to raw cosine distance before the force
        kernel: values < 1 tighten the well, > 1 widen it.

        d_eff(token, probe_B) = d_cosine * distance_scale(B, current_signals)
        """
        if not self.enabled or not current_signals or not self._cofire:
            return 1.0

        weighted_pot = 0.0
        total_weight = 0.0
        for other, sig in current_signals.items():
            if other == probe_name or sig <= 0.0:
                continue
            p = self.potentiation(probe_name, other)
            if p == 0.0:
                continue
            w = sig  # weight by signal strength of the co-activating probe
            weighted_pot += p * w
            total_weight += w

        if total_weight < 1e-9:
            return 1.0

        norm_pot = weighted_pot / total_weight
        # scale < 1 → closer (LTP); scale > 1 → farther (LTD)
        scale = 1.0 - self.strength * norm_pot
        return max(0.2, min(2.0, scale))

    def mass_multiplier(
        self,
        probe_name: str,
        current_signals: Dict[str, float],
    ) -> float:
        """Hebbian gating multiplier on effective mass.

        When historically co-active probes are currently firing, boost the
        effective mass of probe_name so it pulls harder.  When historically
        anti-correlated probes are firing, reduce mass.

        Returns a multiplier on the probe's static mass.
        """
        if not self.enabled or not current_signals or not self._cofire:
            return 1.0

        pot_sum = 0.0
        for other, sig in current_signals.items():
            if other == probe_name:
                continue
            p = self.potentiation(probe_name, other)
            # Weight by absolute signal — both directions of firing can gate
            pot_sum += p * abs(sig)

        mult = 1.0 + self.strength * self.mass_strength * pot_sum
        return max(0.1, min(3.0, mult))

    def effective_distance(self, probe_name, distance, current_signals):
        """Fallback for direct field calls without a compiled live stencil."""
        from invariants.field_geometry import local_distance
        if not self.enabled:
            return distance
        support = 0.
        total = 0.
        for other, signal in (current_signals or {}).items():
            if other != probe_name and math.isfinite(signal) and signal > 0:
                support += self.potentiation(probe_name, other) * signal
                total += signal
        return local_distance(distance, support / max(1., total),
                              strength=self.strength, radius=self.radius)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def top_pairs(self, n: int = 10) -> List[dict]:
        """Return the n strongest pairs by absolute potentiation, desc."""
        pairs = [
            {
                "a": k[0],
                "b": k[1],
                "potentiation": v,
                "kind": "LTP" if v > 0 else "LTD",
            }
            for k, v in self._cofire.items()
        ]
        pairs.sort(key=lambda x: -abs(x["potentiation"]))
        return pairs[:n]

    def pair_info(self, a: str, b: str) -> dict:
        """Return full info for one pair."""
        p = self.potentiation(a, b)
        return {
            "a": a,
            "b": b,
            "potentiation": p,
            "kind": "LTP" if p > 0 else ("LTD" if p < 0 else "neutral"),
            "distance_effect": f"d × {1.0 - self.strength * p:.3f}" if p else "none",
            "mass_effect": f"m × {1.0 + self.strength * self.mass_strength * p:.3f}" if p else "none",
        }

    def status(self) -> dict:
        """Summary statistics for display."""
        vals = list(self._cofire.values())
        ltp = [v for v in vals if v > 0]
        ltd = [v for v in vals if v < 0]
        return {
            "enabled": self.enabled,
            "pairs": len(vals),
            "ltp_pairs": len(ltp),
            "ltd_pairs": len(ltd),
            "max_ltp": max(ltp, default=0.0),
            "max_ltd": min(ltd, default=0.0),
            "mean_abs": sum(abs(v) for v in vals) / max(len(vals), 1),
            "turns": self._turn,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "strength": self.strength,
            "mass_strength": self.mass_strength,
            "max_potentiation": self.max_potentiation,
            "max_depression": self.max_depression,
            "min_signal": self.min_signal,
            "radius": self.radius,
        }

    def format_status(self) -> str:
        """Human-readable status block."""
        s = self.status()
        lines = [
            f"[Plasticity] {'ON' if s['enabled'] else 'OFF'} | "
            f"turns={s['turns']} | pairs={s['pairs']} "
            f"(LTP={s['ltp_pairs']}, LTD={s['ltd_pairs']})",
            f"  max LTP={s['max_ltp']:+.4f}  max LTD={s['max_ltd']:+.4f}  "
            f"mean |p|={s['mean_abs']:.4f}",
            f"  lr={s['learning_rate']:.4f}  decay={s['decay_rate']:.4f}  "
            f"strength={s['strength']:.3f}  mass_strength={s['mass_strength']:.3f}",
        ]
        top = self.top_pairs(5)
        if top:
            lines.append("  Top pairs:")
            for pair in top:
                lines.append(
                    f"    {pair['a']} ↔ {pair['b']}: "
                    f"{pair['potentiation']:+.4f} [{pair['kind']}]"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def reset(self, a: Optional[str] = None, b: Optional[str] = None) -> int:
        """Clear all potentiation, or a specific pair.  Returns entries removed."""
        if a is not None and b is not None:
            key = self._key(a, b)
            removed = 1 if key in self._cofire else 0
            self._cofire.pop(key, None)
            return removed
        removed = len(self._cofire)
        self._cofire.clear()
        return removed

    def set_knob(self, name: str, value: float) -> bool:
        """Set a named knob.  Returns True if recognized."""
        knobs = {
            "learning_rate", "decay_rate", "strength", "mass_strength",
            "max_potentiation", "max_depression", "min_signal", "radius",
        }
        if name not in knobs:
            return False
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Plasticity knobs must be finite and nonnegative")
        if name in {"decay_rate", "max_potentiation", "max_depression"} and value > 1:
            raise ValueError(f"{name} must be between 0 and 1")
        if name == "radius" and not .02 <= value <= 2.:
            raise ValueError("radius must be between 0.02 and 2 cosine-distance units")
        setattr(self, name, float(value))
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> bool:
        """Atomically save to self.path.  Returns True on success."""
        if not self.path:
            return False
        try:
            os.makedirs(os.path.dirname(os.fspath(self.path)) or ".", exist_ok=True)
            payload = {
                "version": self.VERSION,
                "saved_at": time.time(),
                "turns": self._turn,
                "knobs": {
                    "learning_rate": self.learning_rate,
                    "decay_rate": self.decay_rate,
                    "strength": self.strength,
                    "mass_strength": self.mass_strength,
                    "max_potentiation": self.max_potentiation,
                    "max_depression": self.max_depression,
                    "min_signal": self.min_signal,
                    "radius": self.radius,
                    "enabled": self.enabled,
                },
                # Serialise as list of [a, b, value] for JSON compactness.
                "cofire": [[k[0], k[1], v] for k, v in self._cofire.items()],
            }
            tmp = os.fspath(self.path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except Exception:
            return False

    def _load(self) -> None:
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                return
            self._turn = int(payload.get("turns", 0))
            knobs = payload.get("knobs") or {}
            for knob in (
                "learning_rate", "decay_rate", "strength", "mass_strength",
                "max_potentiation", "max_depression", "min_signal", "radius",
            ):
                if knob in knobs:
                    self.set_knob(knob, float(knobs[knob]))
            if "enabled" in knobs:
                self.enabled = bool(knobs["enabled"])
            for entry in payload.get("cofire") or []:
                if len(entry) == 3:
                    a, b, v = entry
                    value = float(v)
                    if math.isfinite(value) and a != b:
                        self._cofire[self._key(str(a), str(b))] = max(
                            -self.max_depression, min(self.max_potentiation, value))
        except Exception:
            pass  # corrupt or missing file → start fresh

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_history(self, n_active: int, n_pairs: int, outcome: float) -> None:
        entry = {
            "turn": self._turn,
            "active_probes": n_active,
            "pairs_updated": n_pairs,
            "outcome": round(outcome, 4),
            "total_pairs": len(self._cofire),
        }
        self._history.append(entry)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max :]

    def recent_history(self, n: int = 10) -> List[dict]:
        return self._history[-n:]


# ---------------------------------------------------------------------------
# Field integration patch for _gravity_pull
# ---------------------------------------------------------------------------
# This section documents exactly where and how to wire plasticity into
# _gravity_pull in interactive_phenomenality.py.
#
# Target function signature (add two optional kwargs):
#
#   def _gravity_pull(h_float, entries, G, eps=GRAVITY_EPS, doppler=0.0,
#                     vstate=None, vkey=None, taus=None, names=None, telem=None,
#                     time_index=0,
#                     plasticity=None,            # <-- ADD
#                     plasticity_signals=None):   # <-- ADD
#
# Inside the force loop, after computing `distance` on line 1677:
#
#   # Original:
#   distance = (1.0 - cos).clamp_min(0.0)
#   w = (m * g_i * tau_i) * _gravity_kernel(distance, shape, eps=eps)
#
#   # With plasticity:
#   distance = (1.0 - cos).clamp_min(0.0)
#   if plasticity is not None and plasticity_signals and _name in probes_set:
#       d_scale = plasticity.distance_scale(_name, plasticity_signals)
#       m_scale = plasticity.mass_multiplier(_name, plasticity_signals)
#       distance = (distance * d_scale).clamp_min(0.0)
#       m_eff = m * m_scale
#   else:
#       m_eff = m
#   w = (m_eff * g_i * tau_i) * _gravity_kernel(distance, shape, eps=eps)
#
# Pass-through from build_gravity_field_handles:
#
#   def hook(module, inp, out, entries=entries, ...,
#            plasticity=plasticity, plasticity_signals=plasticity_signals):
#       h = out[0] if isinstance(out, tuple) else out
#       with torch.no_grad():
#           add = _gravity_pull(
#               h.detach().float(), entries, 1.0, ...,
#               plasticity=plasticity,
#               plasticity_signals=plasticity_signals,
#           )
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shell command handler (drop into the :plasticity command block)
# ---------------------------------------------------------------------------
# Recommended placement: after the :steer block in the main command dispatch.
#
# elif cmd == ":plasticity":
#     _handle_plasticity_command(sargs, plasticity, probes)
#
# where sargs = remaining words after ":plasticity".


class PlasticityOwners:
    """Keep learned geometry and recent stimulation local to each shell tuner."""

    def __init__(self):
        self._owners = {}

    def for_tuner(self, tuner):
        path = Path(tuner.path).resolve().with_suffix(".plasticity.json")
        key = str(path)
        if key not in self._owners:
            from invariants.target_relations import TargetRelations
            matrix = HebbianPlasticityMatrix(path=path)
            matrix.target_controller = TargetRelations(path.with_suffix('.relations.json'))
            from invariants.indirect_steering import IndirectSteering
            matrix.indirect_controller = IndirectSteering(path.with_suffix('.indirect.json'))
            from invariants.exposure_language import ExposureLanguage
            matrix.exposure_language = ExposureLanguage(path.with_suffix('.language.json'))
            from invariants.other_probes import OtherReference
            matrix.other_reference = OtherReference(path.with_suffix('.others.json'))
            from invariants.probe_behavior import ProbeBehavior
            matrix.probe_behavior = ProbeBehavior(path.with_suffix('.behavior.json'))
            self._owners[key] = (matrix, {})
        # Direction selection and generation use the same owner-local policy.
        tuner.indirect_controller = self._owners[key][0].indirect_controller
        return self._owners[key]

    def observe(self, tuner, turn_row, outcome, probes=None):
        matrix, signals = self.for_tuner(tuner)
        matrix.target_controller.observe(turn_row)
        matrix.indirect_controller.observe(turn_row, probes=probes)
        if probes is not None and turn_row.get('probe_measurement', {}).get('source') == 'live_assistant_prediction':
            # Future saved evidence can distinguish reminted directions and
            # geometry changes from measurements made by the same instrument.
            turn_row['probe_definition_fingerprints'] = dict(matrix.indirect_controller.fingerprints)
            turn_row['indirect_steering'] = {
                'roles': dict(matrix.indirect_controller.roles),
                'feedback': dict(matrix.indirect_controller.feedback),
                'joint_plan': dict(matrix.indirect_controller.joint_plan),
            }
        signals.clear()
        for key, value in turn_row.items():
            if not key.startswith("probe_"):
                continue
            if isinstance(value, (dict, list, tuple, set, bool)):
                continue
            try:
                fval = float(value)
                if math.isfinite(fval):
                    signals[key[6:]] = fval
            except (TypeError, ValueError):
                continue
        if outcome is None:
            return 0
        try:
            f_outcome = float(outcome)
            if not math.isfinite(f_outcome):
                return 0
        except (TypeError, ValueError):
            return 0
        count = matrix.update(signals, f_outcome)
        matrix.decay()
        if matrix.enabled and not matrix.save():
            raise OSError(f"Could not save plasticity to {matrix.path}")
        return count


def handle_plasticity_command(
    sargs: list,
    plasticity: HebbianPlasticityMatrix,
    probes: dict,
) -> str:
    """Dispatch handler for :plasticity shell command.

    Returns a string to print, or raises ValueError for bad args.

    Subcommands:
      :plasticity                      — full status
      :plasticity top [n]              — top n pairs
      :plasticity pair <A> <B>         — specific pair info
      :plasticity history [n]          — recent update history
      :plasticity reset                — clear all potentiation
      :plasticity reset <A> <B>        — clear one pair
      :plasticity on                   — enable
      :plasticity off                  — disable
      :plasticity set <knob> <value>   — tune a knob
      :plasticity save                 — flush to disk
    """
    if not sargs:
        return plasticity.format_status()

    sub = sargs[0].lower()

    if sub == "top":
        n = int(sargs[1]) if len(sargs) > 1 else 10
        pairs = plasticity.top_pairs(n)
        if not pairs:
            return "[Plasticity] No potentiated pairs yet."
        lines = [f"[Plasticity] Top {len(pairs)} pairs:"]
        for p in pairs:
            lines.append(
                f"  {p['a']} ↔ {p['b']}: {p['potentiation']:+.4f} [{p['kind']}]"
            )
        return "\n".join(lines)

    elif sub == "pair":
        if len(sargs) < 3:
            raise ValueError(":plasticity pair <A> <B>")
        a, b = sargs[1], sargs[2]
        info = plasticity.pair_info(a, b)
        lines = [
            f"[Plasticity] {info['a']} ↔ {info['b']}",
            f"  potentiation: {info['potentiation']:+.4f} [{info['kind']}]",
            f"  distance effect: {info['distance_effect']}",
            f"  mass effect:     {info['mass_effect']}",
        ]
        return "\n".join(lines)

    elif sub == "history":
        n = int(sargs[1]) if len(sargs) > 1 else 10
        hist = plasticity.recent_history(n)
        if not hist:
            return "[Plasticity] No history yet."
        lines = [f"[Plasticity] Recent {len(hist)} updates:"]
        for h in hist:
            lines.append(
                f"  turn={h['turn']} active={h['active_probes']} "
                f"pairs={h['pairs_updated']} outcome={h['outcome']:+.3f} "
                f"total_bonds={h['total_pairs']}"
            )
        return "\n".join(lines)

    elif sub == "reset":
        if len(sargs) == 3:
            removed = plasticity.reset(sargs[1], sargs[2])
            return f"[Plasticity] Cleared pair {sargs[1]} ↔ {sargs[2]} ({removed} entry)."
        elif len(sargs) == 1:
            removed = plasticity.reset()
            return f"[Plasticity] Cleared {removed} potentiated pair(s)."
        else:
            raise ValueError(":plasticity reset  OR  :plasticity reset <A> <B>")

    elif sub == "on":
        plasticity.enabled = True
        return "[Plasticity] Enabled."

    elif sub == "off":
        plasticity.enabled = False
        return "[Plasticity] Disabled (field physics unchanged)."

    elif sub == "save":
        ok = plasticity.save()
        return (
            f"[Plasticity] Saved to {plasticity.path}."
            if ok
            else "[Plasticity] Save failed (no path configured)."
        )

    elif sub == "set":
        if len(sargs) < 3:
            knobs = (
                "learning_rate", "decay_rate", "strength", "mass_strength",
                "max_potentiation", "max_depression", "min_signal", "radius",
            )
            return "[Plasticity] Knobs: " + ", ".join(knobs)
        knob, raw = sargs[1], sargs[2]
        try:
            val = float(raw)
        except ValueError:
            raise ValueError(f":plasticity set {knob} <float>")
        if not plasticity.set_knob(knob, val):
            raise ValueError(f"Unknown plasticity knob: {knob!r}")
        return f"[Plasticity] {knob} = {val}"

    else:
        raise ValueError(
            f"Unknown :plasticity subcommand {sub!r}. "
            "Use: top, pair, history, reset, on, off, save, set."
        )
