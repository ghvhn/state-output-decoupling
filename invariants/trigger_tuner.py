"""Live-tunable triggers: one surface for every threshold and coefficient.

No trigger owns a frozen magic number. Each routes through the tuner, which
logs the underlying signal every turn -- even when it does not fire -- so a
threshold can be read against its own observed distribution and moved live, or
calibrated to a percentile of what it has actually seen. Coefficients (like a
steering alpha) are held here too, adjustable mid-session. State persists to
JSON so tuning survives sessions.

The point (Gavin's spine): a threshold calibrated to a percentile of its own
history cannot be wrong about the scale -- only about the fraction, which is the
one honest knob left to turn. Sensors first, asserted constants never.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

DEFAULT_PATH = Path(
    os.environ.get(
        "TRIGGER_TUNER_FILE",
        str(Path(__file__).parent / "out" / "trigger_tuner.json"),
    )
)
HISTORY = 200


class Trigger:
    def __init__(self, name, value, kind="threshold", comparator=">=", history=HISTORY):
        self.name = name
        self.value = float(value)
        self.kind = kind              # "threshold" (fires vs signal) or "coefficient" (bare knob)
        self.comparator = comparator  # ">=" or "<="
        self.signals = deque(maxlen=history)
        self.observed = 0
        self.fired = 0

    def _fires(self, signal):
        if self.comparator == "<=":
            return signal <= self.value
        return signal >= self.value

    def observe(self, signal):
        signal = float(signal)
        self.signals.append(signal)
        self.observed += 1
        fired = self._fires(signal)
        if fired:
            self.fired += 1
        return fired

    def calibrate(self, percentile):
        if not self.signals:
            return self.value
        data = sorted(self.signals)
        pct = min(max(float(percentile), 0.0), 100.0)
        idx = int(round((pct / 100.0) * (len(data) - 1)))
        self.value = data[idx]
        return self.value

    def stats(self):
        data = sorted(self.signals)
        n = len(data)
        return {
            "name": self.name,
            "kind": self.kind,
            "value": round(self.value, 4),
            "comparator": self.comparator,
            "observed": self.observed,
            "fired": self.fired,
            "fire_rate": round(self.fired / self.observed, 3) if self.observed else None,
            "signal_min": round(data[0], 4) if n else None,
            "signal_med": round(data[n // 2], 4) if n else None,
            "signal_max": round(data[-1], 4) if n else None,
            "n_signals": n,
        }

    def to_dict(self):
        return {
            "value": self.value,
            "kind": self.kind,
            "comparator": self.comparator,
            "observed": self.observed,
            "fired": self.fired,
            "signals": list(self.signals),
        }

    @classmethod
    def from_dict(cls, name, d):
        t = cls(name, d.get("value", 0.0), d.get("kind", "threshold"), d.get("comparator", ">="))
        t.observed = int(d.get("observed", 0))
        t.fired = int(d.get("fired", 0))
        for s in d.get("signals", []):
            t.signals.append(float(s))
        return t


class TriggerTuner:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.triggers: dict[str, Trigger] = {}
        self.load()

    def register(self, name, default, kind="threshold", comparator=">="):
        """Register a trigger if absent. A persisted value wins over the default,
        so live tuning is never clobbered by a code default on restart."""
        if name not in self.triggers:
            self.triggers[name] = Trigger(name, default, kind, comparator)
        return self.triggers[name]

    def observe(self, name, signal, default=None):
        """Log a signal and return whether it fires. Call EVERY turn, even when
        the raw event is absent (signal 0), so the distribution is visible."""
        t = self.triggers.get(name)
        if t is None:
            t = self.register(name, signal if default is None else default)
        fired = t.observe(signal)
        self.save()
        return fired

    def get(self, name, default=0.0):
        t = self.triggers.get(name)
        return t.value if t is not None else default

    def set(self, name, value):
        t = self.triggers.get(name) or self.register(name, value)
        t.value = float(value)
        self.save()
        return t.value

    def calibrate(self, name, percentile):
        t = self.triggers.get(name)
        if t is None:
            return None
        v = t.calibrate(percentile)
        self.save()
        return v

    def summary(self):
        return [self.triggers[n].stats() for n in sorted(self.triggers)]

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {name: t.to_dict() for name, t in self.triggers.items()}
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for name, d in data.items():
                    self.triggers[name] = Trigger.from_dict(name, d)
            except Exception:
                self.triggers = {}
