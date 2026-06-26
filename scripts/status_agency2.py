"""Report agency2 status from cached JSON without loading a model.

This intentionally checks the JSON mode field instead of trusting filenames:

  python scripts/status_agency2.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "invariants" / "out"
SHORT = "Llama-3.1-8B-Instruct"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def best_line(label: str, sweep: dict) -> str:
    best = sweep["best"]
    return (
        f"{label}: best clean={pct(best['clean'])} at "
        f"L{best['L']} alpha={best['alpha']} "
        f"(flip={pct(best['flip'])}, fluent={pct(best['fluent'])})"
    )


def describe_partial(path: Path):
    if not path.exists():
        return
    data = load(path)
    mode = data.get("mode", "unknown")
    status = data.get("status", "unknown")
    updated = data.get("updated_at", "unknown")
    print(f"partial checkpoint: {path.name}")
    print(f"  mode={mode}, status={status}, updated_at={updated}")


def main():
    calib_path = OUT / f"agency2_calibration_{SHORT}.json"
    full_path = OUT / f"agency2_{SHORT}.json"
    duplicate_path = OUT / f"agency2_{SHORT}.calibration-duplicate.json"

    print("agency2 status")
    print("--------------")

    if calib_path.exists():
        calib = load(calib_path)
        print(f"calibration file: present ({calib.get('mode', 'unknown')})")
        print(best_line("calibration", calib["calibration"]))
    else:
        print("calibration file: absent")

    if full_path.exists():
        full = load(full_path)
        mode = full.get("mode", "unknown")
        print(f"full contrast file: present ({mode})")
        if mode == "full":
            print(best_line("null", full["null"]))
            print(best_line("hedge", full["hedge"]))
            print(f"calibration_valid={full.get('calibration_valid')}")
        else:
            print("  warning: this is not a completed full contrast")
    else:
        print("full contrast file: absent")

    if duplicate_path.exists():
        print(f"preserved duplicate: {duplicate_path.name}")

    describe_partial(OUT / f"agency2_calibration_{SHORT}.partial.json")
    describe_partial(OUT / f"agency2_{SHORT}.partial.json")


if __name__ == "__main__":
    main()
