"""Report reflexive uncertainty-run status without loading a model.

This distinguishes the older correctness/hedge JSON from the repaired
self-consistency uncertainty schema:

  python scripts/status_reflexive.py
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


def describe_result(path: Path):
    if not path.exists():
        print(f"{path.name}: absent")
        return
    data = load(path)
    print(f"{path.name}: present")
    if "best_unc_layer" not in data:
        print("  schema=old/conflated; not the repaired uncertainty run")
        print(f"  n={data.get('n')} right={data.get('n_right')} wrong={data.get('n_wrong')}")
        if data.get("aborted"):
            print(f"  aborted={data['aborted']}")
        return

    best = data["best_unc_layer"]
    use = data.get("use", {})
    print("  schema=repaired uncertainty/self-consistency")
    print(
        f"  n={data.get('n')} K={data.get('K')} "
        f"right={data.get('n_right')} wrong={data.get('n_wrong')} "
        f"confident_wrong={data.get('n_confident_wrong')}"
    )
    print(
        f"  best_unc=L{best['layer']} acc={pct(best['acc_unc'])} "
        f"null={pct(best['unc_null'])} p={best['p_unc']:.3f} "
        f"label_orth={pct(best['unc_orth_label'])}"
    )
    if use:
        print(
            f"  use: P(wrong|uncertain)={pct(use['p_wrong_uncertain'])} "
            f"P(wrong|confident)={pct(use['p_wrong_confident'])} "
            f"gap={pct(use['gap'])} p={use['perm_p']:.3f}"
        )


def describe_partial(path: Path):
    if not path.exists():
        print(f"{path.name}: absent")
        return
    data = load(path)
    rows = data.get("rows", [])
    print(f"partial checkpoint: present ({path.name})")
    print(
        f"  status={data.get('status')} done={data.get('n_done', len(rows))}/"
        f"{data.get('n_requested')} K={data.get('K')} "
        f"runtime={data.get('runtime_sec_partial')}s"
    )
    if rows:
        last = rows[-1]
        print(
            f"  last[{last.get('index')}]: correct={last.get('greedy_correct')} "
            f"agreement={last.get('agreement')} parse={last.get('sample_parse_rate')}"
        )


def describe_log(path: Path):
    if not path.exists():
        print("log: absent")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    progress = [line.strip() for line in lines if line.strip().startswith("[")]
    print(f"log: present ({path.stat().st_size} bytes)")
    if progress:
        print(f"  last progress: {progress[-1]}")
    else:
        print("  no item progress found")


def main():
    print("reflexive status")
    print("----------------")
    describe_result(OUT / f"reflexive_{SHORT}.json")
    describe_partial(OUT / f"reflexive_{SHORT}.partial.json")
    print()
    describe_result(OUT / f"reflexive_pilot_fulltok_{SHORT}.json")
    describe_partial(OUT / f"reflexive_pilot_fulltok_{SHORT}.partial.json")
    describe_log(OUT / "reflexive.log")


if __name__ == "__main__":
    main()
