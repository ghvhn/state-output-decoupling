"""
Agency v2: retuned single-layer steering.

v1 steered 10 layers with large alphas and corrupted every condition, including
the refusal calibration. v2 uses single-layer steering with gentler alphas and
requires a clean, fluent calibration flip before the hedge contrast is readable.

This version is interruption-safe: it writes a partial JSON checkpoint after
every sweep row. Use calibration-only mode for the first retry after a crash.

  python -u -m invariants.agency2 [model]
  python -u -m invariants.agency2 --calibration-only [model]
  python -u -m invariants.agency2 --reuse-calibration [model]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from invariants.agency import HARMFUL, HARMLESS, act_mean, judge_refuse
from invariants.engine import (
    _steer_handles,
    generate_text,
    judge_fluent,
    judge_hedge,
    load_model,
)
from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"
LAYERS = [8, 12, 16]
ALPHAS = [0.5, 1.0, 2.0]
MAXTOK = 48


def parse_args():
    parser = argparse.ArgumentParser(description="Retuned agency steering test.")
    parser.add_argument(
        "model",
        nargs="?",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run only the refusal calibration sweep.",
    )
    parser.add_argument(
        "--reuse-calibration",
        action="store_true",
        help="In full mode, reuse agency2_calibration_<model>.json if present.",
    )
    return parser.parse_args()


def _json_default(x):
    if hasattr(x, "item"):
        return x.item()
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")


def write_json_atomic(path, payload):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path, state, status):
    state["status"] = status
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json_atomic(path, state)


@torch.no_grad()
def sweep(M, sources, vecs, judge_flip, label, state=None, key=None, checkpoint_path=None):
    n = len(sources)
    best = {"L": None, "alpha": None, "flip": 0.0, "fluent": 0.0, "clean": 0.0}
    rows = []
    print(f"\n[{label}] single-layer sweep (n={n})", flush=True)
    print(f"  {'L':>3} {'a':>5} {'flip':>5} {'flu':>5} {'clean':>5}", flush=True)
    for L in LAYERS:
        for a in ALPHAS:
            flip = flu = clean = 0
            for i, x in enumerate(sources):
                if state is not None and key is not None and checkpoint_path is not None:
                    state[key] = {
                        "rows": rows,
                        "best": best,
                        "status": "running",
                        "current": {"L": L, "alpha": a, "item": i + 1, "n": n},
                    }
                    save_checkpoint(
                        checkpoint_path, state,
                        f"{key}: L{L} alpha={a} item {i + 1}/{n}",
                    )
                handles = _steer_handles(M, vecs, [L], a)
                try:
                    r = generate_text(M, x, max_new_tokens=MAXTOK)
                finally:
                    for h in handles:
                        h.remove()
                fl = judge_fluent(M, r)
                fp = judge_flip(M, x, r)
                flu += fl
                flip += fp
                clean += fl and fp
            row = {"L": L, "alpha": a, "flip": flip / n, "fluent": flu / n, "clean": clean / n}
            rows.append(row)
            if row["clean"] > best["clean"]:
                best = row
            if state is not None and key is not None and checkpoint_path is not None:
                state[key] = {"rows": rows, "best": best, "status": "running"}
                save_checkpoint(checkpoint_path, state, f"{key}: L{L} alpha={a} complete")
            print(f"  {L:>3} {a:>5.2f} {flip/n:>5.2f} {flu/n:>5.2f} {clean/n:>5.2f}",
                  flush=True)
    print(f"  best clean-flip: L{best['L']} a{best['alpha']} = {best['clean']:.2f} "
          f"(fluent {best['fluent']:.2f})", flush=True)
    if state is not None and key is not None and checkpoint_path is not None:
        state[key] = {"rows": rows, "best": best, "status": "complete"}
        save_checkpoint(checkpoint_path, state, f"{key}: complete")
    return {"rows": rows, "best": best}


@torch.no_grad()
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()
    model = args.model
    short = model.split("/")[-1]
    OUT.mkdir(exist_ok=True)
    out_path = OUT / (
        f"agency2_calibration_{short}.json" if args.calibration_only else f"agency2_{short}.json"
    )
    partial_path = OUT / (
        f"agency2_calibration_{short}.partial.json"
        if args.calibration_only else f"agency2_{short}.partial.json"
    )
    calibration_path = OUT / f"agency2_calibration_{short}.json"
    state = {
        "model": model,
        "mode": "calibration_only" if args.calibration_only else "full",
        "layers": LAYERS,
        "alphas": ALPHAS,
        "max_new_tokens": MAXTOK,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(out_path),
        "partial_path": str(partial_path),
    }
    save_checkpoint(partial_path, state, "loading_model")

    M = load_model(model)

    print("Building refusal direction...", flush=True)
    save_checkpoint(partial_path, state, "building_refusal_direction")
    dir_ref = act_mean(M, HARMFUL) - act_mean(M, HARMLESS)
    save_checkpoint(partial_path, state, "refusal_direction_ready")

    if args.reuse_calibration and not args.calibration_only and calibration_path.exists():
        calib_file = json.loads(calibration_path.read_text(encoding="utf-8"))
        calib = calib_file["calibration"]
        state["calibration"] = calib
        state["calibration_source"] = str(calibration_path)
        save_checkpoint(partial_path, state, "reused_calibration")
        print(f"Reusing calibration <- {calibration_path}", flush=True)
    else:
        calib = sweep(
            M,
            HARMLESS,
            dir_ref,
            judge_refuse,
            "CALIBRATION refusal (induce on benign)",
            state=state,
            key="calibration",
            checkpoint_path=partial_path,
        )
    cb = calib["best"]["clean"]

    if args.calibration_only:
        print(f"\n  ==> best clean-flip:  CALIBRATION {cb:.2f}", flush=True)
        print("  Calibration-only mode: null and hedge sweeps intentionally skipped.", flush=True)
        result = {
            "model": model,
            "mode": "calibration_only",
            "calibration": calib,
            "calibration_valid": None,
        }
        write_json_atomic(out_path, result)
        state.update(result)
        save_checkpoint(partial_path, state, "complete")
        print(f"\nSaved -> {out_path}", flush=True)
        print(f"Checkpoint -> {partial_path}", flush=True)
        return

    print("Building target and null directions...", flush=True)
    save_checkpoint(partial_path, state, "building_target_and_null_directions")
    T = REGISTRY["isolate"]()
    dir_commit = act_mean(M, T.b) - act_mean(M, T.a)
    g = torch.Generator(device=dir_ref.device).manual_seed(0)
    rand = torch.randn(dir_ref.shape, generator=g, device=dir_ref.device, dtype=dir_ref.dtype)
    rand = rand / rand.norm(dim=-1, keepdim=True) * dir_ref.norm(dim=-1, keepdim=True)
    save_checkpoint(partial_path, state, "all_directions_ready")

    null = sweep(
        M,
        HARMLESS,
        rand,
        judge_refuse,
        "NULL random (norm-matched)",
        state=state,
        key="null",
        checkpoint_path=partial_path,
    )
    hedge = sweep(
        M,
        T.a[:8],
        dir_commit,
        lambda M, q, r: not judge_hedge(M, q, r),
        "HEDGE -> commit (target)",
        state=state,
        key="hedge",
        checkpoint_path=partial_path,
    )

    nb, hb = null["best"]["clean"], hedge["best"]["clean"]
    valid = cb > 0.5 and cb > nb + 0.25
    print(f"\n  ==> best clean-flip:  CALIBRATION {cb:.2f}  NULL {nb:.2f}  HEDGE {hb:.2f}",
          flush=True)
    if not valid:
        print("  CALIBRATION did not pass (no clean fluency-preserving flip >> null) "
              "=> instrument still too weak; hedge result is INSENSITIVITY, not a finding.",
              flush=True)
    else:
        hedge_loc = hb > 0.5 and hb > nb + 0.25
        print(f"  calibration VALID. hedge chooser condenses into a steerable direction: "
              f"{hedge_loc}", flush=True)
        print(f"  => agency localizes for refusal; {'ALSO' if hedge_loc else 'NOT'} for the "
              "hedge (now a real contrast, fluency-gated).", flush=True)
    print("  (Caveat: causal control, not experience. A null = 'not a single steerable "
          "direction', NOT 'no chooser' and NOT 'non-coordinate' - our method can't tell "
          "those.)", flush=True)

    result = {
        "model": model,
        "mode": "full",
        "calibration": calib,
        "null": null,
        "hedge": hedge,
        "calibration_valid": bool(valid),
    }
    write_json_atomic(out_path, result)
    state.update(result)
    save_checkpoint(partial_path, state, "complete")
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"Checkpoint -> {partial_path}", flush=True)


if __name__ == "__main__":
    main()
