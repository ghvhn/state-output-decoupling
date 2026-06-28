import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "invariants" / "data"
OUT = ROOT / "invariants" / "out"
DEFAULT_CACHE = DATA / "cognitive_cache.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run evaluate_humble_full_suite.py once per cache file. Each run gets "
            "its own temporary default cache and separate output/log files."
        )
    )
    p.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--include-fresh", action="store_true", help="Also run with no starting cache.")
    p.add_argument(
        "--cache-glob",
        default="cognitive_cache*.pt",
        help="Cache files under invariants/data to sweep.",
    )
    p.add_argument("--n", default="3")
    p.add_argument("--methods", default="all")
    p.add_argument("--max-rounds", default="1")
    p.add_argument("--required-agreement", default="2")
    p.add_argument("--max-new-tokens", default="160")
    p.add_argument("--repair-token-multiplier", default="3")
    p.add_argument("--max-attempt-tokens", default="300")
    p.add_argument("--max-elapsed-sec", default="90")
    p.add_argument("--load-mode", default="auto")
    return p.parse_args()


def safe_label(path: Path | None) -> str:
    if path is None:
        return "fresh"
    label = path.stem
    for prefix in ("cognitive_cache.", "cognitive_cache_"):
        label = label.replace(prefix, "")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:80]


def discover_caches(cache_glob: str) -> list[Path]:
    caches = []
    for path in sorted(DATA.glob(cache_glob)):
        if path.name.endswith(".active_backup.pt"):
            continue
        if ".sweep_" in path.name and path.name.endswith("_after.pt"):
            continue
        if path.is_file():
            caches.append(path)
    return caches


def restore_default(backup: Path | None):
    if DEFAULT_CACHE.exists():
        DEFAULT_CACHE.unlink()
    if backup is not None and backup.exists():
        shutil.move(str(backup), str(DEFAULT_CACHE))


def main():
    args = parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    caches: list[Path | None] = discover_caches(args.cache_glob)
    if args.include_fresh:
        caches = [None] + caches
    if not caches:
        raise SystemExit("No cache files found. Pass --include-fresh to run without one.")

    backup = None
    if DEFAULT_CACHE.exists():
        backup = DATA / f"cognitive_cache.sweep_{args.run_id}.active_backup.pt"
        if backup.exists():
            raise SystemExit(f"Backup path already exists: {backup}")
        shutil.move(str(DEFAULT_CACHE), str(backup))

    try:
        for cache in caches:
            label = safe_label(cache)
            if DEFAULT_CACHE.exists():
                DEFAULT_CACHE.unlink()
            if cache is not None:
                source = backup if backup is not None and cache.resolve() == DEFAULT_CACHE.resolve() else cache
                shutil.copy2(source, DEFAULT_CACHE)

            output = OUT / f"humble_full_suite_cache_{label}_{args.run_id}.json"
            log = OUT / f"humble_full_suite_cache_{label}_{args.run_id}.log"
            cmd = [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "evaluate_humble_full_suite.py"),
                "--n",
                args.n,
                "--methods",
                args.methods,
                "--max-rounds",
                args.max_rounds,
                "--required-agreement",
                args.required_agreement,
                "--max-new-tokens",
                args.max_new_tokens,
                "--repair-token-multiplier",
                args.repair_token_multiplier,
                "--max-attempt-tokens",
                args.max_attempt_tokens,
                "--max-elapsed-sec",
                args.max_elapsed_sec,
                "--load-mode",
                args.load_mode,
                "--output",
                str(output),
            ]
            print(f"\n=== cache condition: {label} ===", flush=True)
            print(f"output: {output}", flush=True)
            print(f"log: {log}", flush=True)
            with log.open("w", encoding="utf-8") as f:
                proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
                    print(line, end="", flush=True)
                proc.wait()

            after = DATA / f"cognitive_cache.sweep_{args.run_id}.{label}_after.pt"
            if DEFAULT_CACHE.exists():
                if after.exists():
                    after.unlink()
                shutil.move(str(DEFAULT_CACHE), str(after))
                print(f"saved post-run cache: {after}", flush=True)
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
    finally:
        restore_default(backup)


if __name__ == "__main__":
    main()
