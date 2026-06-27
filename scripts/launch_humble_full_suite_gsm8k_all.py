import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "invariants" / "out"
OUT.mkdir(exist_ok=True)

python = Path(
    r"C:\Users\Gavin Powell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"
)
if not python.exists():
    python = ROOT / ".venv" / "Scripts" / "pythonw.exe"
if not python.exists():
    python = ROOT / ".venv" / "Scripts" / "python.exe"
log_path = OUT / "humble_full_suite_gsm8k_all.log"
err_path = OUT / "humble_full_suite_gsm8k_all.err.log"
pid_path = OUT / "humble_full_suite_gsm8k_all.pid"

args = [
    str(python),
    "-u",
    "scripts/evaluate_humble_full_suite.py",
    "--n",
    "all",
    "--methods",
    "all",
    "--max-rounds",
    "2",
    "--required-agreement",
    "2",
    "--max-new-tokens",
    "100",
    "--repair-token-multiplier",
    "3",
    "--max-attempt-tokens",
    "300",
    "--max-elapsed-sec",
    "180",
    "--load-mode",
    "auto",
    "--resume",
    "--output",
    "invariants/out/humble_full_suite_gsm8k_all.json",
]

flags = 0
if sys.platform.startswith("win"):
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

with log_path.open("w", encoding="utf-8") as log, err_path.open("w", encoding="utf-8") as err:
    env = dict(**__import__("os").environ)
    venv = ROOT / ".venv"
    site_packages = venv / "Lib" / "site-packages"
    pythonpath = [str(venv), str(site_packages)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ";".join(pythonpath)
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = str(venv / "Scripts") + ";" + env.get("PATH", "")

    proc = subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=log,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=False,
        env=env,
    )

pid_path.write_text(str(proc.pid), encoding="ascii")
print(proc.pid)
