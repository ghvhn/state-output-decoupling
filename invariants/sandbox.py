"""Local Python sandbox tool: code the model writes can actually run.

Connection contract, matching the shell's bare-mode philosophy:
  - No taught tool syntax. The model emits an ordinary fenced ```python block
    (something it does natively); when the sandbox is enabled the shell runs
    the code and folds the REAL output back into the next turn with an honest
    frame — the same one-turn tool-result pattern as memory/claimmap/docs.
  - Execution success (exit 0, no timeout) is an objective, label-free
    outcome. The shell observes it into the tuner (`sandbox_success`) so the
    distribution accrues like every other signal.

Honest scope, stated plainly: this is process isolation, not a security
boundary. Code runs in a separate `python -I` interpreter (isolated mode: no
user site-packages, no PYTHON* env leakage) inside a dedicated working
directory with a hard timeout and truncated I/O — enough to keep a research
session honest and contained, not enough to contain a determined adversary.
It is OFF by default; the operator enables it per session (`:sandbox on`).

Pure stdlib.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

SANDBOX_DIR = Path(__file__).parent / "out" / "sandbox"
SANDBOX_TOOL_HEADER = "[Sandbox Tool Result]"
DEFAULT_TIMEOUT_SEC = 10.0
MAX_OUTPUT_CHARS = 4000
MAX_CODE_CHARS = 20000

_FENCE_PATTERN = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_block(text: str) -> Optional[str]:
    """The LAST fenced ```python block in a response — the model's most recent
    intent. Returns None when there is none; never guesses at bare code."""
    matches = _FENCE_PATTERN.findall(text or "")
    for candidate in reversed(matches):
        code = candidate.strip()
        if code:
            return code[:MAX_CODE_CHARS]
    return None


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated at {limit} chars]"


def run_python(
    code: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    cwd: Optional[Path] = None,
    max_output_chars: int = MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Run code in an isolated interpreter; return what REALLY happened.
    {ok, exit_code, stdout, stderr, timed_out, duration_sec, code_sha}."""
    workdir = Path(cwd) if cwd is not None else SANDBOX_DIR
    workdir.mkdir(parents=True, exist_ok=True)
    code = (code or "")[:MAX_CODE_CHARS]
    started = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_sec)),
            cwd=str(workdir),
        )
        exit_code = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
    duration = time.time() - started
    return {
        "ok": (exit_code == 0) and not timed_out,
        "exit_code": exit_code,
        "stdout": _truncate(stdout or "", max_output_chars),
        "stderr": _truncate(stderr or "", max_output_chars),
        "timed_out": timed_out,
        "duration_sec": round(duration, 3),
        "code_sha": hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()[:16],
        "workdir": str(workdir),
    }


def format_sandbox_tool_result(result: dict[str, Any], timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> str:
    """Honest frame, minimal and first person — the model's own note of what
    it did and what really happened. Folded context conditions the model as
    its own stream, so no narrator voice, no instructions."""
    if result.get("timed_out"):
        status = f"it timed out at {timeout_sec:g}s"
    else:
        status = f"exit_code={result.get('exit_code')} ({result.get('duration_sec')}s)"
    lines = [
        SANDBOX_TOOL_HEADER,
        f"I ran the code I wrote; {status}.",
    ]
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    lines.append("stdout:")
    lines.append(stdout if stdout else "(empty)")
    if stderr:
        lines.append("stderr:")
        lines.append(stderr)
    return "\n".join(lines)
