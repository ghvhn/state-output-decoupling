"""
Backward-compatible wrapper for the renamed warranted-confidence probe.

Prefer:
    python scripts\\probe_warranted_confidence.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("probe_warranted_confidence.py")
    sys.argv[0] = str(target)
    print(
        "probe_grounded_reassurance.py is deprecated; forwarding to "
        "probe_warranted_confidence.py"
    )
    runpy.run_path(str(target), run_name="__main__")
