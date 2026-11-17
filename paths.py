"""Where the app keeps its files.

* Running from source: next to the program (as before), so nothing moves for existing users.
* Installed / frozen .exe: `%APPDATA%/GestureSense`, because the program folder may be read-only.
* `GESTURESENSE_HOME` overrides both (handy for portable installs and for tests).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """Read-only files shipped with the program (the icon, a bundled model)."""
    return Path(getattr(sys, "_MEIPASS", SOURCE_DIR))


def data_dir() -> Path:
    env = os.environ.get("GESTURESENSE_HOME")
    if env:
        base = Path(env)
    elif frozen():
        root = os.environ.get("APPDATA") or str(Path.home() / ".config")
        base = Path(root) / "GestureSense"
    else:
        base = SOURCE_DIR
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        base = Path.home()
    return base
