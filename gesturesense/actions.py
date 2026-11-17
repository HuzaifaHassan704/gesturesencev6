"""What a gesture can *do*: the action catalogue and the runner that performs an action.

No GUI imports. Keys / hotkeys / app launching are injected, so the runner is unit-testable.
An action is a plain string: one of the names below, or "open:<path to an app>".
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

VOLUME_STEP = 0.10

# name -> label shown in the UI (order = order in the dropdown)
ACTIONS: dict[str, str] = {
    "none": "Do nothing",
    "mute_toggle": "Mute / unmute",
    "volume_up": "Volume +10 %",
    "volume_down": "Volume -10 %",
    "play_pause": "Play / pause",
    "next_track": "Next track",
    "prev_track": "Previous track",
    "slide_next": "Slide: next (Right)",
    "slide_prev": "Slide: previous (Left)",
    "slideshow_start": "Start slideshow (F5)",
    "slideshow_end": "End slideshow (Esc)",
    "teams_mute": "Teams: mute mic",
    "zoom_mute": "Zoom: mute mic",
    "meet_mute": "Meet: mute mic",
    "screenshot": "Screenshot",
    "alt_tab": "Switch window (Alt+Tab)",
    "show_desktop": "Show desktop (Win+D)",
}
OPEN_PREFIX = "open:"

_KEYS: dict[str, tuple[str, Any]] = {
    "play_pause": ("press", "playpause"),
    "next_track": ("press", "nexttrack"),
    "prev_track": ("press", "prevtrack"),
    "slide_next": ("press", "right"),
    "slide_prev": ("press", "left"),
    "slideshow_start": ("press", "f5"),
    "slideshow_end": ("press", "esc"),
    "teams_mute": ("hotkey", ("ctrl", "shift", "m")),
    "zoom_mute": ("hotkey", ("alt", "a")),
    "meet_mute": ("hotkey", ("ctrl", "d")),
    "screenshot": ("hotkey", ("win", "shift", "s")),
    "alt_tab": ("hotkey", ("alt", "tab")),
    "show_desktop": ("hotkey", ("win", "d")),
}
# popup icon for each action (anything else shows a generic "action" icon with its label)
_POPUP = {"play_pause": "playpause", "next_track": "next", "prev_track": "prev",
          "volume_up": "vol_up", "volume_down": "vol_down"}


def is_valid(action: Any) -> bool:
    return isinstance(action, str) and (action in ACTIONS or (
        action.startswith(OPEN_PREFIX) and len(action) > len(OPEN_PREFIX)))


def describe(action: str) -> str:
    if action.startswith(OPEN_PREFIX):
        return "Open: " + (Path(action[len(OPEN_PREFIX):]).stem or "app")
    return ACTIONS.get(action, "Do nothing")


def _default_press(key: str) -> None:
    import pyautogui

    pyautogui.PAUSE = 0
    pyautogui.press(key)


def _default_hotkey(*keys: str) -> None:
    import pyautogui

    pyautogui.PAUSE = 0
    pyautogui.hotkey(*keys)


def _default_launch(path: str) -> None:
    if sys.platform == "win32":
        os.startfile(path)                     # type: ignore[attr-defined]
    else:
        subprocess.Popen([path])


class ActionRunner:
    """Performs actions. `run()` returns (popup_kind, label) or None when nothing visible happened."""

    def __init__(self, audio: Any, press: Optional[Callable[[str], None]] = None,
                 hotkey: Optional[Callable[..., None]] = None,
                 launch: Optional[Callable[[str], None]] = None) -> None:
        self.audio = audio
        self.press = press or _default_press
        self.hotkey = hotkey or _default_hotkey
        self.launch = launch or _default_launch

    def run(self, action: str) -> Optional[tuple[str, str]]:
        if not is_valid(action) or action == "none":
            return None
        if action == "mute_toggle":
            muted = bool(self.audio.toggle_mute())
            return ("mute_on" if muted else "mute_off", "")
        if action in ("volume_up", "volume_down"):
            delta = VOLUME_STEP if action == "volume_up" else -VOLUME_STEP
            self.audio.set(float(np.clip(self.audio.get() + delta, 0.0, 1.0)))
            return (_POPUP[action], f"{int(round(self.audio.get() * 100))} %")
        if action.startswith(OPEN_PREFIX):
            try:
                self.launch(action[len(OPEN_PREFIX):])
            except Exception:                  # a bad path must never crash the camera loop
                return ("action", "Could not open app")
            return ("action", describe(action))
        kind, arg = _KEYS[action]
        self.press(arg) if kind == "press" else self.hotkey(*arg)
        return (_POPUP.get(action, "action"), describe(action))
