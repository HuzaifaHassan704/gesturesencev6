"""Profiles: which action each gesture triggers. Stored as JSON next to the program.

Built-in profiles (Music, Presentation, Video call, Gaming) can be edited, reset and extended
with your own. No GUI imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from actions import is_valid

from paths import data_dir

PROFILES_PATH = data_dir() / "gesturesense_profiles.json"

# gesture key -> label (order = order in the UI)
GESTURES: list[tuple[str, str]] = [
    ("fist_hold", "Fist"),
    ("peace_hold", "Peace sign"),
    ("swipe_right", "Swipe right"),
    ("swipe_left", "Swipe left"),
    ("thumbs_up_hold", "Thumbs up"),
    ("thumbs_down_hold", "Thumbs down"),
]
GESTURE_KEYS = [k for k, _ in GESTURES]

BUILTIN: dict[str, dict[str, str]] = {
    "Music": {"fist_hold": "mute_toggle", "peace_hold": "play_pause",
              "swipe_right": "next_track", "swipe_left": "prev_track",
              "thumbs_up_hold": "volume_up", "thumbs_down_hold": "volume_down"},
    "Presentation": {"fist_hold": "slideshow_end", "peace_hold": "slideshow_start",
                     "swipe_right": "slide_next", "swipe_left": "slide_prev",
                     "thumbs_up_hold": "volume_up", "thumbs_down_hold": "volume_down"},
    "Video call": {"fist_hold": "teams_mute", "peace_hold": "screenshot",
                   "swipe_right": "none", "swipe_left": "none",
                   "thumbs_up_hold": "volume_up", "thumbs_down_hold": "volume_down"},
    "Gaming": {"fist_hold": "mute_toggle", "peace_hold": "none",
               "swipe_right": "none", "swipe_left": "none",
               "thumbs_up_hold": "none", "thumbs_down_hold": "none"},
}
DEFAULT_PROFILE = "Music"


def clean(mapping: Any) -> dict[str, str]:
    """A complete, valid mapping: unknown gestures dropped, bad / missing actions -> 'none'."""
    out = {key: "none" for key in GESTURE_KEYS}
    if isinstance(mapping, dict):
        for key in GESTURE_KEYS:
            if is_valid(mapping.get(key)):
                out[key] = mapping[key]
    return out


def defaults() -> dict[str, Any]:
    return {"active": DEFAULT_PROFILE, "profiles": {n: dict(m) for n, m in BUILTIN.items()}}


def load(path: Optional[Path] = None) -> dict[str, Any]:
    """Return {'active': name, 'profiles': {name: mapping}}; always valid, built-ins always present."""
    data = defaults()
    try:
        stored = json.loads((path or PROFILES_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return data
    if isinstance(stored, dict) and isinstance(stored.get("profiles"), dict):
        for name, mapping in stored["profiles"].items():
            if isinstance(name, str) and name.strip():
                data["profiles"][name.strip()[:40]] = clean(mapping)
        active = stored.get("active")
        if isinstance(active, str) and active in data["profiles"]:
            data["active"] = active
    return data


def save(data: dict[str, Any], path: Optional[Path] = None) -> None:
    (path or PROFILES_PATH).write_text(json.dumps(data, indent=2), encoding="utf-8")


def unique_name(base: str, existing) -> str:
    base = (base or "Profile").strip()[:40] or "Profile"
    name, n = base, 2
    while name in existing:
        name, n = f"{base} {n}", n + 1
    return name


def reset_mapping(name: str) -> dict[str, str]:
    """The factory mapping for a built-in profile (custom profiles reset to the Music mapping)."""
    return dict(BUILTIN.get(name, BUILTIN[DEFAULT_PROFILE]))
