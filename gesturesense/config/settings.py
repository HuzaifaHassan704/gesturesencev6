"""Settings: defaults, load and save (JSON next to the program)."""

from __future__ import annotations

import json
from typing import Any

from gesturesense.paths import data_dir

APP_DIR = data_dir()
SETTINGS_PATH = APP_DIR / "gesturesense_settings.json"

DEFAULTS: dict[str, Any] = {
    # --- gesture tuning ---
    "min_ratio": 0.15,  # thumb-index / palm size when pinched  -> 0 % volume
    "max_ratio": 1.10,  # ... when fully spread                 -> 100 % volume
    "smoothing": 0.25,  # 0..1, lower = smoother but slower
    "hold_seconds": 0.7,  # how long a fist / peace sign must be held
    "swipe_distance": 0.30,  # fraction of the frame width an open-palm swipe must cover
    "filter_strength": 0.5,  # 0..1 jitter filter on the hand landmarks (0 = off)
    "lost_grace": 0.3,  # seconds the last pose is held when detection blinks out
    "settle_seconds": 0.4,  # volume stays frozen this long after the hand re-appears
    "auto_lock_seconds": 0.0,  # lock volume after this many seconds without a hand (0 = off)
    # --- gesture on/off ---
    "enable_volume": True,
    "enable_lock": True,
    "enable_mute": True,
    "enable_media": True,
    "enable_thumbs": True,
    # --- visual effects on/off ---
    "fx_skeleton": True,
    "fx_trail": True,
    "fx_arc": True,
    "fx_ring": True,
    "fx_popups": True,
    # --- 3D view ---
    "view_mode": "pip",  # "camera" | "scene" | "pip" (3D scene with a camera inset)
    "orbit_auto": True,  # camera orbits by itself; dragging in the window takes control
    "orbit_speed": 14.0,  # degrees per second
    "orbit_range": 40.0,  # sweep +/- this many degrees (180 = continuous spin)
    "view_yaw": 0.0,  # manual camera angles (set by dragging in the window)
    "view_pitch": 24.0,
    "view_zoom": 1.0,
    "render_quality": 0.85,  # 3D resolution as a fraction of the viewport (lower = faster)
    "pip_size": 0.38,  # camera inset width as a fraction of the 3D view
    "hand_style": "android",  # robot | android | human | glass
    "skin_tone": "medium",  # light | medium | tan | deep (human hand)
    "theme": "aurora",  # aurora | neon | ocean | sunset | mono
    "fx3d_grid": True,
    "fx3d_shadow": True,
    "fx3d_glow": True,
    "fx3d_motion": True,  # trails, sparks and ripples that react to hand speed
    "fx3d_motion_amount": 0.7,
    "auto_profile": False,  # switch profile by the active app (see autoswitch.py)
    # --- two-hand mode ---
    "two_hand_mode": False,
    "primary_hand": "right",  # which hand (as seen on screen) does volume + gestures
    "second_role": "brightness",  # brightness | none
    # --- sound, tray, first run ---
    "sound_fx": True,
    "sound_volume": 0.5,
    "minimize_to_tray": False,  # closing the window keeps GestureSense running in the tray
    "tutorial_done": False,
    "auto_start": False,  # press Start automatically on launch
    # --- camera ---
    "camera_index": 0,
}


def _valid(default: Any, value: Any) -> bool:
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load() -> dict[str, Any]:
    """Return DEFAULTS overridden by any valid values stored on disk."""
    data = dict(DEFAULTS)
    try:
        stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return data
    if isinstance(stored, dict):
        for key, default in DEFAULTS.items():
            if key in stored and _valid(default, stored[key]):
                data[key] = type(default)(stored[key])
    return data


def save(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
