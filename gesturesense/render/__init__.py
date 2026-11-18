"""The software 3D renderer (NumPy + OpenCV) and the 2D overlay effects.

Modules: ``constants`` (geometry), ``theme`` (colours), ``camera``, ``layers`` (glow/alpha
helpers), ``hands`` (hand styles), ``world`` (grid, tower, beam, HUD) and ``scene`` (`Scene3D`).
"""

from gesturesense.render.camera import Camera
from gesturesense.render.constants import GAUGE_POS, SKIN_TONES, VIEW_H, VIEW_W
from gesturesense.render.scene import Scene3D
from gesturesense.render.theme import DEFAULT_THEME, THEMES, gauge_color

__all__ = [
    "DEFAULT_THEME",
    "GAUGE_POS",
    "SKIN_TONES",
    "THEMES",
    "VIEW_H",
    "VIEW_W",
    "Camera",
    "Scene3D",
    "gauge_color",
]
