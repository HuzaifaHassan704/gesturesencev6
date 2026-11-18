"""Colour themes, skin tones and colour helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from gesturesense.render.constants import (
    BODY,
    GAUGE_STOPS,
)
from gesturesense.render.effects import lerp_color

# ----------------------------------------------------------------------------- helpers
# Colour themes (all BGR). `bg_top/bg_bot` = backdrop gradient, `pool` = light pool on the floor,
# `grid` = (minor, major) line colours, `stops` = tower gradient bottom -> top, `body` = hand skin.
THEMES: dict[str, dict] = {
    "aurora": dict(
        label="Aurora",
        stops=GAUGE_STOPS,
        body=BODY,
        bg_top=(30, 21, 15),
        bg_bot=(12, 9, 7),
        pool=(44, 26, 12),
        grid=((104, 74, 46), (170, 122, 78)),
    ),
    "neon": dict(
        label="Neon",
        stops=[(255, 255, 0), (255, 0, 230), (0, 230, 255)],
        body=(235, 215, 255),
        bg_top=(30, 10, 28),
        bg_bot=(14, 4, 12),
        pool=(70, 18, 56),
        grid=((120, 44, 110), (200, 70, 180)),
    ),
    "ocean": dict(
        label="Ocean",
        stops=[(200, 235, 60), (255, 185, 50), (255, 120, 110)],
        body=(235, 225, 190),
        bg_top=(36, 22, 10),
        bg_bot=(14, 8, 3),
        pool=(56, 38, 8),
        grid=((120, 86, 40), (190, 140, 66)),
    ),
    "sunset": dict(
        label="Sunset",
        stops=[(70, 190, 255), (80, 90, 250), (170, 60, 200)],
        body=(205, 218, 242),
        bg_top=(30, 18, 38),
        bg_bot=(10, 6, 18),
        pool=(40, 34, 78),
        grid=((70, 74, 150), (110, 118, 220)),
    ),
    "mono": dict(
        label="Mono",
        stops=[(105, 105, 105), (160, 160, 160), (215, 215, 215)],
        body=(215, 215, 215),
        bg_top=(26, 26, 26),
        bg_bot=(10, 10, 10),
        pool=(34, 34, 34),
        grid=((78, 78, 78), (130, 130, 130)),
    ),
}
DEFAULT_THEME = "aurora"


def gauge_color(t: float, stops=None) -> tuple[int, int, int]:
    """Colour of the tower at relative height t (0 bottom .. 1 top)."""
    stops = stops or GAUGE_STOPS
    t = min(max(t, 0.0), 1.0) * 2.0
    return lerp_color(stops[0], stops[1], t) if t < 1 else lerp_color(stops[1], stops[2], t - 1.0)


def scale_color(c, k: float) -> tuple[int, int, int]:
    return (
        int(min(max(c[0] * k, 0), 255)),
        int(min(max(c[1] * k, 0), 255)),
        int(min(max(c[2] * k, 0), 255)),
    )


def _ip(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _ok(p, lim: float = 20000.0) -> bool:
    return abs(p[0]) < lim and abs(p[1]) < lim


def make_background(w: int, h: int, theme: Optional[dict] = None) -> np.ndarray:
    """Dark vertical gradient with a faint blue pool of light and a vignette."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    th = theme or THEMES[DEFAULT_THEME]
    top, bot = np.array(th["bg_top"], np.float32), np.array(th["bg_bot"], np.float32)
    img = top[None, None, :] * (1 - y[..., None]) + bot[None, None, :] * y[..., None]
    img = np.broadcast_to(img, (h, w, 3)).copy()
    glow = np.exp(-(((x * 0.9) ** 2) + ((y - 0.68) * 2.2) ** 2) * 2.4)  # pool of light
    img += glow[..., None] * np.array(th["pool"], np.float32)
    vig = 1.0 - 0.45 * np.clip((x**2 + (y - 0.5) ** 2 * 1.4) - 0.35, 0, 1)
    return np.clip(img * vig[..., None], 0, 255).astype(np.uint8)
