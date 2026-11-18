"""Orbit camera with a perspective projection."""

from __future__ import annotations

import math

import numpy as np

from gesturesense.render.constants import (
    UP,
)


class Camera:
    """Orbit camera looking at `target`; yaw/pitch in radians."""

    def __init__(self, w: int, h: int, fov_deg: float = 30.0) -> None:
        self.w, self.h = w, h
        self.focal = 0.5 * h / math.tan(math.radians(fov_deg) / 2)
        self.target = np.array([1.05, 1.15, 0.0])
        self.pos = np.zeros(3)
        self.R = np.eye(3)
        self.set(0.0, 0.4, 7.0)

    def set(self, yaw: float, pitch: float, dist: float) -> None:
        cp = math.cos(pitch)
        self.pos = self.target + dist * np.array(
            [math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp]
        )
        f = self.target - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(UP, f)
        r /= np.linalg.norm(r)
        self.R = np.stack([r, np.cross(f, r), f])  # rows: right, up, forward

    def project(self, pts: np.ndarray):
        """pts (N,3) -> screen xy (N,2), depth (N,), pixels-per-unit (N,)."""
        c = (np.atleast_2d(pts) - self.pos) @ self.R.T
        z = np.maximum(c[:, 2], 1e-3)
        k = self.focal / z
        xy = np.stack([self.w / 2 + c[:, 0] * k, self.h / 2 - c[:, 1] * k], axis=1)
        return xy, c[:, 2], k
