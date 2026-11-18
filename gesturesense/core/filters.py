"""Landmark smoothing with a One-Euro filter (no GUI, camera or audio imports).

The One-Euro filter is the standard answer to hand-tracking jitter: when the hand is almost
still it smooths hard (no shaking), and when the hand moves fast it smooths less (no lag).
`strength` 0..1 is the single knob the user sees: 0 = filter off, 1 = very smooth.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuro:
    """One-Euro filter for a NumPy array of any shape."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x: Optional[np.ndarray] = None
        self.dx: Optional[np.ndarray] = None
        self.t = 0.0

    def __call__(self, value: np.ndarray, t: float) -> np.ndarray:
        value = np.asarray(value, np.float64)
        if self.x is None or t <= self.t:
            if self.x is None:
                self.x, self.dx, self.t = value.copy(), np.zeros_like(value), t
            return self.x.copy()
        dt = t - self.t
        dx = (value - self.x) / dt
        self.dx = _alpha(dt, self.d_cutoff) * dx + (1 - _alpha(dt, self.d_cutoff)) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)  # faster hand -> less smoothing
        a = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))) / dt)
        self.x = a * value + (1 - a) * self.x
        self.t = t
        return self.x.copy()


def _points(arr: np.ndarray) -> list:
    return [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in arr]


def _array(points: Sequence) -> np.ndarray:
    return np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in points], np.float64)


class LandmarkFilter:
    """Smooths the 21 image landmarks and the 21 world landmarks of one hand."""

    RESET_AFTER = 0.5  # seconds without a hand before the filter forgets the old pose

    def __init__(self) -> None:
        self._norm = OneEuro()
        self._world = OneEuro()
        self._last_seen = -1e9
        self._strength = -1.0

    @staticmethod
    def params(strength: float) -> tuple[float, float, float]:
        """strength 0..1 -> (min_cutoff Hz, beta for image coords, beta for world coords)."""
        s = min(max(strength, 0.0), 1.0)
        return 3.0 * (0.15**s), 8.0, 25.0

    def _configure(self, strength: float) -> None:
        if strength != self._strength:
            cutoff, b_norm, b_world = self.params(strength)
            self._norm.min_cutoff, self._norm.beta = cutoff, b_norm
            self._world.min_cutoff, self._world.beta = cutoff, b_world
            self._strength = strength

    def miss(self, t: float) -> None:
        """Call when no hand was detected this frame."""
        if t - self._last_seen > self.RESET_AFTER:
            self._norm.reset()
            self._world.reset()

    def apply(self, norm: Sequence, world: Optional[Sequence], t: float, strength: float):
        """Return (norm_points, world_points) as lists of objects with .x .y .z."""
        if strength <= 0.01:  # filter off
            self._last_seen = t
            return norm, world
        if t - self._last_seen > self.RESET_AFTER:
            self._norm.reset()
            self._world.reset()
        self._configure(strength)
        self._last_seen = t
        out_norm = _points(self._norm(_array(norm), t))
        out_world = _points(self._world(_array(world), t)) if world is not None else None
        return out_norm, out_world
