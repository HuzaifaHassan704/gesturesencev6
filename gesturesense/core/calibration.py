"""Calibration logic for the wizard (no GUI imports, so it can be unit-tested).

The wizard asks the user to hold a pinch, then a full spread. `Collector` gathers the
thumb-index ratio for a few seconds (ignoring the first moments while the hand gets into
place, and pausing while no hand is visible); `compute_range` turns the two sets of samples
into the `min_ratio` / `max_ratio` settings.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

MIN_GAP = 0.35  # the spread must be at least this much larger than the pinch
MIN_RATIO_LIMIT = 0.6  # slider limits in the GUI
MAX_RATIO_LIMITS = (0.5, 2.0)


class CalibrationError(ValueError):
    """The measured pinch and spread are too close together (or too unstable) to use."""


class Collector:
    def __init__(self, seconds: float = 2.0, warmup: float = 0.8) -> None:
        self.seconds, self.warmup = seconds, warmup
        self.samples: list[float] = []
        self.active = 0.0  # seconds with a hand visible

    def add(self, ratio: float, dt: float, has_hand: bool) -> None:
        if not has_hand or ratio <= 0:
            return  # the clock pauses while the hand is missing
        self.active += dt
        if self.active > self.warmup:
            self.samples.append(float(ratio))

    @property
    def progress(self) -> float:
        return min(self.active / (self.warmup + self.seconds), 1.0)

    @property
    def done(self) -> bool:
        return self.progress >= 1.0 and len(self.samples) >= 5


def stability(samples: list[float]) -> float:
    """Relative spread of the samples (0 = perfectly steady)."""
    a = np.asarray(samples, float)
    return float(np.std(a) / max(np.median(a), 1e-6))


def compute_range(
    pinch: list[float], spread: list[float], min_gap: float = MIN_GAP
) -> tuple[float, float]:
    """Return (min_ratio, max_ratio) from the collected samples or raise CalibrationError."""
    if len(pinch) < 5 or len(spread) < 5:
        raise CalibrationError("Not enough hand data - keep your hand in view and try again.")
    lo = float(np.median(pinch)) + 0.02  # small margin so a firm pinch reaches 0 %
    hi = float(np.median(spread)) * 0.95  # and a comfortable spread reaches 100 %
    lo = min(max(lo, 0.0), MIN_RATIO_LIMIT)
    hi = min(max(hi, MAX_RATIO_LIMITS[0]), MAX_RATIO_LIMITS[1])
    if hi - lo < min_gap:
        raise CalibrationError(
            "Pinch and spread were too close together. "
            "Pinch tightly, then spread your fingers wide."
        )
    return round(lo, 3), round(hi, 3)


def quality(pinch: list[float], spread: list[float]) -> Optional[str]:
    """A short warning if either hold was shaky, otherwise None."""
    if stability(pinch) > 0.25 or stability(spread) > 0.25:
        return "Your hand was moving a lot - consider repeating for a more accurate result."
    return None
