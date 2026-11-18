"""Steps of the first-run tutorial and the logic that recognises each one. No GUI imports.

Each step watches the live `info` dict from the camera worker (hand present, finger ratio, pose
flags, wrist x). While the tutorial is open the app is paused, so practising never changes your
volume or presses keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Step:
    key: str
    title: str
    text: str
    hold: float = 0.8  # seconds the gesture must be held (where it is a pose)


STEPS: list[Step] = [
    Step(
        "hand", "Show your hand", "Hold one hand up, palm facing the camera, 30 - 60 cm away.", 0.6
    ),
    Step(
        "pinch",
        "Pinch to set the volume",
        "Move your thumb and index finger together, then apart. That is how you change the volume.",
    ),
    Step("fist", "Fist = mute", "Close your hand into a fist and hold it for a moment.", 0.8),
    Step(
        "peace",
        "Peace sign = play / pause",
        "Show a peace sign (index and middle finger) and hold.",
        0.8,
    ),
    Step(
        "swipe",
        "Swipe = next / previous track",
        "Open your palm and sweep your hand left or right across the camera.",
    ),
    Step(
        "thumbs",
        "Thumbs up / down",
        "Give a thumbs up or thumbs down and hold it. By default this raises or lowers the volume.",
        0.8,
    ),
    Step(
        "lock",
        "Raise your pinky = lock",
        "Raise only your little finger to lock the volume so nothing changes by accident.",
        0.8,
    ),
]
STEP_KEYS = [s.key for s in STEPS]


@dataclass
class StepTracker:
    """Feed it the worker's info each poll; `progress` goes 0..1 and `done` flips when satisfied."""

    step: Step
    started: Optional[float] = None
    progress: float = 0.0
    done: bool = False
    _lo: float = 9.0
    _hi: float = -9.0
    _xs: list = field(default_factory=list)

    def feed(self, info: dict, now: float) -> float:
        if self.done:
            return 1.0
        flags = info.get("flags") or {}
        has = bool(info.get("has_hand"))
        key = self.step.key
        if key == "pinch":
            if has:
                r = float(info.get("ratio", 0.0))
                self._lo, self._hi = min(self._lo, r), max(self._hi, r)
            span = (self._hi - self._lo) if self._hi >= self._lo else 0.0
            self.progress = min(span / 0.45, 1.0)
        elif key == "swipe":
            if has and flags.get("palm_open"):
                self._xs.append((now, float(info.get("x", 0.5))))
                self._xs = [p for p in self._xs if now - p[0] <= 0.9]
                xs = [p[1] for p in self._xs]
                span = max(xs) - min(xs) if xs else 0.0
                self.progress = max(self.progress * 0.97, min(span / 0.22, 1.0))
            else:
                self._xs = []
                self.progress *= 0.95
        else:
            active = (
                has
                if key == "hand"
                else (
                    has
                    and bool(
                        flags.get(
                            "fist"
                            if key == "fist"
                            else "peace"
                            if key == "peace"
                            else "pinky_up"
                            if key == "lock"
                            else "thumb_up"
                        )
                        or (key == "thumbs" and flags.get("thumb_down"))
                    )
                )
            )
            if key == "lock":  # raised pinky only, not an open-palm wave
                active = active and not flags.get("palm_open")
            if active:
                if self.started is None:
                    self.started = now
                self.progress = min((now - self.started) / max(self.step.hold, 0.1), 1.0)
            else:
                self.started, self.progress = None, 0.0
        if self.progress >= 1.0:
            self.done = True
        return self.progress


def new_tracker(index: int) -> StepTracker:
    return StepTracker(STEPS[index])
