"""Two-hand mode: a second hand that sets a level (screen brightness). No GUI imports.

* `assign()` decides which detected hand is the main (volume / gestures) hand and which is the
  second one, by which side of the picture they are on (the view is mirrored, so your right hand
  is on the right).
* `SecondaryEngine` turns the second hand's pinch into a 0..1 level, with the same calibration,
  smoothing and pinky-lock as the volume hand, and hands changes to a setter without blocking.
* `BrightnessControl` sets the real screen brightness through the optional
  `screen_brightness_control` package (laptop panels and most DDC monitors).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from gestures import MAX_VOLUME_STEP, FrameResult, analyze

WRIST = 0


def assign(wrist_xs: Sequence[float], primary_side: str = "right") -> tuple[int, Optional[int]]:
    """Return (main_index, second_index or None) for hands whose wrist x (0..1, mirrored) is given."""
    n = len(wrist_xs)
    if n == 0:
        raise ValueError("no hands")
    if n == 1:
        return 0, None
    order = sorted(range(n), key=lambda i: wrist_xs[i])          # left ... right on screen
    if primary_side == "left":
        return order[0], order[-1]
    return order[-1], order[0]


class BrightnessControl:
    """Set / read the screen brightness (0..1). Silent no-op if the package is missing."""

    def __init__(self) -> None:
        try:
            import screen_brightness_control as sbc
            self._sbc: Any = sbc
        except Exception:
            self._sbc = None

    @property
    def available(self) -> bool:
        return self._sbc is not None

    def get(self) -> Optional[float]:
        if self._sbc is None:
            return None
        try:
            vals = self._sbc.get_brightness()
            return float(vals[0] if isinstance(vals, (list, tuple)) else vals) / 100.0
        except Exception:
            return None

    def set(self, level: float) -> None:
        if self._sbc is None:
            return
        try:
            self._sbc.set_brightness(int(round(float(np.clip(level, 0.0, 1.0)) * 100)))
        except Exception:
            self._sbc = None                  # unsupported display: stop trying


class _LatestSetter:
    """Runs a slow setter on a background thread, always applying only the newest value."""

    def __init__(self, fn: Callable[[float], None]) -> None:
        self.fn, self._value, self._event = fn, None, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, v: float) -> None:
        self._value = v
        self._event.set()

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()
            v, self._value = self._value, None
            if v is not None:
                self.fn(v)
            time.sleep(0.05)


class SecondaryEngine:
    MIN_STEP = 0.01          # smallest change worth sending
    MIN_GAP = 0.12           # seconds between sends

    def __init__(self, setter: Optional[Callable[[float], None]] = None, level: float = 0.5,
                 sync: bool = False) -> None:
        self.level = float(level)
        self._sent = -1.0
        self._sent_t = -1.0
        self._setter = setter
        self._async = None if (setter is None or sync) else _LatestSetter(setter)

    def _send(self, now: float) -> None:
        if self._setter is None:
            return
        if abs(self.level - self._sent) < self.MIN_STEP or now - self._sent_t < self.MIN_GAP:
            return
        self._sent, self._sent_t = self.level, now
        if self._async is not None:
            self._async.push(self.level)
        else:
            self._setter(self.level)

    def update(self, landmarks: Optional[Sequence[Any]], size: tuple[int, int], s: dict,
               now: Optional[float] = None, world: Optional[Sequence[Any]] = None) -> FrameResult:
        now = time.time() if now is None else now
        res = FrameResult()
        res.volume = self.level
        if landmarks is None:
            res.status = "Show your second hand"
            return res
        w, h = size
        pose = analyze(landmarks)
        res.has_hand, res.pose = True, pose
        res.norm = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in landmarks], np.float32)
        if world is not None:
            res.world = np.array([[p.x, p.y, p.z] for p in world], np.float32)
        res.landmarks_px = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        locked = bool(s.get("enable_lock", True)) and pose.pinky_up
        res.locked = locked
        res.paused = bool(s.get("paused"))
        controlling = (not res.paused and not locked and not pose.fist and not pose.peace
                       and not pose.palm_open and not pose.thumb_up and not pose.thumb_down)
        if controlling:
            lo = float(s.get("min_ratio", 0.1))
            hi = max(float(s.get("max_ratio", 1.2)), lo + 0.05)
            target = float(np.clip(np.interp(pose.ratio, [lo, hi], [0.0, 1.0]), 0.0, 1.0))
            step = float(s.get("smoothing", 0.3)) * (target - self.level)
            self.level += float(np.clip(step, -MAX_VOLUME_STEP, MAX_VOLUME_STEP))
            self._send(now)
        res.controlling = controlling
        res.volume = self.level
        res.status = ("Paused" if res.paused else "Second hand locked" if locked
                      else "Second hand: adjusting" if controlling else "Second hand: pinch to adjust")
        return res
