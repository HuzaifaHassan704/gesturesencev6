"""Gesture recognition and the actions they trigger.

This module has no camera, GUI or Windows-audio imports, so it can be unit-tested.
`GestureEngine.update()` takes the 21 MediaPipe landmarks of one frame (or None) and returns
a `FrameResult` that the effects module draws.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from gesturesense.config.profiles import BUILTIN, DEFAULT_PROFILE
from gesturesense.core.actions import ActionRunner

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky fingertips
PIPS = (6, 10, 14, 18)  # middle joint of those fingers
MCPS = (5, 9, 13, 17)  # knuckles of those fingers

THUMB_MARGIN = 0.45  # thumb tip must be this many palm-sizes above / below the index knuckle
THUMB_REPEAT = 0.6  # seconds between repeats while a thumbs gesture is held (volume actions)
MAX_VOLUME_STEP = 0.05  # biggest volume change per frame, protects your ears and speakers


def _dist(a: Any, b: Any) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


@dataclass
class HandPose:
    up: tuple[bool, bool, bool, bool]  # index, middle, ring, pinky extended
    fist: bool
    peace: bool
    palm_open: bool
    pinky_up: bool
    thumb_up: bool
    thumb_down: bool
    ratio: float  # thumb-index distance divided by palm size
    palm_size: float  # wrist -> middle knuckle, normalised units


def analyze(lm: Sequence[Any]) -> HandPose:
    """Classify the pose from landmarks (assumes an upright hand; y grows downward)."""
    up = tuple(lm[t].y < lm[p].y for t, p in zip(TIPS, PIPS, strict=True))  # tip above middle joint
    deep = [lm[t].y > lm[m].y for t, m in zip(TIPS, MCPS, strict=True)]  # tip below knuckle
    index, middle, ring, pinky = up
    palm = max(_dist(lm[WRIST], lm[MIDDLE_MCP]), 1e-6)
    curled = all(deep)
    thumb_up = (
        curled and (lm[5].y - lm[THUMB_TIP].y) > THUMB_MARGIN * palm
    )  # thumb well above knuckle
    thumb_down = curled and (lm[THUMB_TIP].y - lm[5].y) > THUMB_MARGIN * palm  # thumb well below it
    return HandPose(
        up=up,  # type: ignore[arg-type]
        fist=curled and not thumb_up and not thumb_down,
        peace=index and middle and not ring and not pinky,
        palm_open=all(up),
        pinky_up=pinky,
        thumb_up=thumb_up,
        thumb_down=thumb_down,
        ratio=_dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm,
        palm_size=palm,
    )


class Hold:
    """Fires once when `active` stays true for `seconds`; also reports progress 0..1."""

    def __init__(self) -> None:
        self.t0: Optional[float] = None
        self.fired = False

    def update(self, active: bool, seconds: float, now: float) -> tuple[bool, float]:
        if not active:
            self.t0, self.fired = None, False
            return False, 0.0
        if self.t0 is None:
            self.t0 = now
        progress = min((now - self.t0) / max(seconds, 1e-6), 1.0)
        if not self.fired and progress >= 1.0:
            self.fired = True
            return True, 1.0
        return False, 1.0 if self.fired else progress


class SwipeDetector:
    """Detects a fast horizontal wrist movement within a short time window."""

    def __init__(self, window: float = 0.5, cooldown: float = 1.0) -> None:
        self.window, self.cooldown = window, cooldown
        self.points: deque[tuple[float, float]] = deque()
        self.cooldown_until = 0.0

    def reset(self) -> None:
        self.points.clear()

    def update(self, x: float, distance: float, now: float) -> Optional[str]:
        self.points.append((now, x))
        while self.points and now - self.points[0][0] > self.window:
            self.points.popleft()
        if now < self.cooldown_until:
            return None
        dx = self.points[-1][1] - self.points[0][1]
        if abs(dx) > distance:
            self.points.clear()
            self.cooldown_until = now + self.cooldown
            return "next" if dx > 0 else "prev"  # frame is mirrored: moving right = next
        return None


@dataclass
class Event:
    kind: str  # mute_on, mute_off, playpause, next, prev, lock, unlock
    time: float
    label: str = ""  # text for generic action popups


@dataclass
class FrameResult:
    has_hand: bool = False
    landmarks_px: list[tuple[int, int]] = field(default_factory=list)
    palm_px: float = 0.0
    pose: Optional[HandPose] = None
    locked: bool = False
    auto_locked: bool = False
    muted: bool = False
    volume: float = 0.0  # current (smoothed) volume 0..1
    controlling: bool = False  # True while the pinch is adjusting volume
    hold_mute: float = 0.0
    hold_play: float = 0.0
    hold_thumb: float = 0.0
    paused: bool = False
    status: str = "Show your hand"
    events: list[Event] = field(default_factory=list)
    coasting: bool = False  # hand briefly lost: the last pose is being held
    settling: bool = False  # hand just re-appeared: volume control is paused for a moment
    norm: Optional[np.ndarray] = None  # (21, 3) image-normalised landmarks (x, y, z)
    world: Optional[np.ndarray] = None  # (21, 3) real 3D landmarks in metres (hand-centred)


def _default_press(key: str) -> None:
    import pyautogui

    pyautogui.PAUSE = 0
    pyautogui.press(key)


class GestureEngine:
    def __init__(
        self,
        audio: Any,
        press: Optional[Callable[[str], None]] = None,
        now: Optional[float] = None,
        hotkey: Optional[Callable[..., None]] = None,
        launch: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.audio = audio
        self.press = press or _default_press
        self.runner = ActionRunner(audio, self.press, hotkey, launch)
        self.thumb_hold = Hold()
        self._thumb_repeat_at = 0.0
        self.smoothed = float(audio.get())
        self.mute_hold, self.play_hold = Hold(), Hold()
        self.swipe = SwipeDetector()
        self.last_hand_time = time.time() if now is None else now
        self.auto_locked = False
        self.unlock_armed = False
        self._lock_active = False
        self._last_lock_event = 0.0
        self._muted_cache = False
        self._muted_t = -1.0
        self._last_lm: Optional[Sequence[Any]] = None  # for riding out short dropouts
        self._last_world: Optional[Sequence[Any]] = None
        self._settle_until = 0.0  # volume stays frozen until then
        self._ramp_until = 0.0  # then the step limit eases in

    # -- helpers ---------------------------------------------------------
    def _muted(self, now: float) -> bool:
        if now - self._muted_t > 0.3:  # avoid a COM call on every frame
            self._muted_cache = bool(self.audio.muted())
            self._muted_t = now
        return self._muted_cache

    def _no_hand(self, res: FrameResult, s: dict, now: float) -> FrameResult:
        self.mute_hold.update(False, 1.0, now)
        self.play_hold.update(False, 1.0, now)
        self.thumb_hold.update(False, 1.0, now)
        self.swipe.reset()
        self._lock_active = False
        limit = float(s.get("auto_lock_seconds", 0.0))
        if limit > 0 and not self.auto_locked and now - self.last_hand_time >= limit:
            self.auto_locked, self.unlock_armed = True, False
            res.events.append(Event("lock", now))
        self.smoothed = float(self.audio.get())
        res.volume = self.smoothed
        res.muted = self._muted(now)
        res.paused = bool(s.get("paused"))
        res.auto_locked = res.locked = self.auto_locked
        res.status = (
            "Auto-locked: show hand, raise then lower pinky"
            if self.auto_locked
            else "Show your hand"
        )
        return res

    @staticmethod
    def _mapping(s: dict) -> dict:
        return s.get("mapping") or BUILTIN[DEFAULT_PROFILE]

    def _do(self, res: FrameResult, s: dict, gesture: str, now: float) -> None:
        """Run the action the active profile assigns to `gesture` and queue its popup."""
        action = self._mapping(s).get(gesture, "none")
        outcome = self.runner.run(action)
        if outcome is None:
            return
        kind, label = outcome
        if action == "mute_toggle":
            self._muted_cache, self._muted_t = kind == "mute_on", now
        self.smoothed = float(self.audio.get())  # volume may have moved
        res.events.append(Event(kind, now, label))

    # -- main entry point -------------------------------------------------
    def update(
        self,
        landmarks: Optional[Sequence[Any]],
        size: tuple[int, int],
        s: dict,
        now: Optional[float] = None,
        world: Optional[Sequence[Any]] = None,
    ) -> FrameResult:
        now = time.time() if now is None else now
        res = FrameResult()
        coasting = False
        if landmarks is None:
            grace = float(s.get("lost_grace", 0.3))
            if self._last_lm is not None and now - self.last_hand_time <= grace:
                landmarks, world, coasting = self._last_lm, self._last_world, True  # hold last pose
            else:
                self._last_lm = self._last_world = None
                return self._no_hand(res, s, now)
        else:
            self._last_lm, self._last_world = landmarks, world
            gap = now - self.last_hand_time
            if gap > max(0.25, float(s.get("lost_grace", 0.3)) + 0.05):  # really gone: re-acquire
                delay = float(s.get("settle_seconds", 0.4))
                self._settle_until = now + delay
                self._ramp_until = self._settle_until + 0.6
            self.last_hand_time = now
        res.coasting = coasting

        w, h = size
        pose = analyze(landmarks)
        res.has_hand, res.pose = True, pose
        res.landmarks_px = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        res.norm = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in landmarks], np.float32)
        if world is not None:
            res.world = np.array([[p.x, p.y, p.z] for p in world], np.float32)
        (x0, y0), (x1, y1) = res.landmarks_px[WRIST], res.landmarks_px[MIDDLE_MCP]
        res.palm_px = math.hypot(x1 - x0, y1 - y0)

        if s.get("paused"):  # tracking continues, but nothing is triggered
            for hold in (self.mute_hold, self.play_hold, self.thumb_hold):
                hold.update(False, 1.0, now)
            self.swipe.reset()
            self._lock_active = False
            self.smoothed = float(self.audio.get())
            res.paused = True
            res.volume, res.muted = self.smoothed, self._muted(now)
            res.status = "Paused - gestures are off"
            return res

        # --- lock: pinky raised (manual) or auto-lock after the hand disappeared ---
        pinky_lock = bool(s["enable_lock"]) and pose.pinky_up
        if self.auto_locked:
            if pose.pinky_up:
                self.unlock_armed = True
            elif self.unlock_armed:
                self.auto_locked, self.unlock_armed = False, False
                res.events.append(Event("unlock", now))
        locked = pinky_lock or self.auto_locked

        # popup only for a deliberate pinky lock (an open-palm swipe also has the pinky up)
        if now - self._last_lock_event > 0.5:
            if pinky_lock and not pose.palm_open and not self._lock_active:
                self._lock_active, self._last_lock_event = True, now
                res.events.append(Event("lock", now))
            elif not pinky_lock and self._lock_active:
                self._lock_active, self._last_lock_event = False, now
                res.events.append(Event("unlock", now))

        # --- volume: pinch distance normalised by palm size, smoothed, rate-limited ---
        controlling = (
            bool(s["enable_volume"])
            and not locked
            and not pose.fist
            and not pose.peace
            and not pose.palm_open
            and not pose.thumb_up
            and not pose.thumb_down
        )
        res.settling = controlling and now < self._settle_until
        if res.settling:
            controlling = False  # hold the volume while the hand settles
        if controlling:
            lo = float(s["min_ratio"])
            hi = max(float(s["max_ratio"]), lo + 0.05)
            target = float(np.clip(np.interp(pose.ratio, [lo, hi], [0.0, 1.0]), 0.0, 1.0))
            step = float(s["smoothing"]) * (target - self.smoothed)
            limit = MAX_VOLUME_STEP
            if now < self._ramp_until:  # ease in after re-acquiring the hand
                k = 1.0 - (self._ramp_until - now) / 0.6
                limit = 0.01 + (MAX_VOLUME_STEP - 0.01) * min(max(k, 0.0), 1.0)
            self.smoothed += float(np.clip(step, -limit, limit))
            self.audio.set(self.smoothed)
        else:
            self.smoothed = float(self.audio.get())  # follow the real system volume
        res.controlling = controlling

        # --- gestures -> actions (what each one does comes from the active profile) ---
        hold_s = float(s["hold_seconds"])
        fired, res.hold_mute = self.mute_hold.update(
            bool(s["enable_mute"]) and pose.fist, hold_s, now
        )
        if fired:
            self._do(res, s, "fist_hold", now)
        fired, res.hold_play = self.play_hold.update(
            bool(s["enable_media"]) and pose.peace, hold_s, now
        )
        if fired:
            self._do(res, s, "peace_hold", now)

        thumb = "thumbs_up_hold" if pose.thumb_up else "thumbs_down_hold"
        thumb_on = bool(s.get("enable_thumbs", True)) and (pose.thumb_up or pose.thumb_down)
        fired, res.hold_thumb = self.thumb_hold.update(thumb_on, hold_s, now)
        if fired:
            self._thumb_repeat_at = now + THUMB_REPEAT
            self._do(res, s, thumb, now)
        elif thumb_on and self.thumb_hold.fired and now >= self._thumb_repeat_at:
            if self._mapping(s).get(thumb) in (
                "volume_up",
                "volume_down",
            ):  # keep stepping while held
                self._do(res, s, thumb, now)
            self._thumb_repeat_at = now + THUMB_REPEAT

        if s["enable_media"] and pose.palm_open:
            direction = self.swipe.update(landmarks[WRIST].x, float(s["swipe_distance"]), now)
            if direction:  # frame is mirrored: right = "next"
                self._do(res, s, "swipe_right" if direction == "next" else "swipe_left", now)
        else:
            self.swipe.reset()

        res.locked, res.auto_locked = locked, self.auto_locked
        res.muted = self._muted(now)
        res.volume = self.smoothed
        if res.settling:
            res.status = "Hand found - hold still for a moment"
        elif self.auto_locked:
            res.status = "Auto-locked: raise then lower pinky"
        elif locked:
            res.status = "Volume locked"
        elif controlling:
            res.status = "Adjusting volume"
        elif pose.thumb_up or pose.thumb_down:
            res.status = "Thumbs up / down: hold for action"
        elif pose.fist:
            res.status = "Fist: hold for action"
        elif pose.peace:
            res.status = "Peace sign: hold for action"
        elif pose.palm_open:
            res.status = "Open palm: swipe left / right"
        else:
            res.status = "Tracking"
        return res
