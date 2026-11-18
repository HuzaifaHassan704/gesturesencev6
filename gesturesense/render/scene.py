"""Software 3D renderer: a hand model, a volume tower and a floor grid (NumPy + OpenCV only)."""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from gesturesense.core.gestures import FrameResult
from gesturesense.render.camera import Camera
from gesturesense.render.constants import (
    FONT,
    GAUGE_H,
    GAUGE_POS,
    HAND_HOME_Y,
    SCALE,
    VIEW_H,
    VIEW_W,
)
from gesturesense.render.effects import (
    ACTIVE,
    LOCKED,
    MUTED,
    PLAY,
    WHITE,
    put_text,
    put_text_centered,
)
from gesturesense.render.hands import HandsMixin
from gesturesense.render.layers import _Layer
from gesturesense.render.theme import (
    DEFAULT_THEME,
    THEMES,
    gauge_color,
    make_background,
    scale_color,
)
from gesturesense.render.world import WorldMixin


class Scene3D(HandsMixin, WorldMixin):
    _dt = 1 / 30

    def __init__(self, w: int = VIEW_W, h: int = VIEW_H) -> None:
        self._init_size(w, h)
        # camera state (the GUI can read `yaw` to start a manual drag from the current angle)
        self.yaw, self.pitch, self.zoom = 0.0, math.radians(24), 1.0
        self._phase = 0.0
        # hand state
        self.pts: Optional[np.ndarray] = None
        self.hand_alpha = 0.0
        self.vol = 0.0
        self.pulses: list[tuple[float, tuple]] = []  # (start time, colour) floor ripples
        self.speed = 0.0  # hand speed, scene units / second
        self._prev_raw: Optional[np.ndarray] = None
        self.trail: list[np.ndarray] = []  # recent fingertip positions (5, 3)
        self.sparks: list[list] = []  # [pos(3), vel(3), born, life]
        self.ripples: list[tuple[float, float, float]] = []  # (start time, x, z) from the hand
        self._next_ripple = 0.0
        self._rng = np.random.default_rng(7)
        self._slot2 = self._new_slot()  # state of the second hand (two-hand mode)
        self._next_pulse = 0.0
        self._last_t: Optional[float] = None
        self._now = 0.0
        self._pip_masks: dict = {}

    def _init_size(self, w: int, h: int) -> None:
        self.w, self.h = int(w), int(h)
        self.k = self.h / 540.0  # UI scale factor (text, badges, insets)
        self.cam = Camera(self.w, self.h)
        self.hand_style = getattr(self, "hand_style", "robot")
        self.skin_tone = getattr(self, "skin_tone", "medium")
        self.theme_name = getattr(self, "theme_name", DEFAULT_THEME)
        self.theme = THEMES[self.theme_name]
        self.bg = make_background(self.w, self.h, self.theme)
        self.glow = _Layer(self.w, self.h, 3)
        self.shadow = _Layer(self.w, self.h, 1)

    def set_theme(self, name: str) -> None:
        name = name if name in THEMES else DEFAULT_THEME
        if name != self.theme_name:
            self.theme_name, self.theme = name, THEMES[name]
            self.bg = make_background(self.w, self.h, self.theme)

    def _gc(self, t: float) -> tuple[int, int, int]:
        return gauge_color(t, self.theme["stops"])

    def set_size(self, w: int, h: int) -> None:
        """Change the output resolution (keeps the camera angle, hand and volume state)."""
        if (int(w), int(h)) != (self.w, self.h):
            self._init_size(w, h)

    def _pt(self, img, text, org, scale=0.6, color=WHITE, thick=1) -> None:
        put_text(img, text, org, scale * self.k, color, max(1, round(thick * self.k)))

    def _tc(self, img, text, center, scale=0.6, color=WHITE, thick=1) -> None:
        put_text_centered(img, text, center, scale * self.k, color, max(1, round(thick * self.k)))

    def _update_camera(self, s: dict, dt: float) -> None:
        auto = bool(s.get("orbit_auto", True))
        speed = float(s.get("orbit_speed", 14.0))
        span = float(s.get("orbit_range", 40.0))
        if auto:
            if span >= 179:  # full spin
                self.yaw += math.radians(speed) * dt
            else:  # gentle sweep left <-> right
                self._phase += math.radians(speed) / max(math.radians(span), 1e-3) * dt
                self.yaw = math.radians(span) * math.sin(self._phase)
        else:
            self.yaw += 0.35 * (math.radians(float(s.get("view_yaw", 0.0))) - self.yaw)
            self._phase = (
                math.asin(max(-1, min(1, self.yaw / max(math.radians(span), 1e-3))))
                if abs(self.yaw) < math.radians(span)
                else 0.0
            )
        pitch = math.radians(min(max(float(s.get("view_pitch", 24.0)), 6.0), 75.0))
        self.pitch += 0.25 * (pitch - self.pitch)
        zoom = min(max(float(s.get("view_zoom", 1.0)), 0.5), 2.0)
        self.zoom += 0.25 * (zoom - self.zoom)
        self.cam.set(self.yaw, self.pitch, 9.6 / self.zoom)

    def _update_hand(self, res: FrameResult) -> None:
        if res.has_hand and res.world is not None and res.norm is not None:
            w = np.asarray(res.world, np.float64)
            n = np.asarray(res.norm, np.float64)
            p = np.stack([w[:, 0], -w[:, 1], w[:, 2]], axis=1) * SCALE
            cx, cy = n[9, 0], n[9, 1]  # hand position in the camera image
            p[:, 0] += (cx - 0.5) * 3.0
            p[:, 1] += HAND_HOME_Y + (0.5 - cy) * 1.9
            p[:, 1] += max(0.0, 0.14 - p[:, 1].min())  # never sink below the floor
            if self._prev_raw is not None:
                inst = float(np.linalg.norm(p - self._prev_raw, axis=1).mean()) / self._dt
                self.speed += 0.4 * (inst - self.speed)
            self._prev_raw = p
            if self.pts is None:
                self.pts = p
            else:  # time-based smoothing: fast moves follow tightly
                speed = float(np.linalg.norm(p - self.pts, axis=1).mean())
                a = 1.0 - math.exp(-self._dt / (0.030 / (1.0 + 4.0 * min(speed, 1.5))))
                self.pts = self.pts + a * (p - self.pts)
            self.hand_alpha += 0.25 * (1.0 - self.hand_alpha)
        else:
            self.hand_alpha += 0.12 * (0.0 - self.hand_alpha)
            self._prev_raw = None
            self.speed *= 0.8

    _HAND_ATTRS = (
        "pts",
        "hand_alpha",
        "speed",
        "_prev_raw",
        "trail",
        "sparks",
        "ripples",
        "_next_ripple",
    )

    @staticmethod
    def _new_slot() -> dict:
        return {
            "pts": None,
            "hand_alpha": 0.0,
            "speed": 0.0,
            "_prev_raw": None,
            "trail": [],
            "sparks": [],
            "ripples": [],
            "_next_ripple": 0.0,
        }

    def _swap_slot(self) -> None:
        """Exchange the live hand state with the second hand's, so every hand-drawing routine
        can be reused for the second hand unchanged. Call twice to swap back."""
        for k in self._HAND_ATTRS:
            self._slot2[k], live = getattr(self, k), self._slot2[k]
            setattr(self, k, live)

    def render(
        self,
        res: FrameResult,
        s: dict,
        fps: float = 0.0,
        now: Optional[float] = None,
        res2: Optional[FrameResult] = None,
        second_role: str = "brightness",
    ) -> np.ndarray:
        now = time.time() if now is None else now
        dt = 1 / 30 if self._last_t is None else min(max(now - self._last_t, 0.0), 0.1)
        self._last_t = now
        self._now = now
        self._dt = max(dt, 1e-3)
        self.set_theme(str(s.get("theme", DEFAULT_THEME)))
        self.hand_style = str(s.get("hand_style", "robot"))
        self.skin_tone = str(s.get("skin_tone", "medium"))
        self._update_camera(s, dt)
        self._update_hand(res)
        self._swap_slot()  # second hand (fades out when absent)
        self._update_hand(res2 if res2 is not None else FrameResult())
        self._swap_slot()
        self.vol += 0.22 * (res.volume - self.vol)
        state = MUTED if res.muted else LOCKED if res.locked else ACTIVE
        for e in res.events:
            self.pulses.append((e.time, state if e.kind != "playpause" else PLAY))
        if res.controlling and now > self._next_pulse:
            self.pulses.append((now, self._gc(self.vol)))
            self._next_pulse = now + 0.45
        self.pulses = [p for p in self.pulses if now - p[0] < 1.6]

        img = self.bg.copy()
        self.glow.clear()
        self.shadow.clear()
        if s.get("fx3d_grid", True):
            self._draw_grid(img)
        self._draw_pulses(img, now)
        hand_ok = self.pts is not None and self.hand_alpha > 0.03
        if hand_ok and s.get("fx3d_shadow", True):
            self._draw_shadow(img, state)
        hand2_ok = (
            res2 is not None and self._slot2["pts"] is not None and self._slot2["hand_alpha"] > 0.03
        )
        if hand2_ok and s.get("fx3d_shadow", True):
            self._swap_slot()
            self._draw_shadow(img, (255, 196, 64))
            self._swap_slot()
        # tower and hands, farther one first
        tower_depth = self.cam.project(np.array([[GAUGE_POS[0], GAUGE_H / 2, GAUGE_POS[2]]]))[1][0]
        items = [(tower_depth, "tower")]
        if hand_ok:
            items.append((self.cam.project(self.pts)[1].mean(), "hand"))
        if hand2_ok:
            items.append((self.cam.project(self._slot2["pts"])[1].mean(), "hand2"))
        for _, item in sorted(items, key=lambda t: -t[0]):
            if item == "tower":
                self._draw_tower(img, res, state)
            elif item == "hand":
                self._draw_hand(img, res, state)
            else:
                self._swap_slot()
                self._draw_hand(img, res2, (255, 196, 64))
                self._swap_slot()
        if hand_ok:
            self._draw_beam(img, res)
        if hand_ok and s.get("fx3d_motion", True):
            self._draw_motion(img, s, state, now)
        else:
            self.trail.clear()
            self.sparks.clear()
        if hand2_ok:
            self._swap_slot()
            if s.get("fx3d_motion", True):
                self._draw_motion(img, s, state, now)
            self._swap_slot()
            self._draw_level_orb(img, res2, second_role, now)
        if s.get("fx3d_glow", True):
            small = cv2.GaussianBlur(self.glow.img, (0, 0), 2.4)
            wide = cv2.GaussianBlur(small, (0, 0), 7 * self.k * self.glow.s)
            small = cv2.addWeighted(small, 1.0, wide, 0.8, 0)
            halo = cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            cv2.add(img, halo, dst=img)
        if hand_ok and res.has_hand:
            self._draw_hold_ring(img, res)
        self._draw_hud(img, res, s, fps, state)
        return img

    @staticmethod
    def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
        cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
        for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        return m

    def add_inset(self, img, cam_frame, accent, frac: float = 0.28) -> None:
        """Camera picture-in-picture, bottom-left, with a rounded border (frac = width fraction)."""
        aspect = cam_frame.shape[0] / cam_frame.shape[1]
        iw = int(min(max(frac, 0.1), 0.85) * self.w)
        iw = min(iw, int((self.h * 0.88) / aspect))  # always fit inside the view
        ih = int(iw * aspect)
        if iw < 40:
            return
        mask = self._pip_masks.get((iw, ih))
        if mask is None:
            self._pip_masks = {(iw, ih): self._rounded_mask(iw, ih, max(6, int(12 * self.k)))}
            mask = self._pip_masks[(iw, ih)]
        small = cv2.resize(cam_frame, (iw, ih), interpolation=cv2.INTER_AREA)
        pad = int(16 * self.k)
        x0, y0 = pad, self.h - ih - pad
        roi = img[y0 : y0 + ih, x0 : x0 + iw]
        m = (mask.astype(np.float32) / 255.0)[..., None]
        roi[:] = (small * m + roi * (1 - m)).astype(np.uint8)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        cv2.polylines(
            roi, [contour], True, scale_color(accent, 0.9), max(1, int(2 * self.k)), cv2.LINE_AA
        )
        (tw, _), _ = cv2.getTextSize("CAMERA", FONT, 0.4 * self.k, 1)
        self._pt(
            img,
            "CAMERA",
            (x0 + iw - tw - int(10 * self.k), y0 + ih - int(10 * self.k)),
            0.4,
            (235, 235, 235),
            1,
        )
