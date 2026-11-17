"""Visual effects drawn on the camera preview with OpenCV (BGR frames, drawn in place)."""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from gestures import FrameResult

FONT = cv2.FONT_HERSHEY_SIMPLEX

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]
FINGERTIPS = (4, 8, 12, 16, 20)

# BGR colours
ACTIVE = (153, 211, 52)      # emerald
LOCKED = (95, 95, 248)       # soft red
MUTED = (220, 120, 190)      # violet
PLAY = (255, 200, 56)        # azure
VIOLET = (250, 139, 167)     # accent for generic actions
GRAY = (150, 150, 150)
WHITE = (245, 245, 245)
DARK = (28, 28, 28)

POPUP_TOTAL = 1.2    # seconds a popup is visible
TRAIL_LIFE = 0.6     # seconds a trail segment takes to fade

POPUP_LABELS = {
    "mute_on": "Muted", "mute_off": "Unmuted", "playpause": "Play / Pause",
    "next": "Next track", "prev": "Previous track", "lock": "Locked", "unlock": "Unlocked",
    "vol_up": "Volume up", "vol_down": "Volume down", "action": "Action",
}


def lerp_color(a: tuple, b: tuple, t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def volume_color(v: float) -> tuple[int, int, int]:
    """Blue at low volume to red at high volume."""
    return lerp_color((255, 140, 30), (50, 60, 255), v)


def put_text(img, text, org, scale=0.6, color=WHITE, thick=1) -> None:
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)  # outline
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def put_text_centered(img, text, center, scale=0.6, color=WHITE, thick=1) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    put_text(img, text, (int(center[0] - tw / 2), int(center[1] + th / 2)), scale, color, thick)


def draw_pill(img, org, text: str, color, scale: float = 0.5, k: float = 1.0) -> None:
    """Rounded dark badge with a coloured status dot, top-left corner at `org` (k = size factor)."""
    scale *= k
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    h = int(30 * k)
    w = tw + int(44 * k)
    roi = img[y:y + h, x:x + w]
    if roi.shape[0] < h or roi.shape[1] < w:
        return
    over = roi.copy()
    cv2.rectangle(over, (h // 2, 0), (w - h // 2, h), (24, 19, 15), -1)
    cv2.circle(over, (h // 2, h // 2), h // 2, (24, 19, 15), -1, cv2.LINE_AA)
    cv2.circle(over, (w - h // 2, h // 2), h // 2, (24, 19, 15), -1, cv2.LINE_AA)
    cv2.addWeighted(over, 0.82, roi, 0.18, 0, dst=roi)
    cv2.circle(roi, (int(16 * k), h // 2), int(5 * k), color, -1, cv2.LINE_AA)
    cv2.putText(roi, text, (int(30 * k), h // 2 + th // 2), FONT, scale, (235, 232, 228), 1,
                cv2.LINE_AA)


def _poly(img, pts, color) -> None:
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color, cv2.LINE_AA)


def draw_icon(img, kind: str, cx: int, cy: int, s: float, color) -> None:
    """Draw a simple vector icon of half-size `s` centred on (cx, cy)."""
    i = int
    if kind in ("mute_on", "mute_off"):
        cv2.rectangle(img, (i(cx - s), i(cy - s * .4)), (i(cx - s * .35), i(cy + s * .4)), color, -1)
        _poly(img, [(cx - s * .35, cy - s * .4), (cx + s * .25, cy - s * .9),
                    (cx + s * .25, cy + s * .9), (cx - s * .35, cy + s * .4)], color)
        if kind == "mute_on":   # speaker with a cross
            cv2.line(img, (i(cx + s * .5), i(cy - s * .45)), (i(cx + s * 1.1), i(cy + s * .45)), color, 3, cv2.LINE_AA)
            cv2.line(img, (i(cx + s * .5), i(cy + s * .45)), (i(cx + s * 1.1), i(cy - s * .45)), color, 3, cv2.LINE_AA)
        else:                   # speaker with sound waves
            for r in (.55, .95):
                cv2.ellipse(img, (i(cx + s * .25), cy), (i(s * r), i(s * r)), 0, -45, 45, color, 3, cv2.LINE_AA)
    elif kind == "playpause":
        _poly(img, [(cx - s * 1.0, cy - s * .8), (cx - s * 1.0, cy + s * .8), (cx - s * .05, cy)], color)
        cv2.rectangle(img, (i(cx + s * .3), i(cy - s * .8)), (i(cx + s * .55), i(cy + s * .8)), color, -1)
        cv2.rectangle(img, (i(cx + s * .8), i(cy - s * .8)), (i(cx + s * 1.05), i(cy + s * .8)), color, -1)
    elif kind in ("next", "prev"):
        d = 1 if kind == "next" else -1
        _poly(img, [(cx - d * s * .8, cy - s * .8), (cx - d * s * .8, cy + s * .8), (cx + d * s * .5, cy)], color)
        x = cx + d * s * .8
        cv2.rectangle(img, (i(x - s * .12), i(cy - s * .8)), (i(x + s * .12), i(cy + s * .8)), color, -1)
    elif kind in ("vol_up", "vol_down"):
        d = 1 if kind == "vol_up" else -1                       # big arrow + stem
        _poly(img, [(cx - s * .8, cy + d * s * .1), (cx + s * .8, cy + d * s * .1),
                    (cx, cy - d * s * .8)], color)
        y0, y1 = sorted((cy + d * s * .1, cy + d * s * .9))
        cv2.rectangle(img, (i(cx - s * .26), i(y0)), (i(cx + s * .26), i(y1)), color, -1)
    elif kind == "action":                                      # lightning bolt
        _poly(img, [(cx + s * .15, cy - s), (cx - s * .6, cy + s * .15), (cx - s * .05, cy + s * .15),
                    (cx - s * .2, cy + s), (cx + s * .6, cy - s * .2), (cx + s * .05, cy - s * .2)], color)
    elif kind in ("lock", "unlock"):
        cv2.rectangle(img, (i(cx - s * .7), i(cy - s * .05)), (i(cx + s * .7), i(cy + s * .9)), color, -1)
        off = 0 if kind == "lock" else s * .55
        cv2.ellipse(img, (i(cx + off), i(cy - s * .05)), (i(s * .45), i(s * .6)), 0, 180, 360, color, 4, cv2.LINE_AA)


class EffectsManager:
    def __init__(self) -> None:
        self._trail: deque = deque(maxlen=60)       # (time, thumb_xy, index_xy)
        self._popups: list[tuple] = []             # (kind, start time, label)
        self._disp_volume: Optional[float] = None

    # ------------------------------------------------------------------
    def render(self, frame, res: FrameResult, s: dict, fps: float = 0.0,
               now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        h, w = frame.shape[:2]
        if s.get("fx_popups", True):
            self._popups.extend((e.kind, e.time, e.label) for e in res.events)
        self._popups = [p for p in self._popups if now - p[1] < POPUP_TOTAL]

        color = self._state_color(res)
        if s.get("fx_trail", True):
            self._update_trail(res, now)
            self._draw_trail(frame, res, now)
        if res.has_hand and res.pose is not None:
            self._draw_skeleton(frame, res, color, glow=bool(s.get("fx_skeleton", True)))
            self._draw_pinch(frame, res)
            if s.get("fx_ring", True):
                self._draw_hold_ring(frame, res)
        if s.get("fx_arc", True):
            self._draw_arc(frame, res)
        self._draw_badge(frame, res, color)
        if fps:
            put_text(frame, f"{fps:.0f} fps", (w - 72, 22), 0.5, (200, 200, 200))
        if self._popups:
            kind, t0, label = self._popups[-1]
            self._draw_popup(frame, kind, now - t0, label)

    def draw_popups(self, frame, now: Optional[float] = None) -> None:
        """Draw the newest action popup on any frame (used by the 3D view too)."""
        now = time.time() if now is None else now
        self._popups = [p for p in self._popups if now - p[1] < POPUP_TOTAL]
        if self._popups:
            kind, t0, label = self._popups[-1]
            self._draw_popup(frame, kind, now - t0, label)

    def add_events(self, res: FrameResult) -> None:
        self._popups.extend((e.kind, e.time, e.label) for e in res.events)

    # ------------------------------------------------------------------
    @staticmethod
    def _state_color(res: FrameResult):
        if res.locked:
            return LOCKED
        if res.muted:
            return MUTED
        return ACTIVE if res.has_hand else GRAY

    state_color = _state_color

    def _update_trail(self, res: FrameResult, now: float) -> None:
        while self._trail and now - self._trail[0][0] > TRAIL_LIFE:
            self._trail.popleft()
        if res.controlling and res.landmarks_px:
            self._trail.append((now, res.landmarks_px[4], res.landmarks_px[8]))

    def _draw_trail(self, frame, res: FrameResult, now: float) -> None:
        if len(self._trail) < 2:
            return
        layer = np.zeros_like(frame)
        base = volume_color(res.volume)
        pts = list(self._trail)
        for a, b in zip(pts, pts[1:]):
            if b[0] - a[0] > 0.2:      # gap in time: do not connect
                continue
            k = 1.0 - (now - b[0]) / TRAIL_LIFE
            if k <= 0:
                continue
            col = tuple(int(c * k) for c in base)
            thick = max(1, int(6 * k))
            cv2.line(layer, a[1], b[1], col, thick, cv2.LINE_AA)
            cv2.line(layer, a[2], b[2], col, thick, cv2.LINE_AA)
        cv2.add(frame, layer, dst=frame)

    def _draw_skeleton(self, frame, res: FrameResult, color, glow: bool) -> None:
        pts = res.landmarks_px
        if glow:
            layer = np.zeros_like(frame)
            for a, b in HAND_CONNECTIONS:
                cv2.line(layer, pts[a], pts[b], color, 7, cv2.LINE_AA)
            for p in pts:
                cv2.circle(layer, p, 8, color, -1, cv2.LINE_AA)
            layer = cv2.GaussianBlur(layer, (0, 0), 6)
            cv2.add(frame, layer, dst=frame)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (235, 235, 235), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 6 if i in FINGERTIPS else 4
            cv2.circle(frame, p, r, color, -1, cv2.LINE_AA)
            if i in FINGERTIPS:
                cv2.circle(frame, p, r, WHITE, 1, cv2.LINE_AA)

    def _draw_pinch(self, frame, res: FrameResult) -> None:
        thumb, index = res.landmarks_px[4], res.landmarks_px[8]
        col = volume_color(res.volume) if res.controlling else GRAY
        cv2.line(frame, thumb, index, col, 3, cv2.LINE_AA)
        for p in (thumb, index):
            cv2.circle(frame, p, 9, col, 2, cv2.LINE_AA)
        if res.controlling:
            mid = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2 - 22)
            put_text_centered(frame, f"{int(round(res.volume * 100))}%", mid, 0.7, col, 2)

    def _draw_hold_ring(self, frame, res: FrameResult) -> None:
        if res.hold_mute > 0:
            prog, col, label = res.hold_mute, MUTED, "MUTE"
        elif res.hold_play > 0:
            prog, col, label = res.hold_play, PLAY, "PLAY"
        elif res.hold_thumb > 0:
            prog, col, label = res.hold_thumb, ACTIVE, "HOLD"
        else:
            return
        cx, cy = res.landmarks_px[9]
        r = int(res.palm_px) + 18
        cv2.circle(frame, (cx, cy), r, (70, 70, 70), 4, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, -90 + int(360 * prog), col, 7, cv2.LINE_AA)
        if not self._popups:   # the popup already names the action
            put_text_centered(frame, label, (cx, cy - r - 16), 0.6, col, 2)

    def _draw_arc(self, frame, res: FrameResult) -> None:
        h = frame.shape[0]
        cx, cy, r = 74, h - 74, 44
        if self._disp_volume is None:
            self._disp_volume = res.volume
        self._disp_volume += 0.25 * (res.volume - self._disp_volume)   # smooth animation
        v = min(max(self._disp_volume, 0.0), 1.0)

        backdrop = frame.copy()
        cv2.circle(backdrop, (cx, cy), r + 14, DARK, -1, cv2.LINE_AA)
        cv2.addWeighted(backdrop, 0.6, frame, 0.4, 0, dst=frame)
        cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, 270, (85, 85, 85), 6, cv2.LINE_AA)
        col = MUTED if res.muted else volume_color(v)
        if v > 0.004:
            cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, -90 + 360 * v, col, 6, cv2.LINE_AA)
        put_text_centered(frame, "MUTE" if res.muted else f"{int(round(v * 100))}%",
                          (cx, cy), 0.6, WHITE, 2)

    def _draw_badge(self, frame, res: FrameResult, color) -> None:
        if res.auto_locked:
            label = "AUTO-LOCKED"
        elif res.locked:
            label = "LOCKED"
        elif res.muted:
            label = "MUTED"
        else:
            label = "ACTIVE" if res.has_hand else "NO HAND"
        draw_pill(frame, (14, 14), label, color)

    def _draw_popup(self, frame, kind: str, age: float, label: str = "") -> None:
        h, w = frame.shape[:2]
        if age < 0.15:
            alpha, grow = age / 0.15, 0.8 + 0.2 * (age / 0.15)
        elif age < 0.8:
            alpha, grow = 1.0, 1.0
        else:
            alpha, grow = max(0.0, (POPUP_TOTAL - age) / (POPUP_TOTAL - 0.8)), 1.0
        if alpha <= 0:
            return
        col = {"mute_on": MUTED, "mute_off": ACTIVE, "lock": LOCKED, "unlock": ACTIVE,
               "vol_up": ACTIVE, "vol_down": MUTED, "action": VIOLET}.get(kind, PLAY)
        cx, cy, r = w // 2, 78, int(44 * grow)
        ov = frame.copy()
        cv2.circle(ov, (cx, cy), r, DARK, -1, cv2.LINE_AA)
        cv2.circle(ov, (cx, cy), r, col, 3, cv2.LINE_AA)
        draw_icon(ov, kind, cx, cy, r * 0.48, col)
        text = (label if kind in ("action", "vol_up", "vol_down") and label
                else POPUP_LABELS.get(kind, label or kind))
        put_text_centered(ov, text, (cx, cy + r + 22), 0.7, WHITE, 2)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, dst=frame)
