"""Software 3D renderer: a hand model, a volume tower and a floor grid, using only NumPy + OpenCV.

How it works
------------
* `Camera`       orbit camera with a real perspective projection (vectorised with NumPy).
* `Scene3D`      turns one `FrameResult` into one BGR image. It draws, far to near:
                 background -> floor grid -> ground shadow -> (tower | hand, sorted by depth)
                 -> pinch beam -> glow pass -> HUD.
* The hand is built from MediaPipe's *world landmarks* (real 3D positions in metres), so you
  see true finger depth, not just the 2D picture. Bones are shaded tapered capsules, joints
  are lit spheres, and everything is depth-sorted with the painter's algorithm.

Nothing here touches the camera, GUI or audio, so it can be tested and benchmarked on its own.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from effects import (ACTIVE, LOCKED, MUTED, PLAY, WHITE, draw_pill, lerp_color, put_text,
                     put_text_centered)
from gestures import FrameResult

VIEW_W, VIEW_H = 720, 540          # size of the image shown in the window
SCALE = 11.5                       # metres -> scene units (a hand is ~1.6 units long)
HAND_HOME_Y = 1.55                 # height of the hand above the floor
GAUGE_POS = (2.75, 0.0, 0.0)       # base centre of the volume tower
GAUGE_H, GAUGE_R, GAUGE_SLABS, GAUGE_FACETS = 2.7, 0.30, 20, 20
GRID_HALF, GRID_STEP = 5.0, 0.5
GRID_CENTER = (1.1, 0.0)           # (x, z) centre of the floor grid
UP = np.array([0.0, 1.0, 0.0])
FONT = cv2.FONT_HERSHEY_SIMPLEX

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
               (18, 19), (19, 20), (0, 17)]
PALM_EDGES = {(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)}          # drawn as thin lines
PALM_POLY = [0, 1, 5, 9, 13, 17]
# joint radius in scene units, indexed by landmark
RADIUS = [0.115, 0.095, 0.085, 0.075, 0.068,
          0.088, 0.076, 0.066, 0.058,
          0.088, 0.076, 0.066, 0.058,
          0.084, 0.072, 0.062, 0.055,
          0.080, 0.066, 0.056, 0.050]
TIPS = (4, 8, 12, 16, 20)
KNUCKLES = (6, 7, 10, 11, 14, 15, 18, 19)
# skin tones for the "human" hand (RGB)
SKIN_TONES = {"light": (236, 190, 160), "medium": (222, 168, 128), "tan": (190, 134, 98),
              "deep": (132, 88, 62)}
RADIUS = [r * 1.35 for r in RADIUS]    # chunkier, more solid-looking fingers

BODY = (225, 200, 172)             # steel-blue "glass" skin (BGR)
LIGHT_DIR = np.array([-0.45, 0.80, -0.40]) / np.linalg.norm([-0.45, 0.80, -0.40])
GAUGE_STOPS = [(255, 205, 60), (255, 105, 150), (75, 85, 255)]     # azure -> violet -> coral


# ----------------------------------------------------------------------------- helpers
# Colour themes (all BGR). `bg_top/bg_bot` = backdrop gradient, `pool` = light pool on the floor,
# `grid` = (minor, major) line colours, `stops` = tower gradient bottom -> top, `body` = hand skin.
THEMES: dict[str, dict] = {
    "aurora": dict(label="Aurora", stops=GAUGE_STOPS, body=BODY, bg_top=(30, 21, 15),
                   bg_bot=(12, 9, 7), pool=(44, 26, 12), grid=((104, 74, 46), (170, 122, 78))),
    "neon": dict(label="Neon", stops=[(255, 255, 0), (255, 0, 230), (0, 230, 255)],
                 body=(235, 215, 255), bg_top=(30, 10, 28), bg_bot=(14, 4, 12), pool=(70, 18, 56),
                 grid=((120, 44, 110), (200, 70, 180))),
    "ocean": dict(label="Ocean", stops=[(200, 235, 60), (255, 185, 50), (255, 120, 110)],
                  body=(235, 225, 190), bg_top=(36, 22, 10), bg_bot=(14, 8, 3), pool=(56, 38, 8),
                  grid=((120, 86, 40), (190, 140, 66))),
    "sunset": dict(label="Sunset", stops=[(70, 190, 255), (80, 90, 250), (170, 60, 200)],
                   body=(205, 218, 242), bg_top=(30, 18, 38), bg_bot=(10, 6, 18), pool=(40, 34, 78),
                   grid=((70, 74, 150), (110, 118, 220))),
    "mono": dict(label="Mono", stops=[(105, 105, 105), (160, 160, 160), (215, 215, 215)],
                 body=(215, 215, 215), bg_top=(26, 26, 26), bg_bot=(10, 10, 10), pool=(34, 34, 34),
                 grid=((78, 78, 78), (130, 130, 130))),
}
DEFAULT_THEME = "aurora"


def gauge_color(t: float, stops=None) -> tuple[int, int, int]:
    """Colour of the tower at relative height t (0 bottom .. 1 top)."""
    stops = stops or GAUGE_STOPS
    t = min(max(t, 0.0), 1.0) * 2.0
    return lerp_color(stops[0], stops[1], t) if t < 1 else lerp_color(stops[1], stops[2], t - 1.0)


def scale_color(c, k: float) -> tuple[int, int, int]:
    return (int(min(max(c[0] * k, 0), 255)), int(min(max(c[1] * k, 0), 255)),
            int(min(max(c[2] * k, 0), 255)))


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
    glow = np.exp(-(((x * 0.9) ** 2) + ((y - 0.68) * 2.2) ** 2) * 2.4)      # pool of light
    img += glow[..., None] * np.array(th["pool"], np.float32)
    vig = 1.0 - 0.45 * np.clip((x ** 2 + (y - 0.5) ** 2 * 1.4) - 0.35, 0, 1)
    return np.clip(img * vig[..., None], 0, 255).astype(np.uint8)


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
        self.pos = self.target + dist * np.array([math.sin(yaw) * cp, math.sin(pitch),
                                                  -math.cos(yaw) * cp])
        f = self.target - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(UP, f)
        r /= np.linalg.norm(r)
        self.R = np.stack([r, np.cross(f, r), f])           # rows: right, up, forward

    def project(self, pts: np.ndarray):
        """pts (N,3) -> screen xy (N,2), depth (N,), pixels-per-unit (N,)."""
        c = (np.atleast_2d(pts) - self.pos) @ self.R.T
        z = np.maximum(c[:, 2], 1e-3)
        k = self.focal / z
        xy = np.stack([self.w / 2 + c[:, 0] * k, self.h / 2 - c[:, 1] * k], axis=1)
        return xy, c[:, 2], k


class _Layer:
    """Quarter-resolution drawing surface that gets blurred and added back as glow."""

    def __init__(self, w: int, h: int, channels: int) -> None:
        self.s = 0.25
        shape = (int(h * self.s), int(w * self.s)) + ((channels,) if channels > 1 else ())
        self.img = np.zeros(shape, np.uint8)

    def clear(self) -> None:
        self.img[:] = 0

    def pt(self, p):
        return int(p[0] * self.s), int(p[1] * self.s)

    def line(self, a, b, color, thick: float) -> None:
        cv2.line(self.img, self.pt(a), self.pt(b), color, max(1, int(thick * self.s)), cv2.LINE_AA)

    def circle(self, p, r: float, color, fill: bool = True) -> None:
        cv2.circle(self.img, self.pt(p), max(1, int(r * self.s)), color, -1 if fill else 1,
                   cv2.LINE_AA)

    def poly(self, pts, color) -> None:
        cv2.fillConvexPoly(self.img, (np.asarray(pts) * self.s).astype(np.int32), color,
                           cv2.LINE_AA)

    def upscaled(self, w: int, h: int, sigma: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(self.img, (0, 0), sigma)
        return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)


def _fill_alpha(img, pts, color, alpha: float) -> None:
    """Translucent polygon, blended only inside its bounding box (cheap)."""
    pts = np.asarray(pts, np.int32)
    x0, y0 = np.maximum(pts.min(0) - 2, 0)
    x1, y1 = pts.max(0) + 3
    h, w = img.shape[:2]
    x1, y1 = min(x1, w), min(y1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.fillPoly(over, [pts - [x0, y0]], color, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)


# ------------------------------------------------------------------------------ scene
class Scene3D:
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
        self.pulses: list[tuple[float, tuple]] = []      # (start time, colour) floor ripples
        self.speed = 0.0                                 # hand speed, scene units / second
        self._prev_raw: Optional[np.ndarray] = None
        self.trail: list[np.ndarray] = []                # recent fingertip positions (5, 3)
        self.sparks: list[list] = []                     # [pos(3), vel(3), born, life]
        self.ripples: list[tuple[float, float, float]] = []   # (start time, x, z) from the hand
        self._next_ripple = 0.0
        self._rng = np.random.default_rng(7)
        self._slot2 = self._new_slot()                   # state of the second hand (two-hand mode)
        self._next_pulse = 0.0
        self._last_t: Optional[float] = None
        self._now = 0.0
        self._pip_masks: dict = {}

    def _init_size(self, w: int, h: int) -> None:
        self.w, self.h = int(w), int(h)
        self.k = self.h / 540.0                       # UI scale factor (text, badges, insets)
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

    # text helpers that follow the output size
    def _pt(self, img, text, org, scale=0.6, color=WHITE, thick=1) -> None:
        put_text(img, text, org, scale * self.k, color, max(1, round(thick * self.k)))

    def _tc(self, img, text, center, scale=0.6, color=WHITE, thick=1) -> None:
        put_text_centered(img, text, center, scale * self.k, color, max(1, round(thick * self.k)))

    # ------------------------------------------------------------------ camera control
    def _update_camera(self, s: dict, dt: float) -> None:
        auto = bool(s.get("orbit_auto", True))
        speed = float(s.get("orbit_speed", 14.0))
        span = float(s.get("orbit_range", 40.0))
        if auto:
            if span >= 179:                               # full spin
                self.yaw += math.radians(speed) * dt
            else:                                         # gentle sweep left <-> right
                self._phase += math.radians(speed) / max(math.radians(span), 1e-3) * dt
                self.yaw = math.radians(span) * math.sin(self._phase)
        else:
            self.yaw += 0.35 * (math.radians(float(s.get("view_yaw", 0.0))) - self.yaw)
            self._phase = math.asin(max(-1, min(1, self.yaw / max(math.radians(span), 1e-3)))) \
                if abs(self.yaw) < math.radians(span) else 0.0
        pitch = math.radians(min(max(float(s.get("view_pitch", 24.0)), 6.0), 75.0))
        self.pitch += 0.25 * (pitch - self.pitch)
        zoom = min(max(float(s.get("view_zoom", 1.0)), 0.5), 2.0)
        self.zoom += 0.25 * (zoom - self.zoom)
        self.cam.set(self.yaw, self.pitch, 9.6 / self.zoom)

    # ------------------------------------------------------------------ hand points
    def _update_hand(self, res: FrameResult) -> None:
        if res.has_hand and res.world is not None and res.norm is not None:
            w = np.asarray(res.world, np.float64)
            n = np.asarray(res.norm, np.float64)
            p = np.stack([w[:, 0], -w[:, 1], w[:, 2]], axis=1) * SCALE
            cx, cy = n[9, 0], n[9, 1]                      # hand position in the camera image
            p[:, 0] += (cx - 0.5) * 3.0
            p[:, 1] += HAND_HOME_Y + (0.5 - cy) * 1.9
            p[:, 1] += max(0.0, 0.14 - p[:, 1].min())      # never sink below the floor
            if self._prev_raw is not None:
                inst = float(np.linalg.norm(p - self._prev_raw, axis=1).mean()) / self._dt
                self.speed += 0.4 * (inst - self.speed)
            self._prev_raw = p
            if self.pts is None:
                self.pts = p
            else:                                          # time-based smoothing: fast moves follow tightly
                speed = float(np.linalg.norm(p - self.pts, axis=1).mean())
                a = 1.0 - math.exp(-self._dt / (0.030 / (1.0 + 4.0 * min(speed, 1.5))))
                self.pts = self.pts + a * (p - self.pts)
            self.hand_alpha += 0.25 * (1.0 - self.hand_alpha)
        else:
            self.hand_alpha += 0.12 * (0.0 - self.hand_alpha)
            self._prev_raw = None
            self.speed *= 0.8

    # ------------------------------------------------------------------ main entry
    # ------------------------------------------------------------------ second hand
    _HAND_ATTRS = ("pts", "hand_alpha", "speed", "_prev_raw", "trail", "sparks", "ripples",
                   "_next_ripple")

    @staticmethod
    def _new_slot() -> dict:
        return {"pts": None, "hand_alpha": 0.0, "speed": 0.0, "_prev_raw": None, "trail": [],
                "sparks": [], "ripples": [], "_next_ripple": 0.0}

    def _swap_slot(self) -> None:
        """Exchange the live hand state with the second hand's, so every hand-drawing routine
        can be reused for the second hand unchanged. Call twice to swap back."""
        for k in self._HAND_ATTRS:
            self._slot2[k], live = getattr(self, k), self._slot2[k]
            setattr(self, k, live)

    def _draw_level_orb(self, img, res2: FrameResult, role: str, now: float) -> None:
        """A glowing sun above the second hand; it grows and brightens with the level."""
        pts = self._slot2["pts"]
        a = self._slot2["hand_alpha"]
        if pts is None or a < 0.05:
            return
        level = float(res2.volume) if res2 is not None else 0.5
        c = pts[9] + np.array([0.0, 1.15, 0.0])
        xy, z, k = self.cam.project(c[None])
        if z[0] < 0.4 or not _ok(xy[0]):
            return
        col = (255, 196, 64)
        r = (0.16 + 0.34 * level) * float(k[0])
        p = xy[0]
        for scale, bright in ((1.6, 0.18), (1.25, 0.35), (1.0, 0.85), (0.62, 1.0)):
            cv2.circle(img, _ip(p), max(2, int(r * scale)), scale_color(col, bright * a), -1, cv2.LINE_AA)
        cv2.circle(img, _ip(p), max(1, int(r * 0.30)), scale_color(WHITE, a), -1, cv2.LINE_AA)
        n = 12
        spin = now * 0.6
        for i in range(n):
            ang = spin + i * 2 * math.pi / n
            d = np.array([math.cos(ang), math.sin(ang)])
            cv2.line(img, _ip(p + d * r * 1.5), _ip(p + d * r * (1.5 + 0.9 * level)),
                     scale_color(col, 0.8 * a), max(1, int(2 * self.k)), cv2.LINE_AA)
        self.glow.circle(p, r * 2.4, scale_color(col, 0.5 * a))
        title = "BRIGHTNESS" if role == "brightness" else "HAND 2"
        self._tc(img, f"{int(round(level * 100))}%", (p[0], p[1] + r * 2.3 + 14 * self.k), 0.6,
                 scale_color(col, a), 2)
        self._tc(img, title, (p[0], p[1] + r * 2.3 + 32 * self.k), 0.38, scale_color(WHITE, 0.8 * a), 1)

    def render(self, res: FrameResult, s: dict, fps: float = 0.0,
               now: Optional[float] = None, res2: Optional[FrameResult] = None,
               second_role: str = "brightness") -> np.ndarray:
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
        self._swap_slot()                                # second hand (fades out when absent)
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
        hand2_ok = (res2 is not None and self._slot2["pts"] is not None
                    and self._slot2["hand_alpha"] > 0.03)
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

    # ------------------------------------------------------------------ motion animations
    def _draw_motion(self, img, s: dict, state, now: float) -> None:
        """Things that react to how fast the hand moves: fingertip streaks, sparks, ripples."""
        amt = min(max(float(s.get("fx3d_motion_amount", 0.7)), 0.0), 1.0)
        dt = self._dt
        energy = min(self.speed / 6.0, 1.0)                 # 0 = still, 1 = fast
        tips = self.pts[list(TIPS)]
        self.trail.append(tips.copy())
        del self.trail[:-int(6 + 10 * amt)]
        col = self._gc(self.vol)
        a = self.hand_alpha
        # streaks behind each fingertip, brighter and longer the faster the hand goes
        if len(self.trail) > 1 and energy > 0.03:
            hist = np.stack(self.trail)                      # (T, 5, 3)
            T = len(hist)
            xy, z, k = self.cam.project(hist.reshape(-1, 3))
            xy = xy.reshape(T, 5, 2)
            ok = (z.reshape(T, 5) > 0.4)
            for j in range(5):
                for t in range(T - 1):
                    if not (ok[t, j] and ok[t + 1, j]):
                        continue
                    f = (t + 1) / T
                    c = scale_color(col, min(1.0, energy * 1.4) * f * a * (0.5 + 0.5 * amt))
                    w = max(1, int((1 + 4 * f) * self.k))
                    cv2.line(img, _ip(xy[t, j]), _ip(xy[t + 1, j]), c, w, cv2.LINE_AA)
                    self.glow.line(xy[t, j], xy[t + 1, j], scale_color(col, f * a * energy), 6 * self.k)
        # sparks thrown off the fingertips
        if energy > 0.12 and amt > 0.0:
            n = int(round(energy * 5 * amt * min(dt * 30, 2.0)))
            for _ in range(n):
                tip = tips[int(self._rng.integers(0, 5))]
                v = self._rng.normal(0, 0.9, 3) + np.array([0, 0.8, 0])
                self.sparks.append([tip.copy(), v, now, 0.5 + 0.5 * float(self._rng.random())])
        alive = []
        for sp in self.sparks:
            age = (now - sp[2]) / sp[3]
            if age >= 1.0 or len(alive) > 160:
                continue
            sp[1][1] -= 2.2 * dt                             # gravity
            sp[0] = sp[0] + sp[1] * dt
            if sp[0][1] < 0.02:
                sp[0][1] = 0.02
                sp[1] = sp[1] * np.array([0.6, -0.4, 0.6])
            alive.append(sp)
        self.sparks = alive
        if alive:
            xy, z, k = self.cam.project(np.array([sp[0] for sp in alive]))
            for sp, p, zz, kk in zip(alive, xy, z, k):
                if zz < 0.4 or not _ok(p):
                    continue
                f = 1.0 - (now - sp[2]) / sp[3]
                c = scale_color(lerp_color(col, WHITE, 0.5), f * a)
                cv2.circle(img, _ip(p), max(1, int(0.045 * kk * f + 0.5)), c, -1, cv2.LINE_AA)
                self.glow.circle(p, 0.12 * kk * f, scale_color(col, f * a))
        # a ripple on the floor under the hand when it whips through the scene
        if energy > 0.55 and now >= self._next_ripple:
            c = self.pts.mean(0)
            self.ripples.append((now, float(c[0]), float(c[2])))
            self._next_ripple = now + 0.35
        self.ripples = [r_ for r_ in self.ripples if now - r_[0] < 0.9]
        for t0, cx, cz in self.ripples:
            age = (now - t0) / 0.9
            xy, z, _ = self.cam.project(self._floor_ring(cx, cz, 0.2 + 1.3 * age, 40))
            if (z > 0.4).all():
                c = scale_color(col, 0.7 * (1 - age) ** 1.5 * amt)
                for p0, p1 in zip(xy[:-1], xy[1:]):
                    self.glow.line(p0, p1, c, 6 * self.k)

    # ------------------------------------------------------------------ floor
    def _grid_static(self):
        if getattr(self, "_gs", None) is None:
            cx, cz = GRID_CENTER
            n = int(2 * GRID_HALF / GRID_STEP) + 1
            coords = np.linspace(-GRID_HALF, GRID_HALF, n)
            sub = np.linspace(-GRID_HALF, GRID_HALF, 13)
            a, b = sub[:-1], sub[1:]
            starts, ends, major = [], [], []
            for i, c in enumerate(coords):
                m = i % 5 == 0
                for lo, hi in zip(a, b):
                    starts += [(cx + c, 0, cz + lo), (cx + lo, 0, cz + c)]
                    ends += [(cx + c, 0, cz + hi), (cx + hi, 0, cz + c)]
                    major += [m, m]
            starts, ends = np.array(starts, float), np.array(ends, float)
            mid = (starts + ends) / 2
            d = np.hypot(mid[:, 0] - cx, mid[:, 2] - cz) / GRID_HALF
            fade = np.clip(1.0 - d, 0.0, 1.0) ** 1.3
            keep = fade > 0.04
            self._gs = (starts[keep], ends[keep], np.array(major)[keep], fade[keep])
        return self._gs

    def _draw_grid(self, img) -> None:
        starts, ends, major, fade = self._grid_static()
        p0, z0, _ = self.cam.project(starts)
        p1, z1, _ = self.cam.project(ends)
        ok = (z0 > 0.4) & (z1 > 0.4) & (np.abs(p0) < 20000).all(1) & (np.abs(p1) < 20000).all(1)
        base = np.where(major[:, None], np.array(self.theme["grid"][1], float),
                        np.array(self.theme["grid"][0], float))
        cols = np.clip(base * (fade * 0.9)[:, None], 0, 255).astype(int).tolist()
        a = np.rint(p0).astype(int).tolist()
        b = np.rint(p1).astype(int).tolist()
        line = cv2.line
        for i in np.nonzero(ok)[0].tolist():
            line(img, tuple(a[i]), tuple(b[i]), tuple(cols[i]), 1, cv2.LINE_AA)

    def _floor_ring(self, cx, cz, r, n=56) -> np.ndarray:
        a = np.linspace(0, 2 * math.pi, n + 1)
        return np.stack([cx + r * np.cos(a), np.zeros_like(a), cz + r * np.sin(a)], axis=1)

    def _draw_pulses(self, img, now: float) -> None:
        """Expanding light ripples on the floor around the tower (drawn into the glow pass)."""
        for t0, col in self.pulses:
            age = (now - t0) / 1.6
            r = 0.6 + 1.9 * age
            k = (1 - age) ** 1.6
            xy, z, _ = self.cam.project(self._floor_ring(GAUGE_POS[0], GAUGE_POS[2], r))
            if (z > 0.4).all():
                c = scale_color(col, 0.8 * k)
                for a, b in zip(xy[:-1], xy[1:]):
                    self.glow.line(a, b, c, 7)

    def _draw_shadow(self, img, state) -> None:
        pts = self.pts.copy()
        pts[:, 1] = 0.0
        xy, z, k = self.cam.project(pts)
        a = self.hand_alpha
        for i, j in CONNECTIONS:
            if (i, j) in PALM_EDGES:
                continue
            th = (RADIUS[i] + RADIUS[j]) * (k[i] + k[j]) / 2 * 1.5
            self.shadow.line(xy[i], xy[j], int(255 * a), th)
        poly = xy[PALM_POLY]
        self.shadow.poly(poly, int(255 * a))
        for i in range(21):
            self.shadow.circle(xy[i], RADIUS[i] * k[i] * 1.5, int(255 * a))
        m = cv2.GaussianBlur(self.shadow.img, (0, 0), 2.2)
        m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        dark = (m.astype(np.float32) * 0.55).astype(np.uint8)
        cv2.subtract(img, cv2.merge([dark, dark, dark]), dst=img)
        # reticle ring on the floor under the palm + soft coloured pool
        c = pts[[0, 5, 9, 13, 17]].mean(0)
        ring, rz, _ = self.cam.project(self._floor_ring(c[0], c[2], 0.55, 40))
        cv2.polylines(img, [np.rint(ring).astype(np.int32)], False, scale_color(state, 0.55 * a),
                      1, cv2.LINE_AA)
        self.glow.circle(self.cam.project(c[None])[0][0], 0.7 * self.cam.project(c[None])[2][0],
                         scale_color(state, 0.16 * a))

    # ------------------------------------------------------------------ hand
    def _draw_hand_android(self, img, res: FrameResult, state) -> None:
        """White android hand: glossy ceramic plates, black joint collars, a dark mechanical back of
        the hand and blue camera-lens knuckles, with a white forearm and black wrist collar."""
        a = self.hand_alpha
        pts = self.pts
        axis = pts[0] - pts[9]
        axis /= max(np.linalg.norm(axis), 1e-6)
        xy, z, k = self.cam.project(np.vstack([pts, pts[0] + axis * 1.5]))
        if (z < 0.5).any() or not all(_ok(p) for p in xy):
            return
        zmin, zmax = float(z.min()), float(z.max())
        spread = max(zmax - zmin, 0.6)
        white, black, blue = (236, 240, 246), (26, 28, 36), (60, 150, 255)
        x0, y0 = np.maximum(np.floor(xy.min(0)) - 12 * self.k, 0).astype(int)
        x1, y1 = np.minimum(np.ceil(xy.max(0)) + 12 * self.k, [self.w, self.h]).astype(int)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        before = img[y0:y1, x0:x1].copy()

        def lum(depth: float) -> float:
            return (0.74 + 0.26 * (1.0 - (depth - zmin) / spread)) * a

        def ceramic(depth: float):
            L = lum(depth)
            return [(1.0, 0.0, scale_color((96, 112, 148), L)),           # cool rim shadow
                    (0.95, 0.05, scale_color((140, 156, 184), L)),
                    (0.86, 0.11, scale_color((184, 196, 214), L)),
                    (0.74, 0.17, scale_color((214, 223, 236), L)),
                    (0.58, 0.23, scale_color(white, 0.97 * L)),
                    (0.38, 0.29, scale_color(white, 1.05 * L)),
                    (0.20, 0.34, scale_color((250, 252, 255), 1.10 * L)),
                    (0.07, 0.37, scale_color((255, 255, 255), 1.15 * L))]    # specular streak

        def lens(c, r: float, L: float) -> None:
            cv2.circle(img, _ip(c), max(2, int(r)), scale_color((70, 76, 90), L), -1, cv2.LINE_AA)
            cv2.circle(img, _ip(c), max(2, int(r * 0.86)), scale_color(black, L), -1, cv2.LINE_AA)
            cv2.circle(img, _ip(c), max(1, int(r * 0.58)), scale_color(blue, 0.95 * L), -1, cv2.LINE_AA)
            cv2.circle(img, _ip(c), max(1, int(r * 0.30)), scale_color((10, 30, 70), L), -1, cv2.LINE_AA)
            cv2.circle(img, _ip(c - np.array([r * 0.28, r * 0.30])), max(1, int(r * 0.14)),
                       scale_color(WHITE, L), -1, cv2.LINE_AA)
            self.glow.circle(c, r * 1.8, scale_color(blue, 0.30 * a))

        # ---- palm: white armoured shell, recessed dark panel with a blue core, knuckle gap
        poly = xy[PALM_POLY]
        cen = poly.mean(0)
        n_ = np.cross(pts[5] - pts[0], pts[17] - pts[0])
        n_ /= max(np.linalg.norm(n_), 1e-6)
        to_cam = self.cam.pos - pts[9]
        facing = abs(float(n_ @ (to_cam / np.linalg.norm(to_cam))))
        PL = (0.72 + 0.28 * facing) * a
        shell = cen + (poly - cen) * 1.10
        cv2.fillPoly(img, [np.rint(shell).astype(np.int32)], scale_color((96, 112, 148), a), cv2.LINE_AA)
        for s_, tone_, off in ((1.04, (150, 164, 190), 0.0), (0.96, (196, 207, 224), 0.0),
                               (0.84, white, 0.0), (0.66, (246, 249, 253), 1.0), (0.42, (255, 255, 255), 1.0)):
            q = cen + (shell - cen) * s_ + np.array([-2.0, -3.0]) * self.k * off * 2.0
            cv2.fillPoly(img, [np.rint(q).astype(np.int32)], scale_color(tone_, PL), cv2.LINE_AA)
        # recessed dark panel with bright rim and the blue core
        panel = cen + (poly - cen) * 0.50 + (xy[0] - cen) * 0.0
        cv2.fillPoly(img, [np.rint(panel).astype(np.int32)], scale_color((120, 132, 156), PL), cv2.LINE_AA)
        panel2 = cen + (panel - cen) * 0.92
        cv2.fillPoly(img, [np.rint(panel2).astype(np.int32)], scale_color(black, PL), cv2.LINE_AA)
        pr = max(4.0, float(np.hypot(*(xy[9] - xy[0]))) * 0.13)
        core = cen + (xy[0] - cen) * 0.05
        cv2.circle(img, _ip(core), int(pr * 1.5), scale_color((70, 78, 96), PL), -1, cv2.LINE_AA)
        cv2.circle(img, _ip(core), int(pr * 1.25), scale_color((14, 18, 28), PL), -1, cv2.LINE_AA)
        cv2.circle(img, _ip(core), int(pr * 0.85), scale_color(blue, 0.95 * PL), -1, cv2.LINE_AA)
        cv2.circle(img, _ip(core), max(1, int(pr * 0.38)), scale_color((220, 240, 255), PL), -1, cv2.LINE_AA)
        self.glow.circle(core, pr * 2.6, scale_color(blue, 0.32 * a))
        # tendon grooves from the wrist to each knuckle
        for kn in (5, 9, 13, 17):
            g0 = xy[0] + (xy[kn] - xy[0]) * 0.40
            g1 = xy[0] + (xy[kn] - xy[0]) * 0.86
            cv2.line(img, _ip(g0), _ip(g1), scale_color((110, 122, 148), PL), max(1, int(1.6 * self.k)),
                     cv2.LINE_AA)
            cv2.line(img, _ip(g0 + (-1, -1)), _ip(g1 + (-1, -1)), scale_color((255, 255, 255), 0.9 * PL), 1,
                     cv2.LINE_AA)
        # black gap band along the knuckle line
        for kn_a, kn_b in ((5, 9), (9, 13), (13, 17)):
            cv2.line(img, _ip(xy[kn_a]), _ip(xy[kn_b]), scale_color(black, a),
                     max(3, int(RADIUS[5] * k[5] * 1.1)), cv2.LINE_AA)
        # thumb pad
        self._capsule(img, xy[1], (xy[0] + xy[5]) / 2, RADIUS[1] * k[1] * 1.15, RADIUS[5] * k[5] * 0.9,
                      ceramic(max(z[0], z[1])))

        prims = [(z[21] + 0.5, "arm", 0, 21)]
        for i, j in CONNECTIONS:
            if (i, j) in PALM_EDGES:
                continue
            prims.append((max(z[i], z[j]) + 0.03, "bone", i, j))
        for i, j in ((5, 9), (9, 13), (13, 17)):
            prims.append((max(z[i], z[j]) + 0.02, "plate", i, j))
        for i in range(21):
            prims.append((z[i], "joint", i, i))
        prims.sort(key=lambda p: -p[0])
        for depth, kind, i, j in prims:
            if depth < 0.4:
                continue
            L = lum(depth)
            if kind == "arm":
                r0, r1 = 0.24 * k[0], 0.30 * k[21]
                self._capsule(img, xy[0], xy[21], r0, r1, ceramic(depth))
                d = xy[21] - xy[0]
                ln = float(np.hypot(*d))
                if ln > 1:                                        # black wrist collar with rings
                    u = d / ln
                    nrm = np.array([-u[1], u[0]])
                    for t, w in ((0.08, 3.6), (0.16, 2.0), (0.24, 2.6), (0.31, 1.6)):
                        c = xy[0] + d * t
                        rr = (r0 + (r1 - r0) * t) * 1.08
                        cv2.line(img, _ip(c - nrm * rr), _ip(c + nrm * rr), scale_color(black, L),
                                 max(2, int(w * self.k)), cv2.LINE_AA)
                        cv2.line(img, _ip(c - nrm * rr + u * 2), _ip(c + nrm * rr + u * 2),
                                 scale_color((90, 100, 120), L), 1, cv2.LINE_AA)
                    lens(xy[0] + d * 0.18 + nrm * r0 * 0.1, r0 * 0.5, L)
            elif kind == "plate":
                self._capsule(img, xy[i], xy[j], RADIUS[i] * k[i] * 0.72, RADIUS[j] * k[j] * 0.72,
                              ceramic(depth))
            elif kind == "bone":
                r0, r1 = RADIUS[i] * k[i] * 0.92, RADIUS[j] * k[j] * 0.92
                self._capsule(img, xy[i], xy[j], r0, r1, ceramic(depth))
                d = xy[j] - xy[i]
                ln = float(np.hypot(*d))
                if ln > 10:                                       # panel seam + inset groove line
                    u = d / ln
                    nrm = np.array([-u[1], u[0]])
                    c = xy[i] + d * 0.62
                    rr = (r0 + (r1 - r0) * 0.62) * 0.86
                    cv2.line(img, _ip(c - nrm * rr), _ip(c + nrm * rr), scale_color((120, 132, 156), L), 1,
                             cv2.LINE_AA)
                    g0, g1 = xy[i] + u * r0 * 1.1 - nrm * r0 * 0.25, xy[j] - u * r1 * 1.1 - nrm * r1 * 0.25
                    cv2.line(img, _ip(g0), _ip(g1), scale_color((160, 172, 194), 0.9 * L), 1, cv2.LINE_AA)
            else:
                r = RADIUS[i] * k[i]
                if i in TIPS:                                     # smooth white fingertip cap
                    cv2.circle(img, _ip(xy[i]), max(1, int(r * 0.9)), scale_color(white, 0.95 * L), -1,
                               cv2.LINE_AA)
                    cv2.circle(img, _ip(xy[i] - np.array([r * 0.25, r * 0.3])), max(1, int(r * 0.35)),
                               scale_color((255, 255, 255), L), -1, cv2.LINE_AA)
                elif i in (5, 9, 13, 17, 2):                      # lens knuckles
                    cv2.circle(img, _ip(xy[i]), max(2, int(r * 1.05)), scale_color(black, L), -1,
                               cv2.LINE_AA)
                    lens(xy[i], r * 0.85, L)
                elif i != 0:                                      # black collar with a thin bright rim
                    cv2.circle(img, _ip(xy[i]), max(1, int(r * 1.12)), scale_color((120, 132, 156), 0.9 * L),
                               -1, cv2.LINE_AA)
                    cv2.circle(img, _ip(xy[i]), max(1, int(r * 0.85)), scale_color(black, L), -1,
                               cv2.LINE_AA)
                    cv2.circle(img, _ip(xy[i]), max(1, int(r * 0.85)), scale_color((96, 106, 128), L), 1,
                               cv2.LINE_AA)
                    cv2.ellipse(img, _ip(xy[i]), (max(1, int(r * 0.6)), max(1, int(r * 0.6))), 0, 200, 260,
                                scale_color((190, 200, 220), 0.9 * L), 1, cv2.LINE_AA)
        roi = img[y0:y1, x0:x1]
        mask = (roi != before).any(axis=2)
        if mask.any() and self.hand_style in ("gold", "carbon"):
            self._recolor_metal(roi, mask, self.hand_style)
        if mask.any():                                        # blend stacked shapes into one surface
            blur = cv2.GaussianBlur(roi, (0, 0), max(0.6, 0.8 * self.k))
            roi[mask] = blur[mask]
        # blue accent on the tips shows the app state (same colour as the volume beam)
        for i in TIPS:
            self.glow.circle(xy[i], RADIUS[i] * k[i] * 1.3, scale_color(state, 0.16 * a))

    @staticmethod
    def _recolor_metal(roi: np.ndarray, mask: np.ndarray, style: str) -> None:
        """Turn the white android shading into gold or dark carbon: the grey level of every drawn
        pixel picks a colour from the metal's ramp, and the blue lenses become amber / red."""
        px = roi[mask].astype(np.float32)
        lens = (px[:, 2] - px[:, 0]) > 55                      # strongly blue = a lens
        lum = (px @ np.array([0.30, 0.59, 0.11], np.float32)) / 255.0
        if style == "gold":
            out = np.stack([255 * lum ** 0.9 * 1.02, 196 * lum ** 1.15 * 1.02, 72 * lum ** 1.6], axis=1)
        else:
            lum = np.clip(lum, 0, 1.15)
            out = np.stack([26 + 95 * lum ** 2.4, 27 + 96 * lum ** 2.4, 31 + 104 * lum ** 2.4], axis=1)
        if lens.any():
            b = px[lens]
            out[lens] = (np.stack([b[:, 2], b[:, 1] * (0.75 if style == "gold" else 0.25),
                                   b[:, 0] * (0.9 if style == "gold" else 0.25)], axis=1))
        roi[mask] = np.clip(out, 0, 255).astype(np.uint8)

    def _draw_hand_hologram(self, img, res: FrameResult, state) -> None:
        """Translucent cyan hologram: glowing bone lines, joint rings, a faint palm mesh and a
        scan band that sweeps up the hand."""
        a = self.hand_alpha
        pts = self.pts
        axis = pts[0] - pts[9]
        axis /= max(np.linalg.norm(axis), 1e-6)
        xy, z, k = self.cam.project(np.vstack([pts, pts[0] + axis * 1.2]))
        if (z < 0.5).any() or not all(_ok(p) for p in xy):
            return
        zmin, zmax = float(z.min()), float(z.max())
        spread = max(zmax - zmin, 0.6)
        flick = 0.93 + 0.07 * math.sin(self._now * 31.0)
        cyan = (70, 225, 255)
        ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
        band = ymax - (ymax - ymin) * ((self._now * 0.45) % 1.0)      # scan band, moving up

        def L(depth: float, y: Optional[float] = None) -> float:
            base = (0.55 + 0.45 * (1.0 - (depth - zmin) / spread)) * a * flick
            if y is None:
                return base
            return base * (1.0 + 0.9 * math.exp(-((y - band) / (14 * self.k)) ** 2))

        poly = xy[PALM_POLY]
        _fill_alpha(img, poly, scale_color(cyan, 0.9), 0.20 * a)
        cen = poly.mean(0)
        for p in poly:                                         # faint palm mesh
            cv2.line(img, _ip(cen), _ip(p), scale_color(cyan, 0.35 * a), 1, cv2.LINE_AA)
        for i in range(len(poly)):
            cv2.line(img, _ip(poly[i]), _ip(poly[(i + 1) % len(poly)]), scale_color(cyan, 0.7 * a),
                     1, cv2.LINE_AA)
        cv2.line(img, _ip(xy[0]), _ip(xy[21]), scale_color(cyan, 0.5 * a), max(1, int(2 * self.k)),
                 cv2.LINE_AA)
        cv2.line(img, _ip(xy[21] + (-8, 0)), _ip(xy[21] + (8, 0)), scale_color(cyan, 0.8 * a), 1,
                 cv2.LINE_AA)
        order = sorted(range(len(CONNECTIONS)), key=lambda c: -max(z[CONNECTIONS[c][0]], z[CONNECTIONS[c][1]]))
        for c in order:
            i, j = CONNECTIONS[c]
            depth = max(z[i], z[j])
            ym = float((xy[i][1] + xy[j][1]) / 2)
            lv = L(depth, ym)
            col = scale_color(cyan, min(lv, 1.4))
            th = max(1, int(self.k * (1.5 + 3.0 * (RADIUS[i] + RADIUS[j]) * 2)))
            cv2.line(img, _ip(xy[i]), _ip(xy[j]), scale_color(cyan, 0.30 * lv), th + 2, cv2.LINE_AA)
            cv2.line(img, _ip(xy[i]), _ip(xy[j]), col, max(1, th // 2), cv2.LINE_AA)
            self.glow.line(xy[i], xy[j], scale_color(cyan, 0.35 * lv), th * 2)
        for i in range(21):
            lv = L(z[i], float(xy[i][1]))
            r_ = max(2, int(RADIUS[i] * k[i] * (1.0 if i in TIPS else 0.7)))
            cv2.circle(img, _ip(xy[i]), r_, scale_color(cyan, min(lv, 1.4)), 1, cv2.LINE_AA)
            if i in TIPS:
                cv2.circle(img, _ip(xy[i]), max(1, r_ // 2), scale_color(WHITE, min(lv, 1.0)), -1, cv2.LINE_AA)
                self.glow.circle(xy[i], r_ * 2.2, scale_color(state, 0.35 * a))

    def _draw_hand_human(self, img, res: FrameResult, state) -> None:
        """Human-style hand: smooth shaded skin, a joined palm and thumb pad, knuckle creases,
        nails and a sleeve cuff. Drawn, then softened slightly so the shapes blend together."""
        a = self.hand_alpha
        pts = self.pts
        axis = pts[0] - pts[9]
        axis /= max(np.linalg.norm(axis), 1e-6)
        xy, z, k = self.cam.project(np.vstack([pts, pts[0] + axis * 1.5]))
        if (z < 0.5).any() or not all(_ok(p) for p in xy):
            return
        zmin, zmax = float(z.min()), float(z.max())
        spread = max(zmax - zmin, 0.6)
        tone = SKIN_TONES.get(self.skin_tone, SKIN_TONES["medium"])
        x0, y0 = np.maximum(np.floor(xy.min(0)) - 12 * self.k, 0).astype(int)
        x1, y1 = np.minimum(np.ceil(xy.max(0)) + 12 * self.k, [self.w, self.h]).astype(int)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        before = img[y0:y1, x0:x1].copy()

        def lum(depth: float) -> float:
            return (0.78 + 0.22 * (1.0 - (depth - zmin) / spread)) * a

        def skin(depth: float):
            L = lum(depth)
            edge = tuple(min(255, int(c * 0.42 * L + (34 if i == 0 else 0))) for i, c in enumerate(tone))
            return [(1.0, 0.0, edge),
                    (0.92, 0.08, scale_color(tone, 0.62 * L)),
                    (0.76, 0.17, scale_color(tone, 0.86 * L)),
                    (0.52, 0.27, scale_color(tone, 1.04 * L)),
                    (0.24, 0.36, scale_color(lerp_color(tone, WHITE, 0.22), 1.08 * L))]

        # ---- palm: slightly enlarged so the finger bases merge into it, plus the thumb pad
        poly = xy[PALM_POLY]
        cen = poly.mean(0)
        poly = cen + (poly - cen) * 1.10
        n = np.cross(pts[5] - pts[0], pts[17] - pts[0])
        n /= max(np.linalg.norm(n), 1e-6)
        to_cam = self.cam.pos - pts[9]
        facing = abs(float(n @ (to_cam / np.linalg.norm(to_cam))))
        base = (0.70 + 0.22 * facing) * a
        cv2.fillPoly(img, [np.rint(poly).astype(np.int32)], scale_color(tone, 0.5 * a), cv2.LINE_AA)
        inner = cen + (poly - cen) * 0.88
        cv2.fillPoly(img, [np.rint(inner).astype(np.int32)], scale_color(tone, base), cv2.LINE_AA)
        inner2 = cen + (poly - cen) * 0.52 + np.array([-2.0, -3.0]) * self.k
        cv2.fillPoly(img, [np.rint(inner2).astype(np.int32)], scale_color(tone, base + 0.10 * a),
                     cv2.LINE_AA)
        for u, v in ((0.30, 0.55), (0.52, 0.78)):               # soft palm creases
            p0 = xy[17] + (xy[5] - xy[17]) * 0.0 + (xy[0] - xy[17]) * u
            p1 = xy[5] + (xy[0] - xy[5]) * v * 0.55
            cv2.line(img, _ip(p0), _ip(p1), scale_color(tone, 0.5 * a), max(1, int(1.2 * self.k)),
                     cv2.LINE_AA)
        thumb_pad_end = (xy[0] + xy[5]) / 2
        self._capsule(img, xy[1], thumb_pad_end, RADIUS[1] * k[1] * 1.25, RADIUS[5] * k[5] * 0.95,
                      skin(max(z[0], z[1])))

        prims = [(z[21] + 0.5, "arm", 0, 21)]
        for i, j in CONNECTIONS:
            if (i, j) in PALM_EDGES:
                continue
            prims.append((max(z[i], z[j]) + 0.03, "bone", i, j))
        for i in range(21):
            prims.append((z[i], "joint", i, i))
        prims.sort(key=lambda p: -p[0])
        for depth, kind, i, j in prims:
            if depth < 0.4:
                continue
            if kind == "arm":
                r0, r1 = 0.20 * k[0], 0.24 * k[21]
                self._capsule(img, xy[0], xy[21], r0, r1, skin(depth))
                d = xy[21] - xy[0]
                ln = float(np.hypot(*d))
                if ln > 1:                                       # sleeve cuff in the state colour
                    u = d / ln
                    nrm = np.array([-u[1], u[0]])
                    c0, c1 = xy[0] + d * 0.42, xy[21]
                    quad = np.array([c0 + nrm * r0 * 1.22, c1 + nrm * r1 * 1.22,
                                     c1 - nrm * r1 * 1.22, c0 - nrm * r0 * 1.22])
                    cv2.fillConvexPoly(img, np.rint(quad).astype(np.int32),
                                       scale_color(self.theme["body"], 0.35 * a), cv2.LINE_AA)
                    cv2.circle(img, _ip(c1), max(1, int(r1 * 1.22)), scale_color(self.theme["body"], 0.35 * a),
                               -1, cv2.LINE_AA)
                    cv2.line(img, _ip(c0 + nrm * r0 * 1.22), _ip(c0 - nrm * r0 * 1.22),
                             scale_color(state, 0.9 * a), max(2, int(2.5 * self.k)), cv2.LINE_AA)
            elif kind == "bone":
                self._capsule(img, xy[i], xy[j], RADIUS[i] * k[i] * 0.98, RADIUS[j] * k[j] * 0.98,
                              skin(depth))
            else:
                r = RADIUS[i] * k[i] * 0.98
                L = lum(depth)
                cv2.circle(img, _ip(xy[i]), max(1, int(r)), scale_color(tone, 0.80 * L), -1, cv2.LINE_AA)
                cv2.circle(img, _ip(xy[i]), max(1, int(r * 0.80)), scale_color(tone, 0.98 * L), -1,
                           cv2.LINE_AA)
                if i in KNUCKLES and i < 21:                     # crease across the joint
                    prev = {6: 5, 7: 6, 10: 9, 11: 10, 14: 13, 15: 14, 18: 17, 19: 18}[i]
                    d = xy[i] - xy[prev]
                    ln = float(np.hypot(*d))
                    if ln > 4:
                        nrm = np.array([-d[1], d[0]]) / ln
                        for off in (-0.18, 0.18):
                            c = xy[i] + d / ln * r * off
                            cv2.line(img, _ip(c - nrm * r * 0.55), _ip(c + nrm * r * 0.55),
                                     scale_color(tone, 0.52 * L), 1, cv2.LINE_AA)
                if i in TIPS:                                     # nail
                    bone = {4: 3, 8: 7, 12: 11, 16: 15, 20: 19}[i]
                    d = xy[i] - xy[bone]
                    ln = float(np.hypot(*d))
                    if ln > 2:
                        u = d / ln
                        c = xy[i] - u * r * 0.15
                        ax = (max(2, int(r * 0.52)), max(2, int(r * 0.66)))
                        ang = math.degrees(math.atan2(u[1], u[0])) + 90
                        nail = lerp_color(tone, (255, 226, 218), 0.55)
                        cv2.ellipse(img, _ip(c), ax, ang, 0, 360, scale_color(nail, 0.98 * L), -1,
                                    cv2.LINE_AA)
                        cv2.ellipse(img, _ip(c), ax, ang, 0, 360, scale_color(tone, 0.55 * L), 1,
                                    cv2.LINE_AA)
                        cv2.circle(img, _ip(c - u * r * 0.18 + np.array([-1.0, -1.0]) * r * 0.1),
                                   max(1, int(r * 0.14)), scale_color(WHITE, 0.8 * L), -1, cv2.LINE_AA)
        # soften: blur only the pixels the hand touched so stacked shapes read as one surface
        roi = img[y0:y1, x0:x1]
        mask = (roi != before).any(axis=2)
        if mask.any():
            blur = cv2.GaussianBlur(roi, (0, 0), max(0.6, 0.85 * self.k))
            roi[mask] = blur[mask]
        # tip glow so the pinch points still read
        for i in (4, 8):
            self.glow.circle(xy[i], RADIUS[i] * k[i] * 1.6, scale_color(state, 0.18 * a))

    def _draw_hand(self, img, res: FrameResult, state) -> None:
        if self.hand_style == "human":
            self._draw_hand_human(img, res, state)
            return
        if self.hand_style in ("android", "gold", "carbon"):
            self._draw_hand_android(img, res, state)
            return
        if self.hand_style == "hologram":
            self._draw_hand_hologram(img, res, state)
            return
        if self.hand_style != "glass":                      # "robot" (default) or anything unknown
            self._draw_hand_robot(img, res, state)
            return
        pts, a = self.pts, self.hand_alpha
        xy, z, k = self.cam.project(pts)
        zmin, zmax = float(z.min()), float(z.max())
        spread = max(zmax - zmin, 0.6)

        def lum(depth: float) -> float:                      # nearer -> brighter
            return (0.55 + 0.45 * (1.0 - (depth - zmin) / spread)) * a

        body = self.theme["body"]
        # palm: translucent plate, drawn first so fingers sit on top of it
        poly = xy[PALM_POLY]
        if all(_ok(p) for p in poly):
            n = np.cross(pts[5] - pts[0], pts[17] - pts[0])
            n /= max(np.linalg.norm(n), 1e-6)
            facing = abs(float(n @ ((self.cam.pos - pts[9]) / np.linalg.norm(self.cam.pos - pts[9]))))
            _fill_alpha(img, poly, scale_color(body, (0.25 + 0.35 * facing) * a), 0.85)
            for i, j in PALM_EDGES:
                cv2.line(img, _ip(xy[i]), _ip(xy[j]), scale_color(body, 0.8 * a), 1, cv2.LINE_AA)
        prims = []
        for i, j in CONNECTIONS:
            if (i, j) in PALM_EDGES:
                continue
            prims.append((max(z[i], z[j]) + 0.03, "bone", i, j))
        for i in range(21):
            prims.append((z[i], "joint", i, i))
        prims.sort(key=lambda p: -p[0])                      # far first
        light2d = np.array([-0.6, -0.8])
        for depth, kind, i, j in prims:
            if depth < 0.4 or not (_ok(xy[i]) and _ok(xy[j])):
                continue
            if kind == "bone":
                r0, r1 = RADIUS[i] * k[i], RADIUS[j] * k[j]
                d = xy[j] - xy[i]
                length = float(np.hypot(*d))
                if length < 1.0:
                    continue
                nrm = np.array([-d[1], d[0]]) / length
                quad = np.array([xy[i] + nrm * r0, xy[j] + nrm * r1,
                                 xy[j] - nrm * r1, xy[i] - nrm * r0])
                shade = 0.62 + 0.38 * float(abs(nrm @ light2d))
                col = scale_color(body, lum(depth) * shade * 0.85)
                cv2.fillConvexPoly(img, quad.astype(np.int32), col, cv2.LINE_AA)
                hi = xy[i] + (xy[j] - xy[i]) * 0.0 + nrm * (r0 * 0.45) * (1 if nrm @ light2d > 0 else -1)
                hj = xy[j] + nrm * (r1 * 0.45) * (1 if nrm @ light2d > 0 else -1)
                cv2.line(img, _ip(hi), _ip(hj), scale_color(WHITE, lum(depth) * 0.55),
                         max(1, int(r0 * 0.28)), cv2.LINE_AA)
            else:
                r = max(2.0, RADIUS[i] * k[i])
                c = _ip(xy[i])
                L = lum(depth)
                tip = i in TIPS
                cv2.circle(img, c, int(r), scale_color(body, L * 0.62), -1, cv2.LINE_AA)
                off = (int(-r * 0.28), int(-r * 0.28))
                cv2.circle(img, (c[0] + off[0], c[1] + off[1]), max(1, int(r * 0.72)),
                           scale_color(body, L * 0.98), -1, cv2.LINE_AA)
                cv2.circle(img, (c[0] + 2 * off[0] // 1, c[1] + 2 * off[1] // 1),
                           max(1, int(r * 0.22)), scale_color(WHITE, L), -1, cv2.LINE_AA)
                if tip:
                    cv2.circle(img, c, int(r), scale_color(state, a), 2, cv2.LINE_AA)
                    self.glow.circle(xy[i], r * 1.9, scale_color(state, 0.9 * a))
        # faint glow along the bones
        for i, j in CONNECTIONS:
            if (i, j) not in PALM_EDGES and _ok(xy[i]) and _ok(xy[j]):
                self.glow.line(xy[i], xy[j], scale_color(state, 0.22 * a), RADIUS[i] * k[i] * 1.6)


    # ------------------------------------------------------------------ robot hand
    @staticmethod
    def _capsule(img, p0, p1, r0: float, r1: float, layers) -> None:
        """Tapered capsule from p0 to p1 drawn as stacked layers (scale, light offset, colour)
        so it reads as a shiny cylinder: dark edge, mid body, bright highlight strip."""
        d = np.asarray(p1, float) - np.asarray(p0, float)
        length = float(np.hypot(*d))
        if length < 0.5:
            return
        nrm = np.array([-d[1], d[0]]) / length
        light = np.array([-0.55, -0.83])
        side = 1.0 if float(nrm @ light) > 0 else -1.0
        for scale, off, color in layers:
            o = nrm * side * off
            a0, a1 = np.asarray(p0) + o * r0, np.asarray(p1) + o * r1
            q0, q1 = r0 * scale, r1 * scale
            quad = np.array([a0 + nrm * q0, a1 + nrm * q1, a1 - nrm * q1, a0 - nrm * q0])
            cv2.fillConvexPoly(img, np.rint(quad).astype(np.int32), color, cv2.LINE_AA)
            cv2.circle(img, _ip(a0), max(1, int(q0)), color, -1, cv2.LINE_AA)
            cv2.circle(img, _ip(a1), max(1, int(q1)), color, -1, cv2.LINE_AA)

    def _draw_hand_robot(self, img, res: FrameResult, state) -> None:
        """Gunmetal robot hand: capsule fingers with an LED strip, ball joints, glowing fingertips,
        a plated palm with a glowing core and a forearm cuff."""
        a = self.hand_alpha
        pts = self.pts
        axis = pts[0] - pts[9]                                  # forearm points away from the fingers
        axis /= max(np.linalg.norm(axis), 1e-6)
        arm_end = pts[0] + axis * 1.25
        xy, z, k = self.cam.project(np.vstack([pts, arm_end]))
        if (z < 0.5).any():
            return
        zmin, zmax = float(z.min()), float(z.max())
        spread = max(zmax - zmin, 0.6)
        steel = lerp_color((150, 134, 122), self.theme["body"], 0.25)     # gunmetal, lightly tinted by theme

        def lum(depth: float) -> float:
            return (0.62 + 0.38 * (1.0 - (depth - zmin) / spread)) * a

        def metal(depth: float):
            L = lum(depth)
            return [(1.0, 0.0, scale_color(steel, 0.34 * L)),
                    (0.80, 0.16, scale_color(steel, 0.72 * L)),
                    (0.40, 0.34, scale_color(steel, min(1.5 * L, 1.6)))]

        accent = state
        # ---- palm plate (drawn first)
        poly = xy[PALM_POLY]
        if all(_ok(p) for p in poly):
            n = np.cross(pts[5] - pts[0], pts[17] - pts[0])
            n /= max(np.linalg.norm(n), 1e-6)
            to_cam = self.cam.pos - pts[9]
            facing = abs(float(n @ (to_cam / np.linalg.norm(to_cam))))
            base = 0.30 + 0.30 * facing
            cv2.fillPoly(img, [np.rint(poly).astype(np.int32)], scale_color(steel, base * a), cv2.LINE_AA)
            cen = poly[[0, 2, 3, 4, 5]].mean(0)
            inner = cen + (poly - cen) * 0.74 + np.array([-2.0, -3.0]) * self.k
            cv2.fillPoly(img, [np.rint(inner).astype(np.int32)], scale_color(steel, (base + 0.18) * a),
                         cv2.LINE_AA)
            cv2.polylines(img, [np.rint(poly).astype(np.int32)], True, scale_color(steel, 0.95 * a),
                          max(1, int(2 * self.k)), cv2.LINE_AA)
            cv2.polylines(img, [np.rint(inner).astype(np.int32)], True, scale_color(steel, 0.45 * a), 1,
                          cv2.LINE_AA)
            # glowing core in the middle of the palm
            pr = float(np.hypot(*(xy[9] - xy[0])))
            core = tuple(int(v) for v in (cen * 0.55 + xy[9] * 0.0 + xy[0] * 0.45 * 0.0 + cen * 0.45))
            r = max(4, int(pr * 0.16))
            cv2.circle(img, core, int(r * 1.7), scale_color(steel, 0.22 * a), -1, cv2.LINE_AA)
            cv2.circle(img, core, int(r * 1.7), scale_color(steel, 0.8 * a), 1, cv2.LINE_AA)
            cv2.circle(img, core, r, scale_color(accent, 0.95 * a), -1, cv2.LINE_AA)
            cv2.circle(img, core, max(1, r // 3), scale_color(WHITE, a), -1, cv2.LINE_AA)
            self.glow.circle(core, r * 2.2, scale_color(accent, 0.28 * a))

        # ---- sorted parts: forearm, knuckle bar, finger bones, joints
        prims = [(z[21] + 0.5, "arm", 0, 21)]
        for i, j in CONNECTIONS:
            if (i, j) in PALM_EDGES:
                continue
            prims.append((max(z[i], z[j]) + 0.03, "bone", i, j))
        for i, j in ((5, 9), (9, 13), (13, 17)):
            prims.append((max(z[i], z[j]) + 0.02, "plate", i, j))
        for i in range(21):
            prims.append((z[i], "joint", i, i))
        prims.sort(key=lambda p: -p[0])
        for depth, kind, i, j in prims:
            if depth < 0.4 or not (_ok(xy[i]) and _ok(xy[j])):
                continue
            if kind == "arm":
                r0, r1 = 0.17 * k[0], 0.22 * k[21]
                self._capsule(img, xy[0], xy[21], r0, r1, metal(depth))
                for t in (0.22, 0.5):                           # accent rings on the cuff
                    c = xy[0] + (xy[21] - xy[0]) * t
                    d = xy[21] - xy[0]
                    ln = float(np.hypot(*d))
                    if ln < 1:
                        continue
                    nrm = np.array([-d[1], d[0]]) / ln
                    rr = (r0 + (r1 - r0) * t) * 0.92
                    cv2.line(img, _ip(c - nrm * rr), _ip(c + nrm * rr), scale_color(accent, 0.8 * a),
                             max(2, int(3 * self.k)), cv2.LINE_AA)
                    self.glow.line(c - nrm * rr, c + nrm * rr, scale_color(accent, 0.14 * a), 6 * self.k)
            elif kind == "plate":
                self._capsule(img, xy[i], xy[j], RADIUS[i] * k[i] * 0.7, RADIUS[j] * k[j] * 0.7,
                              metal(depth))
            elif kind == "bone":
                r0, r1 = RADIUS[i] * k[i] * 0.95, RADIUS[j] * k[j] * 0.95
                self._capsule(img, xy[i], xy[j], r0, r1, metal(depth))
                d = xy[j] - xy[i]
                ln = float(np.hypot(*d))
                if ln > 8:
                    u = d / ln
                    a0, a1 = xy[i] + u * r0 * 1.2, xy[j] - u * r1 * 1.2
                    cv2.line(img, _ip(a0), _ip(a1), scale_color(accent, 0.85 * a),
                             max(1, int(min(r0, r1) * 0.22)), cv2.LINE_AA)      # LED strip
                    self.glow.line(a0, a1, scale_color(accent, 0.10 * a), min(r0, r1) * 0.9)
                    nrm = np.array([-u[1], u[0]])
                    m = xy[i] + d * 0.5                                           # panel seam
                    cv2.line(img, _ip(m - nrm * r0 * 0.8), _ip(m + nrm * r0 * 0.8),
                             scale_color(steel, 0.2 * a), 1, cv2.LINE_AA)
            else:                                               # joint ball / fingertip
                r = max(3.0, RADIUS[i] * k[i] * 1.2)
                c = _ip(xy[i])
                L = lum(depth)
                if i in TIPS:
                    cv2.circle(img, c, int(r), scale_color(steel, 0.3 * L), -1, cv2.LINE_AA)
                    cv2.circle(img, c, int(r * 0.82), scale_color(accent, 0.9 * a), -1, cv2.LINE_AA)
                    cv2.circle(img, (c[0] - int(r * 0.22), c[1] - int(r * 0.25)), max(1, int(r * 0.38)),
                               scale_color(WHITE, 0.95 * a), -1, cv2.LINE_AA)
                    self.glow.circle(xy[i], r * 1.9, scale_color(accent, 0.42 * a))
                else:
                    cv2.circle(img, c, int(r), scale_color(steel, 0.30 * L), -1, cv2.LINE_AA)
                    cv2.circle(img, (c[0] - int(r * 0.14), c[1] - int(r * 0.16)), max(1, int(r * 0.78)),
                               scale_color(steel, 0.74 * L), -1, cv2.LINE_AA)
                    cv2.circle(img, (c[0] - int(r * 0.30), c[1] - int(r * 0.34)), max(1, int(r * 0.34)),
                               scale_color(steel, min(1.55 * L, 1.7)), -1, cv2.LINE_AA)
                    cv2.circle(img, c, int(r * 0.62), scale_color(accent, 0.55 * a), 1, cv2.LINE_AA)

    def _draw_beam(self, img, res: FrameResult) -> None:
        """Pinch beam (thumb tip - index tip) and a tether that runs to the tower."""
        if not res.has_hand or self.pts is None or self.hand_alpha < 0.3:
            return
        col = self._gc(self.vol) if res.controlling else (150, 150, 150)
        a, b = self.pts[4], self.pts[8]
        xy, z, k = self.cam.project(np.array([a, b]))
        if (z < 0.4).any() or not (_ok(xy[0]) and _ok(xy[1])):
            return
        cv2.line(img, _ip(xy[0]), _ip(xy[1]), scale_color(col, 0.95), 3, cv2.LINE_AA)
        self.glow.line(xy[0], xy[1], scale_color(col, 0.9), 12)
        if not res.controlling:
            return
        mid = (a + b) / 2
        gx, _, gz = GAUGE_POS
        direction = np.array([mid[0] - gx, mid[2] - gz])
        direction /= max(np.linalg.norm(direction), 1e-6)
        end = np.array([gx + GAUGE_R * 1.05 * direction[0], max(self.vol, 0.02) * GAUGE_H,
                        gz + GAUGE_R * 1.05 * direction[1]])
        t = np.linspace(0, 1, 28)[:, None]
        curve = mid[None] * (1 - t) + end[None] * t
        curve[:, 1] += np.sin(t[:, 0] * math.pi) * 0.35           # arc upwards
        cxy, cz, _ = self.cam.project(curve)
        if (cz > 0.4).all():
            phase = int(self._now * 14) % 2                  # dash animation follows the frame clock
            for n in range(len(curve) - 1):
                if (n + phase) % 2 == 0:
                    cv2.line(img, _ip(cxy[n]), _ip(cxy[n + 1]), scale_color(col, 0.9), 2, cv2.LINE_AA)
                    self.glow.line(cxy[n], cxy[n + 1], scale_color(col, 0.55), 8)
        mxy = self.cam.project(mid[None])[0][0]
        self._tc(img, f"{int(round(res.volume * 100))}%", (mxy[0], mxy[1] - 26 * self.k), 0.7,
                          col, 2)

    # ------------------------------------------------------------------ volume tower
    def _draw_tower(self, img, res: FrameResult, state) -> None:
        gx, gy, gz = GAUGE_POS
        N, J = GAUGE_SLABS, GAUGE_FACETS
        pitch_h = GAUGE_H / N
        thick = pitch_h * 0.8
        ang = np.linspace(0, 2 * math.pi, J, endpoint=False)
        ring = np.stack([gx + GAUGE_R * np.cos(ang), np.zeros(J), gz + GAUGE_R * np.sin(ang)], 1)
        ys = np.arange(N) * pitch_h
        V = np.zeros((N, 2, J, 3))
        V[:] = ring[None, None]
        V[:, 0, :, 1] = ys[:, None]
        V[:, 1, :, 1] = (ys + thick)[:, None]
        xy, z, _ = self.cam.project(V.reshape(-1, 3))
        xy = xy.reshape(N, 2, J, 2)
        if (z < 0.5).any() or not np.isfinite(xy).all():
            return
        # facet shading (the same for every slab)
        am = ang + math.pi / J
        normals = np.stack([np.cos(am), np.zeros(J), np.sin(am)], 1)
        centres = np.stack([gx + GAUGE_R * np.cos(am), np.full(J, 1.0), gz + GAUGE_R * np.sin(am)], 1)
        to_cam = self.cam.pos[None] - centres
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        front = (normals * to_cam).sum(1) > 0
        lam = np.clip(normals @ LIGHT_DIR, 0, 1)
        half = to_cam + LIGHT_DIR[None]
        half /= np.linalg.norm(half, axis=1, keepdims=True)
        spec = np.clip((normals * half).sum(1), 0, 1) ** 18

        muted_or_locked = res.muted or res.locked
        fill_total = self.vol * N
        base_pos = self.cam.project(np.array([[gx, 0.0, gz]]))[0][0]
        # base plate
        plate = self.cam.project(self._floor_ring(gx, gz, 0.78, 40))[0]
        cv2.fillPoly(img, [np.rint(plate).astype(np.int32)], (34, 27, 22), cv2.LINE_AA)
        cv2.polylines(img, [np.rint(plate).astype(np.int32)], True, scale_color(state, 0.85), 2,
                      cv2.LINE_AA)
        self.glow.poly(plate, scale_color(state, 0.22))
        inner = self.cam.project(self._floor_ring(gx, gz, 0.52, 36))[0]
        cv2.polylines(img, [np.rint(inner).astype(np.int32)], True, (70, 56, 46), 1, cv2.LINE_AA)

        for i in range(N):                                   # bottom -> top
            f = min(max(fill_total - i, 0.0), 1.0)
            lit = self._gc(i / (N - 1))
            if muted_or_locked:
                lit = lerp_color(lit, state, 0.65)
            dark = (58, 47, 40)
            base = lerp_color(dark, lit, f)
            for j in range(J):
                if not front[j]:
                    continue
                shade = 0.42 + 0.58 * lam[j]
                col = scale_color(base, shade * (0.8 + 0.32 * f))
                col = tuple(min(255, int(c + 255 * spec[j] * (0.12 + 0.28 * f))) for c in col)
                j2 = (j + 1) % J
                quad = np.array([xy[i, 0, j], xy[i, 0, j2], xy[i, 1, j2], xy[i, 1, j]])
                cv2.fillConvexPoly(img, np.rint(quad).astype(np.int32), col, cv2.LINE_AA)
                if f > 0.3:
                    self.glow.poly(quad, scale_color(lit, 0.5 * f))
            cap = xy[i, 1]
            capcol = scale_color(lerp_color((84, 70, 60), lit, f), 1.0 + 0.35 * f)
            cv2.fillConvexPoly(img, np.rint(cap).astype(np.int32), capcol, cv2.LINE_AA)
            if f > 0.3:
                self.glow.poly(cap, scale_color(lit, 0.5 * f))
        # glass cage: top ring + two rails
        top = V[-1, 1].copy()
        top[:, 1] = GAUGE_H + 0.06
        cage_top = self.cam.project(top * [1, 1, 1])[0]
        cv2.polylines(img, [np.rint(cage_top).astype(np.int32)], True, (140, 126, 112), 1, cv2.LINE_AA)
        for j in range(0, J, J // 4):
            a0 = self.cam.project(np.array([[ring[j, 0], 0.0, ring[j, 2]]]))[0][0]
            a1 = self.cam.project(np.array([[ring[j, 0], GAUGE_H + 0.06, ring[j, 2]]]))[0][0]
            cv2.line(img, _ip(a0), _ip(a1), (96, 86, 76), 1, cv2.LINE_AA)

        # tick labels on the camera-facing right side
        right = self.cam.R[0]
        for pct in (0, 25, 50, 75, 100):
            y = pct / 100 * GAUGE_H
            p = np.array([gx, y, gz]) + right * (GAUGE_R + 0.32)
            q = np.array([gx, y, gz]) + right * (GAUGE_R + 0.14)
            pp, qq = self.cam.project(np.array([p, q]))[0]
            cv2.line(img, _ip(qq), _ip(pp), (150, 135, 120), 1, cv2.LINE_AA)
            self._pt(img, str(pct), (int(pp[0] + 5 * self.k), int(pp[1] + 4 * self.k)), 0.38, (185, 170, 155), 1)
        # big readout above the tower
        head = self.cam.project(np.array([[gx, GAUGE_H + 0.55, gz]]))[0][0]
        label = "MUTED" if res.muted else f"{int(round(self.vol * 100))}%"
        col = MUTED if res.muted else self._gc(self.vol)
        self._tc(img, label, (head[0], head[1]), 0.95, col, 2)
        sub = "LOCKED" if res.locked else "VOLUME"
        self._tc(img, sub, (head[0], head[1] + 22 * self.k), 0.4,
                          LOCKED if res.locked else (170, 155, 140), 1)

    # ------------------------------------------------------------------ overlays
    def _draw_hold_ring(self, img, res: FrameResult) -> None:
        if res.hold_mute > 0:
            prog, col = res.hold_mute, MUTED
        elif res.hold_play > 0:
            prog, col = res.hold_play, PLAY
        elif res.hold_thumb > 0:
            prog, col = res.hold_thumb, ACTIVE
        else:
            return
        xy, z, k = self.cam.project(self.pts[[9]])
        r = int(max(30, 0.95 * k[0]))
        c = _ip(xy[0])
        cv2.circle(img, c, r, (70, 62, 55), 4, cv2.LINE_AA)
        cv2.ellipse(img, c, (r, r), 0, -90, -90 + int(360 * prog), col, 7, cv2.LINE_AA)

    def _draw_hud(self, img, res: FrameResult, s: dict, fps: float, state) -> None:
        k, m = self.k, int(14 * self.k)
        if res.auto_locked:
            label = "AUTO-LOCKED"
        elif res.locked:
            label = "LOCKED"
        elif res.muted:
            label = "MUTED"
        else:
            label = "ACTIVE" if res.has_hand else "NO HAND"
        draw_pill(img, (m, m), label,
                  state if res.has_hand or res.locked or res.muted else (150, 150, 150), k=k)
        if fps:
            text = f"{fps:.0f} FPS"
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.45 * k, 1)
            self._pt(img, text, (self.w - tw - m, m + int(16 * k)), 0.45, (190, 178, 165), 1)
        auto = bool(s.get("orbit_auto", True))
        hint = ("AUTO ORBIT  -  drag to take control" if auto
                else "drag to orbit  -  wheel to zoom  -  double-click: auto orbit")
        (tw, _), _ = cv2.getTextSize(hint, FONT, 0.4 * k, 1)
        self._pt(img, hint, (self.w - tw - m, self.h - m), 0.4, (140, 128, 116), 1)
        if not res.has_hand and self.hand_alpha < 0.05:
            self._tc(img, "Show your hand to the camera", (self.w * 0.36, self.h * 0.46), 0.7,
                     (190, 178, 165), 1)

    # ------------------------------------------------------------------ compositing
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
        iw = min(iw, int((self.h * 0.88) / aspect))          # always fit inside the view
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
        roi = img[y0:y0 + ih, x0:x0 + iw]
        m = (mask.astype(np.float32) / 255.0)[..., None]
        roi[:] = (small * m + roi * (1 - m)).astype(np.uint8)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        cv2.polylines(roi, [contour], True, scale_color(accent, 0.9), max(1, int(2 * self.k)),
                      cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize("CAMERA", FONT, 0.4 * self.k, 1)
        self._pt(img, "CAMERA", (x0 + iw - tw - int(10 * self.k), y0 + ih - int(10 * self.k)), 0.4,
                 (235, 235, 235), 1)
