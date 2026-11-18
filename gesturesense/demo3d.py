"""Try the 3D scene without a webcam:  python demo3d.py

An animated synthetic hand pinches and spreads, the volume tower follows, and the camera orbits.
Keys: q = quit, space = pause, a = toggle auto-orbit, m = mute, l = lock.
Options: --save out.png  (render one frame and exit)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from types import SimpleNamespace as NS

import cv2
import numpy as np

from gesturesense.config import settings as cfg
from gesturesense.core.gestures import GestureEngine
from gesturesense.render import VIEW_H, VIEW_W, Scene3D
from gesturesense.render.effects import EffectsManager


def _chain(base, lengths, curls, yaw=0.0):
    """Joint positions of one finger in the y-z plane (y down, fingers point to -y)."""
    pts, p, ang = [np.array(base, float)], np.array(base, float), 0.0
    for length, curl in zip(lengths, curls, strict=True):
        ang += curl
        d = np.array([math.sin(yaw) * math.cos(ang), -math.cos(ang), -math.sin(ang)])
        p = p + d * length
        pts.append(p.copy())
    return pts


def synthetic_hand(spread: float = 0.5, curl: float = 1.0, tilt: float = 0.0) -> np.ndarray:
    """21 world landmarks (metres, y down, z away from camera) for a pinch/spread hand.

    spread 0 = thumb touching index, 1 = thumb far away. curl = how closed the other 3 fingers are.
    """
    lm = np.zeros((21, 3))
    lm[0] = [0.0, 0.085, 0.0]
    idx = _chain([-0.030, -0.012, 0.0], [0.040, 0.026, 0.021], [0.05, 0.25, 0.3], -0.04)
    mid = _chain(
        [-0.004, -0.018, 0.0], [0.044, 0.030, 0.024], [0.05 + 0.85 * curl, 0.95 * curl, 0.8 * curl]
    )
    rng = _chain(
        [0.022, -0.012, 0.0],
        [0.040, 0.027, 0.022],
        [0.05 + 0.85 * curl, 0.95 * curl, 0.8 * curl],
        0.04,
    )
    pnk = _chain(
        [0.043, -0.002, 0.0],
        [0.031, 0.020, 0.018],
        [0.05 + 0.85 * curl, 0.95 * curl, 0.8 * curl],
        0.09,
    )
    for base, ch in ((5, idx), (9, mid), (13, rng), (17, pnk)):
        for k in range(4):
            lm[base + k] = ch[k]
    tip_pinch = idx[3] + [0.0, 0.004, -0.008]
    tip_open = np.array([-0.125, 0.0, -0.02])
    thumb_tip = tip_pinch * (1 - spread) + tip_open * spread
    lm[1] = [-0.030, 0.056, -0.004]
    lm[2] = [-0.052, 0.034, -0.012]
    lm[4] = thumb_tip
    lm[3] = lm[2] * 0.45 + thumb_tip * 0.55 + [0.0, -0.004, -0.008]
    c = np.array([math.cos(tilt), math.sin(tilt)])  # rotate around the vertical axis
    x, z = lm[:, 0].copy(), lm[:, 2].copy()
    lm[:, 0] = x * c[0] + z * c[1]
    lm[:, 2] = -x * c[1] + z * c[0]
    lm[:, 1] -= lm[:, 1].mean() * 0.4
    return lm


def to_landmarks(world: np.ndarray, cx: float = 0.5, cy: float = 0.5):
    """Fake image-normalised landmarks (objects with .x .y .z) for the gesture engine."""
    n = [NS(x=cx + p[0] * 2.2, y=cy + p[1] * 2.2, z=p[2]) for p in world]
    w = [NS(x=p[0], y=p[1], z=p[2]) for p in world]
    return n, w


class _Audio:
    def __init__(self) -> None:
        self.vol, self.mute = 0.5, False

    def get(self):
        return self.vol

    def set(self, v):
        self.vol = float(v)

    def muted(self):
        return self.mute

    def toggle_mute(self):
        self.mute = not self.mute
        return self.mute


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="render one frame to this PNG and exit")
    ap.add_argument("--t", type=float, default=2.0, help="animation time for --save")
    args = ap.parse_args()

    audio, s = _Audio(), cfg.load()
    s["orbit_auto"] = True
    eng, fx, scene = GestureEngine(audio, press=lambda k: None), EffectsManager(), Scene3D()
    paused, t0, clock = False, 0.0, time.time()

    def frame_at(t: float, now: float):
        spread = 0.5 + 0.5 * math.sin(t * 0.9)
        world = synthetic_hand(spread, 1.0, tilt=0.35 * math.sin(t * 0.5))
        norm, wlm = to_landmarks(
            world, 0.5 + 0.12 * math.sin(t * 0.6), 0.5 + 0.05 * math.sin(t * 0.8)
        )
        # keep this synthetic pose in "pinch / spread" mode: only the index finger is extended
        res = eng.update(norm, (VIEW_W, VIEW_H), s, now=now, world=wlm)
        fx.add_events(res)
        img = scene.render(res, s, fps=60.0, now=now)
        fx.draw_popups(img, now)
        return img

    if args.save:
        now = 0.0
        for i in range(int(args.t * 30)):  # run the animation up to t
            now = i / 30
            img = frame_at(now, now)
        cv2.imwrite(args.save, img)
        print("saved", args.save)
        return

    while True:
        now = time.time()
        if not paused:
            t0 += now - clock
        clock = now
        img = frame_at(t0, now)
        cv2.imshow("GestureSense 3D demo", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("a"):
            s["orbit_auto"] = not s["orbit_auto"]
        if k == ord("m"):
            audio.toggle_mute()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
