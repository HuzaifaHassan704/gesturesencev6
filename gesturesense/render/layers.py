"""Alpha layers and polygon blending helpers."""

from __future__ import annotations

import cv2
import numpy as np


def _ip(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _ok(p, lim: float = 20000.0) -> bool:
    return abs(p[0]) < lim and abs(p[1]) < lim


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
        cv2.circle(
            self.img, self.pt(p), max(1, int(r * self.s)), color, -1 if fill else 1, cv2.LINE_AA
        )

    def poly(self, pts, color) -> None:
        cv2.fillConvexPoly(
            self.img, (np.asarray(pts) * self.s).astype(np.int32), color, cv2.LINE_AA
        )

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
