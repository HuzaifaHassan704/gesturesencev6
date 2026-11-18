"""Hand drawing: robot, android, human and hologram styles (mixin for Scene3D)."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from gesturesense.core.gestures import FrameResult
from gesturesense.render.constants import (
    CONNECTIONS,
    KNUCKLES,
    PALM_EDGES,
    PALM_POLY,
    RADIUS,
    SKIN_TONES,
    TIPS,
)
from gesturesense.render.effects import WHITE, lerp_color
from gesturesense.render.layers import _fill_alpha, _ip, _ok
from gesturesense.render.theme import scale_color


class HandsMixin:
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
            return [
                (1.0, 0.0, scale_color((96, 112, 148), L)),  # cool rim shadow
                (0.95, 0.05, scale_color((140, 156, 184), L)),
                (0.86, 0.11, scale_color((184, 196, 214), L)),
                (0.74, 0.17, scale_color((214, 223, 236), L)),
                (0.58, 0.23, scale_color(white, 0.97 * L)),
                (0.38, 0.29, scale_color(white, 1.05 * L)),
                (0.20, 0.34, scale_color((250, 252, 255), 1.10 * L)),
                (0.07, 0.37, scale_color((255, 255, 255), 1.15 * L)),
            ]  # specular streak

        def lens(c, r: float, L: float) -> None:
            cv2.circle(img, _ip(c), max(2, int(r)), scale_color((70, 76, 90), L), -1, cv2.LINE_AA)
            cv2.circle(img, _ip(c), max(2, int(r * 0.86)), scale_color(black, L), -1, cv2.LINE_AA)
            cv2.circle(
                img, _ip(c), max(1, int(r * 0.58)), scale_color(blue, 0.95 * L), -1, cv2.LINE_AA
            )
            cv2.circle(
                img, _ip(c), max(1, int(r * 0.30)), scale_color((10, 30, 70), L), -1, cv2.LINE_AA
            )
            cv2.circle(
                img,
                _ip(c - np.array([r * 0.28, r * 0.30])),
                max(1, int(r * 0.14)),
                scale_color(WHITE, L),
                -1,
                cv2.LINE_AA,
            )
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
        cv2.fillPoly(
            img, [np.rint(shell).astype(np.int32)], scale_color((96, 112, 148), a), cv2.LINE_AA
        )
        for s_, tone_, off in (
            (1.04, (150, 164, 190), 0.0),
            (0.96, (196, 207, 224), 0.0),
            (0.84, white, 0.0),
            (0.66, (246, 249, 253), 1.0),
            (0.42, (255, 255, 255), 1.0),
        ):
            q = cen + (shell - cen) * s_ + np.array([-2.0, -3.0]) * self.k * off * 2.0
            cv2.fillPoly(img, [np.rint(q).astype(np.int32)], scale_color(tone_, PL), cv2.LINE_AA)
        # recessed dark panel with bright rim and the blue core
        panel = cen + (poly - cen) * 0.50 + (xy[0] - cen) * 0.0
        cv2.fillPoly(
            img, [np.rint(panel).astype(np.int32)], scale_color((120, 132, 156), PL), cv2.LINE_AA
        )
        panel2 = cen + (panel - cen) * 0.92
        cv2.fillPoly(img, [np.rint(panel2).astype(np.int32)], scale_color(black, PL), cv2.LINE_AA)
        pr = max(4.0, float(np.hypot(*(xy[9] - xy[0]))) * 0.13)
        core = cen + (xy[0] - cen) * 0.05
        cv2.circle(img, _ip(core), int(pr * 1.5), scale_color((70, 78, 96), PL), -1, cv2.LINE_AA)
        cv2.circle(img, _ip(core), int(pr * 1.25), scale_color((14, 18, 28), PL), -1, cv2.LINE_AA)
        cv2.circle(img, _ip(core), int(pr * 0.85), scale_color(blue, 0.95 * PL), -1, cv2.LINE_AA)
        cv2.circle(
            img,
            _ip(core),
            max(1, int(pr * 0.38)),
            scale_color((220, 240, 255), PL),
            -1,
            cv2.LINE_AA,
        )
        self.glow.circle(core, pr * 2.6, scale_color(blue, 0.32 * a))
        # tendon grooves from the wrist to each knuckle
        for kn in (5, 9, 13, 17):
            g0 = xy[0] + (xy[kn] - xy[0]) * 0.40
            g1 = xy[0] + (xy[kn] - xy[0]) * 0.86
            cv2.line(
                img,
                _ip(g0),
                _ip(g1),
                scale_color((110, 122, 148), PL),
                max(1, int(1.6 * self.k)),
                cv2.LINE_AA,
            )
            cv2.line(
                img,
                _ip(g0 + (-1, -1)),
                _ip(g1 + (-1, -1)),
                scale_color((255, 255, 255), 0.9 * PL),
                1,
                cv2.LINE_AA,
            )
        # black gap band along the knuckle line
        for kn_a, kn_b in ((5, 9), (9, 13), (13, 17)):
            cv2.line(
                img,
                _ip(xy[kn_a]),
                _ip(xy[kn_b]),
                scale_color(black, a),
                max(3, int(RADIUS[5] * k[5] * 1.1)),
                cv2.LINE_AA,
            )
        # thumb pad
        self._capsule(
            img,
            xy[1],
            (xy[0] + xy[5]) / 2,
            RADIUS[1] * k[1] * 1.15,
            RADIUS[5] * k[5] * 0.9,
            ceramic(max(z[0], z[1])),
        )

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
                if ln > 1:  # black wrist collar with rings
                    u = d / ln
                    nrm = np.array([-u[1], u[0]])
                    for t, w in ((0.08, 3.6), (0.16, 2.0), (0.24, 2.6), (0.31, 1.6)):
                        c = xy[0] + d * t
                        rr = (r0 + (r1 - r0) * t) * 1.08
                        cv2.line(
                            img,
                            _ip(c - nrm * rr),
                            _ip(c + nrm * rr),
                            scale_color(black, L),
                            max(2, int(w * self.k)),
                            cv2.LINE_AA,
                        )
                        cv2.line(
                            img,
                            _ip(c - nrm * rr + u * 2),
                            _ip(c + nrm * rr + u * 2),
                            scale_color((90, 100, 120), L),
                            1,
                            cv2.LINE_AA,
                        )
                    lens(xy[0] + d * 0.18 + nrm * r0 * 0.1, r0 * 0.5, L)
            elif kind == "plate":
                self._capsule(
                    img,
                    xy[i],
                    xy[j],
                    RADIUS[i] * k[i] * 0.72,
                    RADIUS[j] * k[j] * 0.72,
                    ceramic(depth),
                )
            elif kind == "bone":
                r0, r1 = RADIUS[i] * k[i] * 0.92, RADIUS[j] * k[j] * 0.92
                self._capsule(img, xy[i], xy[j], r0, r1, ceramic(depth))
                d = xy[j] - xy[i]
                ln = float(np.hypot(*d))
                if ln > 10:  # panel seam + inset groove line
                    u = d / ln
                    nrm = np.array([-u[1], u[0]])
                    c = xy[i] + d * 0.62
                    rr = (r0 + (r1 - r0) * 0.62) * 0.86
                    cv2.line(
                        img,
                        _ip(c - nrm * rr),
                        _ip(c + nrm * rr),
                        scale_color((120, 132, 156), L),
                        1,
                        cv2.LINE_AA,
                    )
                    g0, g1 = (
                        xy[i] + u * r0 * 1.1 - nrm * r0 * 0.25,
                        xy[j] - u * r1 * 1.1 - nrm * r1 * 0.25,
                    )
                    cv2.line(
                        img, _ip(g0), _ip(g1), scale_color((160, 172, 194), 0.9 * L), 1, cv2.LINE_AA
                    )
            else:
                r = RADIUS[i] * k[i]
                if i in TIPS:  # smooth white fingertip cap
                    cv2.circle(
                        img,
                        _ip(xy[i]),
                        max(1, int(r * 0.9)),
                        scale_color(white, 0.95 * L),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        img,
                        _ip(xy[i] - np.array([r * 0.25, r * 0.3])),
                        max(1, int(r * 0.35)),
                        scale_color((255, 255, 255), L),
                        -1,
                        cv2.LINE_AA,
                    )
                elif i in (5, 9, 13, 17, 2):  # lens knuckles
                    cv2.circle(
                        img,
                        _ip(xy[i]),
                        max(2, int(r * 1.05)),
                        scale_color(black, L),
                        -1,
                        cv2.LINE_AA,
                    )
                    lens(xy[i], r * 0.85, L)
                elif i != 0:  # black collar with a thin bright rim
                    cv2.circle(
                        img,
                        _ip(xy[i]),
                        max(1, int(r * 1.12)),
                        scale_color((120, 132, 156), 0.9 * L),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        img,
                        _ip(xy[i]),
                        max(1, int(r * 0.85)),
                        scale_color(black, L),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        img,
                        _ip(xy[i]),
                        max(1, int(r * 0.85)),
                        scale_color((96, 106, 128), L),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.ellipse(
                        img,
                        _ip(xy[i]),
                        (max(1, int(r * 0.6)), max(1, int(r * 0.6))),
                        0,
                        200,
                        260,
                        scale_color((190, 200, 220), 0.9 * L),
                        1,
                        cv2.LINE_AA,
                    )
        roi = img[y0:y1, x0:x1]
        mask = (roi != before).any(axis=2)
        if mask.any() and self.hand_style in ("gold", "carbon"):
            self._recolor_metal(roi, mask, self.hand_style)
        if mask.any():  # blend stacked shapes into one surface
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
        lens = (px[:, 2] - px[:, 0]) > 55  # strongly blue = a lens
        lum = (px @ np.array([0.30, 0.59, 0.11], np.float32)) / 255.0
        if style == "gold":
            out = np.stack([255 * lum**0.9 * 1.02, 196 * lum**1.15 * 1.02, 72 * lum**1.6], axis=1)
        else:
            lum = np.clip(lum, 0, 1.15)
            out = np.stack([26 + 95 * lum**2.4, 27 + 96 * lum**2.4, 31 + 104 * lum**2.4], axis=1)
        if lens.any():
            b = px[lens]
            out[lens] = np.stack(
                [
                    b[:, 2],
                    b[:, 1] * (0.75 if style == "gold" else 0.25),
                    b[:, 0] * (0.9 if style == "gold" else 0.25),
                ],
                axis=1,
            )
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
        band = ymax - (ymax - ymin) * ((self._now * 0.45) % 1.0)  # scan band, moving up

        def L(depth: float, y: Optional[float] = None) -> float:
            base = (0.55 + 0.45 * (1.0 - (depth - zmin) / spread)) * a * flick
            if y is None:
                return base
            return base * (1.0 + 0.9 * math.exp(-(((y - band) / (14 * self.k)) ** 2)))

        poly = xy[PALM_POLY]
        _fill_alpha(img, poly, scale_color(cyan, 0.9), 0.20 * a)
        cen = poly.mean(0)
        for p in poly:  # faint palm mesh
            cv2.line(img, _ip(cen), _ip(p), scale_color(cyan, 0.35 * a), 1, cv2.LINE_AA)
        for i in range(len(poly)):
            cv2.line(
                img,
                _ip(poly[i]),
                _ip(poly[(i + 1) % len(poly)]),
                scale_color(cyan, 0.7 * a),
                1,
                cv2.LINE_AA,
            )
        cv2.line(
            img,
            _ip(xy[0]),
            _ip(xy[21]),
            scale_color(cyan, 0.5 * a),
            max(1, int(2 * self.k)),
            cv2.LINE_AA,
        )
        cv2.line(
            img,
            _ip(xy[21] + (-8, 0)),
            _ip(xy[21] + (8, 0)),
            scale_color(cyan, 0.8 * a),
            1,
            cv2.LINE_AA,
        )
        order = sorted(
            range(len(CONNECTIONS)), key=lambda c: -max(z[CONNECTIONS[c][0]], z[CONNECTIONS[c][1]])
        )
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
                cv2.circle(
                    img,
                    _ip(xy[i]),
                    max(1, r_ // 2),
                    scale_color(WHITE, min(lv, 1.0)),
                    -1,
                    cv2.LINE_AA,
                )
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
            edge = tuple(
                min(255, int(c * 0.42 * L + (34 if i == 0 else 0))) for i, c in enumerate(tone)
            )
            return [
                (1.0, 0.0, edge),
                (0.92, 0.08, scale_color(tone, 0.62 * L)),
                (0.76, 0.17, scale_color(tone, 0.86 * L)),
                (0.52, 0.27, scale_color(tone, 1.04 * L)),
                (0.24, 0.36, scale_color(lerp_color(tone, WHITE, 0.22), 1.08 * L)),
            ]

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
        cv2.fillPoly(
            img, [np.rint(inner2).astype(np.int32)], scale_color(tone, base + 0.10 * a), cv2.LINE_AA
        )
        for u, v in ((0.30, 0.55), (0.52, 0.78)):  # soft palm creases
            p0 = xy[17] + (xy[5] - xy[17]) * 0.0 + (xy[0] - xy[17]) * u
            p1 = xy[5] + (xy[0] - xy[5]) * v * 0.55
            cv2.line(
                img,
                _ip(p0),
                _ip(p1),
                scale_color(tone, 0.5 * a),
                max(1, int(1.2 * self.k)),
                cv2.LINE_AA,
            )
        thumb_pad_end = (xy[0] + xy[5]) / 2
        self._capsule(
            img,
            xy[1],
            thumb_pad_end,
            RADIUS[1] * k[1] * 1.25,
            RADIUS[5] * k[5] * 0.95,
            skin(max(z[0], z[1])),
        )

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
                if ln > 1:  # sleeve cuff in the state colour
                    u = d / ln
                    nrm = np.array([-u[1], u[0]])
                    c0, c1 = xy[0] + d * 0.42, xy[21]
                    quad = np.array(
                        [
                            c0 + nrm * r0 * 1.22,
                            c1 + nrm * r1 * 1.22,
                            c1 - nrm * r1 * 1.22,
                            c0 - nrm * r0 * 1.22,
                        ]
                    )
                    cv2.fillConvexPoly(
                        img,
                        np.rint(quad).astype(np.int32),
                        scale_color(self.theme["body"], 0.35 * a),
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        img,
                        _ip(c1),
                        max(1, int(r1 * 1.22)),
                        scale_color(self.theme["body"], 0.35 * a),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        img,
                        _ip(c0 + nrm * r0 * 1.22),
                        _ip(c0 - nrm * r0 * 1.22),
                        scale_color(state, 0.9 * a),
                        max(2, int(2.5 * self.k)),
                        cv2.LINE_AA,
                    )
            elif kind == "bone":
                self._capsule(
                    img, xy[i], xy[j], RADIUS[i] * k[i] * 0.98, RADIUS[j] * k[j] * 0.98, skin(depth)
                )
            else:
                r = RADIUS[i] * k[i] * 0.98
                L = lum(depth)
                cv2.circle(
                    img, _ip(xy[i]), max(1, int(r)), scale_color(tone, 0.80 * L), -1, cv2.LINE_AA
                )
                cv2.circle(
                    img,
                    _ip(xy[i]),
                    max(1, int(r * 0.80)),
                    scale_color(tone, 0.98 * L),
                    -1,
                    cv2.LINE_AA,
                )
                if i in KNUCKLES and i < 21:  # crease across the joint
                    prev = {6: 5, 7: 6, 10: 9, 11: 10, 14: 13, 15: 14, 18: 17, 19: 18}[i]
                    d = xy[i] - xy[prev]
                    ln = float(np.hypot(*d))
                    if ln > 4:
                        nrm = np.array([-d[1], d[0]]) / ln
                        for off in (-0.18, 0.18):
                            c = xy[i] + d / ln * r * off
                            cv2.line(
                                img,
                                _ip(c - nrm * r * 0.55),
                                _ip(c + nrm * r * 0.55),
                                scale_color(tone, 0.52 * L),
                                1,
                                cv2.LINE_AA,
                            )
                if i in TIPS:  # nail
                    bone = {4: 3, 8: 7, 12: 11, 16: 15, 20: 19}[i]
                    d = xy[i] - xy[bone]
                    ln = float(np.hypot(*d))
                    if ln > 2:
                        u = d / ln
                        c = xy[i] - u * r * 0.15
                        ax = (max(2, int(r * 0.52)), max(2, int(r * 0.66)))
                        ang = math.degrees(math.atan2(u[1], u[0])) + 90
                        nail = lerp_color(tone, (255, 226, 218), 0.55)
                        cv2.ellipse(
                            img,
                            _ip(c),
                            ax,
                            ang,
                            0,
                            360,
                            scale_color(nail, 0.98 * L),
                            -1,
                            cv2.LINE_AA,
                        )
                        cv2.ellipse(
                            img,
                            _ip(c),
                            ax,
                            ang,
                            0,
                            360,
                            scale_color(tone, 0.55 * L),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            img,
                            _ip(c - u * r * 0.18 + np.array([-1.0, -1.0]) * r * 0.1),
                            max(1, int(r * 0.14)),
                            scale_color(WHITE, 0.8 * L),
                            -1,
                            cv2.LINE_AA,
                        )
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
        if self.hand_style != "glass":  # "robot" (default) or anything unknown
            self._draw_hand_robot(img, res, state)
            return
        pts, a = self.pts, self.hand_alpha
        xy, z, k = self.cam.project(pts)
        zmin, zmax = float(z.min()), float(z.max())
        spread = max(zmax - zmin, 0.6)

        def lum(depth: float) -> float:  # nearer -> brighter
            return (0.55 + 0.45 * (1.0 - (depth - zmin) / spread)) * a

        body = self.theme["body"]
        # palm: translucent plate, drawn first so fingers sit on top of it
        poly = xy[PALM_POLY]
        if all(_ok(p) for p in poly):
            n = np.cross(pts[5] - pts[0], pts[17] - pts[0])
            n /= max(np.linalg.norm(n), 1e-6)
            facing = abs(
                float(n @ ((self.cam.pos - pts[9]) / np.linalg.norm(self.cam.pos - pts[9])))
            )
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
        prims.sort(key=lambda p: -p[0])  # far first
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
                quad = np.array(
                    [xy[i] + nrm * r0, xy[j] + nrm * r1, xy[j] - nrm * r1, xy[i] - nrm * r0]
                )
                shade = 0.62 + 0.38 * float(abs(nrm @ light2d))
                col = scale_color(body, lum(depth) * shade * 0.85)
                cv2.fillConvexPoly(img, quad.astype(np.int32), col, cv2.LINE_AA)
                hi = (
                    xy[i]
                    + (xy[j] - xy[i]) * 0.0
                    + nrm * (r0 * 0.45) * (1 if nrm @ light2d > 0 else -1)
                )
                hj = xy[j] + nrm * (r1 * 0.45) * (1 if nrm @ light2d > 0 else -1)
                cv2.line(
                    img,
                    _ip(hi),
                    _ip(hj),
                    scale_color(WHITE, lum(depth) * 0.55),
                    max(1, int(r0 * 0.28)),
                    cv2.LINE_AA,
                )
            else:
                r = max(2.0, RADIUS[i] * k[i])
                c = _ip(xy[i])
                L = lum(depth)
                tip = i in TIPS
                cv2.circle(img, c, int(r), scale_color(body, L * 0.62), -1, cv2.LINE_AA)
                off = (int(-r * 0.28), int(-r * 0.28))
                cv2.circle(
                    img,
                    (c[0] + off[0], c[1] + off[1]),
                    max(1, int(r * 0.72)),
                    scale_color(body, L * 0.98),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    img,
                    (c[0] + 2 * off[0] // 1, c[1] + 2 * off[1] // 1),
                    max(1, int(r * 0.22)),
                    scale_color(WHITE, L),
                    -1,
                    cv2.LINE_AA,
                )
                if tip:
                    cv2.circle(img, c, int(r), scale_color(state, a), 2, cv2.LINE_AA)
                    self.glow.circle(xy[i], r * 1.9, scale_color(state, 0.9 * a))
        # faint glow along the bones
        for i, j in CONNECTIONS:
            if (i, j) not in PALM_EDGES and _ok(xy[i]) and _ok(xy[j]):
                self.glow.line(xy[i], xy[j], scale_color(state, 0.22 * a), RADIUS[i] * k[i] * 1.6)

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
        axis = pts[0] - pts[9]  # forearm points away from the fingers
        axis /= max(np.linalg.norm(axis), 1e-6)
        arm_end = pts[0] + axis * 1.25
        xy, z, k = self.cam.project(np.vstack([pts, arm_end]))
        if (z < 0.5).any():
            return
        zmin, zmax = float(z.min()), float(z.max())
        spread = max(zmax - zmin, 0.6)
        steel = lerp_color(
            (150, 134, 122), self.theme["body"], 0.25
        )  # gunmetal, lightly tinted by theme

        def lum(depth: float) -> float:
            return (0.62 + 0.38 * (1.0 - (depth - zmin) / spread)) * a

        def metal(depth: float):
            L = lum(depth)
            return [
                (1.0, 0.0, scale_color(steel, 0.34 * L)),
                (0.80, 0.16, scale_color(steel, 0.72 * L)),
                (0.40, 0.34, scale_color(steel, min(1.5 * L, 1.6))),
            ]

        accent = state
        # ---- palm plate (drawn first)
        poly = xy[PALM_POLY]
        if all(_ok(p) for p in poly):
            n = np.cross(pts[5] - pts[0], pts[17] - pts[0])
            n /= max(np.linalg.norm(n), 1e-6)
            to_cam = self.cam.pos - pts[9]
            facing = abs(float(n @ (to_cam / np.linalg.norm(to_cam))))
            base = 0.30 + 0.30 * facing
            cv2.fillPoly(
                img, [np.rint(poly).astype(np.int32)], scale_color(steel, base * a), cv2.LINE_AA
            )
            cen = poly[[0, 2, 3, 4, 5]].mean(0)
            inner = cen + (poly - cen) * 0.74 + np.array([-2.0, -3.0]) * self.k
            cv2.fillPoly(
                img,
                [np.rint(inner).astype(np.int32)],
                scale_color(steel, (base + 0.18) * a),
                cv2.LINE_AA,
            )
            cv2.polylines(
                img,
                [np.rint(poly).astype(np.int32)],
                True,
                scale_color(steel, 0.95 * a),
                max(1, int(2 * self.k)),
                cv2.LINE_AA,
            )
            cv2.polylines(
                img,
                [np.rint(inner).astype(np.int32)],
                True,
                scale_color(steel, 0.45 * a),
                1,
                cv2.LINE_AA,
            )
            # glowing core in the middle of the palm
            pr = float(np.hypot(*(xy[9] - xy[0])))
            core = tuple(
                int(v) for v in (cen * 0.55 + xy[9] * 0.0 + xy[0] * 0.45 * 0.0 + cen * 0.45)
            )
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
                for t in (0.22, 0.5):  # accent rings on the cuff
                    c = xy[0] + (xy[21] - xy[0]) * t
                    d = xy[21] - xy[0]
                    ln = float(np.hypot(*d))
                    if ln < 1:
                        continue
                    nrm = np.array([-d[1], d[0]]) / ln
                    rr = (r0 + (r1 - r0) * t) * 0.92
                    cv2.line(
                        img,
                        _ip(c - nrm * rr),
                        _ip(c + nrm * rr),
                        scale_color(accent, 0.8 * a),
                        max(2, int(3 * self.k)),
                        cv2.LINE_AA,
                    )
                    self.glow.line(
                        c - nrm * rr, c + nrm * rr, scale_color(accent, 0.14 * a), 6 * self.k
                    )
            elif kind == "plate":
                self._capsule(
                    img, xy[i], xy[j], RADIUS[i] * k[i] * 0.7, RADIUS[j] * k[j] * 0.7, metal(depth)
                )
            elif kind == "bone":
                r0, r1 = RADIUS[i] * k[i] * 0.95, RADIUS[j] * k[j] * 0.95
                self._capsule(img, xy[i], xy[j], r0, r1, metal(depth))
                d = xy[j] - xy[i]
                ln = float(np.hypot(*d))
                if ln > 8:
                    u = d / ln
                    a0, a1 = xy[i] + u * r0 * 1.2, xy[j] - u * r1 * 1.2
                    cv2.line(
                        img,
                        _ip(a0),
                        _ip(a1),
                        scale_color(accent, 0.85 * a),
                        max(1, int(min(r0, r1) * 0.22)),
                        cv2.LINE_AA,
                    )  # LED strip
                    self.glow.line(a0, a1, scale_color(accent, 0.10 * a), min(r0, r1) * 0.9)
                    nrm = np.array([-u[1], u[0]])
                    m = xy[i] + d * 0.5  # panel seam
                    cv2.line(
                        img,
                        _ip(m - nrm * r0 * 0.8),
                        _ip(m + nrm * r0 * 0.8),
                        scale_color(steel, 0.2 * a),
                        1,
                        cv2.LINE_AA,
                    )
            else:  # joint ball / fingertip
                r = max(3.0, RADIUS[i] * k[i] * 1.2)
                c = _ip(xy[i])
                L = lum(depth)
                if i in TIPS:
                    cv2.circle(img, c, int(r), scale_color(steel, 0.3 * L), -1, cv2.LINE_AA)
                    cv2.circle(img, c, int(r * 0.82), scale_color(accent, 0.9 * a), -1, cv2.LINE_AA)
                    cv2.circle(
                        img,
                        (c[0] - int(r * 0.22), c[1] - int(r * 0.25)),
                        max(1, int(r * 0.38)),
                        scale_color(WHITE, 0.95 * a),
                        -1,
                        cv2.LINE_AA,
                    )
                    self.glow.circle(xy[i], r * 1.9, scale_color(accent, 0.42 * a))
                else:
                    cv2.circle(img, c, int(r), scale_color(steel, 0.30 * L), -1, cv2.LINE_AA)
                    cv2.circle(
                        img,
                        (c[0] - int(r * 0.14), c[1] - int(r * 0.16)),
                        max(1, int(r * 0.78)),
                        scale_color(steel, 0.74 * L),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        img,
                        (c[0] - int(r * 0.30), c[1] - int(r * 0.34)),
                        max(1, int(r * 0.34)),
                        scale_color(steel, min(1.55 * L, 1.7)),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(img, c, int(r * 0.62), scale_color(accent, 0.55 * a), 1, cv2.LINE_AA)
