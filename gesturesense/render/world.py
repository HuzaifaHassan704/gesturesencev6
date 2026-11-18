"""Scene furniture: floor grid, shadow, tower, pinch beam, motion effects and HUD (mixin)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from gesturesense.core.gestures import FrameResult
from gesturesense.render.constants import (
    CONNECTIONS,
    FONT,
    GAUGE_FACETS,
    GAUGE_H,
    GAUGE_POS,
    GAUGE_R,
    GAUGE_SLABS,
    GRID_CENTER,
    GRID_HALF,
    GRID_STEP,
    LIGHT_DIR,
    PALM_EDGES,
    PALM_POLY,
    RADIUS,
    TIPS,
)
from gesturesense.render.effects import (
    ACTIVE,
    LOCKED,
    MUTED,
    PLAY,
    WHITE,
    draw_pill,
    lerp_color,
)
from gesturesense.render.layers import _ip, _ok
from gesturesense.render.theme import scale_color


class WorldMixin:
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
            cv2.circle(
                img, _ip(p), max(2, int(r * scale)), scale_color(col, bright * a), -1, cv2.LINE_AA
            )
        cv2.circle(img, _ip(p), max(1, int(r * 0.30)), scale_color(WHITE, a), -1, cv2.LINE_AA)
        n = 12
        spin = now * 0.6
        for i in range(n):
            ang = spin + i * 2 * math.pi / n
            d = np.array([math.cos(ang), math.sin(ang)])
            cv2.line(
                img,
                _ip(p + d * r * 1.5),
                _ip(p + d * r * (1.5 + 0.9 * level)),
                scale_color(col, 0.8 * a),
                max(1, int(2 * self.k)),
                cv2.LINE_AA,
            )
        self.glow.circle(p, r * 2.4, scale_color(col, 0.5 * a))
        title = "BRIGHTNESS" if role == "brightness" else "HAND 2"
        self._tc(
            img,
            f"{int(round(level * 100))}%",
            (p[0], p[1] + r * 2.3 + 14 * self.k),
            0.6,
            scale_color(col, a),
            2,
        )
        self._tc(
            img, title, (p[0], p[1] + r * 2.3 + 32 * self.k), 0.38, scale_color(WHITE, 0.8 * a), 1
        )

    def _draw_motion(self, img, s: dict, state, now: float) -> None:
        """Things that react to how fast the hand moves: fingertip streaks, sparks, ripples."""
        amt = min(max(float(s.get("fx3d_motion_amount", 0.7)), 0.0), 1.0)
        dt = self._dt
        energy = min(self.speed / 6.0, 1.0)  # 0 = still, 1 = fast
        tips = self.pts[list(TIPS)]
        self.trail.append(tips.copy())
        del self.trail[: -int(6 + 10 * amt)]
        col = self._gc(self.vol)
        a = self.hand_alpha
        # streaks behind each fingertip, brighter and longer the faster the hand goes
        if len(self.trail) > 1 and energy > 0.03:
            hist = np.stack(self.trail)  # (T, 5, 3)
            T = len(hist)
            xy, z, k = self.cam.project(hist.reshape(-1, 3))
            xy = xy.reshape(T, 5, 2)
            ok = z.reshape(T, 5) > 0.4
            for j in range(5):
                for t in range(T - 1):
                    if not (ok[t, j] and ok[t + 1, j]):
                        continue
                    f = (t + 1) / T
                    c = scale_color(col, min(1.0, energy * 1.4) * f * a * (0.5 + 0.5 * amt))
                    w = max(1, int((1 + 4 * f) * self.k))
                    cv2.line(img, _ip(xy[t, j]), _ip(xy[t + 1, j]), c, w, cv2.LINE_AA)
                    self.glow.line(
                        xy[t, j], xy[t + 1, j], scale_color(col, f * a * energy), 6 * self.k
                    )
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
            sp[1][1] -= 2.2 * dt  # gravity
            sp[0] = sp[0] + sp[1] * dt
            if sp[0][1] < 0.02:
                sp[0][1] = 0.02
                sp[1] = sp[1] * np.array([0.6, -0.4, 0.6])
            alive.append(sp)
        self.sparks = alive
        if alive:
            xy, z, k = self.cam.project(np.array([sp[0] for sp in alive]))
            for sp, p, zz, kk in zip(alive, xy, z, k, strict=False):
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
                for p0, p1 in zip(xy[:-1], xy[1:], strict=False):
                    self.glow.line(p0, p1, c, 6 * self.k)

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
                for lo, hi in zip(a, b, strict=False):
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
        base = np.where(
            major[:, None],
            np.array(self.theme["grid"][1], float),
            np.array(self.theme["grid"][0], float),
        )
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
                for a, b in zip(xy[:-1], xy[1:], strict=False):
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
        cv2.polylines(
            img,
            [np.rint(ring).astype(np.int32)],
            False,
            scale_color(state, 0.55 * a),
            1,
            cv2.LINE_AA,
        )
        self.glow.circle(
            self.cam.project(c[None])[0][0],
            0.7 * self.cam.project(c[None])[2][0],
            scale_color(state, 0.16 * a),
        )

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
        end = np.array(
            [
                gx + GAUGE_R * 1.05 * direction[0],
                max(self.vol, 0.02) * GAUGE_H,
                gz + GAUGE_R * 1.05 * direction[1],
            ]
        )
        t = np.linspace(0, 1, 28)[:, None]
        curve = mid[None] * (1 - t) + end[None] * t
        curve[:, 1] += np.sin(t[:, 0] * math.pi) * 0.35  # arc upwards
        cxy, cz, _ = self.cam.project(curve)
        if (cz > 0.4).all():
            phase = int(self._now * 14) % 2  # dash animation follows the frame clock
            for n in range(len(curve) - 1):
                if (n + phase) % 2 == 0:
                    cv2.line(
                        img, _ip(cxy[n]), _ip(cxy[n + 1]), scale_color(col, 0.9), 2, cv2.LINE_AA
                    )
                    self.glow.line(cxy[n], cxy[n + 1], scale_color(col, 0.55), 8)
        mxy = self.cam.project(mid[None])[0][0]
        self._tc(
            img, f"{int(round(res.volume * 100))}%", (mxy[0], mxy[1] - 26 * self.k), 0.7, col, 2
        )

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
        centres = np.stack(
            [gx + GAUGE_R * np.cos(am), np.full(J, 1.0), gz + GAUGE_R * np.sin(am)], 1
        )
        to_cam = self.cam.pos[None] - centres
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        front = (normals * to_cam).sum(1) > 0
        lam = np.clip(normals @ LIGHT_DIR, 0, 1)
        half = to_cam + LIGHT_DIR[None]
        half /= np.linalg.norm(half, axis=1, keepdims=True)
        spec = np.clip((normals * half).sum(1), 0, 1) ** 18

        muted_or_locked = res.muted or res.locked
        fill_total = self.vol * N
        # base plate
        plate = self.cam.project(self._floor_ring(gx, gz, 0.78, 40))[0]
        cv2.fillPoly(img, [np.rint(plate).astype(np.int32)], (34, 27, 22), cv2.LINE_AA)
        cv2.polylines(
            img, [np.rint(plate).astype(np.int32)], True, scale_color(state, 0.85), 2, cv2.LINE_AA
        )
        self.glow.poly(plate, scale_color(state, 0.22))
        inner = self.cam.project(self._floor_ring(gx, gz, 0.52, 36))[0]
        cv2.polylines(img, [np.rint(inner).astype(np.int32)], True, (70, 56, 46), 1, cv2.LINE_AA)

        for i in range(N):  # bottom -> top
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
        cv2.polylines(
            img, [np.rint(cage_top).astype(np.int32)], True, (140, 126, 112), 1, cv2.LINE_AA
        )
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
            self._pt(
                img,
                str(pct),
                (int(pp[0] + 5 * self.k), int(pp[1] + 4 * self.k)),
                0.38,
                (185, 170, 155),
                1,
            )
        # big readout above the tower
        head = self.cam.project(np.array([[gx, GAUGE_H + 0.55, gz]]))[0][0]
        label = "MUTED" if res.muted else f"{int(round(self.vol * 100))}%"
        col = MUTED if res.muted else self._gc(self.vol)
        self._tc(img, label, (head[0], head[1]), 0.95, col, 2)
        sub = "LOCKED" if res.locked else "VOLUME"
        self._tc(
            img,
            sub,
            (head[0], head[1] + 22 * self.k),
            0.4,
            LOCKED if res.locked else (170, 155, 140),
            1,
        )

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
        draw_pill(
            img,
            (m, m),
            label,
            state if res.has_hand or res.locked or res.muted else (150, 150, 150),
            k=k,
        )
        if fps:
            text = f"{fps:.0f} FPS"
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.45 * k, 1)
            self._pt(img, text, (self.w - tw - m, m + int(16 * k)), 0.45, (190, 178, 165), 1)
        auto = bool(s.get("orbit_auto", True))
        hint = (
            "AUTO ORBIT  -  drag to take control"
            if auto
            else "drag to orbit  -  wheel to zoom  -  double-click: auto orbit"
        )
        (tw, _), _ = cv2.getTextSize(hint, FONT, 0.4 * k, 1)
        self._pt(img, hint, (self.w - tw - m, self.h - m), 0.4, (140, 128, 116), 1)
        if not res.has_hand and self.hand_alpha < 0.05:
            self._tc(
                img,
                "Show your hand to the camera",
                (self.w * 0.36, self.h * 0.46),
                0.7,
                (190, 178, 165),
                1,
            )
