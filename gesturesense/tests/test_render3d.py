"""Tests for the software 3D renderer (no camera, GUI or audio device needed).

Run from the project folder:   python -m unittest discover -s tests -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from demo3d import _Audio, synthetic_hand, to_landmarks  # noqa: E402
from gestures import FrameResult, GestureEngine  # noqa: E402
from render3d import GAUGE_POS, VIEW_H, VIEW_W, Camera, Scene3D, gauge_color  # noqa: E402
from settings import DEFAULTS  # noqa: E402

S = dict(DEFAULTS)


def make_result(engine, spread=0.5, curl=1.0, now=0.0, cx=0.5, cy=0.5):
    norm, world = to_landmarks(synthetic_hand(spread, curl), cx, cy)
    return engine.update(norm, (VIEW_W, VIEW_H), S, now=now, world=world)


class CameraTests(unittest.TestCase):
    def test_target_projects_to_screen_centre(self):
        cam = Camera(VIEW_W, VIEW_H)
        for yaw in (0.0, 0.7, -1.2, 3.0):
            cam.set(yaw, 0.4, 8.0)
            xy, depth, _ = cam.project(cam.target[None])
            self.assertAlmostEqual(xy[0, 0], VIEW_W / 2, places=3)
            self.assertAlmostEqual(xy[0, 1], VIEW_H / 2, places=3)
            self.assertAlmostEqual(depth[0], 8.0, places=3)

    def test_up_is_up_and_right_is_right(self):
        cam = Camera(VIEW_W, VIEW_H)
        cam.set(0.0, 0.3, 8.0)
        t = cam.target
        xy, _, _ = cam.project(np.array([t, t + [1, 0, 0], t + [0, 1, 0]]))
        self.assertGreater(xy[1, 0], xy[0, 0])      # +X is to the right on screen
        self.assertLess(xy[2, 1], xy[0, 1])         # +Y is up (smaller pixel row)

    def test_nearer_points_are_larger(self):
        cam = Camera(VIEW_W, VIEW_H)
        cam.set(0.0, 0.3, 8.0)
        _, depth, scale = cam.project(np.array([cam.target + [0, 0, -2], cam.target + [0, 0, 2]]))
        self.assertLess(depth[0], depth[1])
        self.assertGreater(scale[0], scale[1])


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.audio = _Audio()
        self.eng = GestureEngine(self.audio, press=lambda k: None, now=0.0)
        self.scene = Scene3D()

    def test_world_landmarks_reach_the_result(self):
        res = make_result(self.eng)
        self.assertEqual(res.world.shape, (21, 3))
        self.assertEqual(res.norm.shape, (21, 3))

    def test_frame_shape_and_content(self):
        res = make_result(self.eng)
        img = self.scene.render(res, S, 30.0, now=0.0)
        self.assertEqual(img.shape, (VIEW_H, VIEW_W, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertGreater(img.max(), 200)           # something bright (hand / tower) was drawn
        self.assertGreater(float(img.mean()), 8)

    def test_all_states_render(self):
        cases = {
            "pinch": dict(spread=0.8),
            "fist": dict(spread=0.0, curl=1.0),
        }
        t = 0.0
        for _ in cases:
            for _ in range(12):
                t += 1 / 30
                res = make_result(self.eng, now=t)
                self.scene.render(res, S, 30.0, now=t)
        for kind in ("muted", "locked", "auto_locked", "hold", "nohand"):
            res = FrameResult()
            if kind == "muted":
                res.muted = True
            elif kind == "locked":
                res.locked = True
            elif kind == "auto_locked":
                res.locked = res.auto_locked = True
            elif kind == "hold":
                res = make_result(self.eng, now=t + 1)
                res.hold_mute = 0.6
            res.volume = 0.8
            img = self.scene.render(res, S, 30.0, now=t + 2)
            self.assertEqual(img.shape, (VIEW_H, VIEW_W, 3))

    def test_volume_tower_follows_volume(self):
        def lit_pixels(vol):
            sc = Scene3D()
            res = FrameResult(volume=vol)
            s = dict(S, orbit_auto=False)
            for i in range(30):                      # let the smoothing settle
                img = sc.render(res, s, 30.0, now=i / 30)
            return int((img.max(axis=2) > 170).sum())
        self.assertGreater(lit_pixels(0.9), lit_pixels(0.1))

    def test_hand_fades_out_without_detection(self):
        res = make_result(self.eng)
        for i in range(10):
            self.scene.render(res, S, 30.0, now=i / 30)
        self.assertGreater(self.scene.hand_alpha, 0.9)
        for i in range(10, 80):
            self.scene.render(FrameResult(), S, 30.0, now=i / 30)
        self.assertLess(self.scene.hand_alpha, 0.05)

    def test_hand_follows_image_position(self):
        a, b = Scene3D(), Scene3D()
        for _ in range(20):
            a.render(make_result(self.eng, cx=0.3), S, 30.0, now=0.0)
            b.render(make_result(self.eng, cx=0.7), S, 30.0, now=0.0)
        self.assertGreater(b.pts[:, 0].mean(), a.pts[:, 0].mean() + 0.8)

    def test_hand_never_goes_below_the_floor(self):
        sc = Scene3D()
        for _ in range(20):
            sc.render(make_result(self.eng, cy=0.98), S, 30.0, now=0.0)
        self.assertGreaterEqual(sc.pts[:, 1].min(), 0.1)

    def test_auto_orbit_sweeps_and_manual_orbit_holds(self):
        sc = Scene3D()
        res = FrameResult()
        yaws = []
        for i in range(300):
            sc.render(res, dict(S, orbit_auto=True, orbit_speed=30.0, orbit_range=40.0), 30.0, now=i / 30)
            yaws.append(math.degrees(sc.yaw))
        self.assertGreater(max(yaws), 25)
        self.assertLess(min(yaws), -25)
        self.assertLessEqual(max(abs(y) for y in yaws), 40.5)
        manual = dict(S, orbit_auto=False, view_yaw=75.0)
        for i in range(60):
            sc.render(res, manual, 30.0, now=20 + i / 30)
        self.assertAlmostEqual(math.degrees(sc.yaw), 75.0, delta=1.0)

    def test_full_spin_mode(self):
        sc = Scene3D()
        for i in range(90):
            sc.render(FrameResult(), dict(S, orbit_auto=True, orbit_speed=60.0, orbit_range=180.0),
                      30.0, now=i / 30)
        self.assertGreater(math.degrees(sc.yaw), 150)

    def test_inset_and_every_toggle_combination(self):
        res = make_result(self.eng)
        cam = np.full((480, 640, 3), 90, np.uint8)
        for grid in (True, False):
            for shadow in (True, False):
                for glow in (True, False):
                    s = dict(S, fx3d_grid=grid, fx3d_shadow=shadow, fx3d_glow=glow)
                    img = self.scene.render(res, s, 30.0, now=0.0)
                    self.scene.add_inset(img, cam, (153, 211, 52))
                    self.assertEqual(img.shape, (VIEW_H, VIEW_W, 3))

    def test_extreme_camera_settings_do_not_crash(self):
        res = make_result(self.eng)
        for pitch in (6, 75):
            for zoom in (0.5, 2.0):
                s = dict(S, orbit_auto=False, view_pitch=pitch, view_zoom=zoom, view_yaw=180.0)
                for i in range(15):
                    self.scene.render(res, s, 30.0, now=i / 30)

    def test_gauge_colour_gradient_ends(self):
        self.assertNotEqual(gauge_color(0.0), gauge_color(1.0))
        self.assertEqual(len(gauge_color(0.5)), 3)
        self.assertEqual(GAUGE_POS[1], 0.0)


if __name__ == "__main__":
    unittest.main()
