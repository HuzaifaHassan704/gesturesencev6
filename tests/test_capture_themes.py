"""Tests for screenshots, MP4 recording and colour themes.

Run from the project folder:   python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from gesturesense.config.settings import DEFAULTS  # noqa: E402
from gesturesense.core.gestures import FrameResult, GestureEngine  # noqa: E402
from gesturesense.demo3d import _Audio, synthetic_hand, to_landmarks  # noqa: E402
from gesturesense.render import (  # noqa: E402
    DEFAULT_THEME,
    THEMES,
    VIEW_H,
    VIEW_W,
    Scene3D,
    gauge_color,
)
from gesturesense.system import capture  # noqa: E402


def frame(value=120, w=320, h=240):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = value
    cv2.circle(img, (w // 2, h // 2), 40, (255, 80, 80), -1)
    return img


class ScreenshotTests(unittest.TestCase):
    def test_png_is_saved_and_scaled(self):
        with tempfile.TemporaryDirectory() as d:
            path = capture.save_screenshot(frame(), (640, 480), Path(d) / "shots")
            self.assertTrue(path.exists() and path.suffix == ".png")
            with Image.open(path) as im:
                self.assertEqual(im.size, (640, 480))

    def test_two_screenshots_in_one_second_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            a = capture.save_screenshot(frame(), None, Path(d))
            b = capture.save_screenshot(frame(), None, Path(d))
            self.assertNotEqual(a, b)
            self.assertEqual(len(list(Path(d).glob("*.png"))), 2)


class RecorderTests(unittest.TestCase):
    def test_records_a_playable_mp4_in_real_time(self):
        with tempfile.TemporaryDirectory() as d:
            rec = capture.Recorder(fps=25)
            path = rec.start((320, 240), Path(d))
            self.assertTrue(rec.active)
            t = 100.0
            for i in range(30):  # 10 fps arrival for 3 s -> still 25 fps video
                rec.write(frame(i * 8), now=t + i * 0.1)
            out, seconds = rec.stop()
            self.assertFalse(rec.active)
            self.assertEqual(out, path)
            self.assertAlmostEqual(seconds, 2.9, delta=0.2)
            cap = cv2.VideoCapture(str(path))
            self.assertTrue(cap.isOpened())
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), seconds * 25, delta=3)
            ok, first = cap.read()
            self.assertTrue(ok)
            self.assertEqual(first.shape[:2], (240, 320))

    def test_resized_frames_are_fitted_and_odd_sizes_made_even(self):
        with tempfile.TemporaryDirectory() as d:
            rec = capture.Recorder()
            rec.start((321, 241), Path(d))
            self.assertEqual(rec.size, (320, 240))
            rec.write(frame(w=500, h=300), now=1.0)  # window resized mid-recording
            rec.write(frame(w=500, h=300), now=1.2)
            path, _ = rec.stop()
            cap = cv2.VideoCapture(str(path))
            self.assertEqual((int(cap.get(3)), int(cap.get(4))), (320, 240))

    def test_write_and_stop_are_safe_when_idle(self):
        rec = capture.Recorder()
        rec.write(frame())
        self.assertEqual(rec.stop(), (None, 0.0))

    def test_stalled_source_does_not_flood_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            rec = capture.Recorder(fps=25)
            rec.start((320, 240), Path(d))
            rec.write(frame(), now=0.0)
            rec.write(frame(), now=60.0)  # a minute of silence
            self.assertLessEqual(rec.count, 26)
            rec.stop()


class ThemeTests(unittest.TestCase):
    def render(self, theme):
        scene = Scene3D()
        eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
        s = dict(DEFAULTS, theme=theme, orbit_auto=False)
        norm, world = to_landmarks(synthetic_hand(0.7, 1.0))
        res = eng.update(norm, (VIEW_W, VIEW_H), s, now=0.0, world=world)
        res.volume = 0.8
        for i in range(25):
            img = scene.render(res, s, 30.0, now=i / 30)
        return img

    def test_every_theme_renders_and_looks_different(self):
        imgs = {name: self.render(name) for name in THEMES}
        for name, img in imgs.items():
            self.assertEqual(img.shape, (VIEW_H, VIEW_W, 3), name)
        names = list(imgs)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                self.assertGreater(
                    float(np.abs(imgs[a].astype(int) - imgs[b].astype(int)).mean()), 1.0, (a, b)
                )

    def test_unknown_theme_falls_back_and_switching_back_restores(self):
        scene = Scene3D()
        scene.set_theme("not-a-theme")
        self.assertEqual(scene.theme_name, DEFAULT_THEME)
        base = scene.bg.copy()
        scene.set_theme("neon")
        self.assertFalse(np.array_equal(base, scene.bg))
        scene.set_theme(DEFAULT_THEME)
        self.assertTrue(np.array_equal(base, scene.bg))

    def test_theme_survives_a_resize(self):
        scene = Scene3D()
        scene.set_theme("ocean")
        scene.set_size(500, 400)
        self.assertEqual(scene.theme_name, "ocean")
        self.assertEqual(scene.bg.shape[:2], (400, 500))

    def test_theme_definitions_are_complete(self):
        for name, th in THEMES.items():
            self.assertEqual(len(th["stops"]), 3, name)
            for key in ("label", "body", "bg_top", "bg_bot", "pool", "grid"):
                self.assertIn(key, th, (name, key))
            self.assertNotEqual(gauge_color(0.0, th["stops"]), gauge_color(1.0, th["stops"]), name)

    def test_idle_scene_renders_in_every_theme(self):
        for name in THEMES:
            scene = Scene3D()
            scene.render(FrameResult(volume=0.5), dict(DEFAULTS, theme=name), 30.0, now=0.0)


if __name__ == "__main__":
    unittest.main()


class HandStyleTests(unittest.TestCase):
    def render(self, style, spread=0.6, t=25):
        scene = Scene3D()
        eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
        s = dict(DEFAULTS, hand_style=style, orbit_auto=False)
        norm, world = to_landmarks(synthetic_hand(spread, 1.0))
        res = eng.update(norm, (VIEW_W, VIEW_H), s, now=0.0, world=world)
        for i in range(t):
            img = scene.render(res, s, 30.0, now=i / 30)
        return img

    def test_robot_and_glass_both_render_and_differ(self):
        robot, glass = self.render("robot"), self.render("glass")
        self.assertEqual(robot.shape, (VIEW_H, VIEW_W, 3))
        self.assertGreater(float(np.abs(robot.astype(int) - glass.astype(int)).mean()), 0.5)

    def test_unknown_style_falls_back_to_robot(self):
        self.assertTrue(np.array_equal(self.render("nonsense"), self.render("robot")))

    def test_robot_hand_in_every_state_theme_and_angle(self):
        for theme in THEMES:
            for spread in (0.0, 1.0):
                scene = Scene3D()
                eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
                for yaw in (0.0, 90.0, 180.0, -70.0):
                    s = dict(DEFAULTS, theme=theme, orbit_auto=False, view_yaw=yaw, view_pitch=40)
                    norm, world = to_landmarks(synthetic_hand(spread, 1.0))
                    res = eng.update(norm, (VIEW_W, VIEW_H), s, now=0.0, world=world)
                    res.muted = yaw == 90.0
                    res.locked = yaw == 180.0
                    for i in range(6):
                        scene.render(res, s, 30.0, now=i / 30)

    def test_robot_hand_fades_out_with_the_hand(self):
        scene = Scene3D()
        eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
        s = dict(DEFAULTS, orbit_auto=False)
        norm, world = to_landmarks(synthetic_hand(0.5, 1.0))
        res = eng.update(norm, (VIEW_W, VIEW_H), s, now=0.0, world=world)
        for i in range(10):
            scene.render(res, s, 30.0, now=i / 30)
        for i in range(10, 70):
            scene.render(FrameResult(), s, 30.0, now=i / 30)
        self.assertLess(scene.hand_alpha, 0.05)


class HumanHandTests(unittest.TestCase):
    def test_every_skin_tone_renders_and_differs(self):
        from gesturesense.config.settings import DEFAULTS
        from gesturesense.core.gestures import GestureEngine
        from gesturesense.demo3d import _Audio, synthetic_hand, to_landmarks
        from gesturesense.render import SKIN_TONES, Scene3D

        outs = {}
        for tone in SKIN_TONES:
            eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
            sc = Scene3D(480, 360)
            s = dict(DEFAULTS, hand_style="human", skin_tone=tone, orbit_auto=False)
            for i in range(12):
                n, w = to_landmarks(synthetic_hand(0.6, 1.0), 0.5, 0.5)
                res = eng.update(n, (480, 360), s, now=i / 30, world=w)
                img = sc.render(res, s, 30, now=i / 30)
            self.assertEqual(img.shape, (360, 480, 3))
            outs[tone] = img
        self.assertFalse(np.array_equal(outs["light"], outs["deep"]))

    def test_unknown_skin_tone_falls_back(self):
        from gesturesense.config.settings import DEFAULTS
        from gesturesense.core.gestures import GestureEngine
        from gesturesense.demo3d import _Audio, synthetic_hand, to_landmarks
        from gesturesense.render import Scene3D

        eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
        sc = Scene3D(480, 360)
        s = dict(DEFAULTS, hand_style="human", skin_tone="purple", orbit_auto=False)
        n, w = to_landmarks(synthetic_hand(0.6, 1.0), 0.5, 0.5)
        res = eng.update(n, (480, 360), s, now=0.0, world=w)
        self.assertEqual(sc.render(res, s, 30, now=0.0).dtype, np.uint8)


class AndroidHandTests(unittest.TestCase):
    def test_android_renders_and_differs_from_robot(self):
        from gesturesense.config.settings import DEFAULTS
        from gesturesense.core.gestures import GestureEngine
        from gesturesense.demo3d import _Audio, synthetic_hand, to_landmarks
        from gesturesense.render import Scene3D

        outs = {}
        for style in ("android", "robot"):
            eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
            sc = Scene3D(480, 360)
            s = dict(DEFAULTS, hand_style=style, orbit_auto=False, fx3d_motion=False)
            for i in range(12):
                n, w = to_landmarks(synthetic_hand(0.6, 1.0), 0.5, 0.5)
                res = eng.update(n, (480, 360), s, now=i / 30, world=w)
                outs[style] = sc.render(res, s, 30, now=i / 30)
        self.assertFalse(np.array_equal(outs["android"], outs["robot"]))
        self.assertEqual(DEFAULTS["hand_style"], "android")
