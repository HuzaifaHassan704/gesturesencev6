"""Motion animations: they react to hand speed, stay bounded and can be switched off."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from gesturesense.config.settings import DEFAULTS  # noqa: E402
from gesturesense.core.gestures import GestureEngine  # noqa: E402
from gesturesense.demo3d import _Audio, synthetic_hand, to_landmarks  # noqa: E402
from gesturesense.render import VIEW_H, VIEW_W, Scene3D  # noqa: E402


def run(settings, moving, frames=40):
    eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
    sc = Scene3D()
    s = dict(DEFAULTS, orbit_auto=False, **settings)
    for i in range(frames):
        cx = 0.5 + (0.3 * math.sin(i * 0.9) if moving else 0.0)
        norm, world = to_landmarks(synthetic_hand(0.5, 1.0), cx, 0.5)
        res = eng.update(norm, (VIEW_W, VIEW_H), s, now=i / 30, world=world)
        img = sc.render(res, s, 30, now=i / 30)
    return sc, img


class MotionTests(unittest.TestCase):
    def test_fast_hand_makes_sparks_and_speed(self):
        sc, _ = run({}, True)
        self.assertGreater(sc.speed, 1.0)
        self.assertGreater(len(sc.sparks), 0)
        self.assertLessEqual(len(sc.sparks), 400)

    def test_still_hand_is_calm(self):
        sc, _ = run({}, False)
        self.assertLess(sc.speed, 0.2)
        self.assertEqual(len(sc.sparks), 0)

    def test_switch_off(self):
        sc, _ = run({"fx3d_motion": False}, True)
        self.assertEqual(len(sc.sparks), 0)
        self.assertEqual(len(sc.trail), 0)

    def test_output_valid(self):
        _, img = run({"fx3d_motion_amount": 1.0}, True)
        self.assertEqual(img.shape, (VIEW_H, VIEW_W, 3))
        self.assertEqual(img.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
