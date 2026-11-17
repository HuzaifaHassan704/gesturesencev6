"""Tests for the jitter filter, hand-lost recovery and calibration logic.

Run from the project folder:   python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from calibration import CalibrationError, Collector, compute_range, quality  # noqa: E402
from filters import LandmarkFilter, OneEuro  # noqa: E402
from gestures import GestureEngine  # noqa: E402
from settings import DEFAULTS  # noqa: E402
from tests.test_logic import PINCH, FakeAudio, hand  # noqa: E402

S = dict(DEFAULTS)
SIZE = (640, 480)


class FilterTests(unittest.TestCase):
    def test_removes_jitter_from_a_still_signal(self):
        rng = np.random.default_rng(1)
        f = OneEuro(min_cutoff=0.5, beta=0.0)
        raw = 0.5 + rng.normal(0, 0.01, 300)
        out = np.array([float(f(np.array([v]), i / 30)[0]) for i, v in enumerate(raw)])
        self.assertLess(out[60:].std(), raw[60:].std() * 0.5)

    def test_follows_fast_motion_with_little_lag(self):
        f = OneEuro(min_cutoff=0.5, beta=8.0)
        out = [float(f(np.array([i / 30.0 * 2]), i / 30)[0]) for i in range(60)]   # 2 units / second
        self.assertLess(abs(out[-1] - 59 / 30 * 2), 0.15)

    def test_strength_zero_is_a_passthrough(self):
        lf = LandmarkFilter()
        lm = hand(**PINCH, ratio=0.5)
        out, _ = lf.apply(lm, None, 0.0, 0.0)
        self.assertIs(out, lm)

    def test_landmark_filter_smooths_noise_and_resets_after_absence(self):
        rng = np.random.default_rng(2)
        lf = LandmarkFilter()
        xs = []
        for i in range(120):
            lm = hand(**PINCH, ratio=0.5)
            for p in lm:
                p.x += rng.normal(0, 0.004)
                p.z = 0.0
            out, _ = lf.apply(lm, None, i / 30, 0.8)
            xs.append(out[8].x)
        self.assertLess(np.std(xs[30:]), 0.004 * 0.6)
        lf.miss(10.0)                                         # long absence -> forget the old pose
        lm = hand(**PINCH, ratio=0.5, wrist_x=0.9)
        out, _ = lf.apply(lm, None, 10.1, 0.8)
        self.assertAlmostEqual(out[0].x, lm[0].x, places=6)   # no smear from the old position

    def test_output_has_xyz_attributes(self):
        lf = LandmarkFilter()
        out, world = lf.apply(hand(), hand(), 0.0, 0.5)
        self.assertEqual(len(out), 21)
        self.assertTrue(hasattr(world[0], "z"))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.audio = FakeAudio(0.3)
        self.eng = GestureEngine(self.audio, press=lambda k: None, now=0.0)

    def feed(self, lm, t0, seconds, fps=30):
        res = None
        for i in range(int(seconds * fps)):
            res = self.eng.update(lm, SIZE, S, now=t0 + i / fps)
        return res

    def test_short_dropout_holds_the_pose(self):
        self.feed(hand(**PINCH, ratio=0.6), 0.0, 1.0)
        res = self.eng.update(None, SIZE, S, now=1.1)
        self.assertTrue(res.has_hand)
        self.assertTrue(res.coasting)
        res = self.eng.update(None, SIZE, S, now=2.0)         # too long -> really lost
        self.assertFalse(res.has_hand)

    def test_volume_is_frozen_while_the_hand_settles(self):
        self.feed(hand(**PINCH, ratio=0.6), 0.0, 0.5)
        self.feed(None, 1.0, 1.0)
        self.audio.vol = 0.3
        res = self.eng.update(hand(**PINCH, ratio=1.1), SIZE, S, now=3.0)   # big spread right away
        self.assertTrue(res.settling)
        self.assertFalse(res.controlling)
        self.assertAlmostEqual(self.audio.vol, 0.3)
        self.feed(hand(**PINCH, ratio=1.1), 3.0, 0.3)
        self.assertAlmostEqual(self.audio.vol, 0.3)           # still inside the 0.4 s window

    def test_volume_eases_in_after_settling(self):
        self.feed(hand(**PINCH, ratio=0.6), 0.0, 0.5)
        self.audio.vol = 0.2
        self.eng.update(None, SIZE, S, now=1.0)
        s = dict(S, lost_grace=0.0)
        self.eng.update(None, SIZE, s, now=1.5)
        res = self.eng.update(hand(**PINCH, ratio=1.1), SIZE, s, now=3.0)
        t, prev, steps = 3.0, self.audio.vol, []
        for _ in range(45):
            t += 1 / 30
            res = self.eng.update(hand(**PINCH, ratio=1.1), SIZE, s, now=t)
            steps.append(self.audio.vol - prev)
            prev = self.audio.vol
        moving = [x for x in steps if x > 1e-9]
        self.assertLessEqual(moving[0], 0.0125)               # first step is gentle
        self.assertGreater(max(moving), 0.04)                 # and it speeds up to full rate
        self.assertGreater(self.audio.vol, 0.6)

    def test_no_settle_on_the_very_first_continuous_frames(self):
        res = self.feed(hand(**PINCH, ratio=1.1), 0.0, 1.0)
        self.assertTrue(res.controlling)
        self.assertFalse(res.settling)

    def test_settle_can_be_disabled(self):
        self.feed(hand(**PINCH, ratio=0.6), 0.0, 0.5)
        s = dict(S, lost_grace=0.0, settle_seconds=0.0)
        self.eng.update(None, SIZE, s, now=1.0)
        res = self.eng.update(hand(**PINCH, ratio=0.6), SIZE, s, now=3.0)
        self.assertFalse(res.settling)


class CalibrationTests(unittest.TestCase):
    def run_collector(self, ratio, hand_visible=lambda t: True):
        c, t, dt = Collector(seconds=2.0, warmup=0.8), 0.0, 1 / 30
        while not c.done and t < 20:
            c.add(ratio + 0.01 * np.sin(t * 40), dt, hand_visible(t))
            t += dt
        return c, t

    def test_collector_finishes_and_skips_warmup(self):
        c, t = self.run_collector(0.2)
        self.assertTrue(c.done)
        self.assertAlmostEqual(t, 2.8, delta=0.1)
        self.assertLess(len(c.samples), 2.8 * 30)

    def test_collector_pauses_without_a_hand(self):
        c, t = self.run_collector(0.2, hand_visible=lambda t: not (1.0 < t < 3.0))
        self.assertGreater(t, 4.5)                            # the 2 s gap did not count

    def test_range_from_good_samples(self):
        lo, hi = compute_range([0.12] * 20, [1.2] * 20)
        self.assertAlmostEqual(lo, 0.14, places=3)
        self.assertAlmostEqual(hi, 1.14, places=3)

    def test_range_rejects_close_values_and_missing_data(self):
        with self.assertRaises(CalibrationError):
            compute_range([0.5] * 20, [0.6] * 20)
        with self.assertRaises(CalibrationError):
            compute_range([0.1] * 2, [1.0] * 20)

    def test_range_is_clamped_to_slider_limits(self):
        lo, hi = compute_range([0.9] * 20, [3.5] * 20)
        self.assertLessEqual(lo, 0.6)
        self.assertLessEqual(hi, 2.0)

    def test_quality_warning(self):
        self.assertIsNone(quality([0.2, 0.21, 0.2, 0.19], [1.0, 1.02, 0.99, 1.0]))
        self.assertIsNotNone(quality([0.1, 0.5, 0.1, 0.6], [1.0, 1.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
