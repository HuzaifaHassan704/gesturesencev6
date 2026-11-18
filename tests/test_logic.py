"""Logic tests with simulated hands (no camera, no audio device needed).

Run from the project folder:   python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from gesturesense.config.settings import DEFAULTS  # noqa: E402
from gesturesense.core.gestures import MAX_VOLUME_STEP, GestureEngine, analyze  # noqa: E402
from gesturesense.render.effects import EffectsManager  # noqa: E402
from tests.helpers import PINCH, FakeAudio, hand  # noqa: E402

S = dict(DEFAULTS)
SIZE = (640, 480)


class Tests(unittest.TestCase):
    def setUp(self):
        self.audio, self.keys = FakeAudio(), []
        self.eng = GestureEngine(self.audio, press=self.keys.append, now=0.0)

    def run_frames(self, lm, seconds, start=0.0, fps=30, s=S):
        res, n = None, int(seconds * fps)
        for i in range(n):
            res = self.eng.update(lm, SIZE, s, now=start + i / fps)
        return res

    def test_pose_classification(self):
        self.assertTrue(analyze(hand(index=False, middle=False, ring=False, pinky=False)).fist)
        self.assertTrue(analyze(hand(ring=False, pinky=False)).peace)
        self.assertTrue(analyze(hand()).palm_open)
        self.assertTrue(analyze(hand(**PINCH, ratio=0.3)).pinky_up is False)
        self.assertAlmostEqual(analyze(hand(**PINCH, ratio=0.5)).ratio, 0.5, places=2)

    def test_spread_raises_and_pinch_lowers_volume(self):
        res = self.run_frames(hand(**PINCH, ratio=1.1), 2)
        self.assertTrue(res.controlling)
        self.assertGreater(self.audio.vol, 0.95)
        self.run_frames(hand(**PINCH, ratio=0.1), 2, start=2)
        self.assertLess(self.audio.vol, 0.05)

    def test_volume_step_is_rate_limited(self):
        before = self.audio.vol
        self.eng.update(hand(**PINCH, ratio=1.1), SIZE, S, now=0.0)
        self.assertLessEqual(self.audio.vol - before, MAX_VOLUME_STEP + 1e-9)

    def test_pinky_locks_volume(self):
        self.run_frames(hand(index=True, middle=False, ring=False, pinky=True, ratio=1.1), 1)
        self.assertAlmostEqual(self.audio.vol, 0.5)

    def test_fist_hold_toggles_mute_once(self):
        fist = hand(index=False, middle=False, ring=False, pinky=False)
        res = self.run_frames(fist, 0.4)
        self.assertFalse(self.audio.mute)
        self.assertTrue(0.4 < res.hold_mute < 0.7)
        res = self.run_frames(fist, 1.5, start=0.4)
        self.assertTrue(self.audio.mute)  # fired exactly once
        self.run_frames(hand(**PINCH), 0.2, start=2.0)  # release...
        self.run_frames(fist, 1.0, start=3.0)  # ...hold again -> unmute
        self.assertFalse(self.audio.mute)

    def test_peace_hold_presses_playpause(self):
        self.run_frames(hand(ring=False, pinky=False), 1.0)
        self.assertEqual(self.keys, ["playpause"])

    def test_open_palm_swipe(self):
        for i in range(12):
            self.eng.update(hand(wrist_x=0.3 + i * 0.04), SIZE, S, now=i / 30)
        self.assertEqual(self.keys, ["nexttrack"])
        self.keys.clear()
        for i in range(12):
            self.eng.update(hand(wrist_x=0.8 - i * 0.04), SIZE, S, now=3 + i / 30)
        self.assertEqual(self.keys, ["prevtrack"])

    def test_auto_lock_and_pinky_unlock(self):
        s = dict(S, auto_lock_seconds=2.0)
        res = None
        for i in range(90):  # 3 s without a hand
            res = self.eng.update(None, SIZE, s, now=i / 30)
        self.assertTrue(res.auto_locked)
        pinch = hand(**PINCH, ratio=0.1)
        self.audio.vol = 0.8
        res = self.run_frames(pinch, 1, start=3, s=s)  # pinching must NOT change volume
        self.assertAlmostEqual(self.audio.vol, 0.8)
        self.run_frames(hand(index=True, middle=False, ring=False, pinky=True), 0.3, start=4, s=s)
        self.run_frames(pinch, 0.3, start=4.3, s=s)  # pinky lowered -> unlocked
        self.assertFalse(self.eng.auto_locked)

    def test_effects_render_all_states(self):
        fx, frame = EffectsManager(), np.zeros((480, 640, 3), np.uint8)
        cases = [
            None,
            hand(**PINCH, ratio=0.7),
            hand(index=False, middle=False, ring=False, pinky=False),
            hand(ring=False, pinky=False),
            hand(),
        ]
        t = 0.0
        for lm in cases:
            for _ in range(40):
                t += 1 / 30
                res = self.eng.update(lm, SIZE, S, now=t)
                fx.render(frame, res, S, fps=30.0, now=t)
        for kind in ("mute_on", "mute_off", "playpause", "next", "prev", "lock", "unlock"):
            fx._draw_popup(frame, kind, 0.4)
        self.assertEqual(frame.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
