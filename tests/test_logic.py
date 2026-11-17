"""Logic tests with simulated hands (no camera, no audio device needed).

Run from the project folder:   python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from effects import EffectsManager  # noqa: E402
from gestures import MAX_VOLUME_STEP, GestureEngine, analyze  # noqa: E402
from settings import DEFAULTS  # noqa: E402


def hand(index=True, middle=True, ring=True, pinky=True, ratio=0.6, wrist_x=0.5):
    """Build 21 fake landmarks for an upright hand. Palm size is 0.3."""
    lm = [NS(x=wrist_x, y=0.9) for _ in range(21)]
    xs = {5: .40, 9: .50, 13: .60, 17: .70}
    for tip, pip, mcp, up in ((8, 6, 5, index), (12, 10, 9, middle), (16, 14, 13, ring), (20, 18, 17, pinky)):
        x = xs[mcp] - .5 + wrist_x
        lm[mcp] = NS(x=x, y=0.6)
        lm[pip] = NS(x=x, y=0.5)
        lm[tip] = NS(x=x, y=0.3 if up else 0.68)
    lm[0] = NS(x=wrist_x, y=0.9)
    lm[9] = NS(x=wrist_x, y=0.6)
    # thumb tip placed to the left of the index tip so ratio = distance / 0.3
    lm[4] = NS(x=lm[8].x - ratio * 0.3, y=lm[8].y)
    return lm


class FakeAudio:
    def __init__(self, vol=0.5):
        self.vol, self.mute = vol, False

    def get(self): return self.vol
    def set(self, v): self.vol = float(v)
    def muted(self): return self.mute

    def toggle_mute(self):
        self.mute = not self.mute
        return self.mute


S = dict(DEFAULTS)
SIZE = (640, 480)
PINCH = dict(index=True, middle=False, ring=False, pinky=False)   # pinch pose: only index extended


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
        self.assertTrue(self.audio.mute)                     # fired exactly once
        self.run_frames(hand(**PINCH), 0.2, start=2.0)       # release...
        self.run_frames(fist, 1.0, start=3.0)                # ...hold again -> unmute
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
        for i in range(90):                                    # 3 s without a hand
            res = self.eng.update(None, SIZE, s, now=i / 30)
        self.assertTrue(res.auto_locked)
        pinch = hand(**PINCH, ratio=0.1)
        self.audio.vol = 0.8
        res = self.run_frames(pinch, 1, start=3, s=s)          # pinching must NOT change volume
        self.assertAlmostEqual(self.audio.vol, 0.8)
        self.run_frames(hand(index=True, middle=False, ring=False, pinky=True), 0.3, start=4, s=s)
        self.run_frames(pinch, 0.3, start=4.3, s=s)            # pinky lowered -> unlocked
        self.assertFalse(self.eng.auto_locked)

    def test_effects_render_all_states(self):
        fx, frame = EffectsManager(), np.zeros((480, 640, 3), np.uint8)
        cases = [None, hand(**PINCH, ratio=0.7), hand(index=False, middle=False, ring=False, pinky=False),
                 hand(ring=False, pinky=False), hand()]
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
