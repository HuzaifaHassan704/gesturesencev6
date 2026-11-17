"""Tests for actions, profiles, thumbs gestures and the gesture -> action mapping.

Run from the project folder:   python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import actions  # noqa: E402
import profiles  # noqa: E402
from effects import EffectsManager  # noqa: E402
from gestures import GestureEngine, analyze  # noqa: E402
from settings import DEFAULTS  # noqa: E402
from tests.test_logic import FakeAudio, hand  # noqa: E402

SIZE = (640, 480)
FIST = dict(index=False, middle=False, ring=False, pinky=False)


def thumbs(up: bool, **kw):
    """A curled hand whose thumb points well above (up) or below (down) the index knuckle."""
    lm = hand(**FIST, **kw)
    lm[4] = NS(x=0.36, y=0.40 if up else 0.85)       # index knuckle is at y = 0.6, palm size 0.3
    return lm


def settings(**over):
    s = dict(DEFAULTS)
    s.update(over)
    return s


class Rig:
    def __init__(self, vol=0.5):
        self.audio = FakeAudio(vol)
        self.keys, self.hotkeys, self.launched = [], [], []
        self.eng = GestureEngine(self.audio, press=self.keys.append,
                                 hotkey=lambda *k: self.hotkeys.append(k),
                                 launch=self.launched.append, now=0.0)

    def run(self, lm, seconds, s, start=0.0, fps=30):
        res, events = None, []
        for i in range(int(seconds * fps)):
            res = self.eng.update(lm, SIZE, s, now=start + i / fps)
            events += res.events
        return res, events


class PoseTests(unittest.TestCase):
    def test_thumbs_up_and_down_are_not_a_fist(self):
        up, down, fist = analyze(thumbs(True)), analyze(thumbs(False)), analyze(hand(**FIST))
        self.assertTrue(up.thumb_up and not up.fist and not up.thumb_down)
        self.assertTrue(down.thumb_down and not down.fist and not down.thumb_up)
        self.assertTrue(fist.fist and not fist.thumb_up and not fist.thumb_down)

    def test_open_hand_has_no_thumb_gesture(self):
        p = analyze(hand())
        self.assertFalse(p.thumb_up or p.thumb_down)


class ActionTests(unittest.TestCase):
    def runner(self, vol=0.5):
        self.audio = FakeAudio(vol)
        self.keys, self.hot, self.launched = [], [], []
        return actions.ActionRunner(self.audio, self.keys.append, lambda *k: self.hot.append(k),
                                    self.launched.append)

    def test_keys_hotkeys_and_none(self):
        r = self.runner()
        self.assertEqual(r.run("play_pause")[0], "playpause")
        self.assertEqual(self.keys, ["playpause"])
        r.run("slide_next")
        r.run("teams_mute")
        self.assertEqual(self.keys[-1], "right")
        self.assertEqual(self.hot, [("ctrl", "shift", "m")])
        self.assertIsNone(r.run("none"))
        self.assertIsNone(r.run("definitely_not_an_action"))

    def test_volume_steps_clamp(self):
        r = self.runner(0.95)
        r.run("volume_up")
        self.assertEqual(self.audio.vol, 1.0)
        r = self.runner(0.05)
        r.run("volume_down")
        self.assertEqual(self.audio.vol, 0.0)
        r = self.runner(0.5)
        kind, label = r.run("volume_up")
        self.assertEqual(kind, "vol_up")
        self.assertAlmostEqual(self.audio.vol, 0.6)
        self.assertIn("60", label)

    def test_mute_toggle_reports_state(self):
        r = self.runner()
        self.assertEqual(r.run("mute_toggle")[0], "mute_on")
        self.assertEqual(r.run("mute_toggle")[0], "mute_off")

    def test_open_app_and_failure(self):
        r = self.runner()
        kind, label = r.run("open:C:/Windows/notepad.exe")
        self.assertEqual(self.launched, ["C:/Windows/notepad.exe"])
        self.assertEqual(label, "Open: notepad")
        bad = actions.ActionRunner(self.audio, launch=lambda p: (_ for _ in ()).throw(OSError("x")))
        self.assertEqual(bad.run("open:nope")[1], "Could not open app")

    def test_validation_and_labels(self):
        self.assertTrue(actions.is_valid("none"))
        self.assertTrue(actions.is_valid("open:app.exe"))
        self.assertFalse(actions.is_valid("open:"))
        self.assertFalse(actions.is_valid(5))
        for name in actions.ACTIONS:
            self.assertTrue(actions.describe(name))


class ProfileTests(unittest.TestCase):
    def test_builtin_profiles_are_complete_and_valid(self):
        for name, mapping in profiles.BUILTIN.items():
            self.assertEqual(set(mapping), set(profiles.GESTURE_KEYS), name)
            self.assertTrue(all(actions.is_valid(a) for a in mapping.values()), name)

    def test_load_missing_or_corrupt_file_gives_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(profiles.load(Path(d) / "nope.json")["active"], "Music")
            bad = Path(d) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual(set(profiles.load(bad)["profiles"]), set(profiles.BUILTIN))

    def test_save_load_roundtrip_and_sanitising(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            data = profiles.defaults()
            data["profiles"]["Mine"] = {"fist_hold": "screenshot", "peace_hold": "BOGUS", "extra": "x"}
            data["active"] = "Mine"
            profiles.save(data, path)
            got = profiles.load(path)
            self.assertEqual(got["active"], "Mine")
            self.assertEqual(got["profiles"]["Mine"]["fist_hold"], "screenshot")
            self.assertEqual(got["profiles"]["Mine"]["peace_hold"], "none")      # invalid -> none
            self.assertNotIn("extra", got["profiles"]["Mine"])
            self.assertEqual(set(got["profiles"]["Mine"]), set(profiles.GESTURE_KEYS))

    def test_deleted_builtin_comes_back_and_bad_active_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            path.write_text(json.dumps({"active": "Ghost", "profiles": {"Mine": {}}}), encoding="utf-8")
            got = profiles.load(path)
            self.assertEqual(got["active"], "Music")
            self.assertIn("Gaming", got["profiles"])
            self.assertIn("Mine", got["profiles"])

    def test_unique_name_and_reset(self):
        self.assertEqual(profiles.unique_name("Music", ["Music"]), "Music 2")
        self.assertEqual(profiles.unique_name("  ", []), "Profile")
        self.assertEqual(profiles.reset_mapping("Gaming"), profiles.BUILTIN["Gaming"])
        self.assertEqual(profiles.reset_mapping("Custom"), profiles.BUILTIN["Music"])


class EngineMappingTests(unittest.TestCase):
    def test_default_mapping_is_the_music_profile(self):
        rig = Rig()
        rig.run(hand(ring=False, pinky=False), 1.0, settings())          # peace, no mapping given
        self.assertEqual(rig.keys, ["playpause"])

    def test_peace_follows_the_active_profile(self):
        rig = Rig()
        rig.run(hand(ring=False, pinky=False), 1.0, settings(mapping=profiles.BUILTIN["Presentation"]))
        self.assertEqual(rig.keys, ["f5"])

    def test_swipe_directions_use_the_mapping(self):
        rig = Rig()
        s = settings(mapping=profiles.BUILTIN["Presentation"])
        for i in range(12):
            rig.eng.update(hand(wrist_x=0.3 + i * 0.04), SIZE, s, now=i / 30)
        for i in range(12):
            rig.eng.update(hand(wrist_x=0.8 - i * 0.04), SIZE, s, now=3 + i / 30)
        self.assertEqual(rig.keys, ["right", "left"])

    def test_none_action_does_nothing(self):
        rig = Rig()
        res, events = rig.run(hand(**FIST), 1.5, settings(mapping=profiles.BUILTIN["Gaming"] | {"fist_hold": "none"}))
        self.assertEqual(events, [])
        self.assertFalse(rig.audio.mute)

    def test_hotkey_action_and_event_label(self):
        rig = Rig()
        res, events = rig.run(hand(**FIST), 1.2, settings(mapping=profiles.BUILTIN["Video call"]))
        self.assertEqual(rig.hotkeys, [("ctrl", "shift", "m")])
        self.assertEqual(events[0].kind, "action")
        self.assertIn("Teams", events[0].label)

    def test_thumbs_up_steps_volume_once_then_repeats_while_held(self):
        rig = Rig(0.3)
        s = settings()
        _, events = rig.run(thumbs(True), 0.5, s)                        # not held long enough
        self.assertEqual(events, [])
        self.assertAlmostEqual(rig.audio.vol, 0.3)
        _, events = rig.run(thumbs(True), 0.4, s, start=0.5)             # hold reached (0.7 s)
        self.assertEqual([e.kind for e in events], ["vol_up"])
        self.assertAlmostEqual(rig.audio.vol, 0.4)
        _, events = rig.run(thumbs(True), 1.5, s, start=0.9)             # keeps stepping while held
        self.assertGreaterEqual(len(events), 2)
        self.assertGreater(rig.audio.vol, 0.55)

    def test_thumbs_down_lowers_and_pinch_logic_stays_out_of_the_way(self):
        rig = Rig(0.5)
        rig.run(thumbs(False, ratio=1.1), 1.0, settings())
        self.assertAlmostEqual(rig.audio.vol, 0.4)        # exactly one step, no pinch-driven drift

    def test_thumbs_disabled(self):
        rig = Rig(0.5)
        _, events = rig.run(thumbs(True), 2.0, settings(enable_thumbs=False))
        self.assertEqual(events, [])
        self.assertAlmostEqual(rig.audio.vol, 0.5)

    def test_fist_still_toggles_mute_with_default_profile(self):
        rig = Rig()
        rig.run(hand(**FIST), 1.2, settings())
        self.assertTrue(rig.audio.mute)

    def test_new_popups_render(self):
        fx, frame = EffectsManager(), np.zeros((480, 640, 3), np.uint8)
        for kind in ("vol_up", "vol_down", "action", "mute_on", "playpause"):
            fx._draw_popup(frame, kind, 0.4, "Slide: next (Right)")
        self.assertGreater(int(frame.max()), 0)


if __name__ == "__main__":
    unittest.main()
