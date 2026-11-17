"""Sounds, pause, hotkey/tray fallbacks and the tutorial logic (no GUI, speaker or camera)."""
import io
import os
import queue
import sys
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import sounds  # noqa: E402
import tutorial_logic as tl  # noqa: E402
from demo3d import _Audio, synthetic_hand, to_landmarks  # noqa: E402
from gestures import Event, FrameResult, GestureEngine  # noqa: E402
from hotkey import GlobalHotkey, Tray  # noqa: E402
from render3d import VIEW_H, VIEW_W  # noqa: E402
from settings import DEFAULTS  # noqa: E402


class SoundTests(unittest.TestCase):
    def test_wav_is_valid(self):
        data = sounds.tone_wav(sounds.RECIPES["mute_on"], 0.5)
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertGreater(w.getnframes(), 100)

    def test_every_recipe_builds(self):
        for kind in sounds.RECIPES:
            self.assertGreater(len(sounds.tone_wav(sounds.RECIPES[kind])), 44, kind)

    def test_volume_scales_amplitude(self):
        def peak(v):
            with wave.open(io.BytesIO(sounds.tone_wav([(440, 0.1)], v))) as w:
                return int(np.abs(np.frombuffer(w.readframes(w.getnframes()), "<i2")).max())
        self.assertGreater(peak(1.0), peak(0.2))

    def test_player_plays_and_rate_limits(self):
        got = []
        p = sounds.SoundPlayer(backend=got.append)
        self.assertTrue(p.play("next", 0.5, now=10.0))
        self.assertFalse(p.play("prev", 0.5, now=10.01))     # too soon
        self.assertTrue(p.play("prev", 0.5, now=10.5))
        self.assertFalse(p.play("nonsense", 0.5, now=11.0))
        self.assertEqual(len(got), 2)

    def test_no_backend_is_silent_noop(self):
        p = sounds.SoundPlayer(backend=None)
        if sys.platform != "win32":
            self.assertFalse(p.available)
            self.assertFalse(p.play("next", 0.5, now=1.0))

    def test_broken_backend_disables_itself(self):
        def boom(_):
            raise OSError("no device")
        p = sounds.SoundPlayer(backend=boom)
        self.assertFalse(p.play("next", 0.5, now=1.0))
        self.assertFalse(p.available)

    def test_tick_only_when_crossing_a_step(self):
        got = []
        p = sounds.SoundPlayer(backend=got.append)
        p.tick(0.52, 0.5, now=1.0)           # first call just records the level
        self.assertFalse(p.tick(0.55, 0.5, now=2.0))
        self.assertTrue(p.tick(0.62, 0.5, now=3.0))
        self.assertEqual(len(got), 1)

    def test_handle_result_respects_toggle(self):
        got = []
        p = sounds.SoundPlayer(backend=got.append)
        res = FrameResult(events=[Event("lock", 0.0)])
        sounds.handle_result(p, res, {"sound_fx": False}, 5.0)
        self.assertEqual(got, [])
        sounds.handle_result(p, res, {"sound_fx": True, "sound_volume": 0.5}, 6.0)
        self.assertEqual(len(got), 1)


class PauseTests(unittest.TestCase):
    SIZE = (640, 480)

    @staticmethod
    def fist():
        from test_logic import hand
        return hand(index=False, middle=False, ring=False, pinky=False)

    def run_fist(self, settings, seconds=2.0):
        audio = _Audio()
        pressed = []
        eng = GestureEngine(audio, press=pressed.append, now=0.0)
        res = None
        for i in range(int(seconds * 30)):
            res = eng.update(self.fist(), self.SIZE, settings, now=i / 30)
        return audio, res

    def test_fist_mutes_when_not_paused(self):                  # proves the pause test below is real
        audio, res = self.run_fist(dict(DEFAULTS, paused=False))
        self.assertTrue(audio.mute)
        self.assertFalse(res.paused)

    def test_paused_blocks_actions_and_marks_result(self):
        audio, res = self.run_fist(dict(DEFAULTS, paused=True))
        self.assertFalse(audio.mute)
        self.assertTrue(res.paused)
        self.assertEqual(res.events, [])
        self.assertFalse(res.controlling)
        self.assertTrue(res.pose.fist)                          # still tracked, so the tutorial can see it

    def test_pause_keeps_volume_still(self):
        audio = _Audio()
        eng = GestureEngine(audio, press=lambda k: None, now=0.0)
        from test_logic import hand
        before = audio.get()
        for i in range(60):
            eng.update(hand(index=True, middle=False, ring=False, pinky=False, ratio=1.1), self.SIZE,
                       dict(DEFAULTS, paused=True), now=i / 30)
        self.assertEqual(audio.get(), before)

    def test_resuming_works_again(self):
        audio = _Audio()
        eng = GestureEngine(audio, press=lambda k: None, now=0.0)
        for i in range(30):
            eng.update(self.fist(), self.SIZE, dict(DEFAULTS, paused=True), now=i / 30)
        self.assertFalse(audio.mute)
        for i in range(30, 100):
            res = eng.update(self.fist(), self.SIZE, dict(DEFAULTS, paused=False), now=i / 30)
        self.assertTrue(audio.mute)
        self.assertFalse(res.paused)

    def test_no_hand_while_paused_is_flagged(self):
        eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
        res = eng.update(None, self.SIZE, dict(DEFAULTS, paused=True, lost_grace=0.0), now=5.0)
        self.assertTrue(res.paused)


class HotkeyTrayTests(unittest.TestCase):
    def test_hotkey_degrades_off_windows(self):
        hk = GlobalHotkey(queue.Queue())
        if sys.platform != "win32":
            self.assertFalse(hk.supported())
            self.assertFalse(hk.start())
        hk.stop()                                              # never raises

    def test_tray_degrades_without_pystray(self):
        t = Tray(queue.Queue(), None)
        if not t.supported():
            self.assertFalse(t.start())
        t.stop()


def feed(tracker, frames, dt=1 / 30):
    t = 0.0
    for info in frames:
        tracker.feed(info, t)
        t += dt
    return tracker


class TutorialTests(unittest.TestCase):
    def test_steps_cover_core_gestures(self):
        for key in ("hand", "pinch", "fist", "peace", "swipe", "thumbs", "lock"):
            self.assertIn(key, tl.STEP_KEYS)

    def test_hold_gesture_needs_time(self):
        tr = tl.new_tracker(tl.STEP_KEYS.index("fist"))
        info = {"has_hand": True, "flags": {"fist": True}}
        feed(tr, [info] * 10)                                  # 0.33 s: not yet
        self.assertFalse(tr.done)
        feed(tr, [info] * 40)
        self.assertTrue(tr.done)

    def test_releasing_resets_the_hold(self):
        tr = tl.new_tracker(tl.STEP_KEYS.index("peace"))
        on = {"has_hand": True, "flags": {"peace": True}}
        off = {"has_hand": True, "flags": {}}
        feed(tr, [on] * 15 + [off] + [on] * 15)
        self.assertFalse(tr.done)

    def test_wrong_gesture_does_not_count(self):
        tr = tl.new_tracker(tl.STEP_KEYS.index("fist"))
        feed(tr, [{"has_hand": True, "flags": {"peace": True}}] * 60)
        self.assertFalse(tr.done)

    def test_pinch_needs_a_real_range(self):
        tr = tl.new_tracker(tl.STEP_KEYS.index("pinch"))
        feed(tr, [{"has_hand": True, "ratio": 0.7}] * 30)
        self.assertFalse(tr.done)
        feed(tr, [{"has_hand": True, "ratio": r} for r in np.linspace(0.2, 1.3, 20)])
        self.assertTrue(tr.done)

    def test_swipe_needs_open_palm_motion(self):
        tr = tl.new_tracker(tl.STEP_KEYS.index("swipe"))
        fist_moving = [{"has_hand": True, "flags": {"fist": True}, "x": x}
                       for x in np.linspace(0.2, 0.8, 20)]
        feed(tr, fist_moving)
        self.assertFalse(tr.done)
        palm = [{"has_hand": True, "flags": {"palm_open": True}, "x": x}
                for x in np.linspace(0.3, 0.7, 20)]
        feed(tr, palm)
        self.assertTrue(tr.done)

    def test_thumbs_either_direction(self):
        for flag in ("thumb_up", "thumb_down"):
            tr = tl.new_tracker(tl.STEP_KEYS.index("thumbs"))
            feed(tr, [{"has_hand": True, "flags": {flag: True}}] * 40)
            self.assertTrue(tr.done, flag)

    def test_lock_ignores_open_palm_wave(self):
        tr = tl.new_tracker(tl.STEP_KEYS.index("lock"))
        feed(tr, [{"has_hand": True, "flags": {"pinky_up": True, "palm_open": True}}] * 60)
        self.assertFalse(tr.done)
        feed(tr, [{"has_hand": True, "flags": {"pinky_up": True}}] * 40)
        self.assertTrue(tr.done)

    def test_no_hand_never_completes(self):
        for i in range(len(tl.STEPS)):
            tr = tl.new_tracker(i)
            feed(tr, [{"has_hand": False}] * 100)
            self.assertFalse(tr.done, tl.STEP_KEYS[i])


class SettingsTests(unittest.TestCase):
    def test_new_defaults(self):
        self.assertTrue(DEFAULTS["sound_fx"])
        self.assertFalse(DEFAULTS["tutorial_done"])
        self.assertFalse(DEFAULTS["minimize_to_tray"])


if __name__ == "__main__":
    unittest.main()


class TwoHandTests(unittest.TestCase):
    SIZE = (640, 480)

    def test_assign_by_side(self):
        from twohand import assign
        self.assertEqual(assign([0.5]), (0, None))
        self.assertEqual(assign([0.3, 0.7]), (1, 0))
        self.assertEqual(assign([0.7, 0.3]), (0, 1))
        self.assertEqual(assign([0.3, 0.7], "left"), (0, 1))
        with self.assertRaises(ValueError):
            assign([])

    def make(self, ratio, **kw):
        from test_logic import hand
        return hand(index=True, middle=False, ring=False, pinky=False, ratio=ratio, **kw)

    def test_pinch_sets_level_and_calls_setter(self):
        from twohand import SecondaryEngine
        sent = []
        eng = SecondaryEngine(sent.append, level=0.5, sync=True)
        for i in range(90):
            res = eng.update(self.make(1.2), self.SIZE, dict(DEFAULTS), now=i / 30)
        self.assertGreater(res.volume, 0.9)
        self.assertTrue(res.controlling)
        self.assertGreater(len(sent), 3)
        self.assertLessEqual(max(sent), 1.0)
        for i in range(90, 200):
            res = eng.update(self.make(0.05), self.SIZE, dict(DEFAULTS), now=i / 30)
        self.assertLess(res.volume, 0.1)

    def test_setter_is_rate_limited(self):
        from twohand import SecondaryEngine
        sent = []
        eng = SecondaryEngine(sent.append, level=0.0, sync=True)
        for i in range(30):                                  # one second of frames
            eng.update(self.make(1.2), self.SIZE, dict(DEFAULTS), now=i / 30)
        self.assertLessEqual(len(sent), 9)

    def test_locked_paused_and_no_hand_do_not_change_level(self):
        from test_logic import hand
        from twohand import SecondaryEngine
        eng = SecondaryEngine(None, level=0.4, sync=True)
        locked = hand(index=True, middle=False, ring=False, pinky=True, ratio=1.2)
        for i in range(30):
            res = eng.update(locked, self.SIZE, dict(DEFAULTS), now=i / 30)
        self.assertTrue(res.locked)
        self.assertAlmostEqual(eng.level, 0.4)
        for i in range(30):
            res = eng.update(self.make(1.2), self.SIZE, dict(DEFAULTS, paused=True), now=i / 30)
        self.assertTrue(res.paused)
        self.assertAlmostEqual(eng.level, 0.4)
        res = eng.update(None, self.SIZE, dict(DEFAULTS), now=5.0)
        self.assertFalse(res.has_hand)
        self.assertAlmostEqual(res.volume, 0.4)

    def test_brightness_control_degrades_without_package(self):
        from twohand import BrightnessControl
        b = BrightnessControl()
        if not b.available:
            self.assertIsNone(b.get())
            b.set(0.5)                                        # no error

    def test_scene_renders_two_hands(self):
        from render3d import Scene3D
        eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
        from twohand import SecondaryEngine
        sec = SecondaryEngine(None, sync=True)
        sc = Scene3D(480, 360)
        s = dict(DEFAULTS, two_hand_mode=True, orbit_auto=False)
        n1, w1 = to_landmarks(synthetic_hand(0.5, 1.0), 0.7, 0.5)
        n2, w2 = to_landmarks(synthetic_hand(0.3, 1.0), 0.3, 0.5)
        for i in range(15):
            r1 = eng.update(n1, (480, 360), s, now=i / 30, world=w1)
            r2 = sec.update(n2, (480, 360), s, now=i / 30, world=w2)
            two = sc.render(r1, s, 30, now=i / 30, res2=r2)
        sc1 = Scene3D(480, 360)
        eng1 = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
        for i in range(15):
            r1 = eng1.update(n1, (480, 360), s, now=i / 30, world=w1)
            one = sc1.render(r1, s, 30, now=i / 30)
        self.assertEqual(two.shape, (360, 480, 3))
        self.assertFalse(np.array_equal(two, one))             # the second hand really is drawn

    def test_second_hand_fades_when_it_leaves(self):
        from render3d import Scene3D
        from twohand import SecondaryEngine
        eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
        sec = SecondaryEngine(None, sync=True)
        sc = Scene3D(320, 240)
        s = dict(DEFAULTS, two_hand_mode=True, orbit_auto=False)
        n1, w1 = to_landmarks(synthetic_hand(0.5, 1.0), 0.7, 0.5)
        n2, w2 = to_landmarks(synthetic_hand(0.3, 1.0), 0.3, 0.5)
        for i in range(15):
            sc.render(eng.update(n1, (320, 240), s, now=i / 30, world=w1), s, 30, now=i / 30,
                      res2=sec.update(n2, (320, 240), s, now=i / 30, world=w2))
        self.assertGreater(sc._slot2["hand_alpha"], 0.9)
        for i in range(15, 60):
            sc.render(eng.update(n1, (320, 240), s, now=i / 30, world=w1), s, 30, now=i / 30,
                      res2=sec.update(None, (320, 240), s, now=i / 30))
        self.assertLess(sc._slot2["hand_alpha"], 0.05)


class SkinTests(unittest.TestCase):
    def test_all_hand_styles_render_and_differ(self):
        from render3d import Scene3D
        outs = {}
        for style in ("android", "gold", "carbon", "hologram", "robot", "human", "glass"):
            eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
            sc = Scene3D(400, 300)
            s = dict(DEFAULTS, hand_style=style, orbit_auto=False, fx3d_motion=False)
            n, w = to_landmarks(synthetic_hand(0.6, 1.0), 0.5, 0.5)
            for i in range(12):
                res = eng.update(n, (400, 300), s, now=i / 30, world=w)
                img = sc.render(res, s, 30, now=i / 30)
            self.assertEqual(img.dtype, np.uint8, style)
            outs[style] = img
        names = list(outs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertFalse(np.array_equal(outs[a], outs[b]), f"{a} == {b}")

    def test_gold_is_warmer_than_android(self):
        from render3d import Scene3D
        means = {}
        for style in ("android", "gold"):
            eng = GestureEngine(_Audio(), press=lambda k: None, now=0.0)
            sc = Scene3D(400, 300)
            s = dict(DEFAULTS, hand_style=style, orbit_auto=False, fx3d_motion=False)
            n, w = to_landmarks(synthetic_hand(0.6, 1.0), 0.5, 0.5)
            for i in range(12):
                img = sc.render(eng.update(n, (400, 300), s, now=i / 30, world=w), s, 30, now=i / 30)
            c = img[100:220, 120:280].astype(float)
            means[style] = c[..., 0].mean() - c[..., 2].mean()      # red minus blue
        self.assertGreater(means["gold"], means["android"])


class StatsTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.path = __import__("pathlib").Path(self.dir.name) / "stats.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_record_and_summary(self):
        import time
        from stats import UsageStats
        st = UsageStats()
        now = time.time()
        for _ in range(3):
            st.record("mute_on", now)
        st.record("next", now)
        st.record("next", now - 86400)                   # yesterday
        st.add_active(12.5, now)
        sm = st.summary(7, now)
        self.assertEqual(sm["total"], 5)
        self.assertEqual(sm["per_kind"]["mute_on"], 3)
        self.assertEqual(sm["top"], "mute_on")
        self.assertAlmostEqual(sm["active_seconds"], 12.5)
        self.assertEqual(sum(sm["hours"]), 5)
        self.assertEqual(len(sm["daily"]), 7)
        self.assertEqual(sm["daily"][-1][1], 4)            # today is last
        self.assertEqual(sm["daily"][-2][1], 1)
        self.assertEqual(st.summary(1, now)["total"], 4)

    def test_save_load_roundtrip(self):
        from stats import UsageStats
        st = UsageStats()
        st.record("lock")
        self.assertTrue(st.save(self.path))
        again = UsageStats.load(self.path)
        self.assertEqual(again.summary(7)["total"], 1)

    def test_bad_or_missing_file_gives_empty_stats(self):
        from stats import UsageStats
        self.assertEqual(UsageStats.load(self.path).summary(7)["total"], 0)
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(UsageStats.load(self.path).summary(7)["total"], 0)
        self.path.write_text('{"days": 5}', encoding="utf-8")
        self.assertEqual(UsageStats.load(self.path).summary(7)["total"], 0)

    def test_corrupt_day_is_repaired(self):
        from stats import UsageStats, day_key
        import time
        st = UsageStats({"days": {day_key(time.time()): {"events": {"next": 2}, "hours": [1, 2]}}})
        st.record("next")
        sm = st.summary(7)
        self.assertEqual(sm["total"], 3)

    def test_prune_and_reset(self):
        import time
        from stats import UsageStats
        st = UsageStats()
        st.record("next", time.time() - 200 * 86400)
        st.record("next")
        st.prune()
        self.assertEqual(len(st.data["days"]), 1)
        st.reset()
        self.assertEqual(st.summary(7)["total"], 0)

    def test_duration_format(self):
        from stats import fmt_duration
        self.assertEqual(fmt_duration(42), "42 s")
        self.assertEqual(fmt_duration(300), "5 min")
        self.assertEqual(fmt_duration(3720), "1 h 02 min")


class AutoSwitchTests(unittest.TestCase):
    AVAILABLE = ["Music", "Presentation", "Video call", "Gaming"]

    def make(self, exes, rules=None):
        from autoswitch import AutoSwitcher
        it = iter(exes)
        return AutoSwitcher(rules, get_exe=lambda: next(it))

    def run_seq(self, sw, current, n):
        """Poll n times, applying any switch the way the GUI would; returns the switches."""
        out = []
        for _ in range(n):
            name = sw.poll(current, self.AVAILABLE)
            if name:
                current = name
                out.append(name)
        return out, current

    def test_switches_after_app_is_stable_and_goes_back(self):
        sw = self.make(["powerpnt.exe"] * 3 + ["notepad.exe"] * 3, rules={"powerpnt.exe": "Presentation"})
        out, cur = self.run_seq(sw, "Music", 3)
        self.assertEqual(out, ["Presentation"])
        out, cur = self.run_seq_continue(sw, cur, 3)
        self.assertEqual(out, ["Music"])

    def run_seq_continue(self, sw, current, n):
        return self.run_seq(sw, current, n)

    def test_needs_two_consecutive_polls(self):
        sw = self.make(["zoom.exe", "chrome.exe", "zoom.exe", "chrome.exe"])
        out, _ = self.run_seq(sw, "Music", 4)
        self.assertEqual(out, [])                              # alt-tab flicker never switches

    def test_user_choice_is_not_overridden_on_leave(self):
        sw = self.make(["zoom.exe"] * 2 + ["notepad.exe"] * 3)
        out, cur = self.run_seq(sw, "Music", 2)
        self.assertEqual(out, ["Video call"])
        out, cur = self.run_seq(sw, "Gaming", 3)               # user picked Gaming meanwhile
        self.assertEqual(out, [])
        self.assertEqual(cur, "Gaming")

    def test_ignored_windows_change_nothing(self):
        sw = self.make(["zoom.exe"] * 2 + ["python.exe"] * 3 + ["explorer.exe"] * 3)
        out, cur = self.run_seq(sw, "Music", 8)
        self.assertEqual(out, ["Video call"])                   # staying in our own window keeps it
        self.assertEqual(cur, "Video call")

    def test_missing_profile_and_no_exe(self):
        sw = self.make(["zoom.exe"] * 3, rules={"zoom.exe": "Deleted profile"})
        out, _ = self.run_seq(sw, "Music", 3)
        self.assertEqual(out, [])
        sw = self.make([None, None])
        self.assertEqual(self.run_seq(sw, "Music", 2)[0], [])

    def test_rules_file_merges_and_survives_garbage(self):
        import tempfile
        from pathlib import Path
        from autoswitch import DEFAULT_RULES, load_rules
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            self.assertEqual(load_rules(p), DEFAULT_RULES)
            p.write_text('{"OBS64.exe": "Gaming", "bad": 5, "": "x"}', encoding="utf-8")
            r = load_rules(p)
            self.assertEqual(r["obs64.exe"], "Gaming")
            self.assertNotIn("bad", r)
            self.assertEqual(r["zoom.exe"], "Video call")
            p.write_text("[1,2", encoding="utf-8")
            self.assertEqual(load_rules(p), DEFAULT_RULES)

    def test_default_rules_point_at_builtin_profiles(self):
        from autoswitch import DEFAULT_RULES
        from profiles import BUILTIN
        for exe, prof in DEFAULT_RULES.items():
            self.assertIn(prof, BUILTIN, exe)

    def test_foreground_is_none_off_windows(self):
        from autoswitch import foreground_exe
        if sys.platform != "win32":
            self.assertIsNone(foreground_exe())


class LauncherTests(unittest.TestCase):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_launcher_files_exist_with_windows_line_endings(self):
        for name in ("GestureSense.bat", "create_desktop_shortcut.bat", "run.bat"):
            data = open(os.path.join(self.ROOT, name), "rb").read()
            self.assertIn(b"\r\n", data, name)
            self.assertNotIn(b"\n", data.replace(b"\r\n", b""), name)   # no bare LF

    def test_launcher_uses_windowless_python_and_the_icon(self):
        bat = open(os.path.join(self.ROOT, "GestureSense.bat")).read()
        self.assertIn("pythonw.exe", bat)
        self.assertIn("main.py", bat)
        lnk = open(os.path.join(self.ROOT, "create_desktop_shortcut.bat")).read()
        self.assertIn("icon.ico", lnk)
        self.assertIn("GestureSense.bat", lnk)

    def test_auto_start_defaults_off(self):
        self.assertFalse(DEFAULTS["auto_start"])
