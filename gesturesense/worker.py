"""Background thread: camera -> MediaPipe -> gestures -> effects.

Keeping this off the GUI thread means the window never freezes. The GUI only reads
`worker.snapshot` (an RGB frame plus an info dict) and writes `worker.settings`.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesturesense.core.filters import LandmarkFilter
from gesturesense.core.gestures import GestureEngine
from gesturesense.core.twohand import BrightnessControl, SecondaryEngine, assign
from gesturesense.render import VIEW_H, VIEW_W, Scene3D
from gesturesense.render.effects import EffectsManager
from gesturesense.system.audio import AudioController
from gesturesense.system.sounds import SoundPlayer, handle_result

WRIST_ID = 0


def create_landmarker(model_path: Path, num_hands: int = 1):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


class CameraWorker(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, model_path: Path, settings: dict[str, Any]) -> None:
        super().__init__(daemon=True, name="GestureWorker")
        self.cap = cap
        self.model_path = model_path
        self.settings = settings  # GUI replaces this dict (atomic swap)
        self.snapshot: Optional[tuple[Any, dict]] = None
        self.error: Optional[str] = None
        self.event_log: list[str] = []
        self._stop_flag = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        landmarker = None
        com_ready = False
        try:
            try:  # COM must be initialised per thread
                import comtypes

                comtypes.CoInitialize()
                com_ready = True
            except Exception:
                pass
            audio = AudioController()
            want_hands = 2 if self.settings.get("two_hand_mode") else 1
            landmarker = create_landmarker(self.model_path, want_hands)
            engine = GestureEngine(audio)
            effects = EffectsManager()
            scene = Scene3D()
            lmf, lmf2 = LandmarkFilter(), LandmarkFilter()
            bright = BrightnessControl()
            second = SecondaryEngine(
                bright.set if bright.available else None, level=bright.get() or 0.5
            )
            sfx = SoundPlayer()

            start, last_ts, fps, t_prev, failures = time.time(), 0, 0.0, time.time(), 0
            while not self._stop_flag.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    failures += 1
                    if failures > 100:
                        raise RuntimeError("The camera stopped delivering frames.")
                    time.sleep(0.01)
                    continue
                failures = 0

                frame = cv2.flip(frame, 1)  # mirror view
                h, w = frame.shape[:2]
                ts = max(int((time.time() - start) * 1000), last_ts + 1)  # must increase
                last_ts = ts
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
                )
                settings = self.settings
                want = 2 if settings.get("two_hand_mode") else 1
                if want != want_hands:  # switched on / off while running
                    landmarker.close()
                    landmarker = create_landmarker(self.model_path, want)
                    want_hands, last_ts = want, last_ts + 1
                    lmf.miss(ts / 1000.0)
                    lmf2.miss(ts / 1000.0)
                    continue
                hands = list(result.hand_landmarks or [])
                worlds = list(result.hand_world_landmarks or [])
                p_idx, s_idx = (
                    assign(
                        [hl[WRIST_ID].x for hl in hands], str(settings.get("primary_hand", "right"))
                    )
                    if hands
                    else (None, None)
                )
                if want == 1:
                    s_idx = None
                landmarks = hands[p_idx] if p_idx is not None else None
                world = worlds[p_idx] if landmarks is not None and p_idx < len(worlds) else None
                landmarks2 = hands[s_idx] if s_idx is not None else None
                world2 = worlds[s_idx] if landmarks2 is not None and s_idx < len(worlds) else None

                strength = float(settings.get("filter_strength", 0.5))
                if landmarks is not None:  # One-Euro jitter filter (strength 0 = off)
                    landmarks, world = lmf.apply(landmarks, world, ts / 1000.0, strength)
                else:
                    lmf.miss(ts / 1000.0)
                if landmarks2 is not None:
                    landmarks2, world2 = lmf2.apply(landmarks2, world2, ts / 1000.0, strength)
                else:
                    lmf2.miss(ts / 1000.0)
                res2 = None
                if want == 2:
                    res2 = second.update(landmarks2, (w, h), settings, world=world2)
                res = engine.update(landmarks, (w, h), settings, world=world)
                now = time.time()
                handle_result(sfx, res, settings, now)
                for e in res.events:  # the tutorial reads these
                    self.event_log.append(e.kind)
                del self.event_log[:-50]
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now
                effects.render(frame, res, settings, fps)

                # output resolution: the viewport size (sent by the GUI) times the quality setting
                vw, vh = int(settings.get("view_w", VIEW_W)), int(settings.get("view_h", VIEW_H))
                q = min(max(float(settings.get("render_quality", 0.85)), 0.3), 1.0)
                rw, rh = max(int(vw * q) // 2 * 2, 160), max(int(vh * q) // 2 * 2, 120)
                mode = settings.get("view_mode", "pip")
                if mode == "camera":  # plain camera view + 2D effects, letterboxed
                    out = np.full((rh, rw, 3), (15, 11, 8), np.uint8)
                    k = min(rw / w, rh / h)
                    nw, nh = int(w * k), int(h * k)
                    x0, y0 = (rw - nw) // 2, (rh - nh) // 2
                    out[y0 : y0 + nh, x0 : x0 + nw] = cv2.resize(
                        frame, (nw, nh), interpolation=cv2.INTER_LINEAR
                    )
                else:  # 3D scene, optionally with a camera inset
                    scene.set_size(rw, rh)
                    out = scene.render(
                        res,
                        settings,
                        fps,
                        now,
                        res2=res2,
                        second_role=str(settings.get("second_role", "brightness")),
                    )
                    if mode == "pip":
                        scene.add_inset(
                            out,
                            frame,
                            effects.state_color(res),
                            float(settings.get("pip_size", 0.28)),
                        )
                    effects.draw_popups(out, now)

                info = {
                    "volume": res.volume,
                    "muted": res.muted,
                    "locked": res.locked,
                    "status": res.status,
                    "fps": fps,
                    "ratio": res.pose.ratio if res.pose else 0.0,
                    "yaw_deg": math.degrees(scene.yaw),
                    "has_hand": res.has_hand,
                    "paused": res.paused,
                    "second": (
                        {
                            "has_hand": res2.has_hand,
                            "level": res2.volume,
                            "controlling": res2.controlling,
                            "locked": res2.locked,
                            "can_set": bright.available,
                        }
                        if res2 is not None
                        else None
                    ),
                    "x": float(res.norm[0, 0]) if res.norm is not None else 0.5,
                    "flags": (
                        {
                            "fist": res.pose.fist,
                            "peace": res.pose.peace,
                            "palm_open": res.pose.palm_open,
                            "thumb_up": res.pose.thumb_up,
                            "thumb_down": res.pose.thumb_down,
                            "pinky_up": res.pose.pinky_up,
                        }
                        if res.pose
                        else {}
                    ),
                }
                self.snapshot = (cv2.cvtColor(out, cv2.COLOR_BGR2RGB), info)
        except Exception as exc:  # report to the GUI instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if landmarker is not None:
                landmarker.close()
            self.cap.release()
            if com_ready:
                try:
                    import comtypes

                    comtypes.CoUninitialize()
                except Exception:
                    pass
