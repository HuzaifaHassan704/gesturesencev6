"""Sound effects for gestures. No GUI imports.

Short tones are synthesised with NumPy into WAV bytes and played through `winsound` on Windows
(asynchronously, from memory). On other systems, or if sound is unavailable, everything is a quiet
no-op. The player is injectable, so it is unit-testable without a speaker.
"""

from __future__ import annotations

import io
import sys
import wave
from typing import Callable, Optional

import numpy as np

RATE = 22050

# kind -> list of (frequency Hz, seconds); played one after another
RECIPES: dict[str, list[tuple[float, float]]] = {
    "mute_on": [(660, 0.07), (440, 0.11)],
    "mute_off": [(440, 0.07), (660, 0.11)],
    "playpause": [(587, 0.06), (880, 0.09)],
    "next": [(523, 0.05), (784, 0.09)],
    "prev": [(784, 0.05), (523, 0.09)],
    "lock": [(392, 0.06), (294, 0.12)],
    "unlock": [(294, 0.06), (392, 0.12)],
    "vol_up": [(700, 0.05)],
    "vol_down": [(500, 0.05)],
    "action": [(620, 0.07)],
    "pause": [(520, 0.06), (390, 0.06), (260, 0.1)],
    "resume": [(260, 0.06), (390, 0.06), (520, 0.1)],
    "success": [(523, 0.07), (659, 0.07), (784, 0.14)],
}


def tone_wav(notes: list[tuple[float, float]], volume: float = 0.5) -> bytes:
    """Mono 16-bit WAV bytes for the given (frequency, seconds) notes, with soft attack/decay."""
    parts = []
    for freq, secs in notes:
        n = max(int(RATE * secs), 8)
        t = np.arange(n) / RATE
        wave_ = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
        ramp = min(int(0.008 * RATE), n // 2)
        env = np.ones(n)
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        parts.append(wave_ * env)
    data = np.concatenate(parts) if parts else np.zeros(8)
    data = np.clip(data / 1.25 * float(np.clip(volume, 0.0, 1.0)) * 0.7, -1.0, 1.0)
    pcm = (data * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _winsound_backend() -> Optional[Callable[[bytes], None]]:
    if sys.platform != "win32":
        return None
    try:
        import winsound

        def play(data: bytes) -> None:
            winsound.PlaySound(data, winsound.SND_MEMORY | winsound.SND_ASYNC)

        return play
    except Exception:
        return None


class SoundPlayer:
    """`play(kind)` for events and `tick(volume)` for the volume-following tick."""

    TICK_EVERY = 0.10  # volume changes are 10 % apart
    MIN_GAP = 0.07  # seconds between sounds, so rapid events never pile up

    def __init__(self, backend: Optional[Callable[[bytes], None]] = None) -> None:
        self.backend = backend if backend is not None else _winsound_backend()
        self._cache: dict[tuple, bytes] = {}
        self._last = -1.0
        self._bucket: Optional[int] = None

    @property
    def available(self) -> bool:
        return self.backend is not None

    def _emit(self, key: tuple, notes, volume: float, now: float) -> bool:
        if self.backend is None or now - self._last < self.MIN_GAP:
            return False
        data = self._cache.get(key)
        if data is None:
            data = self._cache[key] = tone_wav(notes, volume)
        try:
            self.backend(data)
        except Exception:
            self.backend = None  # a broken sound device must never break the app
            return False
        self._last = now
        return True

    def play(self, kind: str, volume: float = 0.5, now: float = 0.0) -> bool:
        notes = RECIPES.get(kind)
        if notes is None:
            return False
        return self._emit((kind, round(volume, 2)), notes, volume, now)

    def tick(self, level: float, volume: float = 0.5, now: float = 0.0) -> bool:
        """A short blip each time the controlled volume crosses a 10 % step; pitch rises with it."""
        bucket = int(min(max(level, 0.0), 1.0) / self.TICK_EVERY)
        if self._bucket is None:
            self._bucket = bucket
            return False
        if bucket == self._bucket:
            return False
        self._bucket = bucket
        freq = 330 + 14 * bucket * 4
        return self._emit(("tick", bucket, round(volume, 2)), [(freq, 0.03)], volume * 0.7, now)

    def reset_tick(self) -> None:
        self._bucket = None


def handle_result(player: SoundPlayer, res, settings: dict, now: float) -> None:
    """Turn one FrameResult into sounds (events, plus ticks while the pinch sets the volume)."""
    if not settings.get("sound_fx", True) or not player.available:
        return
    vol = float(settings.get("sound_volume", 0.5))
    for e in res.events:
        player.play(e.kind, vol, now)
    if res.controlling:
        player.tick(res.volume, vol, now)
    else:
        player.reset_tick()
