"""Windows master-volume control through Pycaw."""

from __future__ import annotations

from ctypes import POINTER, cast

import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class AudioController:
    """Create this object in the thread that will use it (COM objects are thread-bound)."""

    def __init__(self) -> None:
        speakers = AudioUtilities.GetSpeakers()
        if hasattr(speakers, "EndpointVolume"):  # newer pycaw versions
            self._ep = speakers.EndpointVolume
        else:
            iface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._ep = cast(iface, POINTER(IAudioEndpointVolume))

    def get(self) -> float:
        return float(self._ep.GetMasterVolumeLevelScalar())

    def set(self, value: float) -> None:
        self._ep.SetMasterVolumeLevelScalar(float(np.clip(value, 0.0, 1.0)), None)

    def muted(self) -> bool:
        return bool(self._ep.GetMute())

    def toggle_mute(self) -> bool:
        """Toggle mute and return the new muted state."""
        self._ep.SetMute(0 if self.muted() else 1, None)
        return self.muted()
