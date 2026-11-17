"""Switch the gesture profile automatically when you change the active app. No GUI imports.

PowerPoint -> Presentation, Zoom / Teams / Webex -> Video call, Spotify / VLC / media players ->
Music. Leaving such an app puts back the profile you had before (unless you picked another one
yourself in the meantime). Add or change rules in `gesturesense_autorules.json`:
    {"obs64.exe": "Gaming", "chrome.exe": "Music"}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from paths import data_dir

RULES_PATH = data_dir() / "gesturesense_autorules.json"

DEFAULT_RULES: dict[str, str] = {
    "powerpnt.exe": "Presentation", "keynote.exe": "Presentation", "soffice.bin": "Presentation",
    "zoom.exe": "Video call", "teams.exe": "Video call", "ms-teams.exe": "Video call",
    "webex.exe": "Video call", "skype.exe": "Video call", "slack.exe": "Video call",
    "spotify.exe": "Music", "vlc.exe": "Music", "wmplayer.exe": "Music", "musicbee.exe": "Music",
    "itunes.exe": "Music", "applemusic.exe": "Music",
}
# windows that must never trigger a change (our own window, the desktop and shell)
IGNORE = {"python.exe", "pythonw.exe", "gesturesense.exe", "explorer.exe", "searchhost.exe",
          "shellexperiencehost.exe", "startmenuexperiencehost.exe", "lockapp.exe"}


def load_rules(path: Optional[Path] = None) -> dict[str, str]:
    rules = dict(DEFAULT_RULES)
    try:
        extra = json.loads((path or RULES_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return rules
    if isinstance(extra, dict):
        for exe, profile in extra.items():
            if isinstance(exe, str) and isinstance(profile, str) and exe.strip() and profile.strip():
                rules[exe.strip().lower()] = profile.strip()
    return rules


def foreground_exe() -> Optional[str]:
    """File name of the program that owns the active window (Windows only), e.g. 'zoom.exe'."""
    if sys.platform != "win32":
        return None
    try:                                                     # pragma: no cover (Windows only)
        import ctypes
        from ctypes import wintypes

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = kernel32.OpenProcess(0x1000, False, pid.value)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            buf = ctypes.create_unicode_buffer(520)
            size = wintypes.DWORD(len(buf))
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return Path(buf.value).name.lower()
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        pass
    return None


class AutoSwitcher:
    """Decides, one poll at a time, whether the profile should change."""

    def __init__(self, rules: Optional[dict[str, str]] = None,
                 get_exe: Callable[[], Optional[str]] = foreground_exe) -> None:
        self.rules = rules if rules is not None else load_rules()
        self.get_exe = get_exe
        self._last: Optional[str] = None
        self.base: Optional[str] = None            # the profile to go back to
        self.auto_profile: Optional[str] = None    # the one we switched to

    def poll(self, current: str, available: Sequence[str]) -> Optional[str]:
        """Return a profile name to switch to, or None to leave things as they are."""
        exe = self.get_exe()
        if exe is None:
            return None
        exe = exe.lower()
        if exe in IGNORE:
            return None
        if exe != self._last:                      # act only once the app has stayed in front
            self._last = exe
            return None
        target = self.rules.get(exe)
        if target and target in available:
            if current == target:
                return None
            if self.base is None:
                self.base = current
            self.auto_profile = target
            return target
        # an app without a rule: go back to what the user had, unless they changed it themselves
        if self.base is not None:
            back = self.base if (current == self.auto_profile and self.base in available) else None
            self.base = self.auto_profile = None
            return back
        return None

    def forget(self) -> None:
        self._last = self.base = self.auto_profile = None
