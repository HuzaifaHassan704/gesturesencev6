"""Global pause hotkey (Ctrl+Alt+G by default) and an optional system-tray icon. No Tk imports.

Both run on their own threads and only drop messages into a queue, so the GUI thread stays the
only one that touches Tk. Everything degrades to a no-op where it is not supported (non-Windows,
or `pystray` not installed).
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Optional

MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x0001, 0x0002, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
HOTKEY_ID = 0x4753
HOTKEY_TEXT = "Ctrl+Alt+G"


class GlobalHotkey:
    """Posts "pause" to `events` whenever the hotkey is pressed anywhere in Windows."""

    def __init__(
        self, events: "queue.Queue[str]", vk: int = ord("G"), mods: int = MOD_CONTROL | MOD_ALT
    ) -> None:
        self.events, self.vk, self.mods = events, vk, mods
        self.active = False
        self._thread: Optional[threading.Thread] = None
        self._tid = 0
        self._ready = threading.Event()

    @staticmethod
    def supported() -> bool:
        return sys.platform == "win32"

    def start(self) -> bool:
        if not self.supported() or self._thread is not None:
            return self.active
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(2.0)
        return self.active

    def _run(self) -> None:  # pragma: no cover (Windows only)
        import ctypes
        from ctypes import wintypes

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        self._tid = kernel32.GetCurrentThreadId()
        self.active = bool(
            user32.RegisterHotKey(None, HOTKEY_ID, self.mods | MOD_NOREPEAT, self.vk)
        )
        self._ready.set()
        if not self.active:
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                self.events.put("pause")
        user32.UnregisterHotKey(None, HOTKEY_ID)

    def stop(self) -> None:
        if self._thread is None:
            return
        if self._tid and self.supported():  # pragma: no cover
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        self._thread = None
        self.active = False


class Tray:
    """System-tray icon with Show / Pause / Quit. Needs the optional `pystray` package."""

    def __init__(self, events: "queue.Queue[str]", icon_path) -> None:
        self.events, self.icon_path = events, icon_path
        self._icon = None

    @staticmethod
    def supported() -> bool:
        try:
            import pystray  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def running(self) -> bool:
        return self._icon is not None

    def start(self, paused: bool = False) -> bool:
        if self._icon is not None:
            return True
        try:
            import pystray
            from PIL import Image

            def put(name):
                return lambda icon=None, item=None: self.events.put(name)

            menu = pystray.Menu(
                pystray.MenuItem("Show GestureSense", put("show"), default=True),
                pystray.MenuItem("Pause / resume gestures", put("pause")),
                pystray.MenuItem("Quit", put("quit")),
            )
            image = (
                Image.open(self.icon_path)
                if self.icon_path
                else Image.new("RGB", (64, 64), "#3B82F6")
            )
            self._icon = pystray.Icon("GestureSense", image, "GestureSense", menu)
            self._icon.run_detached()
            return True
        except Exception:
            self._icon = None
            return False

    def stop(self) -> None:
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None
