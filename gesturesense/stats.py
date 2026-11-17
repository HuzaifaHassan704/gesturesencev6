"""Usage statistics: which gestures you use, when, and for how long. No GUI imports.

Everything stays on your computer in `gesturesense_stats.json` (next to your settings).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from paths import data_dir

STATS_PATH = data_dir() / "gesturesense_stats.json"
KEEP_DAYS = 90

# event kind -> label shown in the stats window
LABELS = {
    "mute_on": "Mute", "mute_off": "Unmute", "playpause": "Play / pause", "next": "Next track",
    "prev": "Previous track", "lock": "Lock", "unlock": "Unlock", "vol_up": "Volume up",
    "vol_down": "Volume down", "action": "Other action",
}


def day_key(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


class UsageStats:
    def __init__(self, data: Optional[dict] = None) -> None:
        self.data: dict[str, Any] = data if isinstance(data, dict) else {}
        self.data.setdefault("days", {})
        self.dirty = False

    # ---------------------------------------------------------------- recording
    def _day(self, ts: float) -> dict:
        d = self.data["days"].setdefault(day_key(ts), {"events": {}, "hours": [0] * 24, "active": 0.0})
        d.setdefault("events", {})
        d.setdefault("active", 0.0)
        hours = d.setdefault("hours", [0] * 24)
        if len(hours) != 24:
            d["hours"] = (list(hours) + [0] * 24)[:24]
        return d

    def record(self, kind: str, ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        d = self._day(ts)
        d["events"][kind] = int(d["events"].get(kind, 0)) + 1
        d["hours"][time.localtime(ts).tm_hour] += 1
        self.dirty = True

    def add_active(self, seconds: float, ts: Optional[float] = None) -> None:
        if seconds <= 0:
            return
        ts = time.time() if ts is None else ts
        self._day(ts)["active"] = float(self._day(ts)["active"]) + float(seconds)
        self.dirty = True

    # ---------------------------------------------------------------- reading
    def _recent(self, days: int, now: Optional[float] = None) -> list[tuple[str, dict]]:
        now = time.time() if now is None else now
        keys = [day_key(now - i * 86400) for i in range(days)]
        return [(k, self.data["days"].get(k)) for k in keys]

    def summary(self, days: int = 7, now: Optional[float] = None) -> dict:
        """Totals over the last `days` days (today included)."""
        per_kind: dict[str, int] = {}
        hours = [0] * 24
        daily = []                                  # oldest first: (day, gestures)
        active = 0.0
        for key, d in reversed(self._recent(days, now)):
            n = 0
            if d:
                for kind, c in d.get("events", {}).items():
                    per_kind[kind] = per_kind.get(kind, 0) + int(c)
                    n += int(c)
                for h, c in enumerate(d.get("hours", [])[:24]):
                    hours[h] += int(c)
                active += float(d.get("active", 0.0))
            daily.append((key, n))
        total = sum(per_kind.values())
        top = max(per_kind, key=per_kind.get) if per_kind else None
        return {"total": total, "per_kind": per_kind, "hours": hours, "daily": daily,
                "active_seconds": active, "top": top,
                "busiest_hour": (max(range(24), key=lambda h: hours[h]) if total else None)}

    # ---------------------------------------------------------------- housekeeping
    def prune(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        keep = {day_key(now - i * 86400) for i in range(KEEP_DAYS)}
        for k in list(self.data["days"]):
            if k not in keep:
                del self.data["days"][k]

    def reset(self) -> None:
        self.data = {"days": {}}
        self.dirty = True

    def save(self, path: Optional[Path] = None) -> bool:
        self.prune()
        try:
            (path or STATS_PATH).write_text(json.dumps(self.data), encoding="utf-8")
            self.dirty = False
            return True
        except OSError:
            return False

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "UsageStats":
        try:
            raw = json.loads((path or STATS_PATH).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict) or not isinstance(raw.get("days"), dict):
            return cls()
        return cls(raw)


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"
