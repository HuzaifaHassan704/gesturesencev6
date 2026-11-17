"""The statistics window: gesture counts, busiest hours and the last 7 days."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import TYPE_CHECKING

import ui
from stats import LABELS, fmt_duration
from ui import C

if TYPE_CHECKING:                       # pragma: no cover
    from gui import App


class StatsWindow(tk.Toplevel):
    def __init__(self, app: "App") -> None:
        super().__init__(app.root)
        self.app, self.f = app, app.f
        self.title("GestureSense statistics")
        self.configure(bg=C["bg"])
        self.transient(app.root)
        w, h = ui.sc(640), ui.sc(640)
        x = app.root.winfo_rootx() + max((app.root.winfo_width() - w) // 2, 0)
        y = app.root.winfo_rooty() + max((app.root.winfo_height() - h) // 4, 0)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.range = 7
        box = ui.card(self)
        box.pack(fill="both", expand=True, padx=ui.sc(14), pady=ui.sc(14))
        self.inner = tk.Frame(box, bg=C["surface"])
        self.inner.pack(fill="both", expand=True, padx=ui.sc(20), pady=ui.sc(16))
        head = tk.Frame(self.inner, bg=C["surface"])
        head.pack(fill="x")
        tk.Label(head, text="Your gesture statistics", font=self.f.title, bg=C["surface"],
                 fg=C["text"]).pack(side="left")
        ui.Button(head, "30 days", lambda: self.set_range(30), self.f, kind="ghost", width=84,
                  height=30).pack(side="right")
        ui.Button(head, "7 days", lambda: self.set_range(7), self.f, kind="ghost", width=84,
                  height=30).pack(side="right", padx=(0, 6))
        self.tiles = tk.Frame(self.inner, bg=C["surface"])
        self.tiles.pack(fill="x", pady=(ui.sc(12), 0))
        self.canvas = tk.Canvas(self.inner, bg=C["surface"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(ui.sc(10), 0))
        self.canvas.bind("<Configure>", lambda e: self.draw())
        foot = tk.Frame(self.inner, bg=C["surface"])
        foot.pack(fill="x", pady=(ui.sc(8), 0))
        tk.Label(foot, text="Stored only on this computer.", font=self.f.small, bg=C["surface"],
                 fg=C["faint"]).pack(side="left")
        ui.Button(foot, "Reset stats", self.reset, self.f, kind="danger", width=110, height=32).pack(
            side="right")
        ui.Button(foot, "Close", self.destroy, self.f, kind="secondary", width=90, height=32).pack(
            side="right", padx=(0, 8))
        self.refresh()

    def set_range(self, days: int) -> None:
        self.range = days
        self.refresh()

    def reset(self) -> None:
        if messagebox.askyesno("Reset statistics", "Delete all saved usage statistics?", parent=self):
            self.app.stats.reset()
            self.app.stats.save()
            self.refresh()

    def refresh(self) -> None:
        self.sm = self.app.stats.summary(self.range)
        for child in self.tiles.winfo_children():
            child.destroy()
        sm = self.sm
        top = LABELS.get(sm["top"], "-") if sm["top"] else "-"
        busy = f"{sm['busiest_hour']:02d}:00" if sm["busiest_hour"] is not None else "-"
        for label, value in (("GESTURES", str(sm["total"])), ("ACTIVE TIME", fmt_duration(sm["active_seconds"])),
                             ("MOST USED", top), ("BUSIEST HOUR", busy)):
            tile = tk.Frame(self.tiles, bg=C["surface2"])
            tile.pack(side="left", expand=True, fill="x", padx=(0, 6))
            tk.Label(tile, text=label, font=self.f.tiny, bg=C["surface2"], fg=C["faint"]).pack(
                anchor="w", padx=10, pady=(8, 0))
            tk.Label(tile, text=value, font=self.f.title, bg=C["surface2"], fg=C["text"]).pack(
                anchor="w", padx=10, pady=(0, 8))
        self.draw()

    # ------------------------------------------------------------ charts
    def bars(self, x0, y0, w, h, values, labels, color, title) -> None:
        c = self.canvas
        c.create_text(x0, y0, text=title, anchor="sw", font=self.f.tiny, fill=C["faint"])
        n = max(len(values), 1)
        top = max(max(values, default=0), 1)
        gap = max(2, int(w * 0.012))
        bw = max((w - gap * (n - 1)) / n, 2)
        for i, v in enumerate(values):
            bx = x0 + i * (bw + gap)
            bh = h * v / top
            c.create_rectangle(bx, y0 + h + 4 - h, bx + bw, y0 + h + 4, fill=C["surface2"], outline="")
            if v:
                c.create_rectangle(bx, y0 + 4 + h - bh, bx + bw, y0 + h + 4, fill=color, outline="")
            if labels[i]:
                c.create_text(bx + bw / 2, y0 + h + 16, text=labels[i], font=self.f.tiny, fill=C["faint"])

    def hbars(self, x0, y0, w, rows, color, title) -> None:
        c = self.canvas
        c.create_text(x0, y0, text=title, anchor="sw", font=self.f.tiny, fill=C["faint"])
        top = max([v for _, v in rows], default=1) or 1
        label_w = ui.sc(120)
        for i, (name, v) in enumerate(rows):
            y = y0 + 8 + i * ui.sc(24)
            c.create_text(x0, y + 9, text=name, anchor="w", font=self.f.small, fill=C["text"])
            bw = max((w - label_w - ui.sc(44)) * v / top, 2 if v else 0)
            c.create_rectangle(x0 + label_w, y + 3, x0 + label_w + bw, y + ui.sc(17), fill=color, outline="")
            c.create_text(x0 + label_w + bw + 6, y + 9, text=str(v), anchor="w", font=self.f.small,
                          fill=C["muted"])

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        W, H = max(c.winfo_width(), 300), max(c.winfo_height(), 200)
        sm = self.sm
        if sm["total"] == 0:
            c.create_text(W / 2, H / 2, text="No gestures recorded yet.\nUse GestureSense for a while and "
                          "come back.", font=self.f.body, fill=C["muted"], justify="center")
            return
        pad = ui.sc(4)
        rows = sorted(((LABELS.get(k, k), v) for k, v in sm["per_kind"].items()), key=lambda r: -r[1])[:8]
        self.hbars(pad, ui.sc(14), W - 2 * pad, rows, C["accent"], "GESTURES BY TYPE")
        y = ui.sc(14) + 8 + len(rows) * ui.sc(24) + ui.sc(34)
        bar_h = max(int((H - y - ui.sc(150)) / 2), ui.sc(40))
        self.bars(pad, y, W - 2 * pad, bar_h, sm["hours"],
                  [str(h) if h % 3 == 0 else "" for h in range(24)], C["good"], "BY HOUR OF DAY")
        y2 = y + bar_h + ui.sc(46)
        daily = sm["daily"][-14:] if self.range > 14 else sm["daily"]
        self.bars(pad, y2, W - 2 * pad, bar_h, [n for _, n in daily],
                  [d[8:] for d, _ in daily], C["violet"], "GESTURES PER DAY")
