"""Calibration wizard window: measures your real pinch and spread and saves them as settings."""

from __future__ import annotations

import time
import tkinter as tk
from typing import TYPE_CHECKING

from gesturesense.core import calibration as cal
from gesturesense.ui import widgets as ui
from gesturesense.ui.widgets import C

if TYPE_CHECKING:  # pragma: no cover
    from gesturesense.ui.app import App

STEPS = {
    "pinch": ("STEP 1 OF 2", "Pinch", "Touch your thumb and index finger together and hold still."),
    "spread": (
        "STEP 2 OF 2",
        "Spread",
        "Now open your thumb and index finger as wide as feels comfortable and hold still.",
    ),
}


class CalibrationWizard(tk.Toplevel):
    def __init__(self, app: "App") -> None:
        super().__init__(app.root)
        self.app, self.f = app, app.f
        self.title("Calibration wizard")
        self.configure(bg=C["bg"])
        self.transient(app.root)
        w, h = ui.sc(520), ui.sc(470)
        x = app.root.winfo_rootx() + max((app.root.winfo_width() - w) // 2, 0)
        y = app.root.winfo_rooty() + max((app.root.winfo_height() - h) // 3, 0)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.resizable(False, False)
        self.step = "intro"
        self.collector = cal.Collector()
        self.pinch: list[float] = []
        self.spread: list[float] = []
        self._t_last = time.time()
        self._job = None

        box = ui.card(self)
        box.pack(fill="both", expand=True, padx=ui.sc(16), pady=ui.sc(16))
        inner = tk.Frame(box, bg=C["surface"])
        inner.pack(fill="both", expand=True, padx=ui.sc(22), pady=ui.sc(20))
        self.kicker = tk.Label(inner, font=self.f.tiny, bg=C["surface"], fg=C["accent"], anchor="w")
        self.kicker.pack(fill="x")
        self.title_lbl = tk.Label(
            inner, font=self.f.title, bg=C["surface"], fg=C["text"], anchor="w"
        )
        self.title_lbl.pack(fill="x", pady=(2, 6))
        self.body = tk.Label(
            inner,
            font=self.f.body,
            bg=C["surface"],
            fg=C["muted"],
            anchor="nw",
            justify="left",
            wraplength=ui.sc(440),
        )
        self.body.pack(fill="x")
        self.live_lbl = tk.Label(
            inner,
            text="LIVE HAND OPENING",
            font=self.f.tiny,
            bg=C["surface"],
            fg=C["faint"],
            anchor="w",
        )
        self.live_lbl.pack(fill="x", pady=(ui.sc(18), 4))
        self.live = tk.Canvas(inner, height=ui.sc(14), bg=C["surface"], highlightthickness=0)
        self.live.pack(fill="x")
        self.prog_lbl = tk.Label(
            inner, text="", font=self.f.small, bg=C["surface"], fg=C["muted"], anchor="w"
        )
        self.prog_lbl.pack(fill="x", pady=(ui.sc(14), 4))
        self.prog = tk.Canvas(inner, height=ui.sc(10), bg=C["surface"], highlightthickness=0)
        self.prog.pack(fill="x")
        self.note = tk.Label(
            inner,
            font=self.f.small,
            bg=C["surface"],
            fg=C["amber"],
            anchor="nw",
            justify="left",
            wraplength=ui.sc(440),
        )
        self.note.pack(fill="x", pady=(ui.sc(10), 0))

        buttons = tk.Frame(inner, bg=C["surface"])
        buttons.pack(side="bottom", fill="x")
        self.cancel_btn = ui.Button(buttons, "Cancel", self.close, self.f, kind="ghost", height=38)
        self.cancel_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.main_btn = ui.Button(buttons, "Start", self.on_main, self.f, kind="primary", height=38)
        self.main_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.show_intro()
        self._job = self.after(30, self.tick)

    # -------------------------------------------------------------- screens
    def show_intro(self) -> None:
        self.step = "intro"
        self.kicker.config(text="CALIBRATION")
        self.title_lbl.config(text="Teach GestureSense your hand")
        running = self.app.worker is not None
        self.body.config(
            text=(
                "We will measure how tightly you pinch and how wide you spread, so 0 % "
                "and 100 % volume match your own hand. It takes about 10 seconds.\n\n"
                "Keep your hand 30 - 60 cm from the camera, palm facing it."
                if running
                else "The camera is not running. Close this window, press Start in the main "
                "window, show your hand, then open the wizard again."
            )
        )
        self.note.config(text="")
        self.prog_lbl.config(text="")
        self.main_btn.configure_button("Start" if running else "Close")

    def begin(self, step: str) -> None:
        self.step = step
        self.collector = cal.Collector()
        kicker, title, text = STEPS[step]
        self.kicker.config(text=kicker)
        self.title_lbl.config(text=title)
        self.body.config(text=text)
        self.note.config(text="")
        self.main_btn.configure_button("Skip step", "secondary")
        self._t_last = time.time()

    def show_result(self) -> None:
        self.step = "result"
        self.kicker.config(text="DONE")
        self.prog_lbl.config(text="")
        try:
            self.lo, self.hi = cal.compute_range(self.pinch, self.spread)
        except cal.CalibrationError as exc:
            self.title_lbl.config(text="Let's try that again")
            self.body.config(text=str(exc))
            self.note.config(text="")
            self.lo = self.hi = None
            self.main_btn.configure_button("Retry", "primary")
            return
        self.title_lbl.config(text="Calibration complete")
        self.body.config(
            text=f"Pinched ratio (0 % volume):  {self.lo:.2f}\n"
            f"Spread ratio (100 % volume):  {self.hi:.2f}\n\n"
            "Apply to use these values. Remember to press Save settings in the "
            "main window to keep them next time."
        )
        self.note.config(text=cal.quality(self.pinch, self.spread) or "")
        self.main_btn.configure_button("Apply", "primary")

    # -------------------------------------------------------------- actions
    def on_main(self) -> None:
        if self.step == "intro":
            if self.app.worker is None:
                self.close()
            else:
                self.pinch, self.spread = [], []
                self.begin("pinch")
        elif self.step in ("pinch", "spread"):
            self.advance(force=True)
        elif self.step == "result":
            if self.lo is None:  # retry
                self.pinch, self.spread = [], []
                self.begin("pinch")
            else:
                self.app.vars["min_ratio"].set(self.lo)
                self.app.vars["max_ratio"].set(self.hi)
                self.app.status_lbl.config(
                    text=f"Calibrated: pinched {self.lo:.2f}, spread {self.hi:.2f}"
                )
                self.close()

    def advance(self, force: bool = False) -> None:
        samples = self.collector.samples
        if self.step == "pinch":
            self.pinch = list(samples)
            self.begin("spread")
        elif self.step == "spread":
            self.spread = list(samples)
            self.show_result()

    def close(self) -> None:
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.destroy()

    # -------------------------------------------------------------- live updates
    def tick(self) -> None:
        now = time.time()
        dt, self._t_last = min(now - self._t_last, 0.2), now
        ratio = float(getattr(self.app, "_last_ratio", 0.0))
        has_hand = bool(getattr(self.app, "_last_has_hand", False)) and self.app.worker is not None
        self.draw_live(ratio if has_hand else 0.0)
        if self.step in ("pinch", "spread"):
            self.collector.add(ratio, dt, has_hand)
            if not has_hand:
                self.prog_lbl.config(text="Show your hand to the camera...", fg=C["amber"])
            elif self.collector.active < self.collector.warmup:
                self.prog_lbl.config(text="Get ready...", fg=C["muted"])
            else:
                self.prog_lbl.config(text="Hold still - measuring...", fg=C["good"])
            self.draw_progress(self.collector.progress)
            if self.collector.done:
                self.advance()
        else:
            self.draw_progress(1.0 if self.step == "result" and self.lo is not None else 0.0)
        self._job = self.after(30, self.tick)

    def draw_live(self, ratio: float) -> None:
        c = self.live
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        ui.rrect(c, 0, 0, w, h, h / 2, fill="#1B2438", outline="")
        frac = min(max(ratio / 1.6, 0.0), 1.0)
        if frac > 0.01:
            ui.rrect(c, 0, 0, max(frac * w, h), h, h / 2, fill=C["accent"], outline="")

    def draw_progress(self, frac: float) -> None:
        c = self.prog
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        ui.rrect(c, 0, 0, w, h, h / 2, fill="#1B2438", outline="")
        if frac > 0.01:
            ui.rrect(
                c,
                0,
                0,
                max(frac * w, h),
                h,
                h / 2,
                fill=C["good"] if frac >= 1.0 else C["violet"],
                outline="",
            )
