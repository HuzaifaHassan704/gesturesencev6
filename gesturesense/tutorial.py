"""Guided first-run tutorial window. While it is open the app is paused, so practising is safe."""
from __future__ import annotations

import time
import tkinter as tk
from typing import TYPE_CHECKING, Optional

import ui
from tutorial_logic import STEPS, StepTracker, new_tracker
from ui import C

if TYPE_CHECKING:                       # pragma: no cover
    from gui import App


class TutorialWindow(tk.Toplevel):
    def __init__(self, app: "App") -> None:
        super().__init__(app.root)
        self.app, self.f = app, app.f
        self.title("GestureSense tutorial")
        self.configure(bg=C["bg"])
        self.transient(app.root)
        w, h = ui.sc(540), ui.sc(460)
        x = app.root.winfo_rootx() + max((app.root.winfo_width() - w) // 2, 0)
        y = app.root.winfo_rooty() + max((app.root.winfo_height() - h) // 3, 0)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.resizable(False, False)
        self.index = -1                                  # -1 = intro
        self.tracker: Optional[StepTracker] = None
        self._advance_at: Optional[float] = None
        self._job = None
        app.tutorial_open = True

        box = ui.card(self)
        box.pack(fill="both", expand=True, padx=ui.sc(16), pady=ui.sc(16))
        inner = tk.Frame(box, bg=C["surface"])
        inner.pack(fill="both", expand=True, padx=ui.sc(22), pady=ui.sc(20))
        self.kicker = tk.Label(inner, font=self.f.tiny, bg=C["surface"], fg=C["accent"], anchor="w")
        self.kicker.pack(fill="x")
        self.title_lbl = tk.Label(inner, font=self.f.title, bg=C["surface"], fg=C["text"], anchor="w")
        self.title_lbl.pack(fill="x", pady=(2, 6))
        self.body = tk.Label(inner, font=self.f.body, bg=C["surface"], fg=C["muted"], anchor="nw",
                             justify="left", wraplength=ui.sc(430))
        self.body.pack(fill="x")
        self.live_lbl = tk.Label(inner, text="", font=self.f.small, bg=C["surface"], fg=C["muted"],
                                 anchor="w")
        self.live_lbl.pack(fill="x", pady=(ui.sc(16), 4))
        self.prog = tk.Canvas(inner, height=ui.sc(12), bg=C["surface"], highlightthickness=0)
        self.prog.pack(fill="x")
        self._frac = 0.0
        self.prog.bind("<Configure>", lambda e: self.draw_bar(self._frac))
        self.dots = tk.Canvas(inner, height=ui.sc(12), bg=C["surface"], highlightthickness=0)
        self.dots.pack(fill="x", pady=(ui.sc(14), 0))

        buttons = tk.Frame(inner, bg=C["surface"])
        buttons.pack(side="bottom", fill="x")
        self.skip_all = ui.Button(buttons, "Close", self.close, self.f, kind="ghost", height=38)
        self.skip_all.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.main_btn = ui.Button(buttons, "Start", self.on_main, self.f, kind="primary", height=38)
        self.main_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.show_intro()
        self._job = self.after(30, self.tick)

    # ------------------------------------------------------------ screens
    def show_intro(self) -> None:
        self.index = -1
        self.kicker.config(text="WELCOME")
        self.title_lbl.config(text="A 1-minute tour")
        self.body.config(text="You will try each gesture once. While this window is open GestureSense is "
                              "paused, so nothing you do here changes your volume or presses keys.")
        self.live_lbl.config(text="")
        self.main_btn.configure_button("Start")
        self.draw_bar(0)
        self.draw_dots()

    def show_step(self, i: int) -> None:
        self.index = i
        step = STEPS[i]
        self.tracker = new_tracker(i)
        self._advance_at = None
        self.kicker.config(text=f"STEP {i + 1} OF {len(STEPS)}")
        self.title_lbl.config(text=step.title)
        self.body.config(text=step.text)
        self.live_lbl.config(text="Waiting for your hand...")
        self.main_btn.configure_button("Skip step", "secondary")
        self.draw_bar(0)
        self.draw_dots()

    def show_done(self) -> None:
        self.index = len(STEPS)
        self.tracker = None
        self.kicker.config(text="ALL SET")
        self.title_lbl.config(text="You're ready")
        from hotkey import HOTKEY_TEXT
        self.body.config(text="Tips:\n"
                              f"  -  {HOTKEY_TEXT} pauses and resumes gestures from anywhere.\n"
                              "  -  Raise your pinky to lock the volume.\n"
                              "  -  Change what each gesture does in the Gestures tab, "
                              "and keep several setups as profiles.\n"
                              "  -  Run Calibrate once so 0 % and 100 % match your own hand.")
        self.live_lbl.config(text="")
        self.main_btn.configure_button("Finish", "primary")
        self.draw_bar(1.0)
        self.draw_dots()
        self.app.sfx_play("success")

    # ------------------------------------------------------------ drawing
    def draw_bar(self, frac: float) -> None:
        self._frac = frac
        c = self.prog
        c.delete("all")
        w, h = max(c.winfo_width(), 10), ui.sc(12)
        ui.rrect(c, 0, 0, w, h, h / 2, fill=C["surface2"], outline="")
        if frac > 0.01:
            ui.rrect(c, 0, 0, max(w * frac, h), h, h / 2, fill=C["good"] if frac >= 1 else C["accent"],
                     outline="")

    def draw_dots(self) -> None:
        c = self.dots
        c.delete("all")
        n = len(STEPS)
        r, gap = ui.sc(4), ui.sc(18)
        x = ui.sc(4)
        for i in range(n):
            col = C["good"] if i < self.index else C["accent"] if i == self.index else C["surface2"]
            c.create_oval(x, ui.sc(2), x + 2 * r, ui.sc(2) + 2 * r, fill=col, outline="")
            x += gap

    # ------------------------------------------------------------ flow
    def on_main(self) -> None:
        if self.index == -1:
            if self.app.worker is None:
                self.app.start()
                if self.app.worker is None:              # the camera could not start
                    return
            self.show_step(0)
        elif self.index >= len(STEPS):
            self.finish()
        else:
            self.next_step()

    def next_step(self) -> None:
        if self.index + 1 >= len(STEPS):
            self.show_done()
        else:
            self.show_step(self.index + 1)

    def tick(self) -> None:
        now = time.time()
        if self.tracker is not None and self.app.worker is not None:
            info = self.app._last_info or {}
            if self._advance_at is None:
                p = self.tracker.feed(info, now)
                self.draw_bar(p)
                if not info.get("has_hand") and p == 0:
                    self.live_lbl.config(text="I can't see your hand yet...")
                elif p < 1.0:
                    self.live_lbl.config(text="Good - keep going")
                if self.tracker.done:
                    self.live_lbl.config(text="Got it!")
                    self.app.sfx_play("success")
                    self._advance_at = now + 0.7
            elif now >= self._advance_at:
                self.next_step()
        self._job = self.after(30, self.tick)

    def finish(self) -> None:
        self.app.vars["tutorial_done"].set(True)
        self.app.save_quiet()
        self.close()

    def close(self) -> None:
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.app.tutorial_open = False
        self.destroy()
