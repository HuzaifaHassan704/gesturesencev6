"""Tkinter window (dark theme): live 3D / camera viewport, volume card and settings tabs."""

from __future__ import annotations

import os
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from gesturesense import paths
from gesturesense.config import profiles
from gesturesense.config import settings as cfg
from gesturesense.core import actions
from gesturesense.core.autoswitch import AutoSwitcher
from gesturesense.core.stats import UsageStats
from gesturesense.paths import assets_dir
from gesturesense.render import THEMES, VIEW_H, VIEW_W
from gesturesense.system import capture, model
from gesturesense.system.hotkey import HOTKEY_TEXT, GlobalHotkey, Tray
from gesturesense.system.sounds import SoundPlayer
from gesturesense.ui import widgets as ui
from gesturesense.ui.stats_window import StatsWindow
from gesturesense.ui.tutorial import TutorialWindow
from gesturesense.ui.widgets import C
from gesturesense.ui.wizard import CalibrationWizard
from gesturesense.version import APP_NAME, __version__
from gesturesense.worker import CameraWorker

MODEL_PATH = cfg.APP_DIR / "hand_landmarker.task"
_BUNDLED_MODEL = assets_dir() / "hand_landmarker.task"
MODEL_URL = model.MODEL_URL

HAND_STYLES = [
    ("android", "White android hand"),
    ("gold", "Gold hand"),
    ("carbon", "Carbon hand"),
    ("hologram", "Hologram hand"),
    ("robot", "Robot hand"),
    ("human", "Human hand"),
    ("glass", "Glass hand"),
]
SKIN_TONES = [
    ("light", "Light skin"),
    ("medium", "Medium skin"),
    ("tan", "Tan skin"),
    ("deep", "Deep skin"),
]
VIEW_MODES = [("pip", "3D + Camera"), ("scene", "3D Scene"), ("camera", "Camera")]
TABS = [("gestures", "Gestures"), ("tuning", "Tuning"), ("effects", "Effects"), ("view", "3D View")]

GESTURE_TOGGLES = [
    ("enable_volume", "Volume: pinch / spread thumb + index"),
    ("enable_lock", "Lock volume: raise pinky"),
    ("enable_mute", "Fist hold action"),
    ("enable_media", "Peace sign + palm swipe actions"),
    ("enable_thumbs", "Thumbs up / down actions"),
    ("auto_profile", "Auto-switch profile by active app"),
]
ACTION_OPTIONS = list(actions.ACTIONS.items()) + [("open_app", "Open app...")]

EFFECT_TOGGLES = [
    ("fx_skeleton", "Glowing hand skeleton (camera view)"),
    ("fx_trail", "Fingertip trail"),
    ("fx_arc", "Animated volume arc"),
    ("fx_ring", "Hold-progress ring"),
    ("fx_popups", "Action popups"),
    ("sound_fx", "Sound effects"),
    ("auto_start", "Start the camera when the app opens"),
    ("minimize_to_tray", "Keep running in the tray when closed"),
]
VIEW_TOGGLES = [
    ("orbit_auto", "Auto-orbit camera"),
    ("fx3d_grid", "Floor grid"),
    ("fx3d_shadow", "Hand shadow and floor marker"),
    ("fx3d_glow", "Bloom / glow"),
    ("fx3d_motion", "Motion animations (streaks, sparks)"),
]
SLIDERS = [
    ("filter_strength", "Jitter filter (0 = off)", 0.0, 1.0, "{:.2f}"),
    ("smoothing", "Smoothing (lower = smoother)", 0.05, 1.0, "{:.2f}"),
    ("min_ratio", "Pinched ratio (0 % volume)", 0.0, 0.6, "{:.2f}"),
    ("max_ratio", "Spread ratio (100 % volume)", 0.5, 2.0, "{:.2f}"),
    ("hold_seconds", "Hold time for fist / peace (s)", 0.2, 2.0, "{:.1f}"),
    ("swipe_distance", "Swipe distance (frame width)", 0.1, 0.6, "{:.2f}"),
    ("lost_grace", "Hold pose when hand blinks out (s)", 0.0, 1.0, "{:.2f}"),
    ("settle_seconds", "Re-acquire delay (s)", 0.0, 1.5, "{:.2f}"),
    ("auto_lock_seconds", "Auto-lock after no hand (s, 0 = off)", 0.0, 30.0, "{:.0f}"),
]
VIEW_SLIDERS = [
    ("fx3d_motion_amount", "Motion animation strength", 0.0, 1.0, "{:.2f}"),
    ("orbit_speed", "Orbit speed (deg / s)", 2.0, 60.0, "{:.0f}"),
    ("orbit_range", "Orbit sweep (deg, 180 = full spin)", 10.0, 180.0, "{:.0f}"),
    ("view_pitch", "Camera height (deg)", 6.0, 75.0, "{:.0f}"),
    ("view_zoom", "Zoom", 0.5, 2.0, "{:.2f}"),
    ("pip_size", "Camera inset size (or scroll over it)", 0.15, 0.8, "{:.2f}"),
    ("render_quality", "3D render quality (lower = faster)", 0.4, 1.0, "{:.2f}"),
]


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"{APP_NAME} {__version__}")
        root.configure(bg=C["bg"])
        ui.SCALE = min(
            max(root.winfo_fpixels("1i") / 96.0, 1.0), 3.0
        )  # follow Windows display scaling
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(int(sw * 0.94), 1900), min(int(sh * 0.90), 1150)
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max((sh - h) // 2 - 10, 0)}")
        root.minsize(min(ui.sc(900), sw), min(ui.sc(620), sh))
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<F11>", lambda e: self.toggle_fullscreen())
        root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
        self.stage_size = (VIEW_W, VIEW_H)
        self._panel_visible = True
        self.f = ui.Fonts(root)
        self.worker: Optional[CameraWorker] = None
        self._last_frame = None
        self._last_ratio = 0.0
        self._last_rgb = None
        self.recorder = capture.Recorder()
        self._last_has_hand = False
        self._last_yaw = 0.0
        self._drag: Optional[tuple[int, int]] = None
        self.tutorial_open = False
        self.paused_user = False
        self._last_info: Optional[dict] = None
        self.sfx = SoundPlayer()
        self.stats = UsageStats.load()
        self.autoswitch = AutoSwitcher()
        self._stats_t = time.time()
        self._stats_saved = time.time()
        self.ui_events: "queue.Queue[str]" = queue.Queue()
        self.hotkey = GlobalHotkey(self.ui_events)
        self.tray = Tray(self.ui_events, paths.assets_dir() / "icon.png")

        self.vars: dict[str, tk.Variable] = {}
        for key, value in cfg.load().items():
            kind = (
                tk.BooleanVar
                if isinstance(value, bool)
                else tk.IntVar
                if isinstance(value, int)
                else tk.StringVar
                if isinstance(value, str)
                else tk.DoubleVar
            )
            self.vars[key] = kind(value=value)
        self.tab = tk.StringVar(value="gestures")
        self.pdata = profiles.load()  # {'active': name, 'profiles': {...}}

        self.build_ui()
        self.show_placeholder()
        self.hotkey.start()
        root.bind("<Control-Alt-g>", lambda e: self.toggle_pause())
        root.after(150, self._pump)
        root.after(1000, self._auto_tick)
        if not self.vars["tutorial_done"].get():
            root.after(900, self.open_tutorial)
        elif self.vars["auto_start"].get():
            root.after(700, self.start)  # camera starts by itself when the app opens

    # ---------------------------------------------------------------- layout
    def build_ui(self) -> None:
        f, sc = self.f, ui.sc
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ---- top bar
        bar = tk.Frame(self.root, bg=C["bg"])
        bar.grid(row=0, column=0, sticky="ew", padx=sc(20), pady=(sc(14), sc(8)))
        k = ui.SCALE
        logo = tk.Canvas(bar, width=sc(36), height=sc(36), bg=C["bg"], highlightthickness=0)
        logo.pack(side="left")
        ui.rrect(logo, 1, 1, 35 * k, 35 * k, 11 * k, fill=C["accent"], outline="")
        for r in (6, 11, 16):  # "wave" glyph
            logo.create_arc(
                (18 - r) * k,
                (18 - r) * k,
                (18 + r) * k,
                (18 + r) * k,
                start=-50,
                extent=100,
                style="arc",
                outline="#06121F",
                width=max(2, int(3 * k)),
            )
        logo.create_oval(8 * k, 15 * k, 14 * k, 21 * k, fill="#06121F", outline="")
        titles = tk.Frame(bar, bg=C["bg"])
        titles.pack(side="left", padx=12)
        tk.Label(titles, text="GestureSense", font=f.title, bg=C["bg"], fg=C["text"]).pack(
            anchor="w"
        )
        tk.Label(
            titles,
            text="Hand-gesture media control  -  F11 = fullscreen",
            font=f.small,
            bg=C["bg"],
            fg=C["muted"],
        ).pack(anchor="w")

        self.btn = ui.Button(
            bar, "Start", self.toggle, f, kind="primary", width=110, height=38, bg=C["bg"]
        )
        self.btn.pack(side="right")
        self.profile_dd = ui.Dropdown(
            bar,
            [],
            self.pdata["active"],
            self.switch_profile,
            f,
            bg=C["bg"],
            height=38,
            prefix="Profile:  ",
        )
        self.profile_dd.configure(width=sc(210))
        self.profile_dd.pack(side="right", padx=(0, 10))
        self.pause_btn = ui.Button(
            bar, "Pause", self.toggle_pause, f, kind="ghost", width=84, height=38, bg=C["bg"]
        )
        self.pause_btn.pack(side="right", padx=(0, 10))
        self.stats_btn = ui.Button(
            bar, "Stats", self.open_stats, f, kind="ghost", width=70, height=38, bg=C["bg"]
        )
        self.stats_btn.pack(side="right", padx=(0, 10))
        self.tut_btn = ui.Button(
            bar, "Tutorial", self.open_tutorial, f, kind="ghost", width=88, height=38, bg=C["bg"]
        )
        self.tut_btn.pack(side="right", padx=(0, 10))
        self.panel_btn = ui.Button(
            bar, "Hide panel", self.toggle_panel, f, kind="ghost", width=100, height=38, bg=C["bg"]
        )
        self.panel_btn.pack(side="right", padx=(0, 10))
        self.pill = tk.Canvas(bar, width=sc(190), height=sc(34), bg=C["bg"], highlightthickness=0)
        self.pill.pack(side="right", padx=14)
        self.set_status("Stopped", C["faint"])

        # ---- body: viewport | draggable divider | settings panel
        self.paned = tk.PanedWindow(
            self.root,
            orient="horizontal",
            bg=C["bg"],
            bd=0,
            sashwidth=sc(10),
            sashrelief="flat",
            opaqueresize=True,
            showhandle=False,
        )
        self.paned.grid(row=1, column=0, sticky="nsew", padx=sc(20), pady=(0, sc(18)))

        left = ui.card(self.paned)
        head = tk.Frame(left, bg=C["surface"])
        head.pack(fill="x", padx=sc(14), pady=(sc(12), sc(10)))
        seg = ui.Segmented(head, VIEW_MODES, self.vars["view_mode"], f, fit=True)
        seg.pack(side="left")
        self.fps_lbl = tk.Label(head, text="", font=f.small, bg=C["surface"], fg=C["muted"])
        self.fps_lbl.pack(side="right", padx=(10, 0))
        self.rec_btn = ui.Button(
            head, "Record", self.toggle_record, f, kind="secondary", width=104, height=32
        )
        self.rec_btn.pack(side="right", padx=(8, 0))
        ui.Button(
            head, "Screenshot", self.take_screenshot, f, kind="secondary", width=112, height=32
        ).pack(side="right")
        self.stage = tk.Frame(left, bg="#06080D", bd=0)
        self.stage.pack(fill="both", expand=True, padx=sc(14), pady=(0, sc(14)))
        self.stage.bind("<Configure>", self.on_stage_resize)
        self.video = tk.Label(self.stage, bg="#06080D", bd=0, cursor="fleur")
        self.video.place(relx=0.5, rely=0.5, anchor="center")
        self.video.bind("<ButtonPress-1>", self.on_press)
        self.video.bind("<B1-Motion>", self.on_drag)
        self.video.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        self.video.bind("<Double-Button-1>", self.on_reset_view)
        self.video.bind("<MouseWheel>", self.on_wheel)  # Windows / macOS
        self.video.bind(
            "<Button-4>", lambda e: self.on_wheel(type("E", (), {"x": e.x, "y": e.y, "delta": 1}))
        )
        self.video.bind(
            "<Button-5>", lambda e: self.on_wheel(type("E", (), {"x": e.x, "y": e.y, "delta": -1}))
        )

        self.side = tk.Frame(self.paned, bg=C["bg"])
        self.side.pack_propagate(False)
        self.build_volume_card(self.side)
        self.build_settings_card(self.side)

        self.paned.add(left, stretch="always", minsize=sc(420))
        self.paned.add(self.side, stretch="never", minsize=sc(320), width=sc(350))

    def on_stage_resize(self, e) -> None:
        self.stage_size = (max(e.width, 160), max(e.height, 120))
        if self.worker is None:
            self.show_placeholder()

    def toggle_panel(self) -> None:
        sc = ui.sc
        if self._panel_visible:
            self.paned.forget(self.side)
        else:
            self.paned.add(self.side, stretch="never", minsize=sc(320), width=sc(350))
        self._panel_visible = not self._panel_visible
        self.panel_btn.configure_button("Hide panel" if self._panel_visible else "Show panel")

    def toggle_fullscreen(self) -> None:
        self.root.attributes("-fullscreen", not bool(self.root.attributes("-fullscreen")))

    def build_volume_card(self, parent) -> None:
        f = self.f
        vc = ui.card(parent)
        vc.pack(fill="x")
        inner = tk.Frame(vc, bg=C["surface"])
        inner.pack(fill="x", padx=ui.sc(16), pady=ui.sc(14))
        tk.Label(inner, text="SYSTEM VOLUME", font=f.tiny, bg=C["surface"], fg=C["muted"]).pack(
            anchor="w"
        )
        row = tk.Frame(inner, bg=C["surface"])
        row.pack(fill="x", pady=(2, 8))
        self.vol_lbl = tk.Label(row, text="--", font=f.big, bg=C["surface"], fg=C["text"])
        self.vol_lbl.pack(side="left")
        self.vol_unit = tk.Label(row, text="%", font=f.mid, bg=C["surface"], fg=C["muted"])
        self.vol_unit.pack(side="left", anchor="s", pady=(0, 8), padx=(2, 0))
        chips = tk.Frame(row, bg=C["surface"])
        chips.pack(side="right")
        self.chip_mute = ui.chip(chips, "MUTED", f)
        self.chip_lock = ui.chip(chips, "LOCKED", f)
        self.chip_mute.pack(side="left", padx=3)
        self.chip_lock.pack(side="left")
        self.meter = ui.LedMeter(inner, f)
        self.meter.pack(fill="x")
        self.status_lbl = tk.Label(
            inner,
            text="Press Start to begin",
            font=f.body,
            bg=C["surface"],
            fg=C["muted"],
            anchor="w",
            justify="left",
        )
        self.status_lbl.pack(fill="x", pady=(10, 0))
        self.status_lbl.bind(
            "<Configure>", lambda e: self.status_lbl.config(wraplength=max(e.width - 6, 80))
        )

    def build_settings_card(self, parent) -> None:
        f = self.f
        sc = ui.card(parent)
        sc.pack(fill="both", expand=True, pady=(12, 0))
        top = tk.Frame(sc, bg=C["surface"])
        top.pack(fill="x", padx=12, pady=(12, 4))
        ui.Segmented(top, TABS, self.tab, f, on_change=lambda _k: self.show_tab(), height=32).pack(
            fill="x"
        )
        self.pages: dict[str, tk.Frame] = {}
        foot = tk.Frame(sc, bg=C["surface"])
        foot.pack(side="bottom", fill="x", padx=16, pady=14)
        self.scroll = ui.ScrollFrame(sc)
        self.scroll.pack(fill="both", expand=True, padx=(16, 10), pady=(8, 0))
        holder = self.scroll.inner
        for key, _ in TABS:
            page = tk.Frame(holder, bg=C["surface"])
            self.pages[key] = page

        g = self.pages["gestures"]
        tk.Label(
            g,
            text="GESTURE ACTIONS  (hold fist / peace / thumbs)",
            font=f.tiny,
            bg=C["surface"],
            fg=C["muted"],
        ).pack(anchor="w", pady=(2, 6))
        self.map_dd: dict[str, ui.Dropdown] = {}
        grid = tk.Frame(g, bg=C["surface"])
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(0, weight=1, minsize=ui.sc(95))
        grid.columnconfigure(1, weight=0, minsize=ui.sc(195))
        for r, (key, label) in enumerate(profiles.GESTURES):
            tk.Label(grid, text=label, font=f.body, bg=C["surface"], fg=C["text"], anchor="w").grid(
                row=r, column=0, sticky="w", pady=3
            )
            dd = ui.Dropdown(
                grid,
                ACTION_OPTIONS,
                "none",
                lambda v, k=key: self.on_map(k, v),
                f,
                label_fn=actions.describe,
            )
            dd.configure(width=ui.sc(195))
            dd.grid(row=r, column=1, sticky="e", pady=3, padx=(8, 0))
            self.map_dd[key] = dd
        pbar = tk.Frame(g, bg=C["surface"])
        pbar.pack(fill="x", pady=(10, 4))
        for text, cmd in (
            ("New profile", self.new_profile),
            ("Reset", self.reset_profile),
            ("Delete", self.delete_profile),
        ):
            ui.Button(
                pbar, text, cmd, f, height=32, kind="secondary" if text != "Delete" else "ghost"
            ).pack(side="left", expand=True, fill="x", padx=2)
        tk.Label(
            g,
            text="Pinch / spread = volume and raising the pinky = lock work in every profile.",
            font=f.small,
            bg=C["surface"],
            fg=C["faint"],
            justify="left",
            wraplength=int(320 * ui.SCALE),
        ).pack(anchor="w", pady=(4, 8))
        tk.Label(g, text="GESTURES ON / OFF", font=f.tiny, bg=C["surface"], fg=C["muted"]).pack(
            anchor="w", pady=(4, 2)
        )
        for key, text in GESTURE_TOGGLES:
            self.row(ui.Toggle(g, text, self.vars[key], f))
        tip = "Tip: good front light, hand upright and facing the camera, 30 - 60 cm away."
        tk.Label(
            g,
            text=tip,
            font=f.small,
            bg=C["surface"],
            fg=C["faint"],
            wraplength=int(300 * ui.SCALE),
            justify="left",
        ).pack(anchor="w", pady=(10, 0))
        self.refresh_profiles()

        ui.Button(
            self.pages["tuning"],
            "Calibration wizard...",
            self.open_wizard,
            f,
            kind="primary",
            height=38,
        ).pack(fill="x", pady=(0, 6))
        for key, text, lo, hi, fmt in SLIDERS:
            ui.Slider(self.pages["tuning"], text, self.vars[key], lo, hi, fmt, f).pack(
                fill="x", pady=(2, 0)
            )
        cal = tk.Frame(self.pages["tuning"], bg=C["surface"])
        cal.pack(fill="x", pady=(8, 0))
        ui.Button(cal, "Set pinched", lambda: self.calibrate("min_ratio"), f, height=34).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ui.Button(cal, "Set spread", lambda: self.calibrate("max_ratio"), f, height=34).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

        tk.Label(
            self.pages["tuning"], text="TWO HANDS", font=f.tiny, bg=C["surface"], fg=C["faint"]
        ).pack(anchor="w", pady=(14, 2))
        self.row(
            ui.Toggle(
                self.pages["tuning"],
                "Two-hand mode (2nd hand = brightness)",
                self.vars["two_hand_mode"],
                f,
            )
        )
        self.dd_row(
            self.pages["tuning"],
            "Main hand (volume)",
            [("right", "Right hand"), ("left", "Left hand")],
            "primary_hand",
        )
        self.dd_row(
            self.pages["tuning"],
            "Second hand does",
            [("brightness", "Screen brightness"), ("none", "Nothing (show only)")],
            "second_role",
        )
        tk.Label(
            self.pages["tuning"],
            text="Pinch with the second hand to change brightness. Needs the "
            "screen-brightness-control package (installed by run.bat); desktop monitors without "
            "DDC support may not respond.",
            font=f.small,
            bg=C["surface"],
            fg=C["faint"],
            wraplength=int(300 * ui.SCALE),
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        for key, text in EFFECT_TOGGLES:
            self.row(ui.Toggle(self.pages["effects"], text, self.vars[key], f))
        ui.Slider(
            self.pages["effects"], "Sound volume", self.vars["sound_volume"], 0.0, 1.0, "{:.2f}", f
        ).pack(fill="x", pady=(2, 0))
        tk.Label(
            self.pages["effects"],
            text=f"Pause / resume gestures from anywhere: {HOTKEY_TEXT}",
            font=f.small,
            bg=C["surface"],
            fg=C["faint"],
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        crow = tk.Frame(self.pages["view"], bg=C["surface"])
        crow.pack(fill="x", pady=(2, 4))
        tk.Label(crow, text="Webcam", font=f.body, bg=C["surface"], fg=C["text"]).pack(side="left")
        self.cam_dd = ui.Dropdown(
            crow,
            [(str(i), "Camera %d%s" % (i, " (default)" if i == 0 else "")) for i in range(5)],
            str(self.vars["camera_index"].get()),
            lambda v: self.vars["camera_index"].set(int(v)),
            f,
        )
        self.cam_dd.configure(width=ui.sc(170))
        self.cam_dd.pack(side="right")
        hrow = tk.Frame(self.pages["view"], bg=C["surface"])
        hrow.pack(fill="x", pady=(2, 4))
        tk.Label(hrow, text="Hand style", font=f.body, bg=C["surface"], fg=C["text"]).pack(
            side="left"
        )
        self.hand_dd = ui.Dropdown(
            hrow,
            HAND_STYLES,
            self.vars["hand_style"].get(),
            lambda v: self.vars["hand_style"].set(v),
            f,
        )
        self.hand_dd.configure(width=ui.sc(170))
        self.hand_dd.pack(side="right")
        self.vars["hand_style"].trace_add(
            "write", lambda *_: self.hand_dd.set(self.vars["hand_style"].get())
        )
        srow = tk.Frame(self.pages["view"], bg=C["surface"])
        srow.pack(fill="x", pady=(2, 4))
        tk.Label(srow, text="Skin tone", font=f.body, bg=C["surface"], fg=C["text"]).pack(
            side="left"
        )
        self.skin_dd = ui.Dropdown(
            srow,
            SKIN_TONES,
            self.vars["skin_tone"].get(),
            lambda v: self.vars["skin_tone"].set(v),
            f,
        )
        self.skin_dd.configure(width=ui.sc(170))
        self.skin_dd.pack(side="right")
        self.vars["skin_tone"].trace_add(
            "write", lambda *_: self.skin_dd.set(self.vars["skin_tone"].get())
        )
        trow = tk.Frame(self.pages["view"], bg=C["surface"])
        trow.pack(fill="x", pady=(2, 6))
        tk.Label(trow, text="Colour theme", font=f.body, bg=C["surface"], fg=C["text"]).pack(
            side="left"
        )
        self.theme_dd = ui.Dropdown(
            trow,
            [(k, v["label"]) for k, v in THEMES.items()],
            self.vars["theme"].get(),
            lambda v: self.vars["theme"].set(v),
            f,
        )
        self.theme_dd.configure(width=ui.sc(170))
        self.theme_dd.pack(side="right")
        self.vars["theme"].trace_add(
            "write", lambda *_: self.theme_dd.set(self.vars["theme"].get())
        )
        for key, text in VIEW_TOGGLES:
            self.row(ui.Toggle(self.pages["view"], text, self.vars[key], f))
        for key, text, lo, hi, fmt in VIEW_SLIDERS:
            ui.Slider(self.pages["view"], text, self.vars[key], lo, hi, fmt, f).pack(
                fill="x", pady=(2, 0)
            )
        tk.Label(
            self.pages["view"],
            text="Drag the 3D view to orbit, scroll to zoom, double-click to resume auto-orbit. "
            "Drag the divider between the view and this panel to resize, or hide the panel.",
            font=f.small,
            bg=C["surface"],
            fg=C["faint"],
            justify="left",
            wraplength=int(320 * ui.SCALE),
        ).pack(anchor="w", pady=(6, 0))

        ui.Button(foot, "Save settings", self.save, f, kind="secondary", height=36).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ui.Button(foot, "Reset defaults", self.reset, f, kind="ghost", height=36).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )
        self.show_tab()

    @staticmethod
    def row(widget: tk.Widget) -> None:
        widget.pack(fill="x", pady=7)

    def show_tab(self) -> None:
        for page in self.pages.values():
            page.pack_forget()
        self.pages[self.tab.get()].pack(fill="both", expand=True)
        self.scroll.to_top()

    def set_status(self, text: str, color: str) -> None:
        if getattr(self, "_pill_state", None) == (text, color):
            return  # nothing changed, skip the redraw
        self._pill_state = (text, color)
        k = ui.SCALE
        self.pill.delete("all")
        ui.rrect(self.pill, 1, 1, 229 * k, 33 * k, 16 * k, fill=C["surface"], outline=C["border"])
        self.pill.create_oval(14 * k, 12 * k, 24 * k, 22 * k, fill=color, outline="")
        self.pill.create_text(
            34 * k, 17 * k, text=text, anchor="w", fill=C["text"], font=self.f.body
        )

    # --------------------------------------------------------------- helpers
    def settings(self) -> dict:
        return {k: v.get() for k, v in self.vars.items()}

    def save(self) -> None:
        cfg.save(self.settings())
        self.status_lbl.config(text="Settings saved")

    def reset(self) -> None:
        for key, value in cfg.DEFAULTS.items():
            self.vars[key].set(value)

    # ------------------------------------------------------------ capture
    def take_screenshot(self) -> None:
        if self._last_rgb is None:
            messagebox.showinfo(
                "Screenshot", "Click Start first - there is nothing to capture yet."
            )
            return
        try:
            path = capture.save_screenshot(self._last_rgb, self.stage_size)
        except OSError as exc:
            messagebox.showerror("Screenshot", f"Could not save the screenshot:\n{exc}")
            return
        self.status_lbl.config(text=f"Screenshot saved: captures/{path.name}")

    def toggle_record(self) -> None:
        if self.recorder.active:
            self.stop_recording()
            return
        if self._last_rgb is None:
            messagebox.showinfo("Record", "Click Start first - there is nothing to record yet.")
            return
        h, w = self._last_rgb.shape[:2]
        if self.recorder.start((w, h)) is None:
            messagebox.showerror("Record", "Could not start the video encoder on this computer.")
            return
        self.rec_btn.configure_button("Stop  00:00", "danger")

    def stop_recording(self) -> None:
        path, seconds = self.recorder.stop()
        self.rec_btn.configure_button("Record", "secondary")
        if path is not None:
            self.status_lbl.config(text=f"Recording saved ({seconds:.0f} s): captures/{path.name}")

    # ------------------------------------------------------------ profiles
    @property
    def mapping(self) -> dict:
        return self.pdata["profiles"][self.pdata["active"]]

    def refresh_profiles(self) -> None:
        names = list(self.pdata["profiles"])
        self.profile_dd.set(self.pdata["active"], [(n, n) for n in names])
        for key, dd in self.map_dd.items():
            dd.set(self.mapping.get(key, "none"))

    def save_profiles(self) -> None:
        try:
            profiles.save(self.pdata)
        except OSError:
            pass

    def switch_profile(self, name: str) -> None:
        if name in self.pdata["profiles"]:
            self.pdata["active"] = name
            self.save_profiles()
            self.refresh_profiles()
            self.status_lbl.config(text=f"Profile: {name}")

    def on_map(self, gesture: str, value: str) -> None:
        if value == "open_app":
            path = filedialog.askopenfilename(title="Choose an app or file to open")
            if not path:
                self.refresh_profiles()
                return
            value = actions.OPEN_PREFIX + path
        self.mapping[gesture] = value
        self.save_profiles()
        self.refresh_profiles()

    def new_profile(self) -> None:
        name = ui.ask_text(self.root, self.f, "New profile", "Name for the new profile")
        if not name:
            return
        name = profiles.unique_name(name, self.pdata["profiles"])
        self.pdata["profiles"][name] = dict(self.mapping)  # starts as a copy of the current one
        self.switch_profile(name)

    def reset_profile(self) -> None:
        name = self.pdata["active"]
        if messagebox.askyesno("Reset profile", f"Reset '{name}' to its default gestures?"):
            self.pdata["profiles"][name] = profiles.clean(profiles.reset_mapping(name))
            self.save_profiles()
            self.refresh_profiles()

    def delete_profile(self) -> None:
        name = self.pdata["active"]
        if name in profiles.BUILTIN:
            messagebox.showinfo(
                "Delete profile", "Built-in profiles can't be deleted. Use Reset instead."
            )
            return
        if messagebox.askyesno("Delete profile", f"Delete the profile '{name}'?"):
            del self.pdata["profiles"][name]
            self.switch_profile(profiles.DEFAULT_PROFILE)

    def open_wizard(self) -> None:
        if getattr(self, "_wizard", None) is not None and self._wizard.winfo_exists():
            self._wizard.lift()
            return
        self._wizard = CalibrationWizard(self)

    def calibrate(self, key: str) -> None:
        if self.worker and self._last_ratio > 0:
            self.vars[key].set(round(self._last_ratio, 3))
            self.status_lbl.config(text=f"{key} set to {self._last_ratio:.2f}")
        else:
            messagebox.showinfo("Calibrate", "Click Start and show your hand first.")

    def show_placeholder(self) -> None:
        w, h = self.stage_size
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (13, 10, 8)
        step = max(24, int(36 * ui.SCALE))
        for x in range(0, w, step):  # faint grid
            cv2.line(img, (x, 0), (x, h), (24, 19, 15), 1)
        for y in range(0, h, step):
            cv2.line(img, (0, y), (w, y), (24, 19, 15), 1)
        k = max(h / 540, 0.6)
        msg, sub = "Press Start", "Your hand appears here in 3D"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * k, 2)
        cv2.putText(
            img,
            msg,
            ((w - tw) // 2, int(h / 2 - 6 * k)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1 * k,
            (235, 230, 226),
            2,
            cv2.LINE_AA,
        )
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, 1)
        cv2.putText(
            img,
            sub,
            ((w - tw) // 2, int(h / 2 + 28 * k)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * k,
            (174, 151, 139),
            1,
            cv2.LINE_AA,
        )
        self.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def set_image(self, rgb) -> None:
        img = Image.fromarray(rgb)
        if img.size != tuple(self.stage_size):  # scale to fill the viewport
            img = img.resize(self.stage_size, Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self.video.configure(image=photo, width=photo.width(), height=photo.height())
        self.video.image = photo  # keep a reference

    def ensure_model(self) -> bool:
        if model.is_valid(MODEL_PATH):
            return True
        self.status_lbl.config(text="Preparing hand model (one time)...")
        self.root.update_idletasks()
        try:
            model.ensure_model(MODEL_PATH, MODEL_URL, _BUNDLED_MODEL)
            return True
        except model.ModelError as exc:
            messagebox.showerror(
                "Model missing",
                f"Could not download the hand model:\n{exc}\n\nDownload it manually from:\n"
                f"{MODEL_URL}\nand put it next to main.py.",
            )
            self.status_lbl.config(text="Stopped")
            return False

    # ------------------------------------------------------ 3D view mouse control
    def _is_3d(self) -> bool:
        return self.vars["view_mode"].get() != "camera"

    def on_press(self, e) -> None:
        if not self._is_3d():
            return
        self._drag = (e.x, e.y)
        if self.vars["orbit_auto"].get():  # take over from the current angle
            self.vars["view_yaw"].set(round(self._last_yaw, 2))
            self.vars["orbit_auto"].set(False)

    def on_drag(self, e) -> None:
        if not self._drag or not self._is_3d():
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self.vars["view_yaw"].set(self.vars["view_yaw"].get() - dx * 0.4)
        pitch = self.vars["view_pitch"].get() + dy * 0.3
        self.vars["view_pitch"].set(min(max(pitch, 6.0), 75.0))

    def _over_inset(self, x: int, y: int) -> bool:
        """True when the pointer is over the camera inset (bottom-left of the 3D view)."""
        if self.vars["view_mode"].get() != "pip":
            return False
        w, h = self.stage_size
        iw = min(self.vars["pip_size"].get() * w, h * 0.88 / 0.75)
        ih = iw * 0.75
        return x < iw + 30 and y > h - ih - 30

    def on_wheel(self, e) -> None:
        if self._over_inset(e.x, e.y):
            self.resize_inset(1.08 if e.delta > 0 else 1 / 1.08)
        else:
            self.zoom_by(1.08 if e.delta > 0 else 1 / 1.08)

    def resize_inset(self, k: float) -> None:
        self.vars["pip_size"].set(round(min(max(self.vars["pip_size"].get() * k, 0.15), 0.8), 3))

    def zoom_by(self, k: float) -> None:
        if self._is_3d():
            self.vars["view_zoom"].set(
                round(min(max(self.vars["view_zoom"].get() * k, 0.5), 2.0), 3)
            )

    def on_reset_view(self, _e) -> None:
        self.vars["orbit_auto"].set(True)
        self.vars["view_pitch"].set(cfg.DEFAULTS["view_pitch"])
        self.vars["view_zoom"].set(cfg.DEFAULTS["view_zoom"])

    # --------------------------------------------------------- start / stop
    def toggle(self) -> None:
        self.stop() if self.worker else self.start()

    def start(self) -> None:
        if not self.ensure_model():
            return
        index = int(self.vars["camera_index"].get())
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not cap.isOpened():
            messagebox.showerror(
                "Camera",
                "Could not open the webcam.\n"
                "Close other apps using it (Zoom, Teams) or change camera_index.",
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame -> lower latency
        self._last_frame = None
        self.worker = CameraWorker(cap, MODEL_PATH, self.settings())
        self.worker.start()
        self.btn.configure_button("Stop", "danger")
        self.set_status("Starting...", C["amber"])
        self.status_lbl.config(text="Starting camera...")
        self.poll()

    def stop(self) -> None:
        saved = ""
        if self.recorder.active:
            path, seconds = self.recorder.stop()
            self.rec_btn.configure_button("Record", "secondary")
            saved = f"Recording saved ({seconds:.0f} s): captures/{path.name}" if path else ""
        worker, self.worker = self.worker, None
        if worker:
            worker.stop()
            worker.join(timeout=3)
        self.btn.configure_button("Start", "primary")
        self.set_status("Stopped", C["faint"])
        self.status_lbl.config(text=saved or "Press Start to begin")
        self.fps_lbl.config(text="")
        self.vol_lbl.config(text="--")
        self.meter.set(0.0)
        for chip in (self.chip_mute, self.chip_lock):
            chip.config(bg=C["surface2"], fg=C["faint"])
        self.show_placeholder()

    def poll(self) -> None:
        worker = self.worker
        if worker is None:
            return
        try:
            cur = self.settings()  # push slider/checkbox changes to the thread
            cur["view_w"], cur["view_h"] = self.stage_size
            cur["mapping"] = dict(self.mapping)
            cur["paused"] = self.paused_user or self.tutorial_open
            worker.settings = cur
        except tk.TclError:
            pass  # a field was momentarily empty
        if worker.error:
            message = worker.error
            self.stop()
            messagebox.showerror("GestureSense", message)
            return
        snap = worker.snapshot
        if snap is not None and snap[0] is not self._last_frame:
            frame, info = snap
            self._last_frame = frame
            self._last_ratio = info["ratio"]
            self._last_has_hand = info.get("has_hand", False)
            self._last_yaw = info.get("yaw_deg", 0.0)
            self._last_info = info
            self.track_usage(worker, info)
            self._last_rgb = frame
            self.set_image(frame)
            if self.recorder.active:
                self.recorder.write(frame)
                self.rec_btn.configure_button(
                    f"Stop  {int(self.recorder.seconds) // 60:02d}:"
                    f"{int(self.recorder.seconds) % 60:02d}"
                )
            self.update_panel(info)
        self.root.after(12, self.poll)

    def update_panel(self, info: dict) -> None:
        pct = info["volume"] * 100
        self.vol_lbl.config(text=f"{pct:.0f}", fg=C["violet"] if info["muted"] else C["text"])
        self.meter.set(info["volume"], dim=info["muted"] or info["locked"])
        self.fps_lbl.config(text=f"{info['fps']:.0f} fps")
        self.chip_mute.config(
            bg="#2A2244" if info["muted"] else C["surface2"],
            fg=C["violet"] if info["muted"] else C["faint"],
        )
        self.chip_lock.config(
            bg="#3A1D24" if info["locked"] else C["surface2"],
            fg=C["bad"] if info["locked"] else C["faint"],
        )
        self.status_lbl.config(text=info["status"])
        if info.get("paused"):
            self.set_status("Paused", C["amber"])
        elif info["locked"]:
            self.set_status("Locked", C["bad"])
        elif info["muted"]:
            self.set_status("Muted", C["violet"])
        elif info["has_hand"]:
            self.set_status("Tracking", C["good"])
        else:
            self.set_status("Waiting for hand", C["amber"])

    def dd_row(self, parent, label: str, options, key: str) -> None:
        """A label with a dropdown on the right that edits the string setting `key`."""
        row = tk.Frame(parent, bg=C["surface"])
        row.pack(fill="x", pady=(2, 4))
        tk.Label(row, text=label, font=self.f.body, bg=C["surface"], fg=C["text"]).pack(side="left")
        dd = ui.Dropdown(
            row, options, self.vars[key].get(), lambda v: self.vars[key].set(v), self.f
        )
        dd.configure(width=ui.sc(170))
        dd.pack(side="right")
        self.vars[key].trace_add("write", lambda *_: dd.set(self.vars[key].get()))

    # ------------------------------------------------------------ pause, tray, sound, tutorial
    def sfx_play(self, kind: str) -> None:
        if self.vars["sound_fx"].get():
            self.sfx.play(kind, float(self.vars["sound_volume"].get()), time.time())

    def toggle_pause(self) -> None:
        self.paused_user = not self.paused_user
        self.pause_btn.configure_button(
            "Resume" if self.paused_user else "Pause", "primary" if self.paused_user else "ghost"
        )
        self.sfx_play("pause" if self.paused_user else "resume")
        if self.worker is None:
            self.set_status(
                "Paused" if self.paused_user else "Stopped",
                C["amber"] if self.paused_user else C["faint"],
            )
        self.status_lbl.config(
            text=("Paused - " + HOTKEY_TEXT + " to resume")
            if self.paused_user
            else "Gestures are on again"
        )

    def save_quiet(self) -> None:
        try:
            cfg.save(self.settings())
        except OSError:
            pass

    def track_usage(self, worker, info: dict) -> None:
        """Count gestures from the worker's event log and the time a hand is in view."""
        now = time.time()
        dt = min(now - self._stats_t, 0.25)
        self._stats_t = now
        while worker.event_log:
            try:
                kind = worker.event_log.pop(0)
            except IndexError:
                break
            self.stats.record(kind, now)
        if info.get("has_hand") and not info.get("paused"):
            self.stats.add_active(dt, now)
        if self.stats.dirty and now - self._stats_saved > 30:
            self.stats.save()
            self._stats_saved = now

    def _auto_tick(self) -> None:
        """Once a second: if enabled, follow the active app with the matching profile."""
        try:
            if self.vars["auto_profile"].get():
                name = self.autoswitch.poll(self.pdata["active"], list(self.pdata["profiles"]))
                if name:
                    self.switch_profile(name)
                    self.status_lbl.config(text=f"Auto profile: {name}")
                    self.sfx_play("action")
            else:
                self.autoswitch.forget()
            self.root.after(1000, self._auto_tick)
        except tk.TclError:
            pass

    def open_stats(self) -> None:
        if getattr(self, "_stats_win", None) is not None and self._stats_win.winfo_exists():
            self._stats_win.lift()
            self._stats_win.refresh()
            return
        self._stats_win = StatsWindow(self)

    def open_tutorial(self) -> None:
        if getattr(self, "_tutorial", None) is not None and self._tutorial.winfo_exists():
            self._tutorial.lift()
            return
        self._tutorial = TutorialWindow(self)

    def show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _pump(self) -> None:
        """Handle messages from the hotkey and tray threads on the GUI thread."""
        try:
            while True:
                msg = self.ui_events.get_nowait()
                if msg == "pause":
                    self.toggle_pause()
                elif msg == "show":
                    self.show_window()
                elif msg == "quit":
                    self.quit_app()
                    return
        except queue.Empty:
            pass
        try:
            self.root.after(150, self._pump)
        except tk.TclError:
            pass

    def on_close(self) -> None:
        if self.vars["minimize_to_tray"].get() and self.tray.start():
            self.root.withdraw()  # keep running; use the tray icon to come back
            return
        self.quit_app()

    def quit_app(self) -> None:
        self.stats.save()
        self.hotkey.stop()
        self.tray.stop()
        self.stop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass
