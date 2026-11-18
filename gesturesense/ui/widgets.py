"""Small dark-theme widget kit for Tkinter (flat buttons, toggles, sliders, tabs, LED meter).

Everything is drawn on canvases, so it looks the same on every platform and needs no extra
packages. Colours live in `C` - change them there to re-skin the whole app.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional, Sequence

C = {
    "bg": "#0A0E16",  # window
    "surface": "#111826",  # cards
    "surface2": "#182033",  # inputs, hover
    "border": "#222D44",
    "text": "#E8ECF4",
    "muted": "#8B97AE",
    "faint": "#5A667E",
    "accent": "#4C9AFF",
    "accent_hi": "#6CB0FF",
    "good": "#34D399",
    "bad": "#F87171",
    "violet": "#A78BFA",
    "amber": "#FBBF24",
}

SCALE = 1.0  # display scale (1.0 = 96 dpi); set once by the app at start-up


def sc(n: float) -> int:
    """Scale a pixel size for the current display DPI."""
    return max(1, int(round(n * SCALE)))


LED_COLORS = [
    "#3CCDFF",
    "#46B9FF",
    "#5AA3FF",
    "#7690FF",
    "#9A82FF",
    "#B87AF2",
    "#D277DA",
    "#E574BE",
    "#F27399",
    "#F87271",
]  # azure -> violet -> coral (matches the 3D tower)


def pick_family(root: tk.Misc) -> str:
    have = set(tkfont.families(root))
    for name in (
        "Segoe UI Variable Text",
        "Segoe UI",
        "SF Pro Text",
        "Inter",
        "Helvetica Neue",
        "Helvetica",
        "DejaVu Sans",
    ):
        if name in have:
            return name
    return "TkDefaultFont"


def rrect(cv: tk.Canvas, x0, y0, x1, y1, r, **kw) -> int:
    """Rounded rectangle on a canvas."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = [
        x0 + r,
        y0,
        x1 - r,
        y0,
        x1,
        y0,
        x1,
        y0 + r,
        x1,
        y1 - r,
        x1,
        y1,
        x1 - r,
        y1,
        x0 + r,
        y1,
        x0,
        y1,
        x0,
        y1 - r,
        x0,
        y0 + r,
        x0,
        y0,
    ]
    return cv.create_polygon(pts, smooth=True, **kw)


class Fonts:
    def __init__(self, root: tk.Misc) -> None:
        fam = pick_family(root)
        self.family = fam
        self.title = (fam, 15, "bold")
        self.h = (fam, 10, "bold")
        self.body = (fam, 10)
        self.small = (fam, 9)
        self.tiny = (fam, 8, "bold")
        self.big = (fam, 34, "bold")
        self.mid = (fam, 12, "bold")


class Button(tk.Canvas):
    """Flat rounded button. kind = primary | secondary | danger | ghost."""

    def __init__(
        self,
        parent,
        text: str,
        command: Callable,
        fonts: Fonts,
        kind: str = "secondary",
        height: int = 36,
        width: int = 120,
        bg: str = C["surface"],
    ) -> None:
        super().__init__(
            parent,
            height=sc(height),
            width=sc(width),
            bg=bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.text, self.command, self.kind, self.fonts = text, command, kind, fonts
        self.hover = False
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<Button-1>", lambda e: self.command())

    def _set_hover(self, v: bool) -> None:
        self.hover = v
        self.draw()

    def configure_button(self, text: Optional[str] = None, kind: Optional[str] = None) -> None:
        if text is not None:
            self.text = text
        if kind is not None:
            self.kind = kind
        self.draw()

    def draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        fills = {
            "primary": (C["accent"], C["accent_hi"], "#06121F"),
            "danger": ("#3A1D24", "#4B2430", C["bad"]),
            "secondary": (C["surface2"], "#202B44", C["text"]),
            "ghost": (C["surface"], C["surface2"], C["muted"]),
        }[self.kind]
        rrect(
            self,
            1,
            1,
            w - 1,
            h - 1,
            10,
            fill=fills[1] if self.hover else fills[0],
            outline=C["border"] if self.kind in ("secondary", "ghost") else fills[0],
        )
        self.create_text(w / 2, h / 2, text=self.text, fill=fills[2], font=self.fonts.h)


class Toggle(tk.Frame):
    """Label on the left, switch on the right, bound to a BooleanVar."""

    def __init__(
        self, parent, text: str, var: tk.BooleanVar, fonts: Fonts, bg: str = C["surface"]
    ) -> None:
        super().__init__(parent, bg=bg)
        self.var = var
        lbl = tk.Label(
            self, text=text, bg=bg, fg=C["text"], font=fonts.body, anchor="w", justify="left"
        )
        lbl.pack(side="left", fill="x", expand=True)
        lbl.bind("<Configure>", lambda e: lbl.config(wraplength=max(e.width - 6, 80)))
        self.sw = tk.Canvas(
            self, width=sc(40), height=sc(22), bg=bg, highlightthickness=0, bd=0, cursor="hand2"
        )
        self.sw.pack(side="right", padx=(8, 0))
        self.sw.bind("<Button-1>", lambda e: self.var.set(not self.var.get()))
        var.trace_add("write", lambda *_: self.draw())
        self.draw()

    def draw(self) -> None:
        on = bool(self.var.get())
        self.sw.delete("all")
        k = SCALE
        rrect(
            self.sw, 1, 1, 39 * k, 21 * k, 10 * k, fill=C["accent"] if on else "#26324B", outline=""
        )
        x = (29 if on else 11) * k
        self.sw.create_oval(
            x - 8 * k, 3 * k, x + 8 * k, 19 * k, fill="#FFFFFF" if on else "#9AA6BF", outline=""
        )


class Slider(tk.Canvas):
    """Label + value on top, draggable track below. Bound to a DoubleVar."""

    def __init__(
        self,
        parent,
        text: str,
        var: tk.DoubleVar,
        lo: float,
        hi: float,
        fmt: str,
        fonts: Fonts,
        bg: str = C["surface"],
    ) -> None:
        super().__init__(parent, height=sc(46), bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.text, self.var, self.lo, self.hi, self.fmt, self.fonts = text, var, lo, hi, fmt, fonts
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Button-1>", self._drag)
        self.bind("<B1-Motion>", self._drag)
        var.trace_add("write", lambda *_: self.draw())

    def _drag(self, e) -> None:
        w = self.winfo_width()
        t = min(max((e.x - sc(8)) / max(w - sc(16), 1), 0.0), 1.0)
        self.var.set(round(self.lo + t * (self.hi - self.lo), 4))

    def draw(self) -> None:
        self.delete("all")
        w = self.winfo_width()
        try:
            v = float(self.var.get())
        except tk.TclError:
            return
        t = min(max((v - self.lo) / (self.hi - self.lo), 0.0), 1.0)
        k = SCALE
        self.create_text(
            2, 11 * k, text=self.text, anchor="w", fill=C["muted"], font=self.fonts.small
        )
        self.create_text(
            w - 2, 11 * k, text=self.fmt.format(v), anchor="e", fill=C["text"], font=self.fonts.h
        )
        x0, x1, y = 8 * k, w - 8 * k, 34 * k
        rrect(self, x0 - 4 * k, y - 3 * k, x1 + 4 * k, y + 3 * k, 3 * k, fill="#26324B", outline="")
        if t > 0:
            rrect(
                self,
                x0 - 4 * k,
                y - 3 * k,
                x0 + t * (x1 - x0),
                y + 3 * k,
                3 * k,
                fill=C["accent"],
                outline="",
            )
        kx = x0 + t * (x1 - x0)
        self.create_oval(
            kx - 8 * k,
            y - 8 * k,
            kx + 8 * k,
            y + 8 * k,
            fill="#FFFFFF",
            outline=C["accent"],
            width=2,
        )


class Segmented(tk.Canvas):
    """Pill-shaped tab / mode switch bound to a StringVar."""

    def __init__(
        self,
        parent,
        options: Sequence[tuple[str, str]],
        var: tk.StringVar,
        fonts: Fonts,
        bg: str = C["surface"],
        height: int = 34,
        on_change: Optional[Callable] = None,
        fit: bool = False,
    ) -> None:
        super().__init__(
            parent, height=sc(height), bg=bg, highlightthickness=0, bd=0, cursor="hand2"
        )
        self.options, self.var, self.fonts, self.on_change = list(options), var, fonts, on_change
        if fit:  # width = text width + padding
            font = tkfont.Font(parent, font=fonts.h)
            self.configure(width=sum(font.measure(label) + sc(34) for _, label in self.options) + 8)
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Button-1>", self._click)
        var.trace_add("write", lambda *_: self.draw())

    def _click(self, e) -> None:
        n = len(self.options)
        i = min(max(int(e.x / max(self.winfo_width(), 1) * n), 0), n - 1)
        self.var.set(self.options[i][0])
        if self.on_change:
            self.on_change(self.options[i][0])

    def draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        rrect(self, 1, 1, w - 1, h - 1, h / 2 - 1, fill=C["bg"], outline=C["border"])
        n = len(self.options)
        seg = (w - 6) / n
        for i, (key, label) in enumerate(self.options):
            active = key == self.var.get()
            if active:
                rrect(
                    self,
                    3 + i * seg,
                    3,
                    3 + (i + 1) * seg,
                    h - 3,
                    (h - 6) / 2,
                    fill=C["surface2"],
                    outline=C["border"],
                )
            self.create_text(
                3 + (i + 0.5) * seg,
                h / 2,
                text=label,
                fill=C["text"] if active else C["muted"],
                font=self.fonts.h,
            )


class LedMeter(tk.Canvas):
    """Segmented level meter, a flat echo of the 3D tower."""

    def __init__(self, parent, fonts: Fonts, segments: int = 24, bg: str = C["surface"]) -> None:
        super().__init__(parent, height=sc(14), bg=bg, highlightthickness=0, bd=0)
        self.n, self.value, self.dim = segments, 0.0, False
        self.bind("<Configure>", lambda e: self.draw())

    def set(self, value: float, dim: bool = False) -> None:
        if abs(value - self.value) > 1e-4 or dim != self.dim:
            self.value, self.dim = value, dim
            self.draw()

    def draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        gap = sc(3)
        seg_w = (w - gap * (self.n - 1)) / self.n
        for i in range(self.n):
            lit = (i + 0.5) / self.n <= self.value
            col = LED_COLORS[min(int(i / self.n * len(LED_COLORS)), len(LED_COLORS) - 1)]
            if self.dim:
                col = "#6C5A90" if lit else col
            x = i * (seg_w + gap)
            rrect(self, x, 0, x + seg_w, h, 3, fill=col if lit else "#1B2438", outline="")


def card(parent, bg: str = C["surface"], **kw) -> tk.Frame:
    return tk.Frame(parent, bg=bg, highlightbackground=C["border"], highlightthickness=1, **kw)


def chip(parent, text: str, fonts: Fonts) -> tk.Label:
    return tk.Label(
        parent, text=text, font=fonts.tiny, padx=9, pady=3, bg=C["surface2"], fg=C["faint"]
    )


class ScrollFrame(tk.Frame):
    """Vertically scrollable area (mouse wheel while hovering). Put children in `.inner`."""

    def __init__(self, parent, bg: str = C["surface"]) -> None:
        super().__init__(parent, bg=bg)
        self.bar = tk.Canvas(self, width=sc(5), bg=bg, highlightthickness=0, bd=0)
        self.bar.pack(side="right", fill="y")
        self.cv = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.cv.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.cv, bg=bg)
        self.win = self.cv.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self._update())
        self.cv.bind("<Configure>", self._on_resize)
        self.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.bind("<Leave>", lambda e: self._bind_wheel(False))

    def _on_resize(self, e) -> None:
        self.cv.itemconfigure(self.win, width=e.width)
        self._update()

    def _bind_wheel(self, on: bool) -> None:
        if on:
            self.bind_all("<MouseWheel>", self._wheel)
            self.bind_all("<Button-4>", lambda e: self._scroll(-1))
            self.bind_all("<Button-5>", lambda e: self._scroll(1))
        else:
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.unbind_all(ev)

    def _wheel(self, e) -> None:
        self._scroll(-1 if e.delta > 0 else 1)

    def _scroll(self, direction: int) -> None:
        if self.inner.winfo_reqheight() > self.cv.winfo_height():
            self.cv.yview_scroll(direction * 2, "units")
            self._update()

    def to_top(self) -> None:
        self.cv.yview_moveto(0)
        self._update()

    def _update(self) -> None:
        self.cv.configure(scrollregion=self.cv.bbox("all"))
        self.bar.delete("all")
        total, view = self.inner.winfo_reqheight(), self.cv.winfo_height()
        if total > view > 1:
            top, bot = self.cv.yview()
            h = self.bar.winfo_height()
            rrect(
                self.bar,
                0,
                top * h,
                self.bar.winfo_width(),
                max(bot * h, top * h + 12),
                self.bar.winfo_width() / 2,
                fill="#34415F",
                outline="",
            )


class Dropdown(tk.Canvas):
    """Dark dropdown. `options` = [(value, label)]; the chosen value goes to `on_select(value)`."""

    def __init__(
        self,
        parent,
        options,
        value: str,
        on_select: Callable[[str], None],
        fonts: Fonts,
        bg: str = C["surface"],
        height: int = 34,
        label_fn: Optional[Callable[[str], str]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            parent, height=sc(height), bg=bg, highlightthickness=0, bd=0, cursor="hand2"
        )
        self.options, self.value, self.on_select = list(options), value, on_select
        self.fonts, self.label_fn, self.prefix, self.hover = fonts, label_fn, prefix, False
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Button-1>", self._open)
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))

    def _set_hover(self, v: bool) -> None:
        self.hover = v
        self.draw()

    def set(self, value: str, options=None) -> None:
        self.value = value
        if options is not None:
            self.options = list(options)
        self.draw()

    def label(self) -> str:
        for val, text in self.options:
            if val == self.value:
                return text
        return self.label_fn(self.value) if self.label_fn else self.value

    def _open(self, _e=None) -> None:
        menu = tk.Menu(
            self,
            tearoff=0,
            bg=C["surface2"],
            fg=C["text"],
            activebackground=C["accent"],
            activeforeground="#06121F",
            bd=0,
            relief="flat",
            font=self.fonts.body,
        )
        for val, text in self.options:
            menu.add_command(label=("  " + text + "  "), command=lambda v=val: self.on_select(v))
        try:
            menu.tk_popup(self.winfo_rootx(), self.winfo_rooty() + self.winfo_height())
        finally:
            menu.grab_release()

    def draw(self) -> None:
        self.delete("all")
        w, h, k = self.winfo_width(), self.winfo_height(), SCALE
        rrect(
            self,
            1,
            1,
            w - 1,
            h - 1,
            9 * k,
            fill="#202B44" if self.hover else C["surface2"],
            outline=C["border"],
        )
        self.create_text(
            12 * k,
            h / 2,
            text=self.prefix + self.label(),
            anchor="w",
            fill=C["text"],
            font=self.fonts.body,
        )
        cx = w - 16 * k
        self.create_polygon(
            cx - 4 * k,
            h / 2 - 2 * k,
            cx + 4 * k,
            h / 2 - 2 * k,
            cx,
            h / 2 + 3 * k,
            fill=C["muted"],
            outline="",
        )


def ask_text(parent, fonts: Fonts, title: str, prompt: str, initial: str = "") -> Optional[str]:
    """Small dark modal dialog that asks for one line of text. Returns None when cancelled."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.configure(bg=C["bg"])
    win.transient(parent.winfo_toplevel())
    w, h = sc(420), sc(190)
    root = parent.winfo_toplevel()
    win.geometry(
        f"{w}x{h}+{root.winfo_rootx() + (root.winfo_width() - w) // 2}"
        f"+{root.winfo_rooty() + (root.winfo_height() - h) // 3}"
    )
    win.resizable(False, False)
    box = card(win)
    box.pack(fill="both", expand=True, padx=sc(14), pady=sc(14))
    inner = tk.Frame(box, bg=C["surface"])
    inner.pack(fill="both", expand=True, padx=sc(18), pady=sc(16))
    tk.Label(inner, text=prompt, font=fonts.h, bg=C["surface"], fg=C["text"], anchor="w").pack(
        fill="x"
    )
    var = tk.StringVar(value=initial)
    entry = tk.Entry(
        inner,
        textvariable=var,
        font=fonts.body,
        bg=C["surface2"],
        fg=C["text"],
        insertbackground=C["text"],
        relief="flat",
        highlightthickness=1,
        highlightbackground=C["border"],
        highlightcolor=C["accent"],
    )
    entry.pack(fill="x", ipady=sc(6), pady=(sc(10), sc(14)))
    result: dict = {"value": None}

    def ok(_e=None) -> None:
        result["value"] = var.get().strip() or None
        win.destroy()

    row = tk.Frame(inner, bg=C["surface"])
    row.pack(side="bottom", fill="x")
    Button(row, "Cancel", win.destroy, fonts, kind="ghost", height=36).pack(
        side="left", expand=True, fill="x", padx=(0, 6)
    )
    Button(row, "OK", ok, fonts, kind="primary", height=36).pack(
        side="left", expand=True, fill="x", padx=(6, 0)
    )
    entry.bind("<Return>", ok)
    win.bind("<Escape>", lambda e: win.destroy())
    entry.focus_set()
    entry.select_range(0, "end")
    win.grab_set()
    parent.wait_window(win)
    return result["value"]
