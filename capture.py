"""Screenshots (PNG) and screen recording (MP4) of the view. No GUI imports.

Files go into a `captures` folder next to the program. PNGs are written with Pillow and MP4s
with OpenCV's built-in `mp4v` encoder, so nothing extra needs installing.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from paths import data_dir

APP_DIR = data_dir()
CAPTURE_DIR = APP_DIR / "captures"


def stamp(prefix: str, ext: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"


def unique_path(folder: Path, name: str) -> Path:
    """`folder/name`, or `folder/name_2` etc. when that file already exists."""
    path = folder / name
    n = 2
    while path.exists():
        path = folder / f"{path.stem.rsplit('_dup', 1)[0]}_dup{n}{path.suffix}"
        n += 1
    return path


def save_screenshot(frame_rgb: np.ndarray, size: Optional[tuple[int, int]] = None,
                    folder: Optional[Path] = None) -> Path:
    """Save an RGB frame as PNG (optionally scaled to `size` = (w, h)) and return its path."""
    folder = folder or CAPTURE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame_rgb)
    if size and tuple(size) != img.size:
        img = img.resize(tuple(size), Image.LANCZOS)
    path = unique_path(folder, stamp("gesturesense", "png"))
    img.save(path)
    return path


class Recorder:
    """Records frames to an MP4 at a steady frame rate, whatever rate the frames arrive at."""

    def __init__(self, fps: int = 25) -> None:
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.path: Optional[Path] = None
        self.size = (0, 0)
        self.t0: Optional[float] = None
        self.count = 0

    @property
    def active(self) -> bool:
        return self.writer is not None

    @property
    def seconds(self) -> float:
        return self.count / self.fps

    def start(self, size: tuple[int, int], folder: Optional[Path] = None) -> Optional[Path]:
        """Begin recording frames of `size` = (w, h). Returns the file path, or None on failure."""
        if self.active:
            return self.path
        folder = folder or CAPTURE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        w, h = int(size[0]) // 2 * 2, int(size[1]) // 2 * 2         # the encoder needs even sizes
        path = unique_path(folder, stamp("gesturesense", "mp4"))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not writer.isOpened():
            return None
        self.writer, self.path, self.size, self.t0, self.count = writer, path, (w, h), None, 0
        return path

    def write(self, frame_rgb: np.ndarray, now: Optional[float] = None) -> None:
        if self.writer is None:
            return
        now = time.time() if now is None else now
        if self.t0 is None:
            self.t0 = now
        due = int((now - self.t0) * self.fps) + 1 - self.count      # frames owed to keep real time
        if due <= 0:
            return
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if (frame.shape[1], frame.shape[0]) != self.size:           # the window was resized
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        for _ in range(min(due, self.fps)):                         # repeat the frame to fill gaps
            self.writer.write(frame)
            self.count += 1

    def stop(self) -> tuple[Optional[Path], float]:
        """Finish the file. Returns (path, duration in seconds)."""
        path, seconds = self.path, self.seconds
        if self.writer is not None:
            self.writer.release()
        self.writer, self.path, self.t0, self.count = None, None, None, 0
        return path, seconds
