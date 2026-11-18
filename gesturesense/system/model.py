"""Getting the MediaPipe hand model onto disk without ever leaving a broken file behind.

The model is first written to ``<name>.part`` and only renamed to its final name once the
whole download has arrived and passed a size check.  If the app is closed half way, the
final path never exists, so the next start simply downloads again.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Callable, Optional

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MIN_MODEL_BYTES = 1_000_000  # the real file is ~7.5 MB; anything smaller is a broken download
CHUNK = 64 * 1024


class ModelError(Exception):
    """The model could not be downloaded or failed verification."""


def part_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".part")


def is_valid(path: Path, min_bytes: int = MIN_MODEL_BYTES) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def download(
    url: str,
    dest: Path,
    *,
    opener: Optional[Callable] = None,
    min_bytes: int = MIN_MODEL_BYTES,
    sha256: Optional[str] = None,
    timeout: float = 30.0,
) -> None:
    """Download ``url`` to ``dest`` atomically (``dest`` only appears when complete and valid)."""
    opener = opener or (lambda u: urllib.request.urlopen(u, timeout=timeout))
    tmp = part_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        digest = hashlib.sha256()
        total = 0
        with opener(url) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                total += len(chunk)
        if total < min_bytes:
            raise ModelError(f"download too small ({total} bytes), expected at least {min_bytes}")
        if sha256 and digest.hexdigest().lower() != sha256.lower():
            raise ModelError("download is corrupt (SHA-256 mismatch)")
        tmp.replace(dest)  # atomic on the same drive
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def ensure_model(
    path: Path,
    url: str = MODEL_URL,
    bundled: Optional[Path] = None,
    *,
    opener: Optional[Callable] = None,
    min_bytes: int = MIN_MODEL_BYTES,
) -> Path:
    """Make sure a valid model exists at ``path``; copy the bundled one or download it."""
    if is_valid(path, min_bytes):
        return path
    path.unlink(missing_ok=True)  # a broken leftover from an older version
    part_path(path).unlink(missing_ok=True)
    if bundled is not None and is_valid(bundled, min_bytes):
        tmp = part_path(path)
        try:
            shutil.copyfile(bundled, tmp)
            tmp.replace(path)
            return path
        except OSError:
            tmp.unlink(missing_ok=True)
    try:
        download(url, path, opener=opener, min_bytes=min_bytes)
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(str(exc)) from exc
    return path
