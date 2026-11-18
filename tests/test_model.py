"""The model download must never leave a broken file at the final path."""

import io
import tempfile
import unittest
from pathlib import Path

from gesturesense.system import model


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _Dies(_Resp):
    """Delivers some bytes then fails, like a dropped connection."""

    def read(self, n=-1):
        if self.tell() >= 10:
            raise ConnectionError("link lost")
        return super().read(min(n, 10))


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "hand_landmarker.task"

    def test_good_download_is_renamed_into_place(self):
        model.ensure_model(self.path, "u", opener=lambda u: _Resp(b"x" * 2000), min_bytes=1000)
        self.assertEqual(self.path.stat().st_size, 2000)
        self.assertFalse(model.part_path(self.path).exists())

    def test_interrupted_download_leaves_nothing(self):
        with self.assertRaises(model.ModelError):
            model.ensure_model(self.path, "u", opener=lambda u: _Dies(b"x" * 5000), min_bytes=1000)
        self.assertFalse(self.path.exists())
        self.assertFalse(model.part_path(self.path).exists())

    def test_truncated_download_is_rejected(self):
        with self.assertRaises(model.ModelError):
            model.ensure_model(self.path, "u", opener=lambda u: _Resp(b"x" * 10), min_bytes=1000)
        self.assertFalse(self.path.exists())

    def test_broken_leftover_is_replaced(self):
        self.path.write_bytes(b"half")
        model.ensure_model(self.path, "u", opener=lambda u: _Resp(b"y" * 2000), min_bytes=1000)
        self.assertEqual(self.path.read_bytes()[:1], b"y")

    def test_valid_existing_file_is_not_downloaded_again(self):
        self.path.write_bytes(b"z" * 2000)

        def boom(u):
            raise AssertionError("should not download")

        model.ensure_model(self.path, "u", opener=boom, min_bytes=1000)

    def test_bundled_copy_is_used_first(self):
        bundled = Path(self.tmp.name) / "bundled.task"
        bundled.write_bytes(b"b" * 2000)

        def boom(u):
            raise AssertionError("should not download")

        model.ensure_model(self.path, "u", bundled, opener=boom, min_bytes=1000)
        self.assertEqual(self.path.read_bytes()[:1], b"b")

    def test_sha256_mismatch_is_rejected(self):
        with self.assertRaises(model.ModelError):
            model.download(
                "u", self.path, opener=lambda u: _Resp(b"x" * 2000), min_bytes=1000, sha256="0" * 64
            )
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
