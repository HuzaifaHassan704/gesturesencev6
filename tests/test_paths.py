import importlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PathTests(unittest.TestCase):
    def test_env_override_and_creation(self):
        import paths
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "gs")
            os.environ["GESTURESENSE_HOME"] = target
            try:
                self.assertEqual(str(paths.data_dir()), target)
                self.assertTrue(os.path.isdir(target))
            finally:
                del os.environ["GESTURESENSE_HOME"]

    def test_source_default(self):
        import paths
        self.assertEqual(paths.data_dir(), paths.SOURCE_DIR)

    def test_version(self):
        import version
        self.assertRegex(version.__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
