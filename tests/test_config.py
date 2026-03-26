import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from logistics_approval_agent.config import DEFAULT_YOLO_WEIGHTS_FILENAME, ensure_yolo_weights, get_settings


class ConfigTests(unittest.TestCase):
    def test_get_settings_uses_models_directory_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["PROJECT_ROOT"] = temp_dir
            settings = get_settings()
            expected_path = Path(temp_dir) / "models" / DEFAULT_YOLO_WEIGHTS_FILENAME
            self.assertEqual(settings.project_root, Path(temp_dir))
            self.assertEqual(settings.yolo_weights_path, expected_path)

    def test_ensure_yolo_weights_skips_download_when_file_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = Path(temp_dir) / DEFAULT_YOLO_WEIGHTS_FILENAME
            existing_path.write_bytes(b"existing")

            calls = []

            def downloader(url, destination):
                calls.append((url, destination))

            resolved_path = ensure_yolo_weights(
                weights_path=existing_path,
                download_url="https://example.com/model.pt",
                downloader=downloader,
            )

            self.assertEqual(resolved_path, existing_path)
            self.assertEqual(calls, [])

    def test_ensure_yolo_weights_downloads_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "nested" / DEFAULT_YOLO_WEIGHTS_FILENAME

            def downloader(url, destination):
                self.assertEqual(url, "https://example.com/model.pt")
                destination.write_bytes(b"downloaded")

            resolved_path = ensure_yolo_weights(
                weights_path=weights_path,
                download_url="https://example.com/model.pt",
                downloader=downloader,
            )

            self.assertEqual(resolved_path, weights_path)
            self.assertTrue(weights_path.exists())
            self.assertEqual(weights_path.read_bytes(), b"downloaded")


if __name__ == "__main__":
    unittest.main()
