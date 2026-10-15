"""Architecture experiments must not mutate or consume incomplete exports."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_room3d_architecture as pilot
from room3d_checkpoint import save


class ArchitecturePilotTests(unittest.TestCase):
    def test_modified_export_is_rejected_before_model_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "batch"
            room = batch / "room1070"
            room.mkdir(parents=True)
            artifact = room / "survey.json"
            artifact.write_text("{}")
            save(batch / "checkpoints/room1070.json", "original",
                 {"status": "image_assisted", "cameras": []}, room)
            artifact.write_text('{"changed": true}')
            out = Path(folder) / "pilot"
            with patch.object(pilot, "detect_images") as detect:
                with self.assertRaisesRegex(ValueError, "intact, completed"):
                    pilot.probe_room(batch, "room1070", out)
                detect.assert_not_called()
            self.assertFalse(out.exists())
            self.assertEqual(artifact.read_text(), '{"changed": true}')

    def test_cli_rejects_output_inside_batch_without_writing_it(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "batch"
            batch.mkdir()
            marker = batch / "original.txt"
            marker.write_text("original")
            out = batch / "experiment"
            result = subprocess.run(
                [sys.executable, pilot.__file__, "--batch", str(batch),
                 "--rooms", "room1070", "--out", str(out)],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the source batch", result.stderr)
            self.assertFalse(out.exists())
            self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
