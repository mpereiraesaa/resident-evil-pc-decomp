"""Resume must reject changed inputs, truncated exports and unsuccessful rooms."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_checkpoint import restore, save


class CheckpointTests(unittest.TestCase):
    def test_changes_and_missing_artifacts_force_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room = root / "room"
            room.mkdir()
            mesh = room / "mesh.obj"
            mesh.write_text("complete")
            result = {"status": "image_assisted", "cameras": []}
            checkpoint = root / "checkpoints" / "room.json"
            save(checkpoint, "input1", result, room)
            self.assertEqual(restore(checkpoint, "input1", room), result)
            self.assertIsNone(restore(checkpoint, "input2", room))
            mesh.write_text("truncated")
            self.assertIsNone(restore(checkpoint, "input1", room))
            mesh.unlink()
            self.assertIsNone(restore(checkpoint, "input1", room))

    def test_failed_and_partial_results_are_not_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.json"
            for result in ({"status": "error"},
                           {"status": "baseline", "cameras": [{"error": "decode"}]},
                           {"status": "image_assisted", "runtime": {"passed": False}}):
                save(checkpoint, "key", result, root / "room")
                self.assertFalse(checkpoint.exists())

    def test_empty_slots_and_interrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.json"
            result = {"status": "empty_slot", "cameras": []}
            save(checkpoint, "key", result, root / "room")
            self.assertEqual(restore(checkpoint, "key", root / "room"), result)
            checkpoint.write_text('{"schema":')
            self.assertIsNone(restore(checkpoint, "key", root / "room"))


if __name__ == "__main__":
    unittest.main()
