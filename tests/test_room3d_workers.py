"""Spawned rooms must keep independent outputs and preserve failure reporting."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / 'tools/batch_room3d.py'


class WorkerTests(unittest.TestCase):
    def test_parallel_empty_slots_resume_and_invalid_room_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assets = root / 'assets/Stage1'
            assets.mkdir(parents=True)
            first, second = assets / 'ROOM1000.RDT', assets / 'ROOM1010.RDT'
            first.write_bytes(bytes(4))
            second.write_bytes(bytes(4))
            out = root / 'output'
            command = [sys.executable, str(TOOL), '--assets', str(assets.parent),
                       '--out', str(out), '--survey-only', '--jobs', '2', '--resume']
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual([item['room'] for item in manifest['results']], ['ROOM1000', 'ROOM1010'])
            self.assertTrue(all(item['status'] == 'empty_slot' for item in manifest['results']))
            # Changing worker count cannot change a room's checkpoint key.
            command[command.index('--jobs') + 1] = '1'
            resumed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(resumed.stdout.count('resumed (empty_slot)'), 2)
            second.write_bytes(b'invalid')
            command[command.index('--jobs') + 1] = '2'
            failed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(failed.returncode, 1, failed.stderr)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual([item['status'] for item in manifest['results']], ['empty_slot', 'error'])


if __name__ == '__main__':
    unittest.main()
