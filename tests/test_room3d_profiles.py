"""Profiles must preserve overrides and keep checkpoint recipes distinct."""

import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_checkpoint import signature
from room3d_profiles import PROFILES, apply_profile


class ProfileTests(unittest.TestCase):
    def test_default_is_exact_and_copies_nested_recipes(self):
        original = {'room1000': {'primitives': [{'kind': 'lamp'}]}}
        output = apply_profile('existing', original, ['room1000', 'room1010'])
        self.assertEqual(original, output)
        output['room1000']['primitives'][0]['kind'] = 'column'
        self.assertEqual(original['room1000']['primitives'][0]['kind'], 'lamp')

    def test_explicit_overrides_and_room_independence(self):
        recipes = {'room1000': {'fixture_part_fit': False, 'handrail_candidates': False, 'primitives': []},
                   'room2000': {'custom': 'untouched'}}
        original = copy.deepcopy(recipes)
        output = apply_profile('creative-v2', recipes, ['room1000', 'room1010'])
        self.assertFalse(output['room1000']['fixture_part_fit'])
        self.assertFalse(output['room1000']['handrail_candidates'])
        self.assertTrue(output['room1010']['handrail_candidates'])
        self.assertTrue(output['room1010']['fixture_part_fit'])
        self.assertTrue(output['room1010']['creative_stair_landings'])
        self.assertEqual(output['room1010']['_detail_profile'], 'creative-v2')
        self.assertEqual(output['room2000'], original['room2000'])
        self.assertEqual(recipes, original)
        self.assertNotEqual(signature(original.get('room1010', {})), signature(output['room1010']))
        output['room1010']['scene_depth_polish'] = False
        self.assertTrue(PROFILES['creative-v2']['scene_depth_polish'])

    def test_unknown_or_malformed_profile_inputs_are_rejected(self):
        for name, recipes in (('missing', {}), ('existing', []), ('creative-v2', {'room1000': []})):
            with self.subTest(name=name), self.assertRaises(ValueError):
                apply_profile(name, recipes, ['room1000'])

    def test_cli_documents_profile_and_requires_reconstruction(self):
        script = Path(__file__).resolve().parents[1] / 'tools/batch_room3d.py'
        result = subprocess.run([sys.executable, str(script), '--help'], check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('creative-v2', result.stdout)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'ROOM1000.RDT').write_bytes(bytes(4))
            result = subprocess.run([sys.executable, str(script), '--assets', str(root),
                                     '--out', str(root.parent / (root.name + '-unused')),
                                     '--mesh-only', '--detail-profile', 'creative-v2'],
                                    check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('requires --reconstruct', result.stderr)
            result = subprocess.run([sys.executable, str(script), '--assets', str(root),
                                     '--out', str(root.parent / (root.name + '-unused')),
                                     '--reconstruct', '--detail-profile', 'creative-v2'],
                                    check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('requires cached --segmentation and --depth', result.stderr)
            self.assertFalse((root.parent / (root.name + '-unused')).exists())


if __name__ == '__main__':
    unittest.main()
