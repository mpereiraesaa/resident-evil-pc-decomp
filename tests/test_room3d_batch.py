"""Synthetic geometry tests: no proprietary assets or running game required."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from batch_room3d import generate_mesh, process
from room3d_preview import camera_basis, clip_near, render


class PreviewTests(unittest.TestCase):
    def test_inset_background_matches_runtime_logical_viewport(self):
        from unittest.mock import patch

        from batch_room3d import background

        path = self.root / "test.pak"
        path.write_bytes(b"packed")
        for width, height in ((316, 236), (320, 240)):
            data = struct.pack("<IIIHHHH", 16, 2, width*height*2, 0, 0, width, height)
            data += struct.pack("<H", 31) * (width*height)
            with patch("batch_room3d.PakDecoder") as decoder:
                decoder.return_value.run.return_value = data
                image = background(path)
            self.assertEqual(image.size, (320, 240))
            self.assertEqual(image.getpixel((319, 239)), image.getpixel((0, 0)))
            self.assertGreater(max(image.getpixel((0, 0))), 240)
        self.assertEqual(path.read_bytes(), b"packed")

    def test_railing_material_is_not_stair_material(self):
        from room3d_segment import matches_material

        self.assertFalse(matches_material("staircase railing", ("staircase",)))
        self.assertTrue(matches_material("staircase railing", ("railing",)))

    def test_free_space_at_depth_boundary_is_unknown(self):
        from room3d_structure import evidence

        camera = {"origin": [0, 0, 0], "target": [0, 0, 1000], "roll": 0, "fov": 100}
        depth = np.full((240, 320), 1000.0)
        depth[:, 159] = 400
        _, conflict = evidence(
            [[0, 0, 500]],
            {"cameras": [camera, camera]},
            [(depth, np.ones_like(depth, bool))] * 2,
        )
        self.assertEqual(conflict[0], 0)

    def test_empty_fixture_does_not_become_a_solid_box(self):
        mesh = self.root / "omitted.obj"
        room = {
            "cameras": [],
            "boundaries": [
                {
                    "index": 0,
                    "xmin": 0,
                    "xmax": 1000,
                    "zmin": 0,
                    "zmax": 1000,
                    "type": 4,
                    "flags": 0,
                }
            ],
        }
        generate_mesh(
            room,
            mesh,
            {
                "primitives": [
                    {
                        "kind": "lamp",
                        "parts": [],
                        "box": [0, 0, 500, 500, -1000, 0],
                        "omit_geometry": True,
                    }
                ]
            },
        )
        self.assertNotIn("primitive_0", mesh.read_text())

    def test_shared_wall_union_has_no_internal_faces(self):
        from room3d_structure import wall_outline

        bounds = [
            {
                "index": i,
                "xmin": x,
                "xmax": x + 1000,
                "zmin": 0,
                "zmax": 1000,
                "flags": 256,
                "type": 1,
            }
            for i, x in enumerate([0, 500])
        ]
        edges = wall_outline(bounds)
        self.assertFalse(
            any(e["axis"] == 0 and e["position"] in (500, 1000) for e in edges)
        )
        self.assertEqual(sum(e["hi"] - e["lo"] for e in edges), 5000)

    def test_free_space_evidence_distinguishes_occlusion_and_missing_data(self):
        from room3d_structure import evidence

        camera = {"origin": [0, 0, 0], "target": [0, 0, 1000], "roll": 0, "fov": 100}
        room = {"cameras": [camera, camera]}
        depth = np.full((240, 320), 1000.0)
        valid = np.ones_like(depth, bool)
        support, conflict = evidence(
            [[0, 0, 500], [0, 0, 1000], [0, 0, 1500]], room, [(depth, valid)] * 2
        )
        np.testing.assert_array_equal(support, [0, 2, 0])
        np.testing.assert_array_equal(conflict, [2, 0, 0])
        _, conflict = evidence([[0, 0, 500]], room, [(depth, valid & False)] * 2)
        self.assertEqual(conflict[0], 0)

    def test_floor_labels_and_depth_scale_validation(self):
        from room3d_refine import calibrate_floor_depth
        from room3d_structure import semantic_masks

        mask = np.ones((240, 320), bool)
        floor_entry = ({"label": "wooden floor", "score": .39, "predicted_iou": .97}, mask)
        self.assertTrue(semantic_masks([floor_entry])["floor"].all())
        lamp = ({"label": "floor lamp", "score": .9, "predicted_iou": .9}, mask)
        self.assertFalse(semantic_masks([floor_entry, lamp])["floor"].any())
        target = np.full(mask.shape, 1000.)
        floor = np.full(mask.shape, 1250.)
        corrected, report = calibrate_floor_depth((target, mask), floor, floor, mask)
        self.assertTrue(report["accepted"])
        np.testing.assert_allclose(corrected[0], floor)
        np.testing.assert_allclose(target, 1000)
        y, x = np.indices(mask.shape)
        inconsistent = floor.copy()
        inconsistent[(x//16+y//16) % 2 == 1] = 1900
        _, report = calibrate_floor_depth((target, mask), inconsistent, inconsistent, mask)
        self.assertFalse(report["accepted"])
        _, report = calibrate_floor_depth((target, mask), floor, floor-200, mask)
        self.assertFalse(report["accepted"])
        _, report = calibrate_floor_depth((target, mask), floor, floor, mask & False)
        self.assertFalse(report["accepted"])

    def test_door_prior_resolves_crate_label_without_overriding_foreground(self):
        from room3d_segment import texture_support

        mask = np.ones((240, 320), bool)
        xy = np.array([[160, 120]])
        crate = ({"label": "chest wooden crate", "score": .52, "predicted_iou": .98}, mask)
        door = ({"label": "door", "score": .31, "predicted_iou": .95}, mask)
        for entries in ([crate, door], [door, crate]):
            self.assertTrue(texture_support(entries, "door_panel", xy)[0])
            self.assertFalse(texture_support(entries, "wall", xy)[0])
            lamp = ({"label": "wall lamp", "score": .6, "predicted_iou": .9}, mask)
            self.assertFalse(texture_support([*entries, lamp], "door_panel", xy)[0])
        for change in ({"score": .29}, {"predicted_iou": .89}):
            weak = ({**door[0], **change}, mask)
            self.assertFalse(texture_support([crate, weak], "door_panel", xy)[0])

    def test_fixture_mask_overrides_overlapping_wall_in_either_order(self):
        from room3d_segment import texture_support
        from room3d_structure import semantic_masks

        mask = np.ones((240, 320), bool)
        wall = ({"label": "wall", "score": 0.9, "predicted_iou": 0.9}, mask)
        lamp = ({"label": "wall lamp", "score": 0.9, "predicted_iou": 0.9}, mask)
        for entries in ([wall, lamp], [lamp, wall]):
            self.assertFalse(
                texture_support(entries, "wall", np.array([[160, 120]]))[0]
            )
            self.assertFalse(semantic_masks(entries)["wall"].any())

    def test_ceiling_fan_pixels_do_not_fit_overlapping_background_planes(self):
        from room3d_structure import semantic_masks

        background = np.ones((240, 320), bool)
        fan_mask = np.zeros_like(background)
        fan_mask[30:45, 70:250] = True
        entries = [({"label": label, "score": .9, "predicted_iou": .95}, background)
                   for label in ("wall", "wooden floor")]
        fan = ({"label": " Ceiling Fan ", "score": .7, "predicted_iou": .9}, fan_mask)
        original = fan_mask.copy()
        for observations in ([*entries, fan], [fan, *entries]):
            result = semantic_masks(observations)
            np.testing.assert_array_equal(result["fixture"], fan_mask)
            for kind in ("wall", "floor"):
                np.testing.assert_array_equal(result[kind], ~fan_mask)
        np.testing.assert_array_equal(fan_mask, original)
        self.assertTrue(background.all())

    def test_unreliable_fan_detection_cannot_exclude_architectural_evidence(self):
        from room3d_segment import texture_support
        from room3d_structure import semantic_masks

        mask = np.ones((240, 320), bool)
        wall = ({"label": "wall", "score": .9, "predicted_iou": .95}, mask)
        for score, iou in ((.29, .9), (.7, .79), (float('nan'), .9),
                           (.7, float('nan')), (float('inf'), .9), (.7, float('inf'))):
            with self.subTest(score=score, iou=iou):
                fan = ({"label": "ceiling fan", "score": score, "predicted_iou": iou}, mask)
                result = semantic_masks([wall, fan])
                self.assertTrue(result["wall"].all())
                self.assertFalse(result["fixture"].any())
                if not np.isfinite(score) or not np.isfinite(iou):
                    for guarded in (False, True):
                        self.assertTrue(texture_support([wall, fan], "wall", np.array([[160, 120]]),
                                                        foreground_fixture_guard=guarded)[0])

    def test_thin_fixture_depth_jump_does_not_bridge_background(self):
        from room3d_objects import masked_volume

        camera = {"origin": [0, 0, 0], "target": [0, 0, 1000], "roll": 0, "fov": 100}
        depth = np.full((6, 6), 1000.0)
        depth[:, 3:] = 5000
        self.assertEqual(masked_volume(camera, np.ones((6, 6), bool), depth), [])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.camera = {
            "index": 0,
            "origin": [0, 0, 0],
            "target": [0, 0, 1000],
            "roll": 0,
            "fov": 100,
        }

    def mesh(self, bodies):
        path = self.root / "test.obj"
        path.write_text("mtllib test.mtl\n" + bodies)
        path.with_suffix(".mtl").write_text(
            "newmtl red\nKd 1 0 0\nnewmtl green\nKd 0 1 0\n"
        )
        return path

    def test_isolated_object_render_excludes_occluding_group(self):
        mesh = self.mesh(
            "v -1000 -1000 1000\nv 1000 -1000 1000\nv 0 1000 1000\n"
            "v -1000 -1000 1500\nv 1000 -1000 1500\nv 0 1000 1500\n"
            "g wall\nusemtl red\nf 1 2 3\n"
            "g lamp\nusemtl green\nf 4 5 6\n"
        )
        full, _ = render(mesh, self.camera)
        isolated, coverage = render(mesh, self.camera, only_groups={"lamp"})
        self.assertEqual(full.getpixel((160, 120)), (255, 0, 0))
        self.assertEqual(isolated.getpixel((160, 120)), (0, 255, 0))
        self.assertTrue(coverage[120, 160])
        _, empty = render(mesh, self.camera, only_groups={"absent"})
        self.assertFalse(empty.any())

    def test_masked_volume_preserves_empty_cells_and_infers_thickness(self):
        from room3d_objects import masked_volume

        mask = np.zeros((240, 320), dtype=bool)
        mask[120:126, 150:156] = True
        mask[120:126, 162:168] = True
        parts = masked_volume(self.camera, mask, np.full(mask.shape, 1000.0))
        self.assertEqual(len(parts), 2)
        for vertices in map(np.asarray, parts):
            self.assertEqual(vertices.shape, (8, 3))
            np.testing.assert_allclose(vertices[:4, 2], 1000)
            np.testing.assert_allclose(vertices[4:, 2], 1035)
        self.assertLess(
            np.max(np.asarray(parts[0])[:, 0]), np.min(np.asarray(parts[1])[:, 0])
        )

    def test_runtime_audit_detects_changed_texture_with_unchanged_obj(self):
        import shutil

        from check_room3d_pipeline import check_runtime_materials

        mesh = self.mesh("g wall\n")
        mesh.with_suffix(".mtl").write_text("newmtl wall\nmap_Kd wall.png\n")
        texture = self.root / "wall.png"
        texture.write_bytes(b"original texture bytes")
        runtime = self.root / "runtime"
        mods = runtime / "mods" / "Stage1"
        mods.mkdir(parents=True)
        for source in (mesh.with_suffix(".mtl"), texture):
            shutil.copy2(source, mods / source.name)
        check_runtime_materials(mesh, runtime, "ROOM1180")
        (mods / "wall.png").write_bytes(b"stale texture bytes")
        with self.assertRaisesRegex(ValueError, "runtime texture differs"):
            check_runtime_materials(mesh, runtime, "ROOM1180")

    def test_new_viewpoints_report_when_collision_bounds_leave_no_samples(self):
        from room3d_novel_views import inspect

        room = {"boundaries": [{"xmin": 0, "xmax": 3000, "zmin": 0, "zmax": 3000}]}
        report = inspect(self.root / "unused.obj", room, self.root / "new-views")
        self.assertEqual(report["status"], "no_clear_sample_positions")
        self.assertEqual(report["views"], [])

    def test_review_cache_requires_exact_inputs_and_prompt(self):
        import hashlib
        import json

        from PIL import Image
        from room3d_review import MODELS, PROMPT, reuse_review

        reference = self.root / "reference.png"
        reconstruction = self.root / "reconstruction.png"
        for path in (reference, reconstruction):
            Image.new("RGB", (8, 8), "green").save(path)
        model, revision = MODELS["4b"]
        record = {
            "model": model,
            "revision": revision,
            "prompt": PROMPT,
            "inputs": [
                {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in (reference, reconstruction)
            ],
            "response": "NO_CLEAR_DIFFERENCE",
        }
        source, out = self.root / "source.json", self.root / "out.json"
        source.write_text(json.dumps(record))
        self.assertTrue(reuse_review(source, reference, reconstruction, out))
        out.unlink()
        record["prompt"] = "different comparison task"
        source.write_text(json.dumps(record))
        self.assertFalse(reuse_review(source, reference, reconstruction, out))
        record["prompt"] = PROMPT
        source.write_text(json.dumps(record))
        Image.new("RGB", (8, 8), "red").save(reconstruction)
        self.assertFalse(reuse_review(source, reference, reconstruction, out))
        self.assertFalse(out.exists())

    def test_depth_is_order_independent(self):
        verts = (
            "v -1000 -1000 1000\nv 1000 -1000 1000\nv 0 1000 1000\n"
            "v -1000 -1000 1500\nv 1000 -1000 1500\nv 0 1000 1500\n"
        )
        faces = ["usemtl red\nf 1 2 3\n", "usemtl green\nf 4 5 6\n"]
        first, _ = render(self.mesh(verts + "".join(faces)), self.camera)
        second, _ = render(self.mesh(verts + "".join(reversed(faces))), self.camera)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.getpixel((160, 120)), (255, 0, 0))
        self.assertEqual(first.getpixel((0, 0)), (0, 0, 0))

    def test_camera_hide_and_near_clipping(self):
        path = self.mesh(
            "g wall\nusemtl red\nv -300 -300 100\nv 300 -300 500\nv 0 300 500\nf -3 -2 -1\n"
        )
        _, visible = render(path, self.camera)
        self.assertGreater(visible.sum(), 0)
        path.with_suffix(".cameras").write_text("hide 0 wall\n")
        _, hidden = render(path, self.camera)
        self.assertEqual(hidden.sum(), 0)
        clipped = clip_near(
            np.array(
                [[0.0, 0.0, 100.0, 0.0], [1.0, 0.0, 500.0, 1.0], [0.0, 1.0, 500.0, 1.0]]
            ),
            200,
        )
        self.assertEqual(len(clipped), 4)
        self.assertTrue(all(p[2] >= 200 for p in clipped))
        self.assertAlmostEqual(clipped[0][3], 0.25)

    def test_basis_and_unsupported_roll(self):
        np.testing.assert_allclose(camera_basis(self.camera), np.eye(3))
        with self.assertRaises(ValueError):
            camera_basis(dict(self.camera, roll=1))

    def test_recipe_cutaway_and_unresolved_shapes(self):
        room = {
            "cameras": [dict(self.camera, origin=[5, -100, 5])],
            "boundaries": [
                {
                    "index": 0,
                    "xmin": 0,
                    "xmax": 10,
                    "zmin": 0,
                    "zmax": 10,
                    "type": 1,
                    "flags": 768,
                },
                {
                    "index": 1,
                    "xmin": 20,
                    "xmax": 30,
                    "zmin": 20,
                    "zmax": 30,
                    "type": 4,
                    "flags": 0,
                },
            ],
        }
        path = self.root / "test.obj"
        info = generate_mesh(room, path, {"wall_height": 300})
        self.assertEqual(info["automatic_cutaways"], [[0, "boundary_0"]])
        self.assertEqual(info["unresolved_boundaries"], [1])
        self.assertIn("hide 0 boundary_0", path.with_suffix(".cameras").read_text())
        self.assertEqual(info["wall_height"], 300)

    def test_revisit_background_resolution_and_empty_slot(self):
        stage1 = self.root / "Stage1"
        stage6 = self.root / "Stage6"
        stage1.mkdir()
        stage6.mkdir()
        (stage1 / "RC1000.pak").write_bytes(b"presence-only inventory test")
        data = bytearray(148 + 44 + 24)
        data[1] = 1
        struct.pack_into("<9i", data, 156, 0, 0, 0, 0, 0, 1000, 0, 0, 100)
        struct.pack_into("<I", data, 0x4C, 192)
        rdt = stage6 / "ROOM6000.RDT"
        rdt.write_bytes(data)
        out = self.root / "out"
        out.mkdir()
        result = process(rdt, out, None, {}, True)
        self.assertTrue(result["cameras"][0]["available"])
        self.assertEqual(result["cameras"][0]["background"], "Stage1/rc1000.pak")
        rdt.write_bytes(bytes(4))
        self.assertEqual(process(rdt, out, None, {}, True)["status"], "empty_slot")
        rdt.write_bytes(bytes(3))
        with self.assertRaises(ValueError):
            process(rdt, out, None, {}, True)


class ReconstructionTests(unittest.TestCase):
    setUp = PreviewTests.setUp
    mesh = PreviewTests.mesh

    def test_triangulation_rejects_inconsistent_and_parallel_rays(self):
        from room3d_reconstruct import project, triangulate

        a = self.camera
        b = dict(a, origin=[500, 0, 0], target=[500, 0, 1000])
        points = np.array([[100.0, -100.0, 2000.0], [200.0, 100.0, 3000.0]])
        pa, _ = project(points, a)
        pb, _ = project(points, b)
        recovered, valid, _error = triangulate(a, b, pa, pb)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(recovered, points, atol=1e-6)
        pb[:, 1] += 50
        self.assertFalse(triangulate(a, b, pa, pb)[1].any())
        self.assertFalse(triangulate(a, a, pa, pa)[1].any())

    def test_sparse_evidence_does_not_change_height(self):
        from room3d_reconstruct import fit_heights

        room = {
            "boundaries": [
                {
                    "index": 0,
                    "type": 1,
                    "flags": 512,
                    "xmin": 0,
                    "xmax": 1000,
                    "zmin": 0,
                    "zmax": 1000,
                }
            ]
        }
        fitted, proposals = fit_heights(
            room, {"points": [{"xyz": [200, -1400, 200]}]}, {}
        )
        self.assertEqual(fitted["boundary_heights"], {})
        self.assertEqual(proposals, [])

    def test_texture_filter_rejects_object_inside_triangle(self):
        from PIL import Image
        from room3d_reconstruct import bake_projected

        path = self.mesh(
            "g wall\nusemtl red\nv -1000 -1000 1000\nv 1000 -1000 1000\nv 0 1000 1000\nf 1 2 3\n"
        )
        path.with_suffix(".cameras").write_text("")
        pixels = np.zeros((240, 320, 3), np.uint8)
        pixels[:, :, 1] = 255
        pixels[115:126, 155:166] = [255, 0, 0]
        mask = np.zeros((240, 320), bool)
        mask[115:126, 155:166] = True
        segments = [
            [({"label": "storage chest", "score": 1, "predicted_iou": 1}, mask)]
        ]
        out = self.root / "filtered.obj"
        info = bake_projected(
            path,
            {"cameras": [self.camera]},
            [Image.fromarray(pixels)],
            out,
            segments=segments,
            group_kinds={"wall": "wall"},
        )
        image, _ = render(out, self.camera)
        self.assertGreater(info["filtered_texture_pixels"], 0)
        self.assertGreater(image.getpixel((160, 120))[1], 200)
        self.assertLess(image.getpixel((160, 120))[0], 20)

    def test_creative_geometry_uses_material_without_projecting_original_shape(self):
        from PIL import Image
        from room3d_reconstruct import bake_projected

        path = self.mesh(
            "g lamp\nusemtl red\nv -1000 -1000 1000\nv 1000 -1000 1000\nv 0 1000 1000\nf 1 2 3\n"
        )
        path.with_suffix(".cameras").write_text("")
        mask = np.zeros((240, 320), bool)
        mask[80:160, 120:200] = True
        image = Image.new("RGB", (320, 240), (0, 255, 0))
        segments = [[({"label": "lamp", "score": 1, "predicted_iou": 1}, mask)]]
        for excluded in (None, 0):
            out = self.root / f"creative-{excluded}.obj"
            report = bake_projected(
                path,
                {"cameras": [self.camera]},
                [image],
                out,
                segments=segments,
                group_kinds={"lamp": "lamp"},
                interpreted_groups={"lamp"},
                exclude_camera=excluded,
            )
            self.assertEqual(report["sampled_vertices"], 0)
            self.assertNotIn("usemtl camera", out.read_text())
            result, _ = render(out, self.camera)
            if excluded is None:
                self.assertGreater(report["inferred_vertices"], 0)
                self.assertGreater(result.getpixel((160, 120))[1], 200)
            else:
                self.assertEqual(report["inferred_vertices"], 0)
                self.assertEqual(result.getpixel((160, 120)), (45, 45, 45))

    def test_creative_stair_supports_use_explicit_artistic_material(self):
        from PIL import Image
        from room3d_reconstruct import bake_projected

        path = self.mesh(
            "g stair\nusemtl red\nv -1000 -1000 1000\nv 1000 -1000 1000\nv 0 1000 1000\nf 1 2 3\n"
        )
        path.with_suffix(".cameras").write_text("")
        for interpreted in (set(), {"stair"}):
            out = self.root / f"style-{bool(interpreted)}.obj"
            report = bake_projected(
                path,
                {"cameras": [self.camera]},
                [Image.new("RGB", (320, 240), "green")],
                out,
                group_kinds={"stair": "staircase"},
                interpreted_groups=interpreted,
            )
            self.assertEqual(report["styled_faces"] > 0, bool(interpreted))
            self.assertEqual(
                "usemtl creative_stone" in out.read_text(), bool(interpreted)
            )

    def test_inferred_chart_does_not_collapse_at_tile_multiples(self):
        from room3d_reconstruct import material_chart

        points = np.array([[0, 0, 0], [1000, 0, 0], [1000, 0, 1000]])
        axes, origin, scale = material_chart(points, [0, 1, 0])
        uv = (points[:, axes] - origin) / scale
        np.testing.assert_allclose(uv, [[0, 0], [1, 0], [1, 1]])
        midpoint = (points[0] + points[2]) / 2
        np.testing.assert_allclose((midpoint[axes] - origin) / scale, [0.5, 0.5])

    def test_grazing_texture_projection_is_rejected(self):
        from PIL import Image
        from room3d_reconstruct import bake_projected

        path = self.mesh(
            "g wall\nusemtl red\nv 200 -500 1000\nv 200 500 1000\nv 200 0 4000\nf 1 2 3\n"
        )
        path.with_suffix(".cameras").write_text("")
        room = {"cameras": [self.camera]}
        image = Image.new("RGB", (320, 240), (0, 255, 0))
        legacy = bake_projected(
            path,
            room,
            [image],
            self.root / "grazing-old.obj",
            minimum_projection_cosine=0,
        )
        filtered = bake_projected(
            path, room, [image], self.root / "grazing-filtered.obj"
        )
        self.assertGreater(legacy["sampled_vertices"], 0)
        self.assertEqual(filtered["sampled_vertices"], 0)
        self.assertGreater(filtered["grazing_rejections"], 0)
        self.assertNotIn(
            "usemtl camera", (self.root / "grazing-filtered.obj").read_text()
        )

    def test_appearance_excludes_evaluation_camera(self):
        from PIL import Image
        from room3d_reconstruct import bake_projected

        path = self.mesh(
            "g wall\nusemtl red\nv -1000 -1000 1000\nv 1000 -1000 1000\nv 0 1000 1000\nf 1 2 3\n"
        )
        path.with_suffix(".cameras").write_text("")
        image = Image.new("RGB", (320, 240), (0, 255, 0))
        room = {"cameras": [self.camera]}
        all_info = bake_projected(path, room, [image], self.root / "all.obj")
        heldout = bake_projected(
            path, room, [image], self.root / "heldout.obj", exclude_camera=0
        )
        self.assertGreater(all_info["sampled_vertices"], 0)
        self.assertEqual(heldout["sampled_vertices"], 0)
        colored, _ = render(self.root / "all.obj", self.camera)
        excluded, _ = render(self.root / "heldout.obj", self.camera)
        self.assertGreater(colored.getpixel((160, 120))[1], 200)
        self.assertEqual(excluded.getpixel((160, 120)), (45, 45, 45))


class EvidenceTests(unittest.TestCase):
    def test_wall_lamp_is_not_wall_material(self):
        from room3d_segment import matches_material

        self.assertFalse(matches_material("wall lamp chandelier", ("wall",)))
        self.assertTrue(matches_material("wall lamp chandelier", ("lamp",)))
        self.assertTrue(matches_material("wall", ("wall",)))

    def test_stair_proposal_needs_inclined_cloud(self):
        from room3d_objects import stair_proposals

        room = {
            "boundaries": [
                {
                    "index": 3,
                    "type": 4,
                    "xmin": 0,
                    "xmax": 6000,
                    "zmin": 0,
                    "zmax": 2000,
                }
            ]
        }
        x = np.linspace(1000, 5000, 100)
        points = np.column_stack((x, -x * 0.5, np.full(100, 500)))
        proposed = stair_proposals(room, {"index": 1}, {"label": "staircase"}, points)
        self.assertEqual(len(proposed), 1)
        self.assertTrue(proposed[0]["rise_positive"])
        self.assertEqual(proposed[0]["axis"], "x")
        points[:, 1] = 0
        self.assertEqual(
            stair_proposals(room, {"index": 1}, {"label": "staircase"}, points), []
        )

    def test_object_proposals_keep_input_room_unchanged(self):
        import copy

        from room3d_objects import object_proposals

        camera = {
            "index": 0,
            "origin": [0, -1000, 0],
            "target": [0, -1000, 1000],
            "fov": 100,
            "roll": 0,
        }
        room = {
            "cameras": [camera],
            "boundaries": [
                {
                    "index": 0,
                    "xmin": -5000,
                    "xmax": 5000,
                    "zmin": 500,
                    "zmax": 5000,
                    "type": 1,
                    "flags": 512,
                }
            ],
        }
        original = copy.deepcopy(room)
        mask = np.zeros((240, 320), bool)
        mask[100:170, 145:175] = True
        segments = [
            [({"label": "wooden barrel", "score": 0.8, "predicted_iou": 0.9}, mask)]
        ]
        depths = [(np.full((240, 320), 1000.0), np.ones((240, 320), bool))]
        proposals = object_proposals(room, segments, depths, [])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["kind"], "barrel")
        self.assertEqual(proposals[0]["replace_visual_boundaries"], [0])
        self.assertEqual(room, original)
        room["boundaries"][0]["flags"] = 256
        self.assertEqual(
            object_proposals(room, segments, depths, [])[0][
                "replace_visual_boundaries"
            ],
            [],
        )

    def test_hidden_material_patch_excludes_heldout_camera(self):
        from PIL import Image
        from room3d_complete import material_patches

        image = Image.new("RGB", (32, 24), (20, 80, 120))
        entries = [
            (
                {"label": "wall", "score": 0.9, "predicted_iou": 0.95},
                np.ones((24, 32), bool),
            )
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            patches = material_patches([image], [entries], root)
            self.assertIn("wall", patches)
            self.assertTrue((root / patches["wall"]["file"]).is_file())
            self.assertEqual(
                material_patches([image], [entries], root, exclude_camera=0), {}
            )

    def test_depth_score_penalizes_holes_and_multiview_regression(self):
        from room3d_refine import depth_error, improvements

        target = np.full((2, 2), 1000.0)
        valid = np.ones((2, 2), dtype=bool)
        self.assertEqual(depth_error(target, target, valid), 0)
        holes = np.full((2, 2), np.inf)
        self.assertEqual(depth_error(holes, target, valid), 1)
        self.assertFalse(improvements([0.3, 0.3], [0.1, 0.4]))
        self.assertFalse(improvements([0.3, 0.3], [0.3, 0.3]))
        self.assertTrue(improvements([0.3, 0.3], [0.2, 0.25]))

    def test_depth_candidates_leave_anchor_footprints_unchanged(self):
        import copy

        from room3d_refine import candidates

        recipe = {
            "primitives": [{"kind": "chest_body", "box": [0, 0, 1000, 1500, -1400, 0]}]
        }
        original = copy.deepcopy(recipe)
        for _, proposal in candidates(recipe):
            self.assertEqual(
                proposal["primitives"][0]["box"][:4],
                original["primitives"][0]["box"][:4],
            )
            self.assertEqual(proposal["primitives"][0]["box"][5], 0)
        self.assertEqual(recipe, original)

    def test_semantic_guard_rejects_chest_pixels_on_floor(self):
        from room3d_segment import texture_support

        mask = np.array([[True, False]])
        entries = [
            ({"label": "storage chest", "score": 0.9, "predicted_iou": 0.95}, mask)
        ]
        xy = np.array([[0, 0], [1, 0]])
        self.assertEqual(texture_support(entries, "floor", xy).tolist(), [False, True])
        self.assertEqual(
            texture_support(entries, "chest_body", xy).tolist(), [True, True]
        )
        entries[0][0]["predicted_iou"] = 0.2
        self.assertTrue(texture_support(entries, "floor", xy).all())

    def test_depth_camera_inputs_preserve_game_projection(self):
        from room3d_depth import camera_inputs
        from room3d_reconstruct import project

        camera = {
            "origin": [300, -1800, 100],
            "target": [0, -900, 5000],
            "fov": 230,
            "roll": 0,
        }
        ext, intr = camera_inputs({"cameras": [camera]})
        point = np.array([200, -1200, 3000, 1])
        view = ext[0] @ point
        pixel = intr[0] @ view[:3]
        expected, depth = project([point[:3]], camera)
        np.testing.assert_allclose(pixel[:2] / pixel[2], expected[0], atol=1e-4)
        self.assertAlmostEqual(view[2], depth[0], places=3)
        self.assertAlmostEqual(np.linalg.det(ext[0, :3, :3]), 1, places=5)

    def test_segmentation_rejects_stale_images_and_invalid_boxes(self):
        import hashlib

        from PIL import Image
        from room3d_segment import verified_boxes

        image = Image.new("RGB", (8, 6))
        record = {"rgb_sha256": "stale", "detections": []}
        with self.assertRaises(ValueError):
            verified_boxes(image, record)
        record["rgb_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
        record["detections"] = [{"box": [-1, -2, 10, 12]}]
        self.assertEqual(verified_boxes(image, record), [[0, 0, 8, 6]])
        for invalid in ([3, 0, 2, 4], [0, 0, float("nan"), 4]):
            record["detections"] = [{"box": invalid}]
            with self.assertRaises(ValueError):
                verified_boxes(image, record)

    def test_runtime_requires_room_and_ordered_menu_close(self):
        from check_room3d_runtime import observations

        log = "[room3d] loaded mods/room1180.obj\n"
        log += "[room3d-test] stage=1 room=18 camera=0 menu=0 door=0 standalone=0\n"
        log += "[room3d-test] stage=1 room=18 camera=1 menu=0 door=0 standalone=0\n"
        log += "[room3d-test] stage=1 room=18 camera=1 menu=1 door=0 standalone=1\n"
        self.assertFalse(observations(log, 0x18)["menu_closed"])
        log += "[room3d-test] stage=1 room=18 camera=1 menu=0 door=0 standalone=0\n"
        self.assertTrue(observations(log, 0x18)["menu_closed"])
        self.assertFalse(observations(log, 0x0B)["menu_opened"])
        self.assertFalse(observations(log, 0x0B)["mesh_loaded"])
        self.assertTrue(
            observations("[room3d] loaded mods/room10B0.obj", 0x0B)["mesh_loaded"]
        )

    def test_visual_height_needs_matching_script_anchor(self):
        from room3d_scene import fit_detected_primitives

        camera = {
            "index": 0,
            "origin": [0, 0, 0],
            "target": [0, 0, 1000],
            "roll": 0,
            "fov": 100,
        }
        item = {"kind": "door_panel", "box": [-100, 990, 100, 1010, -1500, 0]}
        outside = [
            {"detections": [{"label": "door", "score": 0.9, "box": [0, 0, 10, 10]}]}
        ]
        self.assertEqual(
            fit_detected_primitives([item], {"cameras": [camera]}, outside), []
        )
        inside = [
            {
                "detections": [
                    {"label": "door", "score": 0.9, "box": [145, 20, 175, 120]}
                ]
            }
        ]
        fit = fit_detected_primitives([item], {"cameras": [camera]}, inside)
        self.assertTrue(fit[0]["accepted"])
        inside[0]["detections"].append(dict(inside[0]["detections"][0], score=0.8))
        self.assertEqual(
            fit_detected_primitives([item], {"cameras": [camera]}, inside)[0]["views"],
            1,
        )
        self.assertAlmostEqual(item["box"][4], -1000)

    def test_furniture_inherits_only_overlapping_wall_cutaway(self):
        from room3d_scene import primitives

        room = {
            "boundaries": [
                {
                    "index": 4,
                    "type": 1,
                    "flags": 256,
                    "xmin": -2000,
                    "xmax": 2000,
                    "zmin": 1000,
                    "zmax": 2000,
                }
            ]
        }
        script = {
            "anchors": [
                {"kind": "typewriter", "zone": [0, 800, 100, 100], "source_offset": 0}
            ]
        }
        item = primitives(room, script)[0]
        self.assertEqual(item["hide_with"], "boundary_4")
        script["anchors"][0]["zone"] = [0, 0, 100, 100]
        self.assertNotIn("hide_with", primitives(room, script)[0])

    def parse_anchor_commands(self, commands):
        from room3d_scene import anchors

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "room.RDT"
            data = bytearray(148)
            struct.pack_into("<II", data, 0x60, 148, 150 + len(commands))
            data.extend(struct.pack("<H", 2 + len(commands)))
            data.extend(commands)
            path.write_bytes(data)
            return anchors(path)

    def test_height_rule_signed_rise_and_reset_zone_provenance(self):
        declaration = struct.pack(
            "<BB4HBBHHh", 0x0D, 2, 100, 200, 900, 1200, 17, 4, 2, 300, -180
        )
        reset = struct.pack("<BBBBHHh", 0x12, 2, 17, 5, 3, 400, -200)
        result = self.parse_anchor_commands(declaration + reset)
        self.assertTrue(result["complete"])
        original, changed = result["anchors"]
        self.assertEqual(original["height_increment"], -180)
        self.assertEqual(original["horizontal_interval"], 300)
        self.assertTrue(original["valid_height_rule"])
        self.assertFalse(original["conditional"])
        self.assertEqual(changed["zone"], original["zone"])
        self.assertTrue(changed["conditional"])
        self.assertTrue(changed["reset_requires_prior_zone"])
        self.assertEqual(changed["height_increment"], -200)

    def test_height_reset_does_not_guess_ambiguous_or_missing_zone(self):
        reset = struct.pack("<BBBBHHh", 0x12, 2, 17, 0, 0, 300, -180)
        for prefix in (
            b"",
            b"".join(
                struct.pack("<BB4HBBHHh", 0x0D, 2, x, 200, 900, 1200, 0, 0, 0, 0, 0)
                for x in (100, 500)
            ),
        ):
            result = self.parse_anchor_commands(prefix + reset)
            self.assertEqual(result["anchors"], [])
            self.assertEqual(len(result["unresolved_height_resets"]), 1)

    def test_invalid_height_rules_are_marked_not_normalized(self):
        for direction, interval in ((4, 300), (0, 0)):
            result = self.parse_anchor_commands(
                struct.pack(
                    "<BB4HBBHHh",
                    0x0D,
                    0,
                    0,
                    0,
                    900,
                    1200,
                    17,
                    0,
                    direction,
                    interval,
                    -180,
                )
            )
            self.assertFalse(result["anchors"][0]["valid_height_rule"])

    def test_player_property_widths_preserve_following_anchor(self):
        anchor = struct.pack(
            "<BB4HBBHHh", 0x0D, 0, 0, 0, 900, 1200, 17, 0, 0, 300, -180
        )
        for subtype in (0, 1, 3, 4, 5, 6, 7, 8, 9, 10):
            command = bytes((0x33, subtype))
            if subtype in (1, 3, 5, 8, 9, 10):
                command += b"\xfe\xff"
            result = self.parse_anchor_commands(command + anchor)
            self.assertTrue(result["complete"])
            self.assertEqual(result["anchors"][0]["source_offset"], 150 + len(command))
        result = self.parse_anchor_commands(b"\x33\x02" + anchor)
        self.assertFalse(result["complete"])
        self.assertEqual(result["anchors"], [])
        for truncated in (b"\x33", b"\x33\x01\x00", anchor[:-1]):
            with self.assertRaises(ValueError):
                self.parse_anchor_commands(truncated)

    def test_unknown_script_opcode_stops_without_scanning(self):
        from room3d_scene import anchors

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "room.RDT"
            data = bytearray(160)
            struct.pack_into("<II", data, 0x60, 148, 160)
            struct.pack_into("<H", data, 148, 12)
            data[150] = 0xFE
            data[151] = 0x0C
            path.write_bytes(data)
            result = anchors(path)
            self.assertFalse(result["complete"])
            self.assertEqual(result["anchors"], [])
            self.assertEqual(result["stopped_offset"], 150)


if __name__ == "__main__":
    unittest.main()
