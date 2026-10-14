#!/usr/bin/env python3
"""Batch USA PC room surveys, provisional OBJ meshes and camera comparisons.

Dependencies: numpy, Pillow. Run from any directory. Inputs stay read-only.
Collision extrusion is a baseline, not automatic recovery of scenery.
"""

import argparse
import hashlib
import html
import itertools
import json
import math
import os
import struct
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
from pak_view import PakDecoder, rgb555
from PIL import Image, ImageDraw
from room3d_preview import render
from survey_room3d import survey, svg_plan


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def render_visibility(cameras):
    """Pixel coverage detects empty exports; it never establishes fidelity."""
    measured = [c for c in cameras if 'assisted_screen_coverage' in c]
    empty = [c['index'] for c in measured if c['assisted_screen_coverage'] == 0]
    return {'status': 'empty_geometry' if measured and len(empty) == len(cameras)
            else 'partially_empty' if empty else 'visible_geometry' if measured else 'not_measured',
            'measured_cameras': len(measured), 'empty_cameras': empty,
            'missing_cameras': [c['index'] for c in cameras if 'assisted_screen_coverage' not in c],
            'qualification': 'Visibility only; fidelity, hidden geometry and navigation remain unverified'}


def generate_mesh(room, path, recipe):
    """Only solid rectangles are extruded; other shape types remain unresolved."""
    height = float(recipe.get("wall_height", 6000))
    floor = float(recipe.get("floor_y", 0))
    if not 0 < height < 100000:
        raise ValueError("wall_height must be between 0 and 100000")
    bounds = room["boundaries"]
    if not bounds:
        raise ValueError("No boundaries available for baseline floor extent")
    lines = [
        "# PROVISIONAL collision extrusion; see reconstruction.json",
        "mtllib baseline.mtl",
    ]
    count = 0

    def face(points, material, group):
        nonlocal count
        lines.extend(["g " + group, "usemtl " + material])
        for p in points:
            lines.append("v " + " ".join(str(v) for v in p))
        lines.append("f " + " ".join(str(count + i + 1) for i in range(len(points))))
        count += len(points)

    def box(x0, z0, x1, z1, top, base, group):
        face(
            [(x1, base, z0), (x1, base, z1), (x0, base, z1), (x0, base, z0)],
            "top",
            group,
        )
        face([(x0, top, z0), (x0, top, z1), (x1, top, z1), (x1, top, z0)], "top", group)
        for a, b in [
            ((x0, z0), (x0, z1)),
            ((x0, z1), (x1, z1)),
            ((x1, z1), (x1, z0)),
            ((x1, z0), (x0, z0)),
        ]:
            face(
                [
                    (a[0], base, a[1]),
                    (a[0], top, a[1]),
                    (b[0], top, b[1]),
                    (b[0], base, b[1]),
                ],
                "wall",
                group,
            )

    def tiled_face(points, material, group):
        a, b, _c, d = np.asarray(points, float)
        rows = min(64, max(1, math.ceil(np.linalg.norm(b - a) / 1000)))
        cols = min(64, max(1, math.ceil(np.linalg.norm(d - a) / 1000)))
        for row in range(rows):
            for col in range(cols):
                p = a + (b - a) * row / rows + (d - a) * col / cols
                u, v = (b - a) / rows, (d - a) / cols
                face([p, p + u, p + u + v, p + v], material, group)

    x0 = min(b["xmin"] for b in bounds)
    x1 = max(b["xmax"] for b in bounds)
    z0 = min(b["zmin"] for b in bounds)
    z1 = max(b["zmax"] for b in bounds)
    floor_face = tiled_face if recipe.get("structural_surfaces") else face
    if recipe.get("floor_openings"):
        from room3d_floor import floor_patches

        for patch in floor_patches((x0, z0, x1, z1), recipe["floor_openings"]):
            face([(x, floor, z) for x, z in patch], "floor", "floor")
    else:
        floor_face(
            [(x0, floor, z0), (x0, floor, z1), (x1, floor, z1), (x1, floor, z0)],
            "floor",
            "floor",
        )
    unresolved = []
    cutaways = []
    for b in bounds:
        index = b["index"]
        if index in recipe.get("skip_boundaries", []):
            continue
        if b["type"] & 255 not in (1, 5):
            unresolved.append(index)
            continue
        estimated = (
            height if b["flags"] & 0x100 else float(recipe.get("obstacle_height", 1500))
        )
        top = floor - float(
            recipe.get("boundary_heights", {}).get(str(index), estimated)
        )
        for camera in room["cameras"]:
            x, y, z = camera["origin"]
            if (
                b["xmin"] <= x <= b["xmax"]
                and b["zmin"] <= z <= b["zmax"]
                and top <= y <= floor
            ):
                cutaways.append([camera["index"], f"boundary_{index}"])
        if index not in recipe.get("shared_wall_boundaries", []):
            box(
                b["xmin"],
                b["zmin"],
                b["xmax"],
                b["zmax"],
                top,
                floor,
                f"boundary_{index}",
            )
    from room3d_structure import door_openings

    for surface in door_openings(recipe.get("structural_surfaces", []),
                                 recipe.get("primitives", [])):
        # Shared planes remain continuous, but finite patches let the texture
        # baker select visible cameras without one giant off-screen triangle.
        tiled_face(surface["points"], surface["material"], surface["group"])
    for i, b in enumerate(recipe.get("boxes", [])):
        box(*b, f"authored_box_{i}")
    for i, item in enumerate(recipe.get("primitives", [])):
        if item.get("omit_geometry"):
            continue
        group = f"primitive_{i}"
        parent = item.get("hide_with")
        if parent:
            cutaways.extend(
                [camera, group]
                for camera, hidden in list(cutaways) + recipe.get("hide", [])
                if hidden == parent
            )
        if item.get("parts"):
            for part_index, vertices in enumerate(item["parts"]):
                material = "wall"
                profile = item.get("shape", {}).get("parameters", {}).get("profile")
                if (recipe.get("local_fixture_appearance", False)
                        and item["kind"] == "lamp"
                        and profile in ("wall_sconce", "wall_candelabra", "table_lamp", "ceiling_fan", "wall_linear", "ceiling_linear", "ceiling_panel")):
                    if profile in ("wall_candelabra", "ceiling_fan", "wall_linear", "ceiling_linear", "ceiling_panel"):
                        roles = item.get("part_roles", [])
                        if len(roles) != len(item["parts"]) or any(r not in ("shade", "support") for r in roles):
                            raise ValueError("Multilight fixture requires explicit shade/support roles")
                        material = "fixture_" + roles[part_index]
                    else:
                        material = "fixture_shade" if part_index == 2 else "fixture_support"
                for indices in [
                    (0, 1, 2, 3),
                    (7, 6, 5, 4),
                    (0, 4, 5, 1),
                    (1, 5, 6, 2),
                    (2, 6, 7, 3),
                    (3, 7, 4, 0),
                ]:
                    face([vertices[j] for j in indices], material, group)
        elif item["kind"] == "staircase":
            x0, z0, x1, z1, top, base = item["box"]
            steps = min(48, max(3, math.ceil((base - top) / 220)))
            for step in range(steps):
                level = (step + 1) if item["rise_positive"] else (steps - step)
                y = base - (base - top) * level / steps
                if item["axis"] == "x":
                    box(
                        x0 + (x1 - x0) * step / steps,
                        z0,
                        x0 + (x1 - x0) * (step + 1) / steps,
                        z1,
                        y,
                        base,
                        group,
                    )
                else:
                    box(
                        x0,
                        z0 + (z1 - z0) * step / steps,
                        x1,
                        z0 + (z1 - z0) * (step + 1) / steps,
                        y,
                        base,
                        group,
                    )
        elif item["kind"] == "barrel":
            x0, z0, x1, z1, top, base = item["box"]
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
            rings = []
            for y, scale in [(top, 0.85), ((top + base) / 2, 1), (base, 0.85)]:
                rings.append(
                    [
                        (
                            cx + (x1 - x0) / 2 * scale * math.cos(a * math.tau / 12),
                            y,
                            cz + (z1 - z0) / 2 * scale * math.sin(a * math.tau / 12),
                        )
                        for a in range(12)
                    ]
                )
            face(rings[0], "top", group)
            face(list(reversed(rings[-1])), "top", group)
            for upper, lower in itertools.pairwise(rings):
                for j in range(12):
                    k = (j + 1) % 12
                    face([upper[j], upper[k], lower[k], lower[j]], "wall", group)
        elif item["kind"] in ("door_panel", "chest_body", "ceiling", "lamp"):
            box(*item["box"], group)
        elif item["kind"] == "typewriter_table":
            x, z = item["center"]
            table_top = item.get("table_top", -1550)
            box(x - 650, z - 400, x + 650, z + 400, table_top, table_top + 120, group)
            for dx, dz in [(-520, -270), (-520, 270), (520, -270), (520, 270)]:
                box(
                    x + dx - 50,
                    z + dz - 50,
                    x + dx + 50,
                    z + dz + 50,
                    table_top + 120,
                    0,
                    group,
                )
            box(x - 360, z - 240, x + 360, z + 180, table_top - 240, table_top, group)
            box(
                x - 330,
                z + 100,
                x + 330,
                z + 210,
                table_top - 460,
                table_top - 240,
                group,
            )
            for row in range(3):
                for col in range(8):
                    kx = x - 280 + col * 80
                    kz = z - 190 + row * 65
                    box(
                        kx - 22,
                        kz - 20,
                        kx + 22,
                        kz + 20,
                        table_top - 270,
                        table_top - 240,
                        group,
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path.parent / "baseline.mtl").write_text(
        "newmtl floor\nKd 0.25 0.22 0.18\nnewmtl wall\nKd 0.34 0.42 0.50\nnewmtl top\nKd 0.20 0.24 0.28\n",
        encoding="utf-8",
    )
    path.with_suffix(".cameras").write_text(
        "".join(
            f"hide {camera} {group}\n"
            for camera, group in cutaways + recipe.get("hide", [])
        ),
        encoding="utf-8",
    )
    return {
        "wall_height": height,
        "floor_y": floor,
        "unresolved_boundaries": unresolved,
        "automatic_cutaways": cutaways,
        "assumptions": [
            "Rectangular collision volumes extruded as solid blocks.",
            "Estimated wall height; 1500-unit obstacle default when flag 0x100 is absent.",
            "Collision blocks containing the camera are hidden in that view.",
            "These flag/height and cutaway rules are heuristics, not recovered scenery.",
            "Flat floor at the stated estimated Y; no automatic stairs.",
            "Neutral diagnostic materials; no recovered textures or lighting.",
            "No furnishings, door apertures, scripts or dynamic objects reconstructed.",
        ],
        "recipe": recipe,
        "vertices": count,
    }


def background(path):
    data = PakDecoder(path.read_bytes()).run()
    # PC background TIMs store payload bytes (not payload+12) at +8.
    # Read the declared dimensions rather than dropping the last six pixels.
    if len(data) < 20 or struct.unpack_from("<II", data) != (16, 2):
        raise ValueError("Only direct-color PC background TIMs are supported")
    w, h = struct.unpack_from("<HH", data, 16)
    if len(data) != 20 + w * h * 2:
        raise ValueError("Unexpected background payload size")
    pixels = [rgb555(v[0])[:3] for v in struct.iter_unpack("<H", data[20:])]
    image = Image.new("RGB", (w, h))
    image.putdata(pixels)
    if image.size == (316, 236):
        # The current runtime's AddBackgroundQuad maps the complete texture
        # (UV 0..1) onto its 320x240 logical viewport, also for inset TIMs.
        # Normalize for camera projection without modifying the source asset.
        image = image.resize((320, 240), Image.Resampling.BILINEAR)
    elif image.size != (320, 240):
        raise ValueError(f"Unsupported original background size {image.size}")
    return image


def process(
    rdt,
    out,
    authored,
    recipes,
    inventory_only,
    mesh_only=False,
    reconstruct=False,
    detections_root=None,
    vision_python=None,
    runtime_exe=None,
    segmentation_root=None,
    depth_root=None,
    review_python=None,
    local_models=False,
    review_cache=None,
    defer_review=False,
):
    if rdt.read_bytes() == bytes(4):
        return {"room": rdt.stem, "status": "empty_slot", "cameras": []}
    room = survey(rdt)
    dest = out / rdt.stem.lower()
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "survey.json", room)
    (dest / "plan.svg").write_text(svg_plan(room), encoding="utf-8")
    stage = int(rdt.stem[4], 16)
    if not 1 <= stage <= 7:
        raise ValueError("Only original USA PC stages 1 through 7 are supported")
    background_stage = stage - 5 if stage > 5 else stage
    background_dir = next(
        p
        for p in rdt.parent.parent.iterdir()
        if p.name.lower() == f"stage{background_stage}"
    )
    files = {p.name.lower(): p for p in background_dir.iterdir() if p.is_file()}
    prefix = f"rc{background_stage}{rdt.stem[5:-1].lower()}"
    result = {
        "room": rdt.stem,
        "source_sha256": room["sha256"],
        "cameras": [],
        "status": "survey" if inventory_only else ("mesh" if mesh_only else "baseline"),
    }
    if not inventory_only:
        mesh = dest / (rdt.stem.lower() + ".obj")
        recipe = recipes.get(rdt.stem.lower(), {})
        result['detail_profile'] = recipe.get('_detail_profile', 'existing')
        write_json(dest / "reconstruction.json", generate_mesh(room, mesh, recipe))
    manual = (
        authored / rdt.parent.name / (rdt.stem.lower() + ".obj") if authored else None
    )
    assisted = None
    if reconstruct:
        from room3d_reconstruct import (
            BakeDepthCache,
            bake_projected,
            feature_cloud,
            fit_heights,
        )
        from room3d_scene import (
            anchors,
            fit_ceiling,
            fit_detected_primitives,
            primitives,
        )

        images = [
            background(files[prefix + f"{c['index']:X}".lower() + ".pak"])
            for c in room["cameras"]
        ]
        if vision_python:
            image_paths = []
            for camera, image in zip(room["cameras"], images, strict=True):
                image_path = dest / f"camera{camera['index']}-original.png"
                image.save(image_path)
                image_paths.append(image_path.resolve())
            detections_root = out / "detections"
            env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1")
            subprocess.run(
                [
                    str(vision_python),
                    str(Path(__file__).with_name("room3d_vision.py")),
                    *map(str, image_paths),
                    "--out",
                    str(detections_root / rdt.stem.lower()),
                ],
                check=True,
                env=env,
            )
        if local_models:
            segmentation_root = out / "segmentation"
            depth_root = out / "depth"
            env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1")
            subprocess.run(
                [
                    str(vision_python),
                    str(Path(__file__).with_name("room3d_segment.py")),
                    *map(str, image_paths),
                    "--detections",
                    str(detections_root / rdt.stem.lower()),
                    "--out",
                    str(segmentation_root / rdt.stem.lower()),
                ],
                check=True,
                env=env,
            )
            subprocess.run(
                [
                    str(vision_python),
                    str(Path(__file__).with_name("room3d_depth.py")),
                    "--survey",
                    str(dest / "survey.json"),
                    "--images",
                    str(dest),
                    "--out",
                    str(depth_root / rdt.stem.lower()),
                ],
                check=True,
                env=env,
            )
        cloud = feature_cloud(room, images)
        write_json(dest / "feature-cloud.json", cloud)
        fitted, proposals = fit_heights(room, cloud, recipe)
        detections = []
        if detections_root:
            for camera, image in zip(room["cameras"], images, strict=True):
                detection_file = (
                    detections_root
                    / rdt.stem.lower()
                    / f"camera{camera['index']}-original.json"
                )
                record = json.loads(detection_file.read_text())
                if (
                    record.get("rgb_sha256")
                    != hashlib.sha256(image.tobytes()).hexdigest()
                ):
                    raise ValueError(
                        f"Stale/mismatched detection input: {detection_file}"
                    )
                detections.append(record)
        script = anchors(rdt)
        fitted["primitives"] = recipe.get("primitives", primitives(room, script))
        visual_fits = (
            fit_detected_primitives(fitted["primitives"], room, detections)
            if detections and "primitives" not in recipe
            else []
        )
        ceiling = (
            fit_ceiling(room, detections)
            if detections
            else {"accepted": False, "reason": "Detector not run"}
        )
        if ceiling["accepted"]:
            fitted.setdefault("wall_height", ceiling["height"])
            ceiling["applied_height"] = fitted["wall_height"]
            ceiling["box"][4:6] = [-fitted["wall_height"] - 50, -fitted["wall_height"]]
            fitted["primitives"].append(
                {"kind": "ceiling", "box": ceiling["box"], "hide_with": "ceiling"}
            )
        write_json(dest / "scene-anchors.json", script)
        segments = None
        fixture_segments = None
        fixture_mask_root = None
        mask_root = segmentation_root / rdt.stem.lower() if segmentation_root else None
        if recipe.get('stair_prompt_recovery', False) and not (segmentation_root and depth_root):
            raise ValueError('stair_prompt_recovery requires segmentation and depth')
        if recipe.get('ceiling_fan_prompt_recovery', False) and not (
                segmentation_root and depth_root and recipe.get('parametric_details', False)
                and recipe.get('ceiling_fan_fixtures', False) and recipe.get('ceiling_fan_anchors', False)):
            raise ValueError('ceiling_fan_prompt_recovery requires masks/depth, parametric_details, ceiling_fan_fixtures and ceiling_fan_anchors')
        if recipe.get('visual_stair_proxy_cleanup', False) and not recipe.get('parametric_details', False):
            raise ValueError('visual_stair_proxy_cleanup requires parametric_details')
        if recipe.get('local_ladder_appearance', False) and not (segmentation_root and recipe.get('reconstructed_object_appearance', False)):
            raise ValueError('local_ladder_appearance requires segmentation and reconstructed_object_appearance')
        if recipe.get('object_wall_fitting', False) and not recipe.get('parametric_details', False):
            raise ValueError('object_wall_fitting requires parametric_details')
        if recipe.get('vertical_ladders', False) and not recipe.get('parametric_details', False):
            raise ValueError('vertical_ladders requires parametric_details')
        if segmentation_root:
            from room3d_segment import load_masks

            segments = load_masks(mask_root, room, images)
        if depth_root:
            from room3d_refine import load_depth, refine

            depth_views = load_depth(
                depth_root / rdt.stem.lower(), dest / "survey.json", room, images
            )
            depth_record = json.loads((depth_root / rdt.stem.lower() / 'depth.json').read_text())
            result['depth_scale'] = {'scale': depth_record.get('scale', 'not_recorded'),
                                     'evidence': depth_record.get('scale_evidence')}
            if recipe.get('stair_prompt_recovery', False):
                recovery_root = out / 'stair-recovery' / rdt.stem.lower() / uuid.uuid4().hex[:8]
                for camera, image in zip(room['cameras'], images, strict=True):
                    image.save(dest / f"camera{camera['index']}-original.png")
                subprocess.run([
                    str(vision_python or sys.executable),
                    str(Path(__file__).with_name('room3d_stair_recovery.py')),
                    '--survey', str(dest / 'survey.json'), '--images', str(dest),
                    '--segmentation', str(mask_root),
                    '--depth', str(depth_root / rdt.stem.lower()), '--out', str(recovery_root),
                ], check=True, env=dict(os.environ, HF_HUB_DISABLE_TELEMETRY='1',
                                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
                mask_root = recovery_root / 'segmentation'
                from room3d_segment import load_masks

                segments = load_masks(mask_root, room, images)
                result['stair_prompt_recovery'] = json.loads((recovery_root / 'recovery.json').read_text())
                result['stair_prompt_recovery']['segmentation_root'] = str(mask_root)
            if recipe.get('ceiling_fan_prompt_recovery', False):
                recovery_root = out/'fixture-recovery'/rdt.stem.lower()/uuid.uuid4().hex[:8]
                for camera, image in zip(room['cameras'], images, strict=True):
                    image.save(dest/f"camera{camera['index']}-original.png")
                command = [str(vision_python or sys.executable),
                           str(Path(__file__).with_name('room3d_fixture_recovery.py')),
                           '--survey', str(dest/'survey.json'), '--images', str(dest),
                           '--segmentation', str(mask_root),
                           '--depth', str(depth_root/rdt.stem.lower()), '--out', str(recovery_root)]
                if recipe.get('ceiling_fan_recovery_cache'):
                    command.extend(['--proposals', str(Path(recipe['ceiling_fan_recovery_cache'])/rdt.stem.lower())])
                subprocess.run(command, check=True, env=dict(os.environ, HF_HUB_DISABLE_TELEMETRY='1',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
                fixture_mask_root = recovery_root/'segmentation'
                fixture_segments = load_masks(fixture_mask_root, room, images)
                result['ceiling_fan_prompt_recovery'] = json.loads((recovery_root/'recovery.json').read_text())
                result['ceiling_fan_prompt_recovery']['segmentation_root'] = str(fixture_mask_root)
            from room3d_objects import object_proposals

            objects = object_proposals(
                room, segments, depth_views, fitted["primitives"]
            )
            write_json(dest / "object-proposals.json", objects)
            fitted, correction = refine(
                room,
                fitted,
                depth_views,
                dest / "corrections",
                generate_mesh,
                objects,
                segments,
            )
            if recipe.get("consolidate_structure", True):
                from room3d_structure import consolidate

                write_json(dest / "before-structure-recipe.json", fitted)
                fitted, structural_report = consolidate(
                    room, fitted, depth_views, segments
                )
                write_json(dest / "structure.json", structural_report)
                result["structure"] = structural_report
        if recipe.get('visual_surface_recovery', False):
            if segments is None or not depth_root:
                raise ValueError('Visual surface recovery requires segmentation and depth')
            from room3d_visual_surfaces import recover

            fitted, surface_report = recover(room, fitted, depth_views, segments,
                                            dest/'visual-surfaces', generate_mesh)
            write_json(dest/'visual-surfaces/results.json', surface_report)
            result['visual_surface_recovery'] = surface_report
        if recipe.get("wall_attachment_proposals", False):
            from room3d_wall_attachment import propose_attachments

            # Evaluate alternatives separately; do not silently promote an unvalidated topology.
            attachment_cloud = feature_cloud(room, images, use_collision_bounds=False)
            alternatives, attachment_report = propose_attachments(room, fitted, attachment_cloud)
            attachment_dir = dest / "wall-attachments"
            attachment_dir.mkdir(exist_ok=True)
            write_json(attachment_dir / "feature-cloud.json", attachment_cloud)
            write_json(attachment_dir / "report.json", attachment_report)
            for alternative in alternatives:
                write_json(attachment_dir / f"{alternative['name']}.json", alternative["recipe"])
                generate_mesh(room, attachment_dir / f"{alternative['name']}.obj", alternative["recipe"])
            if recipe.get("wall_attachment_evaluation", False) and alternatives:
                from room3d_attachment_evaluation import evaluate_attachments

                attachment_report["render_evaluation"] = evaluate_attachments(
                    room, fitted, alternatives, images, segments, depth_views if depth_root else None,
                    attachment_dir / "evaluation", generate_mesh,
                )
                write_json(attachment_dir / "report.json", attachment_report)
            result["wall_attachment_proposals"] = attachment_report
        if recipe.get("parametric_details", False):
            if segments is None or not depth_root:
                raise ValueError("Parametric details require segmentation and depth")
            from room3d_shape_fit import fit_objects

            if recipe.get("creative_fallback", False):
                from room3d_levels import infer_levels

                level_report = infer_levels(room, segments)
                write_json(dest / "level-hypotheses.json", level_report)
                result["level_hypotheses"] = level_report
                fitted["level_hypotheses"] = level_report
            detail_segments = fixture_segments if fixture_segments is not None else segments
            refinement = recipe.get("stair_refinement", False)
            if refinement:
                from room3d_stair_edges import (
                    load_refinements,
                    normalize_horizontal_axes,
                )

                horizontal_axes = recipe.get('tread_horizontal_axes')
                axis_args = []
                if horizontal_axes is not None:
                    horizontal_axes = normalize_horizontal_axes(horizontal_axes).tolist()
                    for axis in horizontal_axes:
                        axis_args.extend(['--horizontal-axis', *map(str, axis)])

                if refinement is True:
                    refinement_root = dest / "stair-refinement"
                    for camera, image in zip(room["cameras"], images, strict=True):
                        image.save(dest / f"camera{camera['index']}-original.png")
                    subprocess.run(
                        [
                            str(vision_python or sys.executable),
                            str(Path(__file__).with_name("room3d_stair_edges.py")),
                            "--survey",
                            str(dest / "survey.json"),
                            "--images",
                            str(dest),
                            "--depth",
                            str(depth_root / rdt.stem.lower()),
                            "--segments",
                            str(fixture_mask_root or mask_root),
                            "--out",
                            str(refinement_root),
                            *axis_args,
                        ],
                        check=True,
                        env=dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1"),
                    )
                else:
                    refinement_root = Path(refinement)
                detail_segments = load_refinements(
                    refinement_root, room, images, detail_segments, depth_views,
                    **({'horizontal_axes': horizontal_axes} if horizontal_axes is not None else {}),
                )
            write_json(dest / "before-details-recipe.json", fitted)
            fitted, detail_report = fit_objects(
                room,
                fitted,
                detail_segments,
                depth_views,
                dest / "details",
                generate_mesh,
                images=images,
            )
            write_json(dest / "after-details-recipe.json", fitted)
            result["parametric_details"] = detail_report
            if recipe.get("visual_stair_proxy_cleanup", False):
                from room3d_proxy_cleanup import fit as clean_visual_proxies

                reserved = {c for obj in detail_report["objects"]
                            for c in obj.get("heldout_cameras", [])}
                fitted, proxy_report = clean_visual_proxies(
                    room, fitted, detail_segments, depth_views,
                    dest / "visual-proxy-cleanup", generate_mesh,
                    heldout_cameras=reserved,
                )
                result["visual_proxy_cleanup"] = proxy_report
                detail_report["visual_proxy_cleanup"] = proxy_report
                if proxy_report["accepted"]:
                    for obj in detail_report["objects"]:
                        obj["visibility_evidence_scope"] = "before_visual_proxy_cleanup"
                    with (dest / "details/index.html").open("a") as page:
                        page.write('<p>Detail visibility measurements above precede visual collision '
                                   'proxy cleanup. See the final batch renders and cleanup report.</p>')
                write_json(dest / "details/results.json", detail_report)
                write_json(dest / "after-proxy-cleanup-recipe.json", fitted)
        if recipe.get('vertical_ladders', False):
            from room3d_vertical_ladders import fit_vertical_ladders

            fitted, ladder_report = fit_vertical_ladders(
                room, fitted, detail_segments, depth_views,
                dest/'vertical-ladders', generate_mesh)
            result['vertical_ladders'] = ladder_report
            write_json(dest/'after-ladders-recipe.json', fitted)
        if recipe.get('object_wall_fitting', False):
            from room3d_object_wall_fit import fit as fit_object_walls

            fitted, wall_report = fit_object_walls(
                room, fitted, detail_segments, depth_views,
                dest/'object-wall-fitting', generate_mesh)
            result['object_wall_fitting'] = wall_report
            write_json(dest/'after-object-walls-recipe.json', fitted)
        if fixture_segments is not None:
            from room3d_fixture_recovery import appearance_segments

            segments, appearance_report = appearance_segments(room, segments, fixture_segments, fitted)
            result['fixture_recovery_appearance'] = appearance_report
            write_json(dest/'fixture-recovery-appearance.json', appearance_report)
        fitted_mesh = dest / "fitted.obj"
        generate_mesh(room, fitted_mesh, fitted)
        if recipe.get('final_detail_metrics', False):
            if segments is None or not depth_root:
                raise ValueError('Final detail metrics require segmentation and depth')
            from room3d_final_details import measure

            final_metrics = measure(room, fitted, segments, depth_views, fitted_mesh)
            write_json(dest/'final-details.json', final_metrics)
            result['final_detail_metrics'] = final_metrics
        assisted = dest / "assisted.obj"
        kinds = {
            f"primitive_{i}": item["kind"]
            for i, item in enumerate(fitted["primitives"])
        }
        kinds.update(
            {
                f"boundary_{b['index']}": "wall"
                for b in room["boundaries"]
                if b["flags"] & 0x100
            }
        )
        kinds["floor"] = "floor"
        kinds.update({s['group']: s['material'] for s in fitted.get('structural_surfaces', [])
                      if s['material'] in ('wall', 'floor', 'ceiling')})
        interpreted_groups = {
            f"primitive_{i}"
            for i, item in enumerate(fitted["primitives"])
            if item.get("interpretation")
        }
        fixture_observations = {
            f"primitive_{i}": item.get("evidence", {}).get("shared_observations", [])
            for i, item in enumerate(fitted["primitives"])
            if item.get("interpretation") and item.get("kind") == "lamp"
        } if recipe.get("local_fixture_appearance", False) else None
        from room3d_texture_ownership import reconstructed_sources

        reconstructed_observations = reconstructed_sources(fitted)
        bake_depth_cache = BakeDepthCache()
        bake = bake_projected(
            fitted_mesh,
            room,
            images,
            assisted,
            segments=segments,
            group_kinds=kinds,
            interpreted_groups=interpreted_groups,
            fixture_observations=fixture_observations,
            reconstructed_observations=reconstructed_observations,
            local_ladder_appearance=recipe.get("local_ladder_appearance", False),
            depth_cache=bake_depth_cache,
            foreground_fixture_guard=recipe.get("foreground_fixture_guard", False),
            fixture_core_appearance=recipe.get("fixture_core_appearance", False),
            observed_depths=depth_views
            if recipe.get("parametric_details", False)
            else None,
        )
        write_json(
            dest / "assisted.json",
            {
                "depth_correction": correction if depth_root else {"status": "not_run"},
                "height_proposals": proposals,
                "bake": bake,
                "primitive_proposals": fitted["primitives"],
                "geometry_status": "provisional",
                "depth_scale": result.get('depth_scale', {'scale': 'not_used'}),
                "semantic_object_recognition": "local_detector_proposals"
                if detections
                else "not_run",
                "visual_fits": visual_fits,
                "ceiling_fit": ceiling,
                "runtime_interactions": "See result.json for optional runtime evidence",
                "fitted_recipe": fitted,
            },
        )
        result["status"] = "image_assisted"
        result["accepted_feature_points"] = len(cloud["points"])
    rows = []
    if assisted:
        rows.append(
            "<p>Excluded-view comparisons withhold that camera from texture projection only. "
            "Room geometry may use all input views; these are not held-out geometry tests. "
            "Per-object reserved-camera evidence, when available, is reported in detail trials.</p>"
        )
    for camera in room["cameras"]:
        name = prefix + f"{camera['index']:X}".lower() + ".pak"
        source = files.get(name)
        record = {
            "index": camera["index"],
            "background": f"{background_dir.name}/{name}",
            "available": source is not None,
        }
        result["cameras"].append(record)
        if inventory_only or mesh_only:
            continue
        if source is None:
            record["error"] = "Missing original background"
            continue
        original = background(source)
        auto, mask = render(mesh, camera)
        record["background_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        record["baseline_screen_coverage"] = round(float(mask.mean()), 4)
        images = [("Original", original), ("Automatic baseline", auto)]
        if manual and manual.exists():
            image, _ = render(manual, camera)
            images.append(("Authored mesh (offline)", image))
        if assisted and detections:
            import shutil

            detected_image = (
                detections_root
                / rdt.stem.lower()
                / f"camera{camera['index']}-original-detections.png"
            )
            if detected_image.exists():
                local_detection = dest / f"camera{camera['index']}-detections.png"
                shutil.copy2(detected_image, local_detection)
                rows.append(
                    f'<details><summary>Camera {camera["index"]}: local detector proposals</summary><img src="{local_detection.name}"></details>'
                )
        if assisted:
            image, assisted_mask = render(assisted, camera)
            record['assisted_screen_coverage'] = float(assisted_mask.mean())
            images.append(("Image-assisted (all views)", image))
            heldout = dest / f"heldout{camera['index']}.obj"
            bake_projected(
                fitted_mesh,
                room,
                [
                    background(files[prefix + f"{c['index']:X}".lower() + ".pak"])
                    for c in room["cameras"]
                ],
                heldout,
                exclude_camera=camera["index"],
                segments=segments,
                group_kinds=kinds,
                interpreted_groups=interpreted_groups,
                fixture_observations=fixture_observations,
                reconstructed_observations=reconstructed_observations,
                local_ladder_appearance=recipe.get("local_ladder_appearance", False),
                depth_cache=bake_depth_cache,
                foreground_fixture_guard=recipe.get("foreground_fixture_guard", False),
                fixture_core_appearance=recipe.get("fixture_core_appearance", False),
                observed_depths=depth_views
                if recipe.get("parametric_details", False)
                else None,
            )
            image, _ = render(heldout, camera)
            images.append(("Excluded-view appearance only", image))
        sheet = Image.new("RGB", (320 * max(3, len(images)), 266), (22, 25, 30))
        draw = ImageDraw.Draw(sheet)
        for j, (label, image) in enumerate(images):
            sheet.paste(image, (320 * j, 26))
            draw.text((320 * j + 8, 7), label, fill="white")
        if len(images) == 2:
            draw.text((648, 110), "No authored mesh available", fill="white")
        stem = f"camera{camera['index']}"
        sheet.save(dest / (stem + ".png"))
        original.save(dest / (stem + "-original.png"))
        auto.save(dest / (stem + "-auto.png"))
        if assisted and defer_review:
            reviewed_image, _ = render(assisted, camera)
            reviewed_image.save(dest / (stem + "-assisted.png"))
        if assisted and review_python and not defer_review:
            reviewed_image, _ = render(assisted, camera)
            reviewed_path = dest / (stem + "-assisted.png")
            reviewed_image.save(reviewed_path)
            review_path = dest / (stem + "-review.json")
            from room3d_review import reuse_review

            cached = (
                review_cache / rdt.stem.lower() / review_path.name
                if review_cache
                else None
            )
            reused = reuse_review(
                cached, dest / (stem + "-original.png"), reviewed_path, review_path
            )
            if not reused:
                with (dest / (stem + "-review.log")).open("w") as log:
                    subprocess.run(
                        [
                            str(review_python),
                            str(Path(__file__).with_name("room3d_review.py")),
                            str(dest / (stem + "-original.png")),
                            str(reviewed_path),
                            "--out",
                            str(review_path),
                            "--size",
                            "4b",
                        ],
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1"),
                    )
            review = json.loads(review_path.read_text())
            rows.append(
                f'<details><summary>Local visual review: camera {camera["index"]} (unverified proposals)</summary><pre>{html.escape(review["response"])}</pre><a href="{review_path.name}">Model and input evidence</a></details>'
            )
        rows.append(
            f'<h3>Camera {camera["index"]}</h3><img class="comparison" src="{stem}.png"><details><summary>Overlay: original / automatic</summary><div class="overlay"><img src="{stem}-original.png"><img class="auto" src="{stem}-auto.png"></div><input type="range" min="0" max="100" value="50" aria-label="Automatic geometry opacity" oninput="this.previousElementSibling.lastElementChild.style.opacity=this.value/100"></details>'
        )
    if assisted and defer_review:
        from room3d_review_queue import enqueue

        enqueue(dest, room["cameras"])
        result["visual_review"] = {"status": "pending", "queue": "review-queue.json"}
        rows.append('<p>Local visual review pending: <a href="review-queue.json">queued inputs</a>. '
                    'Export and fitting have finished; geometry quality remains unverified.</p>')
    if assisted:
        result['render_visibility'] = render_visibility(result['cameras'])
        if result['render_visibility']['empty_cameras']:
            rows.insert(0, '<p>WARNING: no assisted geometry is visible in camera(s) '
                        + ', '.join(map(str, result['render_visibility']['empty_cameras']))
                        + '. This export needs geometry correction.</p>')
        from room3d_novel_views import inspect as inspect_new_views

        result["new_viewpoints"] = inspect_new_views(assisted, room, dest / "new-views")
        rows.append(
            '<p><a href="new-views/index.html">Inferred surfaces from new viewpoints</a></p>'
        )
    if assisted and depth_root:
        rows.append(
            '<p><a href="corrections/index.html">Depth refinement: before, after and rejected proposals</a></p>'
        )
    if assisted and runtime_exe:
        if rdt.stem.lower() in ("room1180", "room10b0"):
            from check_room3d_runtime import run as runtime_probe

            runtime_folder = dest / ("runtime-" + uuid.uuid4().hex[:8])
            result["runtime"] = runtime_probe(
                runtime_exe,
                rdt.parent.parent.parent,
                assisted,
                runtime_folder,
                target_room=int(rdt.stem[5:7], 16),
            )
            result["runtime_report"] = runtime_folder.name + "/result.json"
            rows.append(
                f'<p>Wine interaction test: {"PASS" if result["runtime"]["passed"] else "FAIL"} — <a href="{result["runtime_report"]}">evidence</a></p>'
            )
        else:
            result["runtime"] = {
                "passed": False,
                "status": "not_run",
                "reason": "No validated interaction scenario for this room",
            }
    write_json(dest / "result.json", result)
    if assisted:
        rows.insert(
            0,
            '<p><a href="assisted.json">Geometry and appearance evidence</a> · <a href="feature-cloud.json">Triangulated points</a> · <a href="scene-anchors.json">Original script anchors</a></p>',
        )
    page = (
        f'<h1>{html.escape(rdt.stem)}</h1><p>Original vs automatic collision extrusion vs existing authored geometry. Offline previews exclude actors, dynamic objects and game postprocessing. Coverage is NOT a fidelity score.</p><p><a href="survey.json">Measured data</a> · <a href="plan.svg">Plan</a> · <a href="result.json">Camera availability</a></p>'
        + (
            '<p><a href="details/index.html">Parametric details: accepted and rejected candidates</a></p>'
            if (dest / "details/index.html").exists()
            else ""
        )
        + (
            '<p><a href="structure.json">Shared structure and contradiction evidence</a></p>'
            if (dest / "structure.json").exists()
            else ""
        )
        + "".join(rows)
    )
    (dest / "index.html").write_text(html_page(page), encoding="utf-8")
    return result


def html_page(body):
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Room 3D batch comparison</title><style>body{background:#151920;color:#eee;font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}a{color:#8acbff}.comparison{width:100%;image-rendering:pixelated}.overlay{position:relative;width:640px;max-width:100%;aspect-ratio:4/3}.overlay img{position:absolute;width:100%;image-rendering:pixelated}.auto{opacity:.5}input{width:320px}li{margin:12px}</style><body>'
        + body
        + "</body></html>"
    )


def process_one(path, args, recipes, batch_key):
    from room3d_checkpoint import restore, save, signature

    room_name = path.stem.lower()
    checkpoint = args.out / "checkpoints" / (room_name + ".json")
    room_dir = args.out / room_name
    key = signature({"batch": batch_key, "room": room_name,
                     "recipe": recipes.get(room_name, {})})
    cached = restore(checkpoint, key, room_dir) if args.resume else None
    if cached is not None:
        print(f"{path.stem}: resumed ({cached['status']})", flush=True)
        return apply_overrides(args, room_name, cached)
    checkpoint.unlink(missing_ok=True)
    try:
        result = process(
            path,
            args.out,
            args.authored,
            recipes,
            args.survey_only,
            args.mesh_only,
            args.reconstruct,
            args.detections,
            args.vision_python if args.local_vision else None,
            args.runtime_exe,
            args.segmentation,
            args.depth,
            args.review_python,
            args.local_models,
            args.review_cache,
            args.defer_review,
        )
    except (
        ValueError,
        OSError,
        IndexError,
        KeyError,
        struct.error,
        StopIteration,
        subprocess.CalledProcessError,
    ) as error:
        result = {"room": path.stem, "status": "error", "error": str(error)}
    save(checkpoint, key, result, room_dir)
    print(f"{path.stem}: {result['status']}", flush=True)
    return apply_overrides(args, room_name, result)


def apply_overrides(args, room_name, result):
    """Replay portable corrections into a distinct corrected output after the automatic export.

    Corrections never touch the automatic room directory or its checkpoint, and
    an unresolved object or source mismatch is reported instead of guessed.
    """
    if not getattr(args, "overrides", None) or result.get("status") != "image_assisted":
        return result
    from room3d_overrides import replay_for_batch

    result = dict(result)
    result["corrections"] = replay_for_batch(args.overrides, args.out, room_name,
                                             args.segmentation, args.depth)
    print(f"{room_name.upper()}: corrections {result['corrections']['status']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, required=True, help="USA asset directory"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="Independent room processes (default: 1)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed rooms only when inputs, code, recipe and outputs match",
    )
    parser.add_argument(
        "--rooms", nargs="+", help="RDT stems, e.g. ROOM1180 ROOM10B0; omit for all"
    )
    parser.add_argument(
        "--authored",
        type=Path,
        help="Existing OBJ mod directory containing Stage1 etc.",
    )
    parser.add_argument(
        "--recipes", type=Path, help="JSON object keyed by lower-case RDT stem"
    )
    from room3d_profiles import PROFILES

    parser.add_argument('--detail-profile', choices=sorted(PROFILES), default='existing',
                        help='creative-v2 combines experimental detail improvements; default preserves existing recipes')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--mesh-only",
        action="store_true",
        help="Export all meshes without rendering comparisons",
    )
    modes.add_argument(
        "--survey-only",
        action="store_true",
        help="Inventory every room without image decoding/rendering",
    )
    modes.add_argument(
        "--reconstruct",
        action="store_true",
        help="Experimental multiview fitting and appearance transfer",
    )
    parser.add_argument(
        "--review-cache",
        type=Path,
        help="Reuse matching model/prompt/image reviews; mismatches run inference",
    )
    parser.add_argument(
        "--defer-review", action="store_true",
        help="Queue visual reviews for a separate pass; keep geometry fitting enabled",
    )
    parser.add_argument(
        "--review-python",
        type=Path,
        help="Run the local 4B visual reviewer for every reconstructed camera",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        help="Local DA3 output root for reversible geometry fitting",
    )
    parser.add_argument(
        "--segmentation", type=Path, help="Verified local SAM mask root"
    )
    parser.add_argument(
        "--detections",
        type=Path,
        help="Local detector output root, verified against image pixel hashes",
    )
    parser.add_argument(
        "--local-vision",
        action="store_true",
        help="Run the local detector before reconstruction",
    )
    parser.add_argument(
        "--vision-python",
        type=Path,
        default=Path(sys.executable),
        help="Python with local detector dependencies",
    )
    parser.add_argument(
        "--runtime-exe",
        type=Path,
        help="Opt-in Wine probes: ROOM1180 chest and ROOM10B0 navigation",
    )
    parser.add_argument(
        "--local-models",
        action="store_true",
        help="Run DINO, SAM, DA3 and Qwen locally using --vision-python",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        help="Directory of portable override files (ROOM1180.json); each is replayed "
             "into <out>/corrected/<room> after the automatic export",
    )
    args = parser.parse_args()
    if args.overrides and not args.reconstruct:
        parser.error("--overrides requires --reconstruct")
    if args.overrides and not args.overrides.is_dir():
        parser.error("--overrides must be an existing directory")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.jobs > 1 and args.runtime_exe:
        parser.error("Interactive runtime probes require --jobs 1")
    if args.local_models:
        if not args.reconstruct:
            parser.error("--local-models requires --reconstruct")
        if args.detections or args.segmentation or args.depth:
            parser.error(
                "--local-models generates model artifacts; omit cached model roots"
            )
        args.local_vision = True
        args.review_python = args.vision_python
    if (args.local_vision or args.runtime_exe) and not args.reconstruct:
        parser.error("--local-vision and --runtime-exe require --reconstruct")
    if args.detections and args.local_vision:
        parser.error("Choose --detections or --local-vision, not both")
    if args.review_cache and not args.review_python:
        parser.error("--review-cache requires --review-python or --local-models")
    if args.review_python and not args.reconstruct:
        parser.error("--review-python requires --reconstruct")
    if args.defer_review and not args.reconstruct:
        parser.error("--defer-review requires --reconstruct")
    if args.depth and not args.reconstruct:
        parser.error("--depth requires --reconstruct")
    if args.segmentation and not args.reconstruct:
        parser.error("--segmentation requires --reconstruct")
    if args.detections and not args.reconstruct:
        parser.error("--detections requires --reconstruct")
    if args.out.resolve().is_relative_to(args.assets.resolve()):
        parser.error("Output must be outside the original asset directory")
    files = sorted(p for p in args.assets.rglob("*") if p.suffix.lower() == ".rdt")
    if args.rooms:
        wanted = {s.lower().removesuffix(".rdt") for s in args.rooms}
        files = [p for p in files if p.stem.lower() in wanted]
        missing = wanted - {p.stem.lower() for p in files}
        if missing:
            parser.error("Rooms not found: " + ", ".join(sorted(missing)))
    if not files:
        parser.error("No RDT files found")
    if args.detail_profile != 'existing' and not args.reconstruct:
        parser.error('--detail-profile creative-v2 requires --reconstruct')
    if args.detail_profile != 'existing' and not args.local_models and not (args.segmentation and args.depth):
        parser.error('--detail-profile creative-v2 requires cached --segmentation and --depth, or --local-models')
    recipes = json.loads(args.recipes.read_text()) if args.recipes else {}
    from room3d_profiles import apply_profile

    try:
        recipes = apply_profile(args.detail_profile, recipes, [p.stem.lower() for p in files])
    except ValueError as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    from room3d_checkpoint import run_signature

    print("Checking batch input fingerprints...", flush=True)
    batch_key = run_signature(args)
    if args.jobs == 1:
        results = [process_one(path, args, recipes, batch_key) for path in files]
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context

        completed = {}
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(process_one, path, args, recipes, batch_key): path for path in files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    completed[path] = future.result()
                except (RuntimeError, TypeError, MemoryError, OSError, ValueError,
                        IndexError, KeyError, struct.error, StopIteration,
                        subprocess.CalledProcessError) as error:
                    completed[path] = {"room": path.stem, "status": "error",
                                       "error": f"Worker failed: {type(error).__name__}: {error}"}
                    print(f"{path.stem}: error (worker)", flush=True)
        results = [completed[path] for path in files]
    write_json(
        args.out / "manifest.json",
        {
            "schema": 1,
            "mode": "survey"
            if args.survey_only
            else (
                "mesh"
                if args.mesh_only
                else ("image_assisted" if args.reconstruct else "baseline")
            ),
            "results": results,
        },
    )
    links = []
    failures = 0
    for result in results:
        failed = result["status"] == "error" or any(
            "error" in c for c in result.get("cameras", [])
        )
        runtime = result.get("runtime", {})
        failed |= (
            runtime.get("status") != "not_run"
            and bool(runtime)
            and not runtime.get("passed", False)
        )
        failures += int(failed)
        room = html.escape(result["room"])
        if result["status"] == "empty_slot":
            links.append(f"<li>{room}: empty RDT slot (four zero bytes)</li>")
            continue
        corrections = result.get("corrections", {})
        corrected = ""
        if corrections.get("status") in ("applied", "partial"):
            corrected = (f' · <a href="{html.escape(corrections["page"], quote=True)}">corrected</a>'
                         f' ({corrections["status"]}, {corrections["steps"]} steps)')
        elif corrections.get("status") in ("conflict", "error"):
            corrected = f' · corrections {corrections["status"]}: ' + html.escape(
                corrections.get("error") or "; ".join(c.get("message", "") for c in corrections.get("conflicts", [])))
        links.append(
            f"<li>{room}: {html.escape(result['error'])}</li>"
            if result["status"] == "error"
            else f'<li><a href="{room.lower()}/index.html">{room}</a> — {len(result["cameras"])} cameras'
            + (" (incomplete)" if failed else "")
            + corrected
            + "</li>"
        )
    (args.out / "index.html").write_text(
        html_page(
            "<h1>Room reconstruction batch</h1><p>Generated geometry is a provisional baseline. Original gameplay is untouched.</p><ul>"
            + "".join(links)
            + "</ul>"
        ),
        encoding="utf-8",
    )
    print(
        f"{len(results)} rooms; {failures} failed or incomplete. Report: {args.out / 'index.html'}"
    )
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
