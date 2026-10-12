"""Image-assisted reconstruction experiments with explicit evidence and uncertainty.

Known RDT cameras constrain feature matching. Reprojection tests cannot establish
semantic identity: an accepted point is NOT proof of a door, lamp or chair.
"""

import hashlib
import itertools
import json

import cv2
import numpy as np
from room3d_preview import camera_basis, load_obj, render


def projection(camera):
    rotation = camera_basis(camera)
    intrinsic = np.array(
        [[camera["fov"], 0, 160], [0, camera["fov"], 120], [0, 0, 1.0]]
    )
    return intrinsic @ np.column_stack((rotation, -rotation @ camera["origin"]))


def project(points, camera, basis=None):
    points = np.asarray(points, dtype=float)
    view = (points - camera["origin"]) @ (camera_basis(camera) if basis is None else basis).T
    uv = view[:, :2] / np.maximum(view[:, 2:3], 1e-9) * camera["fov"] + [160, 120]
    return uv, view[:, 2]


def triangulate(camera_a, camera_b, pixels_a, pixels_b):
    """Reject poor depth, ray angle and reprojection before accepting a match."""
    if not len(pixels_a):
        return np.empty((0, 3)), np.empty(0, dtype=bool), np.empty(0)
    a, b = np.asarray(pixels_a, float), np.asarray(pixels_b, float)
    raw = cv2.triangulatePoints(projection(camera_a), projection(camera_b), a.T, b.T).T
    valid_w = np.abs(raw[:, 3]) > 1e-10
    points = raw[:, :3] / np.where(valid_w, raw[:, 3], 1)[:, None]
    pa, za = project(points, camera_a)
    pb, zb = project(points, camera_b)
    error = np.maximum(np.linalg.norm(pa - a, axis=1), np.linalg.norm(pb - b, axis=1))
    va = points - camera_a["origin"]
    vb = points - camera_b["origin"]
    cosine = np.sum(va * vb, axis=1) / np.maximum(
        np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    valid = (
        valid_w
        & np.isfinite(points).all(axis=1)
        & (za > camera_a["fov"] * 2)
        & (zb > camera_b["fov"] * 2)
    )
    valid &= (error <= 1.5) & (angle >= 3) & (angle <= 150)
    return points, valid, error


def unique_correspondences(records, radius=2.0):
    """Avoid counting multiple SIFT orientations at the same image locations.

    Keep the lowest reprojection error for each pair-local pixel neighborhood.
    This reduces duplicated evidence; it does not establish physical identity.
    """
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Expected a finite nonnegative pixel radius")
    kept = []
    for record in sorted(records, key=lambda item: item['reprojection_error']):
        pixels = np.asarray(record['pixels'], float)
        duplicate = any(
            prior['views'] == record['views']
            and np.all(np.linalg.norm(pixels - prior['pixels'], axis=1) <= radius)
            for prior in kept
        )
        if not duplicate:
            kept.append(record)
    return kept


def feature_cloud(room, images, *, use_collision_bounds=True):
    """Match original-camera features; collision bounds are an optional prior."""
    detector = cv2.SIFT_create(nfeatures=1600, contrastThreshold=0.025)
    features = [
        detector.detectAndCompute(np.asarray(im.convert("L")), None) for im in images
    ]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    evidence = []
    pairs = []
    bounds = room["boundaries"]
    if use_collision_bounds and not bounds:
        raise ValueError("Collision-bounded matching requires room boundaries")
    xlo = min((b["xmin"] for b in bounds), default=0) - 1000
    xhi = max((b["xmax"] for b in bounds), default=0) + 1000
    zlo = min((b["zmin"] for b in bounds), default=0) - 1000
    zhi = max((b["zmax"] for b in bounds), default=0) + 1000
    for i, j in itertools.combinations(range(len(images)), 2):
        ka, da = features[i]
        kb, db = features[j]
        if da is None or db is None or min(len(da), len(db)) < 2:
            pairs.append({"cameras": [i, j], "matches": 0, "accepted": 0})
            continue
        forward = [
            m[0]
            for m in matcher.knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance
        ]
        backward = {
            m[0].queryIdx: m[0].trainIdx
            for m in matcher.knnMatch(db, da, k=2)
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance
        }
        matches = [m for m in forward if backward.get(m.trainIdx) == m.queryIdx]
        pa = [ka[m.queryIdx].pt for m in matches]
        pb = [kb[m.trainIdx].pt for m in matches]
        xyz, valid, error = triangulate(room["cameras"][i], room["cameras"][j], pa, pb)
        geometric_accepted = int(valid.sum())
        if use_collision_bounds and len(xyz):
            valid &= (
                (xyz[:, 0] >= xlo)
                & (xyz[:, 0] <= xhi)
                & (xyz[:, 2] >= zlo)
                & (xyz[:, 2] <= zhi)
                & (xyz[:, 1] >= -18000)
                & (xyz[:, 1] <= 1000)
            )
        pairs.append(
            {"cameras": [i, j], "matches": len(matches), "accepted": int(valid.sum()),
             "geometric_accepted": geometric_accepted,
             "bounds_rejected": geometric_accepted - int(valid.sum())}
        )
        for n in np.flatnonzero(valid):
            evidence.append(
                {
                    "xyz": xyz[n].tolist(),
                    "views": [i, j],
                    "pixels": [pa[n], pb[n]],
                    "reprojection_error": float(error[n]),
                }
            )
    unique = unique_correspondences(evidence)
    for pair in pairs:
        pair["unique_accepted"] = sum(p["views"] == pair["cameras"] for p in unique)
    return {
        "points": unique,
        "duplicate_observations_removed": len(evidence) - len(unique),
        "pairs": pairs,
        "collision_bounds_applied": use_collision_bounds,
        "method": "Mutual SIFT ratio 0.7; known-camera triangulation; <=1.5px error; >=3deg ray angle",
        "limitations": [
            "Repeated patterns can survive geometric filtering.",
            "Sparse points do not identify object categories.",
            "No overlapping visible features means no multiview reconstruction.",
        ],
    }


def fit_heights(room, cloud, recipe):
    """Produce conservative proposals; retain explicit user recipe values."""
    proposals = []
    fitted = dict(recipe)
    heights = dict(recipe.get("boundary_heights", {}))
    points = np.array([p["xyz"] for p in cloud["points"]]).reshape(-1, 3)
    for b in room["boundaries"]:
        if b["flags"] & 0x100 or b["type"] & 255 not in (1, 5):
            continue
        if not len(points):
            continue
        # Only points inside the collision footprint; a collision volume is
        # not necessarily the object's visible silhouette, so expose support.
        mask = (
            (points[:, 0] >= b["xmin"])
            & (points[:, 0] <= b["xmax"])
            & (points[:, 2] >= b["zmin"])
            & (points[:, 2] <= b["zmax"])
        )
        selected = points[mask]
        if len(selected) < 8:
            continue
        y = selected[:, 1]
        top = float(np.quantile(y, 0.15))
        support = selected[np.abs(y - top) < 150]
        # Require horizontal spatial support, not one repeated vertical edge.
        spread = np.ptp(support[:, [0, 2]], axis=0) if len(support) else np.zeros(2)
        accept = len(support) >= 5 and min(spread) > 150 and -3500 < top < -400
        proposals.append(
            {
                "boundary": b["index"],
                "top_y": top,
                "support": len(support),
                "accepted": bool(accept),
                "reason": "Supported horizontal extent"
                if accept
                else "Insufficient spatial support",
            }
        )
        if accept:
            heights.setdefault(
                str(b["index"]), round(float(recipe.get("floor_y", 0)) - top)
            )
    fitted["boundary_heights"] = heights
    return fitted, proposals


def observed_visibility(z, xy, observation):
    """Veto confident, smooth foreground pixels when baking a farther surface.

    Model depth can be wrong; unknown and discontinuity pixels do not veto.
    This rejects texture sources, never changes geometry.
    """
    target, confident = observation
    x, y = xy[:, 0], xy[:, 1]
    h, w = target.shape
    inside = (x > 0) & (x < w - 1) & (y > 0) & (y < h - 1)
    x, y = np.clip(x, 1, w - 2), np.clip(y, 1, h - 2)
    local = np.stack(
        [
            target[y + dy, x + dx]
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        ]
    )
    valid = inside & confident[y, x] & np.isfinite(local).all(0) & (local > 0).all(0)
    valid &= local.max(0) <= local.min(0) * 1.12
    nearer = target[y, x] < z - np.maximum(180, z * 0.12)
    return ~(valid & nearer)


def material_chart(points, normal):
    """A non-folding local chart for inferred patches with clamped samplers.

    Per-vertex modulo collapses faces whose extents are multiples of the tile
    period. Keep a continuous chart across subdivisions of the source triangle.
    """
    axes = np.argsort(np.abs(normal))[:2]
    projected = np.asarray(points)[:, axes]
    return axes, projected.min(axis=0), np.maximum(np.ptp(projected, axis=0), 500)


class BakeDepthCache:
    """Per-room depth reuse keyed by geometry, visibility sidecar and camera values."""

    def __init__(self):
        self.geometry = None
        self.views = {}

    def render_depth(self, mesh, camera, only_groups=None):
        sidecar = mesh.with_suffix('.cameras')
        geometry = (hashlib.sha256(mesh.read_bytes()).digest(),
                    hashlib.sha256(sidecar.read_bytes()).digest() if sidecar.exists() else None)
        if geometry != self.geometry:
            self.views.clear()
            self.geometry = geometry
        key = (json.dumps(camera, sort_keys=True, default=lambda value: value.tolist()),
               tuple(sorted(only_groups)) if only_groups is not None else None)
        if key not in self.views:
            depth = render(mesh, camera, return_depth=True, only_groups=only_groups)[2]
            depth.setflags(write=False)
            self.views[key] = depth
        return self.views[key]


def bake_projected(
    mesh,
    room,
    images,
    output,
    exclude_camera=None,
    segments=None,
    group_kinds=None,
    observed_depths=None,
    interpreted_groups=None,
    minimum_projection_cosine=0.25,
    fixture_observations=None,
    depth_cache=None,
    foreground_fixture_guard=False,
    fixture_core_appearance=False,
    preserve_fixture_appearance=False,
    reconstructed_observations=None,
):
    """Bake visible source textures onto subdivided mesh triangles, with depth tests.

    This transfers baked appearance, not a physical separation of albedo and light.
    Leave-one-camera-out output is used for evaluation to reveal reprojection cheats.
    """
    from room3d_complete import fixture_tints, material_patches
    from room3d_segment import texture_support

    if not 0 <= minimum_projection_cosine <= 1:
        raise ValueError("Projection cosine must be within 0..1")
    if preserve_fixture_appearance and exclude_camera is not None:
        raise ValueError("Preserved fixture textures cannot be used for heldout appearance evaluation")
    grazing_rejections = 0
    interpreted_groups = set(interpreted_groups or ())
    fixture_colors = fixture_tints(room, images, segments, {
        group: sources for group, sources in (fixture_observations or {}).items()
        if group in interpreted_groups and (group_kinds or {}).get(group) == 'lamp'
    }, exclude_camera)
    styled_faces = 0
    completed_vertices = 0
    semantic_rejections = 0
    cache = depth_cache if depth_cache is not None else BakeDepthCache()
    depth = []
    for camera in room["cameras"]:
        depth.append(cache.render_depth(mesh, camera))
    depth_calibration = []
    if observed_depths is not None and segments is not None:
        from room3d_refine import calibrate_floor_depth
        from room3d_structure import semantic_masks

        calibrated = []
        for k, (camera, observation) in enumerate(zip(room["cameras"], observed_depths, strict=True)):
            if k == exclude_camera:
                calibrated.append(observation)
                depth_calibration.append({"camera": camera["index"], "accepted": False,
                                          "reason": "excluded_source_camera"})
                continue
            floor = cache.render_depth(mesh, camera, only_groups={"floor"})
            corrected, evidence = calibrate_floor_depth(
                observation, floor, depth[k], semantic_masks(segments[k])["floor"]
            )
            calibrated.append(corrected)
            depth_calibration.append({"camera": camera["index"], **evidence})
        observed_depths = calibrated
    # Load every group once, retaining visibility sidecar separately. Index -1
    # deliberately matches no valid sidecar camera.
    loaded = load_obj(mesh, -1, with_face_charts=fixture_core_appearance)
    triangles = loaded[0]
    face_charts = loaded[2] if fixture_core_appearance else [None] * len(triangles)
    # Keep the original OBJ groups and triangle order to preserve cutaways.
    groups = []
    group = "default"
    for line in mesh.read_text().splitlines():
        words = line.split()
        if words and words[0] == "g":
            group = " ".join(words[1:])
        elif words and words[0] == "f":
            groups.extend([group] * (len(words) - 3))
    from room3d_texture_ownership import ARCHITECTURE, completion_segments, owned_masks

    retained_observations = {group: sources
                             for group, sources in (reconstructed_observations or {}).items()
                             if group in set(groups)}
    ownership, ownership_evidence = owned_masks(
        room, segments, retained_observations, group_kinds, exclude_camera)
    completion = material_patches(images, completion_segments(segments, ownership),
                                  output.parent, exclude_camera)
    arrays = [np.asarray(im, dtype=float) / 255 for im in images]
    preserved_materials = {}
    if preserve_fixture_appearance:
        from room3d_fixture_appearance import baked_fixture_role

        for (_, material), group in zip(triangles, groups, strict=True):
            role = baked_fixture_role(material)
            key = (group, material)
            if (role is not None and material in loaded[1] and group in interpreted_groups
                    and (group_kinds or {}).get(group) == 'lamp' and key not in preserved_materials):
                preserved_materials[key] = f'preserved_fixture_{len(preserved_materials)}_{role}'
    camera_rotations = [camera_basis(camera) for camera in room['cameras']]
    lines = ["# Projected original appearance; not recovered albedo or lighting."]
    material_lines = [
        "newmtl neutral",
        "Kd 1 1 1",
        "newmtl creative_stone",
        "Kd 0.34 0.31 0.27",
    ]
    for k, image in enumerate(images):
        image.save(output.parent / f"source-camera{k}.png")
        material_lines += [
            f"newmtl camera{k}",
            "Kd 1 1 1",
            f"map_Kd source-camera{k}.png",
        ]
    filtered_arrays = {}
    filtered_pixels = 0
    if segments is not None:
        yy, xx = np.indices((240, 320))
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        for kind in sorted(set((group_kinds or {}).values())):
            if not kind.isidentifier():
                raise ValueError("Invalid semantic material name")
            for k, entries in enumerate(segments):
                support = texture_support(entries, kind, xy,
                                          foreground_fixture_guard=foreground_fixture_guard).reshape(240, 320)
                if kind in ARCHITECTURE:
                    support &= ~ownership[k]
                filtered = arrays[k].copy()
                patch = completion.get(kind)
                fallback = (
                    patch["mean"]
                    if patch
                    else (
                        np.median(filtered[support], axis=0)
                        if support.any()
                        else [0.18] * 3
                    )
                )
                filtered[~support] = fallback
                filtered_pixels += int((~support).sum())
                filtered_arrays[k, kind] = filtered
                name = f"{output.stem}-camera{k}-{kind}.png"
                from PIL import Image

                Image.fromarray((np.clip(filtered, 0, 1) * 255).astype(np.uint8)).save(
                    output.parent / name
                )
                material_lines += [
                    f"newmtl camera{k}_{kind}",
                    "Kd 1 1 1",
                    f"map_Kd {name}",
                ]
    for kind, patch in completion.items():
        material_lines += [
            f"newmtl inferred_{kind}",
            "Kd 1 1 1",
            f"map_Kd {patch['file']}",
        ]
    for (_, source_material), alias in preserved_materials.items():
        kd, texture = loaded[1][source_material]
        material_lines += ['newmtl ' + alias, 'Kd ' + ' '.join(map(str, kd))]
        if texture is not None:
            from PIL import Image

            filename = f'{output.stem}-{alias}.png'
            Image.fromarray(np.rint(texture * 255).astype('uint8')).save(output.parent / filename)
            material_lines.append('map_Kd ' + filename)
    fixture_materials = {group: f'fixture_tint_{i}' for i, group in enumerate(sorted(fixture_colors))}
    fixture_cores = {}
    if fixture_core_appearance and segments is not None:
        from room3d_fixture_appearance import SHADE_MATERIALS, core_patch

        for group, palette in fixture_colors.items():
            wanted_cameras = {s['camera'] for s in palette['sources']}
            areas = {}
            for index, camera in enumerate(room['cameras']):
                if index == exclude_camera or camera['index'] not in wanted_cameras:
                    continue
                _, visible, shade = render(mesh, camera, return_depth=True,
                                           only_groups={group}, only_materials=SHADE_MATERIALS)
                visible &= shade <= depth[index] + .001
                areas[camera['index']] = int(visible.sum())
            patch = core_patch(palette, room, images, segments, areas, exclude_camera)
            if patch is not None:
                texture, evidence = patch
                filename = f'{output.stem}-{fixture_materials[group]}-core.png'
                texture.save(output.parent / filename)
                fixture_cores[group] = {**evidence, 'file': filename}
    for group, name in fixture_materials.items():
        material_lines += [f'newmtl {name}', 'Kd ' + ' '.join(map(str, fixture_colors[group]['rgb']))]
        for role, rgb in fixture_colors[group]['roles'].items():
            if role == 'shade' and group in fixture_cores:
                material_lines += [f'newmtl {name}_{role}', 'Kd 1 1 1', 'map_Kd ' + fixture_cores[group]['file']]
            else:
                material_lines += [f'newmtl {name}_{role}', 'Kd ' + ' '.join(map(str, rgb))]
    output.with_suffix(".mtl").write_text("\n".join(material_lines) + "\n")
    lines.append("mtllib " + output.with_suffix(".mtl").name)
    vertex_count = 0
    colored = 0
    total = 0
    for (tri, source_material), group, face_uv in zip(triangles, groups, face_charts, strict=True):
        preserved = preserved_materials.get((group, source_material))
        # Subdivision makes perspective projection colors useful at runtime,
        # without assigning a different camera texture to every original face.
        a, b, c = tri[:, :3]
        span = max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))
        n = min(
            max(1, int(np.sqrt(10000 / max(1, len(triangles))))),
            12,
            max(2, int(np.ceil(span / 500))),
        )
        if preserved is not None:
            # Retain the existing chart exactly; subdividing a previously baked
            # face would otherwise invent new local bright cores per triangle.
            n = 1
        local = []
        for row in range(n):
            for col in range(n - row):
                p = a + (b - a) * row / n + (c - a) * col / n
                q = p + (b - a) / n
                r = p + (c - a) / n
                local.append([p, q, r])
                if col + row < n - 1:
                    local.append([q, q + (c - a) / n, r])
        xyz = np.asarray(local).reshape(-1, 3)
        colors = np.full((len(xyz), 3), 0.18)
        best = np.zeros(len(xyz))
        scores = np.zeros((len(images), len(xyz)))
        coordinates = np.zeros((len(images), len(xyz), 2))
        normal = np.cross(b - a, c - a)
        normal /= max(np.linalg.norm(normal), 1e-12)
        chart_axes, chart_origin, chart_scale = material_chart(tri[:, :3], normal)
        for k, camera in enumerate(room["cameras"]):
            if k == exclude_camera or group in interpreted_groups:
                continue
            uv, z = project(xyz, camera, camera_rotations[k])
            visible = (
                (z > camera["fov"] * 2)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < 320)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < 240)
            )
            # Avoid converting overflowed coordinates for points behind camera.
            xy = np.floor(np.clip(uv, [0, 0], [319, 239])).astype(int)
            sampled_depth = depth[k][xy[:, 1], xy[:, 0]]
            visible &= np.isfinite(sampled_depth) & (
                np.abs(sampled_depth - z) < np.maximum(100, z * 0.025)
            )
            if observed_depths is not None:
                visible &= observed_visibility(z, xy, observed_depths[k])
            if segments is not None:
                support = texture_support(
                    segments[k], (group_kinds or {}).get(group), xy,
                    foreground_fixture_guard=foreground_fixture_guard,
                )
                if (group_kinds or {}).get(group) in ARCHITECTURE:
                    support &= ~ownership[k][xy[:, 1], xy[:, 0]]
                semantic_rejections += int((visible & ~support).sum())
                visible &= support
            rays = np.array(camera["origin"]) - xyz
            cosine = np.abs(rays @ normal) / np.maximum(np.linalg.norm(rays, axis=1), 1)
            grazing = cosine < minimum_projection_cosine
            grazing_rejections += int((visible & grazing).sum())
            visible &= ~grazing
            score = cosine / np.maximum(z, 1) ** 2
            scores[k] = np.where(visible, score, 0)
            coordinates[k] = uv
            update = visible & (score > best)
            source = filtered_arrays.get((k, (group_kinds or {}).get(group)), arrays[k])
            colors[update] = source[xy[update, 1], xy[update, 0]]
            best[update] = score[update]
        patch = completion.get((group_kinds or {}).get(group))
        if patch:
            unknown = best == 0
            colors[unknown] = patch["mean"]
            completed_vertices += int(unknown.sum())
        total += len(xyz)
        colored += int((best > 0).sum())
        lines.append("g " + group)
        for index, (points, cols) in enumerate(
            zip(np.asarray(local), colors.reshape(-1, 3, 3), strict=True)
        ):
            sl = slice(index * 3, index * 3 + 3)
            support = scores[:, sl].min(axis=1)
            chosen = int(support.argmax())
            textured = support[chosen] > 0
            inferred = patch is not None and not (best[sl] > 0).any()
            semantic_kind = (group_kinds or {}).get(group)
            camera_material = (
                f"camera{chosen}_{semantic_kind}"
                if (chosen, semantic_kind) in filtered_arrays
                else f"camera{chosen}"
            )
            material = (
                camera_material
                if textured
                else (
                    f"inferred_{(group_kinds or {})[group]}" if inferred else "neutral"
                )
            )
            styled = (
                group in interpreted_groups
                and semantic_kind == "staircase"
                and abs(normal[1]) < 0.5
            )
            if styled:
                material = "creative_stone"
                styled_faces += 1
            if group in fixture_materials:
                material = fixture_materials[group]
                if source_material in ('fixture_support', 'fixture_shade'):
                    material += '_' + source_material.removeprefix('fixture_')
                elif source_material in ('retouch_added_support', 'retouch_added_shade'):
                    material += '_' + source_material.removeprefix('retouch_added_')
            if preserved is not None:
                material = preserved
            lines.append(f"usemtl {material}")
            for corner, (point, color) in enumerate(zip(points, cols, strict=True)):
                rgb = [1, 1, 1] if textured or inferred or styled or group in fixture_materials else color
                uv = (
                    coordinates[chosen, index * 3 + corner] / [319, 239]
                    if textured
                    else [0, 1]
                )
                if inferred:
                    # Continuous inferred patch, compatible with clamped samplers.
                    uv = (point[chart_axes] - chart_origin) / chart_scale
                if (group in fixture_cores and face_uv is not None
                        and source_material in ('fixture_shade', 'retouch_added_shade')):
                    origin, axes, low, face_scale = face_uv
                    uv = ((point-origin) @ axes.T - low) / face_scale
                if preserved is not None:
                    rgb = tri[corner, 5:8]
                    uv = tri[corner, 3:5]
                lines.append("v " + " ".join(f"{v:.5f}" for v in [*point, *rgb]))
                lines.append(f"vt {uv[0]:.7f} {uv[1] if preserved is not None else 1 - uv[1]:.7f}")
            lines.append(
                "f "
                + " ".join(
                    f"{v}/{v}" for v in range(vertex_count + 1, vertex_count + 4)
                )
            )
            vertex_count += 3
    contents = "\n".join(lines) + "\n"
    if len(contents.encode()) > 4 * 1024 * 1024 or vertex_count > 100000:
        raise ValueError("Generated mesh exceeds current runtime OBJ limits")
    output.write_text(contents)
    output.with_suffix(".cameras").write_text(mesh.with_suffix(".cameras").read_text())
    return {
        "depth_calibration": depth_calibration,
        "styled_faces": styled_faces,
        "fixture_tints": fixture_colors,
        **({'fixture_cores': fixture_cores} if fixture_core_appearance else {}),
        **({'preserved_fixture_materials': [
            {'group': group, 'source_material': material, 'material': alias}
            for (group, material), alias in preserved_materials.items()
        ]} if preserve_fixture_appearance else {}),
        "creative_support_material": "classic_mansion stone; artistic choice, not recovered albedo",
        "minimum_projection_cosine": minimum_projection_cosine,
        "grazing_rejections": grazing_rejections,
        "interpreted_groups": sorted(interpreted_groups),
        "interpretation_appearance": "Local material continuation; no camera projection onto redesigned objects",
        "filtered_texture_pixels": filtered_pixels,
        **({"foreground_fixture_guard": {
            "detector_minimum": .3, "mask_minimum": .8,
            "qualification": "Explicit fixture masks veto incompatible texture samples; geometry and material-source eligibility unchanged",
        }} if foreground_fixture_guard else {}),
        **({"reconstructed_object_appearance": {
            "sources": ownership_evidence,
            "excluded_camera": exclude_camera,
            "qualification": "Retained ladder masks veto architecture projection and completion samples; unseen material remains estimated",
        }} if reconstructed_observations is not None else {}),
        "completion_patches": completion,
        "inferred_vertices": completed_vertices,
        "semantic_rejections": semantic_rejections,
        "segmentation_applied": segments is not None,
        "sampled_vertices": colored,
        "vertices": total,
        "excluded_camera": exclude_camera,
        "limitations": [
            "Appearance includes original shadows and highlights.",
            "Unobserved vertices use inferred material where supported; otherwise neutral gray.",
            "Incorrect geometry can inherit objects painted onto walls.",
        ],
    }
