"""Shared architectural surfaces and conservative multi-view evidence gates.

Collision footprints constrain topology, not visual truth. DA3/SAM remain
estimated evidence. Occluded and unknown samples never count as free space.
"""

import copy

import numpy as np
from room3d_preview import camera_basis


def semantic_masks(entries):
    masks = {k: np.zeros((240, 320), bool) for k in ("wall", "floor", "fixture")}
    for item, mask in entries:
        if (not np.isfinite(item["score"]) or not np.isfinite(item["predicted_iou"])
                or item["score"] < 0.3 or item["predicted_iou"] < 0.8):
            continue
        label = item["label"].strip().lower()
        if label == "ceiling fan" or any(word in label for word in ("lamp", "chandelier", "railing")):
            masks["fixture"] |= mask
        elif "floor" in label.split():
            masks["floor"] |= mask
        elif label == "wall":
            masks["wall"] |= mask
    for kind in ("wall", "floor"):
        masks[kind] &= ~masks["fixture"]
    return masks


def evidence(points, room, depths, exclusions=None):
    """Count support and free-space votes, excluding depth discontinuities.

    Two contradicting views are required by callers. A nearer observed surface
    occludes the candidate and is therefore neither support nor contradiction.
    """
    points = np.asarray(points, float)
    support = np.zeros(len(points), int)
    conflict = np.zeros(len(points), int)
    for i, (camera, (depth, valid)) in enumerate(
        zip(room["cameras"], depths, strict=True)
    ):
        view = (points - camera["origin"]) @ camera_basis(camera).T
        z = view[:, 2]
        xy = view[:, :2] / np.maximum(z[:, None], 1) * camera["fov"] + [160, 120]
        inside = (
            (z > 2 * camera["fov"])
            & (xy[:, 0] >= 1)
            & (xy[:, 0] < 319)
            & (xy[:, 1] >= 1)
            & (xy[:, 1] < 239)
        )
        ids = np.flatnonzero(inside)
        x, y = np.rint(xy[ids]).astype(int).T
        x, y = np.clip(x, 1, 318), np.clip(y, 1, 238)
        neighbors = np.stack(
            [
                depth[y + dy, x + dx]
                for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            ]
        )
        known = valid[y, x] & np.isfinite(neighbors).all(0) & (neighbors.min(0) > 0)
        known &= neighbors.max(0) <= neighbors.min(0) * 1.12
        if exclusions is not None:
            known &= ~exclusions[i][y, x]
        target = depth[y, x]
        tolerance = np.maximum(180, target * 0.12)
        support[ids] += known & (np.abs(z[ids] - target) <= tolerance)
        conflict[ids] += known & (z[ids] < target - tolerance)
    return support, conflict


def wall_outline(bounds):
    """Union rectangular wall footprints into continuous exposed boundary runs."""
    walls = [b for b in bounds if b["flags"] & 0x100 and b["type"] & 255 in (1, 5)]
    if not walls:
        return []
    xs = sorted({b[k] for b in walls for k in ("xmin", "xmax")})
    zs = sorted({b[k] for b in walls for k in ("zmin", "zmax")})
    cells = {}
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            x, z = (xs[i] + xs[i + 1]) / 2, (zs[j] + zs[j + 1]) / 2
            owners = [
                b["index"]
                for b in walls
                if b["xmin"] < x < b["xmax"] and b["zmin"] < z < b["zmax"]
            ]
            if owners:
                cells[i, j] = min(owners)
    edges = {}
    for (i, j), owner in cells.items():
        for neighbor, axis, position, lo, hi in [
            ((i - 1, j), 0, xs[i], zs[j], zs[j + 1]),
            ((i + 1, j), 0, xs[i + 1], zs[j], zs[j + 1]),
            ((i, j - 1), 2, zs[j], xs[i], xs[i + 1]),
            ((i, j + 1), 2, zs[j + 1], xs[i], xs[i + 1]),
        ]:
            if neighbor not in cells:
                edges.setdefault((axis, position, owner), []).append((lo, hi))
    result = []
    for (axis, position, owner), runs in sorted(edges.items()):
        merged = []
        for lo, hi in sorted(runs):
            if merged and lo == merged[-1][1]:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        result.extend(
            {"axis": axis, "position": position, "lo": lo, "hi": hi, "owner": owner}
            for lo, hi in merged
        )
    return result


def consolidate(room, recipe, depths, segments):
    result = copy.deepcopy(recipe)
    report = {
        "planes": [],
        "fixtures": [],
        "qualification": "Estimated shared planes; collision topology and DA3/SAM are constraints, not ground truth.",
    }
    masks = (
        [semantic_masks(entries) for entries in segments]
        if segments is not None
        else None
    )
    clouds = []
    for i, (camera, (depth, valid)) in enumerate(
        zip(room["cameras"], depths, strict=True)
    ):
        y, x = np.mgrid[0:240:3, 0:320:3]
        z = depth[y, x]
        points = (
            np.stack(
                ((x - 160) * z / camera["fov"], (y - 120) * z / camera["fov"], z), -1
            )
            @ camera_basis(camera)
            + camera["origin"]
        )
        for kind in ("wall", "floor"):
            keep = valid[y, x] & (masks[i][kind][y, x] if masks is not None else False)
            clouds.append((i, kind, points[keep]))
    floor = float(recipe.get("floor_y", 0))
    floor_votes = [
        float(np.median(points[:, 1]))
        for _, kind, points in clouds
        if kind == "floor"
        and len(points) >= 30
        and abs(float(np.median(points[:, 1])) - floor) < 300
    ]
    if len(floor_votes) >= 2 and np.ptp(floor_votes) <= 200:
        floor = float(np.clip(np.median(floor_votes), floor - 100, floor + 100))
        result["floor_y"] = floor
    report["floor"] = {"fitted": floor, "supporting_views": len(floor_votes)}
    top = floor - float(recipe.get("wall_height", 6000))
    surfaces = []
    shifts = {}
    bounds = [
        b
        for b in room["boundaries"]
        if b["index"] not in recipe.get("skip_boundaries", [])
    ]
    outline = wall_outline(bounds)
    report["wall_footprints"] = sum(
        bool(b["flags"] & 0x100) and b["type"] & 255 in (1, 5) for b in bounds
    )
    for edge in outline:
        axis, p = edge["axis"], edge["position"]
        # One shared plane per coordinate across all wall runs and views.
        key = (axis, p)
        if key not in shifts:
            medians = []
            for _, kind, points in clouds:
                if kind != "wall":
                    continue
                near = points[np.abs(points[:, axis] - p) < 450]
                if len(near) >= 30:
                    medians.append(float(np.median(near[:, axis])))
            value = p
            if len(medians) >= 2 and np.ptp(medians) <= 300:
                value = float(np.clip(np.median(medians), p - 200, p + 200))
            shifts[key] = value
            report["planes"].append(
                {
                    "axis": axis,
                    "prior": p,
                    "fitted": value,
                    "supporting_views": len(medians),
                }
            )
    # Preserve plane ordering in narrow gaps; bounded fits must not invert
    # a corridor or disconnect a corner shared by adjacent runs.
    for axis in (0, 2):
        positions = sorted(p for a, p in shifts if a == axis)
        for i, p in enumerate(positions):
            lower = (positions[i - 1] + p) / 2 + 0.01 if i else -np.inf
            upper = (
                (positions[i + 1] + p) / 2 - 0.01 if i + 1 < len(positions) else np.inf
            )
            shifts[axis, p] = float(np.clip(shifts[axis, p], lower, upper))
    for plane in report["planes"]:
        plane["fitted"] = shifts[plane["axis"], plane["prior"]]
    # Snap intersections by the same mapping, keeping shared corners joined.
    for edge in outline:
        axis, cross = edge["axis"], 2 - edge["axis"]
        p = shifts[axis, edge["position"]]
        lo = shifts.get((cross, edge["lo"]), edge["lo"])
        hi = shifts.get((cross, edge["hi"]), edge["hi"])
        if hi <= lo:
            lo, hi = edge["lo"], edge["hi"]

        def vertex(along, y, p=p, axis=axis):
            return [p, y, along] if axis == 0 else [along, y, p]

        wall_top = floor - recipe.get("boundary_heights", {}).get(
            str(edge["owner"]), floor - top
        )
        surfaces.append(
            {
                "points": [
                    vertex(lo, floor),
                    vertex(lo, wall_top),
                    vertex(hi, wall_top),
                    vertex(hi, floor),
                ],
                "group": f"boundary_{edge['owner']}",
                "material": "wall",
            }
        )
    result["structural_surfaces"] = surfaces
    result["shared_wall_boundaries"] = [
        b["index"] for b in bounds if b["flags"] & 0x100 and b["type"] & 255 in (1, 5)
    ]
    # Retain architectural planes and script-owned geometry. Prune only small
    # inferred fixture cells, never punching holes into the room shell.
    for index, obj in enumerate(result.get("primitives", [])):
        if obj["kind"] not in ("lamp", "railing") or not obj.get("parts"):
            continue
        parts = np.asarray(obj["parts"], float)
        samples = np.concatenate((parts, parts.mean(1)[:, None, :]), axis=1)
        support, conflict = evidence(samples.reshape(-1, 3), room, depths)
        contradictions = ((conflict >= 2) & (conflict > support)).reshape(-1, 9)
        keep = contradictions.sum(1) < 7
        obj["parts"] = parts[keep].tolist()
        # Empty parts must not fall through to a bounding-box renderer.
        obj["omit_geometry"] = not bool(keep.any())
        report["fixtures"].append(
            {
                "primitive": index,
                "before": len(parts),
                "after": int(keep.sum()),
                "contradiction_threshold_views": 2,
                "sample_rule": "At least 7 of 9 cell samples have more conflicting than supporting views",
            }
        )
    report["wall_surfaces"] = len(surfaces)
    return result, report


def door_openings(surfaces, primitives, maximum_snap=100):
    """Cut script-owned closed-door rectangles into their nearest wall plane.

    Wall collision footprints and door panels remain unchanged. This removes the
    duplicate visual wall in front of a panel, not the panel or its gameplay lock.
    Only axis-aligned surfaces of the panel's designated wall are eligible.
    """
    result = copy.deepcopy(surfaces)
    for door in primitives:
        if door.get('kind') != 'door_panel' or 'source_offset' not in door:
            continue
        box = door.get('box')
        if box is None or not door.get('hide_with'):
            continue
        x0, z0, x1, z1, y0, y1 = box
        axis = 0 if x1-x0 < z1-z0 else 2
        cross = 2-axis
        center = (x0+x1)/2 if axis == 0 else (z0+z1)/2
        lower, upper = (z0, z1) if axis == 0 else (x0, x1)
        eligible = []
        for surface in result:
            points = np.asarray(surface['points'], float)
            if (surface['group'] == door['hide_with']
                    and np.ptp(points[:, axis]) < 1e-6
                    and points[:, cross].max() > lower and points[:, cross].min() < upper
                    and points[:, 1].max() > y0 and points[:, 1].min() < y1):
                eligible.append(float(points[0, axis]))
        if not eligible:
            continue
        plane = min(eligible, key=lambda p: abs(p-center))
        if abs(plane-center) > maximum_snap:
            continue
        split = []
        for surface in result:
            points = np.asarray(surface['points'], float)
            if (surface['group'] != door['hide_with']
                    or np.ptp(points[:, axis]) > 1e-6
                    or abs(points[0, axis]-plane) > 1e-6):
                split.append(surface)
                continue
            lo, hi = points[:, cross].min(), points[:, cross].max()
            bottom, top = points[:, 1].min(), points[:, 1].max()
            a, b = max(lo, lower), min(hi, upper)
            c, d = max(bottom, y0), min(top, y1)
            if a >= b or c >= d:
                split.append(surface)
                continue
            original_normal = np.cross(points[1]-points[0], points[2]-points[0])
            for l, h, low_y, high_y in ((lo, a, bottom, top), (b, hi, bottom, top),
                                      (a, b, bottom, c), (a, b, d, top)):
                if h-l <= 1e-6 or high_y-low_y <= 1e-6:
                    continue
                patch = np.zeros((4, 3))
                patch[:, axis] = plane
                patch[:, cross] = [l, l, h, h]
                patch[:, 1] = [high_y, low_y, low_y, high_y]
                if np.cross(patch[1]-patch[0], patch[2]-patch[0]) @ original_normal < 0:
                    patch = patch[::-1]
                split.append({**surface, 'points': patch.tolist()})
        result = split
    return result
