"""Experimental visual wall hypotheses from depth clouds, independent of collisions.

Depth residuals measure agreement with an estimated model, not ground truth or
calibrated uncertainty. Keep source-view diagnostics and never infer confidence
from the number of generated vertices.
"""

import numpy as np


def fit_wall(points):
    """Fit a vertical plane with robust offset; return finite patch and residuals."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an N x 3 point cloud")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("Insufficient finite wall samples")
    xz = points[:, [0, 2]]
    values, vectors = np.linalg.eigh(np.cov(xz.T))
    if values[-1] <= 1e-12:
        raise ValueError("Degenerate wall cloud")
    along = vectors[:, -1]
    normal = np.array([-along[1], along[0]])
    offset = float(np.median(xz @ normal))
    extent = np.percentile(xz @ along, [2, 98])
    heights = np.percentile(points[:, 1], [2, 98])
    if extent[1] - extent[0] <= 1e-6 or heights[1] - heights[0] <= 1e-6:
        raise ValueError("Degenerate wall patch")
    residual = np.abs(xz @ normal - offset)
    corners = []
    for distance, height in [(extent[0], heights[1]), (extent[0], heights[0]),
                             (extent[1], heights[0]), (extent[1], heights[1])]:
        position = along * distance + normal * offset
        corners.append([float(position[0]), float(height), float(position[1])])
    return {"points": corners, "normal": normal.tolist(), "offset": offset,
            "p90_residual": float(np.percentile(residual, 90)),
            "span": float(extent[1] - extent[0]), "samples": len(points)}


def reconcile_walls(observations, max_angle=5.0, relative_slack=0.01):
    """Propose shared patches with equal-view fitting and per-view evidence gates.

    Each observation contains camera, points, and median_depth. Only overlapping
    patches may merge. The 1% depth slack is a provisional model-error allowance,
    not a probability. A merged plane must fit every original source cloud; a
    chain of individually compatible neighbors cannot bypass this check.
    """
    if not 0 <= max_angle < 90 or relative_slack < 0:
        raise ValueError("Invalid wall reconciliation tolerances")
    prepared = []
    for observation in observations:
        points = np.asarray(observation["points"], float)
        points = points[np.isfinite(points).all(axis=1)]
        depth = float(observation["median_depth"])
        if not np.isfinite(depth) or depth <= 0:
            raise ValueError("Expected positive finite median depth")
        plane = fit_wall(points)
        prepared.append({**observation, "points": points, "plane": plane})
    # Stable camera order avoids depending on caller iteration order.
    prepared.sort(key=lambda item: item["camera"])
    groups = [[item] for item in prepared]
    decisions = []

    def candidate(members):
        count = min(4096, min(len(m["points"]) for m in members))
        samples = [m["points"][np.linspace(0, len(m["points"]) - 1, count).astype(int)]
                   for m in members]
        return fit_wall(np.concatenate(samples))

    def overlap(a, b):
        normal = np.array(a["normal"])
        along = np.array([-normal[1], normal[0]])
        pa, pb = np.array(a["points"]), np.array(b["points"])
        ia, ib = pa[:, [0, 2]] @ along, pb[:, [0, 2]] @ along
        return (min(ia.max(), ib.max()) > max(ia.min(), ib.min())
                and min(pa[:, 1].max(), pb[:, 1].max())
                > max(pa[:, 1].min(), pb[:, 1].min()))

    index = 0
    while index < len(groups):
        other = index + 1
        while other < len(groups):
            members = groups[index] + groups[other]
            angles = [float(np.degrees(np.arccos(np.clip(abs(
                np.dot(a["plane"]["normal"], b["plane"]["normal"])), 0, 1))))
                for a in groups[index] for b in groups[other]]
            connected = any(overlap(a["plane"], b["plane"])
                            for a in groups[index] for b in groups[other])
            proposal = candidate(members)
            diagnostics = []
            for member in members:
                residual = np.abs(member["points"][:, [0, 2]] @ proposal["normal"]
                                  - proposal["offset"])
                p90 = float(np.percentile(residual, 90))
                limit = member["plane"]["p90_residual"] + relative_slack * member["median_depth"]
                diagnostics.append({"camera": member["camera"], "candidate_p90": p90,
                                    "source_p90": member["plane"]["p90_residual"],
                                    "limit": float(limit), "passes": p90 <= limit})
            accepted = connected and max(angles) <= max_angle and all(d["passes"] for d in diagnostics)
            decisions.append({"cameras": [m["camera"] for m in members],
                              "max_angle": max(angles), "overlap": bool(connected),
                              "accepted": bool(accepted), "views": diagnostics})
            if accepted:
                groups[index] = members
                groups.pop(other)
            else:
                other += 1
        index += 1
    result = []
    for members in groups:
        plane = candidate(members) if len(members) > 1 else members[0]["plane"]
        result.append({**plane, "cameras": [m["camera"] for m in members],
                       "qualification": "depth_estimated_shared_plane" if len(members) > 1
                       else "single_view_depth_plane"})
    return result, decisions


def extract_wall_planes(points, median_depth, *, max_planes=4, min_samples=300,
                        relative_tolerance=0.01, trials=192):
    """Split a cloud into vertical planar proposals using deterministic RANSAC.

    Returned indices address the original cloud; nonfinite samples and residual
    clutter remain unassigned. A proposal is evidence for a surface, not evidence
    that the entire enclosing rectangle is visible or free of openings.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an N x 3 point cloud")
    if (not np.isfinite(median_depth) or median_depth <= 0
            or not np.isfinite(relative_tolerance) or relative_tolerance <= 0
            or min_samples < 3 or max_planes < 1 or trials < 1):
        raise ValueError("Invalid plane extraction parameters")
    tolerance = float(median_depth * relative_tolerance)
    remaining = np.flatnonzero(np.isfinite(points).all(axis=1))
    rng = np.random.default_rng(0)
    proposals = []
    for _ in range(max_planes):
        if len(remaining) < min_samples:
            break
        xz = points[remaining][:, [0, 2]]
        # Bound scoring cost while retaining deterministic, equal point sampling.
        sampled = xz[np.linspace(0, len(xz) - 1, min(4096, len(xz))).astype(int)]
        best_normal = None
        best_offset = 0.0
        best_score = -1
        for _ in range(trials):
            a, b = sampled[rng.choice(len(sampled), 2, replace=False)]
            delta = b - a
            length = np.linalg.norm(delta)
            if length < tolerance * 4:
                continue
            normal = np.array([-delta[1], delta[0]]) / length
            offset = float(a @ normal)
            score = int((np.abs(sampled @ normal - offset) <= tolerance).sum())
            if score > best_score:
                best_normal, best_offset, best_score = normal, offset, score
        if best_normal is None:
            break
        selected = np.abs(xz @ best_normal - best_offset) <= tolerance
        if selected.sum() < min_samples:
            break
        try:
            for _ in range(2):
                plane = fit_wall(points[remaining[selected]])
                selected = np.abs(xz @ plane['normal'] - plane['offset']) <= tolerance
                if selected.sum() < min_samples:
                    break
            if selected.sum() < min_samples:
                break
            plane = fit_wall(points[remaining[selected]])
        except ValueError:
            break
        if plane['span'] < tolerance * 4:
            break
        proposals.append({**plane, 'indices': remaining[selected].tolist(),
                          'tolerance': tolerance,
                          'qualification': 'single_view_planar_proposal'})
        remaining = remaining[~selected]
    return proposals, remaining.tolist()


def supported_wall_tiles(plane, points, cell_size, min_samples=8):
    """Bound planar patches to occupied cells instead of filling a rectangle.

    Tiles remain approximate within one cell. Empty cells stay unknown; this
    preserves larger holes without asserting that they are doors or free space.
    """
    if not np.isfinite(cell_size) or cell_size <= 0 or min_samples < 1:
        raise ValueError("Invalid wall tile parameters")
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an N x 3 point cloud")
    points = points[np.isfinite(points).all(axis=1)]
    normal = np.asarray(plane['normal'], float)
    along = np.array([-normal[1], normal[0]])
    coordinates = np.column_stack((points[:, [0, 2]] @ along, points[:, 1]))
    cells, counts = np.unique(np.floor(coordinates / cell_size).astype(np.int64),
                              axis=0, return_counts=True)
    tiles = []
    for cell, count in zip(cells, counts, strict=True):
        if count < min_samples:
            continue
        lo = cell * cell_size
        hi = lo + cell_size
        corners = []
        for distance, height in [(lo[0], hi[1]), (lo[0], lo[1]),
                                 (hi[0], lo[1]), (hi[0], hi[1])]:
            xz = along * distance + normal * plane['offset']
            corners.append([float(xz[0]), float(height), float(xz[1])])
        tiles.append({'points': corners, 'samples': int(count)})
    return tiles


def wall_view_evidence(points, camera, depth, valid, wall_mask):
    """Classify samples in one view: unknown/support/occluded/conflict/nonwall.

    Codes 0..4 respectively. Support requires a wall label and smooth, valid
    neighboring depth. A foreground observed object occludes a candidate; it
    does not disprove it. These are model-agreement labels, not truth labels.
    """
    from room3d_preview import camera_basis

    points = np.asarray(points, float)
    depth, valid, wall_mask = np.asarray(depth), np.asarray(valid), np.asarray(wall_mask)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an N x 3 point cloud")
    if depth.shape != (240, 320) or valid.shape != depth.shape or wall_mask.shape != depth.shape:
        raise ValueError("Expected matching logical-viewport depth and masks")
    result = np.zeros(len(points), np.uint8)
    view = (points - camera['origin']) @ camera_basis(camera).T
    z = view[:, 2]
    xy = view[:, :2] / np.maximum(z[:, None], 1) * camera['fov'] + [160, 120]
    inside = (np.isfinite(view).all(axis=1) & (z > 2 * camera['fov'])
              & (xy[:, 0] >= 1) & (xy[:, 0] < 319)
              & (xy[:, 1] >= 1) & (xy[:, 1] < 239))
    ids = np.flatnonzero(inside)
    if not len(ids):
        return result
    x, y = np.rint(xy[ids]).astype(int).T
    x, y = np.clip(x, 1, 318), np.clip(y, 1, 238)
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = np.stack([depth[y + dy, x + dx] for dy, dx in offsets])
    validity = np.stack([valid[y + dy, x + dx] for dy, dx in offsets])
    known = (validity.all(axis=0) & np.isfinite(neighbors).all(axis=0)
             & (neighbors.min(axis=0) > 0)
             & (neighbors.max(axis=0) <= neighbors.min(axis=0) * 1.12))
    target = depth[y, x]
    tolerance = np.maximum(180, target * .12)
    matching = known & (np.abs(z[ids] - target) <= tolerance)
    result[ids[matching & wall_mask[y, x]]] = 1
    result[ids[known & (z[ids] > target + tolerance)]] = 2
    result[ids[known & (z[ids] < target - tolerance)]] = 3
    result[ids[matching & ~wall_mask[y, x]]] = 4
    return result


def compare_view_evidence(before, after):
    """Compare corresponding samples without treating lost visibility as a fix.

    A contradiction only gains positive wall evidence when its same sample is
    supported afterward. Occlusion, missing evidence and unlabeled depth matches
    remain separately reported. This report alone never accepts a geometry edit.
    """
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("Expected corresponding one-dimensional evidence arrays")
    if not np.isin(before, np.arange(5)).all() or not np.isin(after, np.arange(5)).all():
        raise ValueError("Unknown view evidence code")
    conflict = before == 3
    return {
        'samples': int(before.size),
        'conflict_to_wall_support': int((conflict & (after == 1)).sum()),
        'conflict_to_unknown': int((conflict & (after == 0)).sum()),
        'conflict_to_occluded': int((conflict & (after == 2)).sum()),
        'conflict_to_unlabeled_match': int((conflict & (after == 4)).sum()),
        'remaining_conflicts': int((conflict & (after == 3)).sum()),
        'new_conflicts': int(((before != 3) & (after == 3)).sum()),
        'lost_wall_support': int(((before == 1) & (after != 1)).sum()),
    }


def identify_wall_attachment(boundaries, plane, ray_records):
    """Identify a backing footprint and a narrower protrusion from ray evidence.

    This proposes topology only, not object category or height. Requires multiple
    unique target points and views per obstructing group. Ambiguous pairs remain
    alternatives; the caller must not silently choose the first one.
    """
    normal = np.asarray(plane['normal'], float)
    axis = int(np.argmax(np.abs(normal)))
    if abs(normal[axis]) < .98:
        return {'status': 'unsupported_orientation', 'candidates': []}
    tangent = 1 - axis
    points = np.asarray(plane['points'], float)[:, [0, 2]]
    span = float(np.ptp(points[:, tangent]))
    if span <= 0:
        return {'status': 'insufficient_extent', 'candidates': []}
    tolerance = span * .01
    normal_position = float(np.median(points[:, axis]))
    interval = [float(points[:, tangent].min()), float(points[:, tangent].max())]
    evidence = {}
    for ray in ray_records:
        for hit in ray['occluders']:
            group = evidence.setdefault(hit['group'], {'targets': set(), 'views': set()})
            group['targets'].add(tuple(np.round(ray['target'], 6)))
            group['views'].add(ray['camera'])
    eligible = []
    for boundary in boundaries:
        group = evidence.get(f"boundary_{boundary['index']}")
        if group is None or len(group['targets']) < 3 or len(group['views']) < 2:
            continue
        lo = np.array([boundary['xmin'], boundary['zmin']], float)
        hi = np.array([boundary['xmax'], boundary['zmax']], float)
        if np.any(hi <= lo) or not lo[axis] - tolerance <= normal_position <= hi[axis] + tolerance:
            continue
        eligible.append((boundary, lo, hi, group))
    candidates = []
    for backing, blo, bhi, be in eligible:
        if blo[tangent] > interval[0] + tolerance or bhi[tangent] < interval[1] - tolerance:
            continue
        for attachment, alo, ahi, ae in eligible:
            if attachment['index'] == backing['index']:
                continue
            # A narrower footprint overlaps the backing and extends out on one side.
            if (alo[tangent] < blo[tangent] - tolerance or ahi[tangent] > bhi[tangent] + tolerance
                    or ahi[tangent] - alo[tangent] >= .8 * (bhi[tangent] - blo[tangent])):
                continue
            overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
            if np.any(overlap <= 0):
                continue
            low_extension = blo[axis] - alo[axis] > tolerance
            high_extension = ahi[axis] - bhi[axis] > tolerance
            if low_extension == high_extension:
                continue
            supported_span = min(ahi[tangent], interval[1]) - max(alo[tangent], interval[0])
            if supported_span < .5 * (ahi[tangent] - alo[tangent]):
                continue
            candidates.append({'backing': backing['index'], 'attachment': attachment['index'],
                               'normal_axis': 'x' if axis == 0 else 'z',
                               'protrusion_side': -1 if low_extension else 1,
                               'backing_points': len(be['targets']),
                               'attachment_points': len(ae['targets']),
                               'qualification': 'topology_hypothesis_not_object_identity'})
    candidates.sort(key=lambda item: (item['backing'], item['attachment']))
    status = 'unique_hypothesis' if len(candidates) == 1 else 'ambiguous' if candidates else 'no_hypothesis'
    return {'status': status, 'candidates': candidates}
