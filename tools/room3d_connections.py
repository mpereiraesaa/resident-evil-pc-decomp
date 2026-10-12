"""Geometric stair-end contact checks; these do not prove gameplay traversal."""

import math

import numpy as np


def closest_triangle_point(point, triangle):
    """Closest point on a finite triangle, including edges/degeneracy."""
    point = np.asarray(point, float)
    a, b, c = np.asarray(triangle, float)[:, :3]
    candidates = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length2 = float(edge @ edge)
        t = float(np.clip((point - start) @ edge / length2, 0, 1)) if length2 else 0
        candidates.append(start + t * edge)
    matrix = np.column_stack((b - a, c - a))
    if np.linalg.matrix_rank(matrix) == 2:
        uv = np.linalg.lstsq(matrix, point - a, rcond=None)[0]
        if uv.min() >= 0 and uv.sum() <= 1:
            candidates.append(a + matrix @ uv)
    return min(candidates, key=lambda candidate: np.linalg.norm(point - candidate))


def point_triangle_distance(point, triangle):
    """Euclidean distance to a finite triangle, including edges/degeneracy."""
    return float(np.linalg.norm(np.asarray(point) - closest_triangle_point(point, triangle)))


def lamp_attachment(obj, triangles, groups, attachment_groups, tolerance=100):
    """Measure the generated lamp's stem attachment; proximity is not fastening."""
    if obj.get("kind") != "lamp" or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Expected a lamp and finite nonnegative tolerance")
    # All current lamp profiles use the first beam as their central stem.
    # beam() stores its four start vertices first, independent of world yaw.
    anchor = np.asarray(obj["parts"][0], float)[:4].mean(axis=0)
    allowed = set(attachment_groups)
    distances = [
        (point_triangle_distance(anchor, triangle), group)
        for triangle, group in zip(triangles, groups, strict=True)
        if group in allowed
    ]
    nearest = min(distances, key=lambda item: item[0]) if distances else None
    return {
        "anchor": anchor.tolist(), "distance": nearest[0] if nearest else None,
        "nearest_group": nearest[1] if nearest else None,
        "within_tolerance": bool(nearest and nearest[0] <= tolerance),
        "tolerance": tolerance,
        "qualification": "Stem endpoint proximity only; fixture orientation and mechanical support unverified",
    }


def support_mesh(path):
    """Read triangle geometry and its OBJ group provenance in matching order."""
    from room3d_preview import load_obj

    groups, current = [], "default"
    for line in path.read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        if words[0] == "g":
            current = " ".join(words[1:])
        elif words[0] == "f":
            groups.extend([current] * (len(words) - 3))
    triangles = [t for t, _ in load_obj(path, -1)[0]]
    if len(triangles) != len(groups):
        raise ValueError("Support mesh group/triangle mismatch")
    return triangles, groups


def stair_connections(
    obj,
    triangles,
    tolerance=180,
    clearance=200,
    groups=None,
    walkable_groups=(),
    excluded_groups=(),
):
    """Require room surfaces just beyond both ends across most of the width.

    Input triangles must exclude the staircase itself. Three samples prevent a
    single corner contact being reported as a usable landing. Heights use game Y.
    """
    shape = obj["shape"]
    p = shape["parameters"]
    yaw = shape["yaw"]
    along = np.array([math.sin(yaw), 0, math.cos(yaw)])
    across = np.array([math.cos(yaw), 0, -math.sin(yaw)])
    origin = np.asarray(shape["origin"], float)
    slope = math.degrees(math.atan2(p["rise"], p["run"] - p.get("landing", 0)))
    surfaces = []
    if groups is None:
        groups = [None] * len(triangles)
    walkable_groups = set(walkable_groups)
    excluded_groups = set(excluded_groups)
    for index, (tri, group) in enumerate(zip(triangles, groups, strict=True)):
        if group in excluded_groups:
            continue
        xyz = np.asarray(tri, float)[:, :3]
        if np.ptp(xyz[:, 1]) > 1:
            continue
        a, b, c = xyz[:, [0, 2]]
        matrix = np.column_stack((b - a, c - a))
        if abs(np.linalg.det(matrix)) < 1e-6:
            continue
        surfaces.append(
            (a, np.linalg.inv(matrix), float(xyz[:, 1].mean()), index, group)
        )
    ends = {}
    for name, center in (
        ("bottom", origin - along * clearance),
        ("top", origin + along * (p["run"] + clearance) + [0, -p["rise"], 0]),
    ):
        samples = []
        for fraction in (-0.4, 0, 0.4):
            point = center + across * p["width"] * fraction
            heights = []
            for a, inverse, height, index, group in surfaces:
                uv = inverse @ (point[[0, 2]] - a)
                if uv.min() >= -1e-6 and uv.sum() <= 1 + 1e-6:
                    heights.append((float(abs(height - point[1])), index, group))
            nearest = min(heights, key=lambda h: h[0]) if heights else None
            error = nearest[0] if nearest else None
            contacts = [h for h in heights if h[0] <= tolerance]
            samples.append(
                {
                    "point": point.tolist(),
                    "height_error": error,
                    "nearest_group": nearest[2] if nearest else None,
                    "contact_groups": sorted(
                        {h[2] for h in contacts if h[2] is not None}
                    ),
                    "walkable": any(h[2] in walkable_groups for h in contacts),
                    "supported": bool(error is not None and error <= tolerance),
                }
            )
        ends[name] = {
            "samples": samples,
            "supported": all(s["supported"] for s in samples),
            "walkable": all(s["walkable"] for s in samples),
        }
    return {
        "inclination_degrees": slope,
        "inclination_valid": 15 <= slope <= 42,
        "ends": ends,
        "valid": bool(15 <= slope <= 42 and all(e["supported"] for e in ends.values())),
        "walkable_valid": bool(
            15 <= slope <= 42 and all(e["walkable"] for e in ends.values())
        ),
        "qualification": "Geometric horizontal-surface contact only; no headroom, collision or gameplay traversal proof",
    }


def stair_headroom(obj, triangles, groups, overhead_groups, required=1800):
    """Sample vertical clearance above treads and landing, with surface provenance.

    Only explicitly supplied overhead groups participate. Missing overhead
    observations are unknown, not proof of an enclosed or traversable stair.
    """
    if not np.isfinite(required) or required <= 0:
        raise ValueError("Required headroom must be finite and positive")
    overhead_groups = set(overhead_groups)
    surfaces = []
    for triangle, group in zip(triangles, groups, strict=True):
        if group not in overhead_groups:
            continue
        xyz = np.asarray(triangle, float)[:, :3]
        a, b, c = xyz[:, [0, 2]]
        matrix = np.column_stack((b - a, c - a))
        if abs(np.linalg.det(matrix)) > 1e-6:
            surfaces.append((a, np.linalg.inv(matrix), xyz[:, 1], group))
    shape = obj["shape"]
    p = shape["parameters"]
    along = np.array([math.sin(shape["yaw"]), 0, math.cos(shape["yaw"])])
    across = np.array([math.cos(shape["yaw"]), 0, -math.sin(shape["yaw"])])
    origin = np.asarray(shape["origin"], float)
    samples, centers = [], []
    for step in range(p["steps"]):
        center = origin + along * ((p["run"] - p.get("landing", 0))
                                   * (step + 0.5) / p["steps"])
        center[1] -= p["rise"] * (step + 1) / p["steps"]
        centers.append(("tread", step, center))
    landing = p.get("landing", 0)
    if landing > 0:
        for fraction in (0.1, 0.5, 0.9):
            center = origin + along * (p["run"] - landing + fraction * landing)
            center[1] -= p["rise"]
            centers.append(("landing", None, center))
    for section, step, center in centers:
        for fraction in (-0.4, 0, 0.4):
            point = center + across * p["width"] * fraction
            hits = []
            for a, inverse, heights, group in surfaces:
                uv = inverse @ (point[[0, 2]] - a)
                if uv.min() < -1e-6 or uv.sum() > 1 + 1e-6:
                    continue
                y = heights[0] + uv @ (heights[1:] - heights[0])
                clearance = float(point[1] - y)
                if clearance >= 0:
                    hits.append((clearance, group))
            nearest = min(hits, key=lambda h: h[0]) if hits else None
            samples.append({"section": section, "step": step, "point": point.tolist(),
                            "clearance": nearest[0] if nearest else None,
                            "group": nearest[1] if nearest else None,
                            "obstructed": bool(nearest and nearest[0] < required)})
    known = [s for s in samples if s["clearance"] is not None]
    blocked = [s for s in samples if s["obstructed"]]
    return {
        "required": required, "samples": samples,
        "minimum_clearance": min((s["clearance"] for s in known), default=None),
        "obstructed_samples": len(blocked), "unknown_samples": len(samples) - len(known),
        "obstructing_groups": sorted({s["group"] for s in blocked}),
        "clear": not blocked and len(known) == len(samples),
        "qualification": "Sampled vertical clearance only; no continuous collision or traversal proof",
    }
