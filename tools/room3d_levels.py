"""Horizontal-line hypotheses from calibrated views; no inferred depth required."""

import cv2
import numpy as np
from room3d_preview import camera_basis


def base_lines(mask):
    """Piecewise lines on the lower envelope of a railing mask."""
    edge = np.zeros(mask.shape, np.uint8)
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, x])
        if len(ys):
            edge[ys[-1], x] = 255
    lines = cv2.HoughLinesP(edge, 1, np.pi / 360, 18, minLineLength=30, maxLineGap=8)
    if lines is None:
        return []
    result = []
    for line in lines.reshape(-1, 4):
        a, b = line.reshape(2, 2).astype(float)
        if a[0] > b[0]:
            a, b = b, a
        if b[0] - a[0] < 30:
            continue
        # Deterministic near-duplicate suppression in image space.
        if any(np.linalg.norm(np.array([a, b]) - old) < 12 for old in result):
            continue
        result.append(np.array([a, b]))
    return result


def rays(camera, points):
    points = np.asarray(points)
    local = np.column_stack(
        (
            (points[:, 0] - 160) / camera["fov"],
            (points[:, 1] - 120) / camera["fov"],
            np.ones(len(points)),
        )
    )
    return local @ camera_basis(camera)


def shared_line(camera_a, line_a, camera_b, line_b):
    """Intersect interpretation planes and require overlapping observed spans.

    Different physical lines can share an epipolar solution. Returned hypotheses
    still need appearance association and other-camera validation.
    """
    ca, cb = (
        np.asarray(camera_a["origin"], float),
        np.asarray(camera_b["origin"], float),
    )
    ra, rb = rays(camera_a, line_a), rays(camera_b, line_b)
    na, nb = np.cross(*ra), np.cross(*rb)
    norms = np.array([np.linalg.norm(na), np.linalg.norm(nb)])
    if min(norms) < 1e-9:
        return None
    na, nb = na / norms[0], nb / norms[1]
    direction = np.cross(na, nb)
    separation = np.linalg.norm(direction)
    if separation < 0.05:
        return None
    direction /= separation
    if abs(direction[1]) > 0.08:
        return None
    point = np.linalg.lstsq(
        np.array([na, nb, direction]), [na @ ca, nb @ cb, 0], rcond=None
    )[0]
    spans = []
    for origin, ray_set in ((ca, ra), (cb, rb)):
        values = []
        for ray in ray_set:
            solution = np.linalg.lstsq(
                np.column_stack((direction, -ray)), origin - point, rcond=None
            )[0]
            if solution[1] <= 0:
                return None
            values.append(solution[0])
        spans.append(sorted(values))
    lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
    if hi - lo < 300:
        return None
    endpoints = point + np.array([[lo], [hi]]) * direction
    return {
        "endpoints": endpoints.tolist(),
        "height": float(endpoints[:, 1].mean()),
        "vertical_drift": float(np.ptp(endpoints[:, 1])),
        "length": float(hi - lo),
        "plane_separation": float(separation),
        "qualification": "Geometric correspondence hypothesis; object identity and landing extent unverified",
    }


def corroborate(hypothesis, camera, image_line):
    """Check an unused camera's segment against the recovered finite 3D line."""
    from room3d_reconstruct import project

    uv, depth = project(np.asarray(hypothesis["endpoints"]), camera)
    line = np.asarray(image_line, float)
    axis = line[1] - line[0]
    length = np.linalg.norm(axis)
    if length < 1 or not np.isfinite(uv).all() or (depth <= 2 * camera["fov"]).any():
        return {"valid": False}
    axis /= length
    offsets = uv - line[0]
    error = float(np.abs(offsets @ np.array([-axis[1], axis[0]])).max())
    span = np.sort(offsets @ axis)
    overlap = float(max(0, min(length, span[1]) - max(0, span[0])))
    return {
        "valid": error <= 3 and overlap >= 20,
        "pixel_error": error,
        "overlap_pixels": overlap,
    }


def infer_levels(room, segments):
    """Collect shared railing/base edges without per-room IDs or depth inputs."""
    from itertools import combinations

    observations = []
    for camera, entries in zip(room["cameras"], segments, strict=True):
        for detection, mask in entries:
            if (
                detection["score"] < 0.3
                or detection["predicted_iou"] < 0.8
                or not any(
                    word in detection["label"] for word in ("railing", "balcony")
                )
            ):
                continue
            for line in base_lines(mask):
                observations.append(
                    {"camera": camera, "line": line, "mask": detection["mask"]}
                )
    hypotheses = []
    for a, b in combinations(observations, 2):
        if a["camera"]["index"] == b["camera"]["index"]:
            continue
        hypothesis = shared_line(a["camera"], a["line"], b["camera"], b["line"])
        if hypothesis is None:
            continue
        source_cameras = [a["camera"]["index"], b["camera"]["index"]]
        checks = []
        for obs in observations:
            camera = obs["camera"]
            if camera["index"] in source_cameras:
                continue
            checks.append(
                {
                    "camera": camera["index"],
                    "mask": obs["mask"],
                    **corroborate(hypothesis, camera, obs["line"]),
                }
            )
        corroborating = sorted({r["camera"] for r in checks if r["valid"]})
        hypotheses.append(
            {
                **hypothesis,
                "cameras": source_cameras,
                "image_lines": [a["line"].tolist(), b["line"].tolist()],
                "checks": checks,
                "corroborating_cameras": corroborating,
                "corroborated": bool(corroborating),
            }
        )
    return {
        "thresholds": {"detector_score": 0.3, "mask_iou": 0.8},
        "observations": len(observations),
        "hypotheses": hypotheses,
        "corroborated": sum(h["corroborated"] for h in hypotheses),
        "qualification": "Development geometric hypotheses; identity, landing extent and held-out generalization unverified",
    }


def floor_anchors(camera, mask, floor_y=0):
    """Candidate feet from the mask's low image band, not monocular depth.

    Assumes that the visible low band touches this floor; occluded feet and
    multi-level floors must be handled by subsequent fitting/validation.
    """
    y, x = np.nonzero(mask)
    if len(x) < 24:
        return []
    band = y >= np.percentile(y, 95)
    pixels = np.column_stack(
        (np.quantile(x[band], [0.2, 0.5, 0.8]), np.repeat(np.median(y[band]), 3))
    )
    directions = rays(camera, pixels)
    origin = np.asarray(camera["origin"], float)
    result = []
    for pixel, direction in zip(pixels, directions, strict=True):
        if abs(direction[1]) < 1e-6:
            continue
        distance = (floor_y - origin[1]) / direction[1]
        if distance <= 2 * camera["fov"]:
            continue
        point = origin + distance * direction
        result.append(
            {
                "point": point.tolist(),
                "pixel": pixel.tolist(),
                "qualification": "Low mask band assumed to touch supplied floor",
            }
        )
    return result


def stair_level_starts(group, report, floor_y=0, excluded_cameras=()):
    """Bounded floor-ray starts with independently corroborated level priors."""
    import math

    from room3d_shapes import generate

    excluded = set(excluded_cameras)
    levels = []
    for hypothesis in report.get("hypotheses", []):
        used = set(hypothesis["cameras"]) | set(
            hypothesis.get("corroborating_cameras", [])
        )
        height = hypothesis["height"]
        if (
            not hypothesis.get("corroborated")
            or used & excluded
            or not 400 <= floor_y - height <= 12000
            or any(abs(height - prior["height"]) < 250 for prior in levels)
        ):
            continue
        levels.append(hypothesis)
        if len(levels) == 2:
            break
    for level in levels:
        rise = floor_y - level["height"]
        for observation in group:
            for anchor in floor_anchors(
                observation["camera_model"], observation["mask"], floor_y
            ):
                for yaw in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
                    for gradient in (0.5, 0.7):
                        obj = generate(
                            "staircase",
                            {
                                "width": 1800,
                                "run": rise / gradient,
                                "rise": rise,
                                "steps": int(np.clip(round(rise / 180), 2, 64)),
                                "landing": 0,
                            },
                            anchor["point"],
                            yaw,
                        )
                        obj["initialization"] = f"floor_ray_level_{gradient}"
                        obj["level_prior"] = {
                            "height": level["height"],
                            "cameras": sorted(
                                set(level["cameras"])
                                | set(level["corroborating_cameras"])
                            ),
                            "qualification": "Corroborated height hypothesis; landing identity unverified",
                        }
                        yield obj
