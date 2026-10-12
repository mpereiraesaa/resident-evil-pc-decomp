"""Propose approximate closed objects from local masks and conditioned depth.

Hidden sides use geometric priors. Every proposal must still pass multiview
refinement gates; these meshes do not alter gameplay collision or scripts.
"""

import numpy as np
from room3d_preview import camera_basis


def masked_volume(camera, mask, depth, stride=6):
    """Closed local prisms preserve holes in thin fixtures instead of boxing air.

    Front positions follow estimated depth. Hidden thickness is an explicit
    small prior; this is approximate geometry, not a recovered lamp design.
    """
    parts = []
    rotation = camera_basis(camera)
    for y0 in range(0, mask.shape[0], stride):
        for x0 in range(0, mask.shape[1], stride):
            y1, x1 = min(y0 + stride, mask.shape[0]), min(x0 + stride, mask.shape[1])
            support = mask[y0:y1, x0:x1]
            if support.mean() < 0.35:
                continue
            z = float(np.median(depth[y0:y1, x0:x1][support]))
            observed = depth[y0:y1, x0:x1][support]
            if not np.isfinite(observed).all() or np.min(observed) <= 0:
                continue
            if np.percentile(observed, 90) / np.percentile(observed, 10) > 1.15:
                continue
            thickness = max(35, z * 0.005)
            xy = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
            front = np.column_stack(
                (
                    (xy[:, 0] - 160) * z / camera["fov"],
                    (xy[:, 1] - 120) * z / camera["fov"],
                    np.full(4, z),
                )
            )
            back = front.copy()
            back[:, 2] += thickness
            vertices = np.vstack((front, back)) @ rotation + camera["origin"]
            parts.append(vertices.tolist())
    return parts


def stair_proposals(room, camera, detection, points):
    """Extrapolate a stepped volume only where an inclined masked cloud supports it.

    The footprint comes from an otherwise unmodeled boundary; this does not
    assert that every boundary of that type is a staircase. Width and unseen
    run length are explicitly geometric priors.
    """
    result = []
    for boundary in room["boundaries"]:
        if boundary["type"] & 255 != 4:
            continue
        x0, x1, z0, z1 = (boundary[k] for k in ("xmin", "xmax", "zmin", "zmax"))
        inside = points[
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 2] >= z0)
            & (points[:, 2] <= z1)
        ]
        if len(inside) < 40:
            continue
        axis = 0 if x1 - x0 >= z1 - z0 else 2
        along = inside[:, axis]
        if np.ptp(along) < 1000 or np.ptp(inside[:, 1]) < 500:
            continue
        correlation = float(np.corrcoef(along, inside[:, 1])[0, 1])
        if not np.isfinite(correlation) or abs(correlation) < 0.65:
            continue
        slope, intercept = np.polyfit(along, inside[:, 1], 1)
        if not 0.2 < abs(slope) < 1.5:
            continue
        start, end = (x0, x1) if axis == 0 else (z0, z1)
        crossing = -intercept / slope
        if start < crossing < end:
            if slope < 0:
                start = crossing
            else:
                end = crossing
        top = min(slope * start + intercept, slope * end + intercept)
        if not -10000 < top < -600:
            continue
        cross = float(np.median(inside[:, 2 if axis == 0 else 0]))
        width = min(1500, (z1 - z0) if axis == 0 else (x1 - x0))
        cross_lo, cross_hi = (z0, z1) if axis == 0 else (x0, x1)
        side = float(np.clip(cross - width / 2, cross_lo, cross_hi - width))
        box = (
            [start, side, end, side + width, top, 0]
            if axis == 0
            else [side, start, side + width, end, top, 0]
        )
        result.append(
            {
                "kind": "staircase",
                "box": [float(v) for v in box],
                "axis": "x" if axis == 0 else "z",
                "rise_positive": bool(slope < 0),
                "estimated": "Incline from masked depth; width, hidden run and 220-unit step rise inferred",
                "evidence": {
                    "camera": camera["index"],
                    "label": detection["label"],
                    "boundary": boundary["index"],
                    "depth_points": len(inside),
                    "correlation": correlation,
                },
            }
        )
    return result


def object_proposals(room, segments, depths, existing):
    proposals = []
    if segments is None:
        return proposals
    bounds = room["boundaries"]
    room_lo = np.array(
        [min(b["xmin"] for b in bounds), -18000, min(b["zmin"] for b in bounds)]
    )
    room_hi = np.array(
        [max(b["xmax"] for b in bounds), 1000, max(b["zmax"] for b in bounds)]
    )
    for camera, entries, (depth, confident) in zip(
        room["cameras"], segments, depths, strict=True
    ):
        for detection, mask in entries:
            label = detection["label"]
            if detection["score"] < 0.3 or detection["predicted_iou"] < 0.8:
                continue
            kind = (
                "barrel"
                if "barrel" in label
                else (
                    "lamp"
                    if "lamp" in label or "chandelier" in label
                    else "railing"
                    if "railing" in label
                    else None
                )
            )
            if kind is None and "staircase" not in label:
                continue
            # Global depth-confidence thresholds can retain only an object's
            # bottom edge, collapsing its dimensions. Use its whole silhouette
            # for a proposal; the independent refinement gate still evaluates
            # on fixed confident pixels across every camera.
            y, x = np.nonzero(mask)
            if len(x) < 24:
                continue
            z = depth[y, x]
            view = np.column_stack(
                ((x - 160) * z / camera["fov"], (y - 120) * z / camera["fov"], z)
            )
            points = view @ camera_basis(camera) + camera["origin"]
            if "staircase" in label and kind != "railing":
                proposals.extend(stair_proposals(room, camera, detection, points))
                continue
            lo, hi = np.percentile(points, [5, 95], axis=0)
            center = (lo + hi) / 2
            if np.any(center < room_lo) or np.any(center > room_hi):
                continue
            span = hi - lo
            if kind == "barrel":
                # Circular hidden footprint inferred from the visible front width.
                radius = float(max(span[0], span[2]) / 2)
                if not 60 <= radius <= 1100 or not 200 <= span[1] <= 3000:
                    continue
                if abs(hi[1]) < 700:
                    hi[1] = 0
                box = [
                    center[0] - radius,
                    center[2] - radius,
                    center[0] + radius,
                    center[2] + radius,
                    lo[1],
                    hi[1],
                ]
            else:
                if not 50 <= max(span) <= (12000 if kind == "railing" else 3500):
                    continue
                span = np.maximum(span, 80)
                lo, hi = center - span / 2, center + span / 2
                box = [lo[0], lo[2], hi[0], hi[2], lo[1], hi[1]]
            # Deduplicate observations and avoid replacing script-owned furniture.
            duplicate = False
            for obj in [*existing, *proposals]:
                if kind == "railing" and obj["kind"] == "staircase":
                    continue
                if "box" not in obj:
                    continue
                b = obj["box"]
                overlap = max(0, min(box[2], b[2]) - max(box[0], b[0])) * max(
                    0, min(box[3], b[3]) - max(box[1], b[1])
                )
                area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
                if overlap / area > 0.5 and min(box[5], b[5]) > max(box[4], b[4]):
                    duplicate = True
                    break
            if duplicate:
                continue
            replacements = []
            for boundary in bounds:
                if boundary["flags"] & 0x100 or boundary["type"] & 255 not in (1, 5):
                    continue
                if (
                    boundary["xmin"] <= center[0] <= boundary["xmax"]
                    and boundary["zmin"] <= center[2] <= boundary["zmax"]
                ):
                    replacements.append(boundary["index"])
            proposals.append(
                {
                    "parts": masked_volume(camera, mask, depth)
                    if kind in ("lamp", "railing")
                    else [],
                    "kind": kind,
                    "replace_visual_boundaries": replacements,
                    "box": [float(v) for v in box],
                    "estimated": "Depth-supported visible extent; hidden sides completed by geometric prior",
                    "evidence": {
                        "camera": camera["index"],
                        "label": label,
                        "mask": detection.get("mask"),
                        "mask_pixels": int(mask.sum()),
                        "depth_points": len(x),
                        "confident_fraction": float(confident[mask].mean()),
                    },
                }
            )
            if kind in ("lamp", "railing") and not proposals[-1]["parts"]:
                proposals.pop()
    return proposals
