"""Extract semantic anchors from a deliberately bounded subset of init commands.

These are original gameplay anchors, not visual object recognition. Unsupported
commands stop extraction instead of resynchronizing on arbitrary byte patterns.
"""

import struct

# Total widths checked against current CmdFunctions.cpp for this subset.
WIDTHS = {
    0x00: 1,
    0x01: 2,
    0x02: 2,
    0x03: 2,
    0x04: 4,
    0x05: 4,
    0x06: 4,  # ResidentEvil.exe 0x00460760, verified through Ghidra.
    0x08: 4,
    0x0C: 26,
    0x0D: 18,
    0x12: 10,  # USA 0x00460fc0, verified action reset (zone pointer retained).
    0x14: 4,  # USA 0x00461040, verified event creation arguments.
    0x18: 26,
    0x1B: 22,
    0x1F: 28,
    0x20: 14,  # USA 0x00430f60, verified player position/rotation record.
    0x23: 2,  # USA 0x00431280, verified cut lock byte write.
    0x2B: 4,  # USA 0x00431990, verified attack animation arguments.
    0x31: 4,  # USA 0x004608d0, verified state word assignment.
    0x37: 4,  # USA 0x00460a30, verified room BGM state arguments.
    0x3B: 6,  # ResidentEvil.exe 0x00431ea0: three u16 operands including opcode.
}


def anchors(path):
    data = path.read_bytes()
    start, end = struct.unpack_from("<II", data, 0x60)
    result = {
        "anchors": [],
        "complete": True,
        "conditional_execution": "Not evaluated; anchors are potential script declarations",
    }
    if not 0x94 <= start <= end <= len(data):
        raise ValueError("Invalid init-script range")
    branch_depth = 0
    slot_zones = {}
    p = start
    while p + 2 <= end:
        size = struct.unpack_from("<H", data, p)[0]
        if size == 0:
            break
        if size < 2 or p + size > end:
            raise ValueError("Invalid init-script block")
        q = p + 2
        while q < p + size:
            op = data[q]
            width = WIDTHS.get(op)
            if op == 0x33:
                # USA 0x004314b0: player properties have subtype-dependent
                # widths. Unknown subtypes do not advance the original VM.
                if q + 2 > p + size:
                    raise ValueError("Truncated player property instruction")
                subtype = data[q + 1]
                if subtype in (0, 4, 6, 7):
                    width = 2
                elif subtype in (1, 3, 5, 8, 9, 10):
                    width = 4
            if width is None:
                result.update(complete=False, stopped_offset=q, unsupported_opcode=op)
                return result
            if q + width > p + size:
                raise ValueError("Init instruction crosses block end")
            if op == 0x01:
                branch_depth += 1
            elif op == 0x03:
                branch_depth = max(0, branch_depth - 1)
            if op in (0x0C, 0x0D):
                zone = list(struct.unpack_from("<4H", data, q + 2))
                slot_zones.setdefault(data[q + 1], set()).add(tuple(zone))
                kind = (
                    "door"
                    if op == 0x0C
                    else {8: "itembox", 16: "typewriter", 17: "stair_height"}.get(
                        data[q + 10]
                    )
                )
                if kind:
                    entry = {
                        "kind": kind,
                        "zone": zone,
                        "source_offset": q,
                        "slot": data[q + 1],
                        "conditional": branch_depth > 0,
                    }
                    if kind == "stair_height":
                        # USA 0x00461130 maps SCD +12/+14/+16 to action
                        # +2/+4/+6. USA 0x0041bf90 consumes direction, unsigned
                        # horizontal interval and signed height increment.
                        direction, interval, rise = struct.unpack_from(
                            "<HHh", data, q + 12
                        )
                        entry.update(
                            direction=direction,
                            horizontal_interval=interval,
                            height_increment=rise,
                            action_flags=data[q + 11],
                            valid_height_rule=direction < 4 and interval > 0,
                        )
                    if op == 0x0C:
                        entry["transition_type"] = data[q + 12]
                    result["anchors"].append(entry)
            elif op == 0x12 and data[q + 2] == 17:
                # USA 0x00460fc0 changes action fields but retains its zone.
                # Without evaluating branches, differing prior zones are ambiguous.
                zones = slot_zones.get(data[q + 1], set())
                if len(zones) != 1:
                    result.setdefault("unresolved_height_resets", []).append(q)
                else:
                    direction, interval, rise = struct.unpack_from("<HHh", data, q + 4)
                    result["anchors"].append(
                        {
                            "kind": "stair_height",
                            "zone": list(next(iter(zones))),
                            "source_offset": q,
                            "slot": data[q + 1],
                            "conditional": True,
                            "reset_requires_prior_zone": True,
                            "direction": direction,
                            "horizontal_interval": interval,
                            "height_increment": rise,
                            "action_flags": data[q + 3],
                            "valid_height_rule": direction < 4 and interval > 0,
                        }
                    )
            elif op == 0x1F:
                result["anchors"].append(
                    {
                        "kind": "dynamic_model",
                        "conditional": branch_depth > 0,
                        "slot": data[q + 1] & 0x3F,
                        "position": list(struct.unpack_from("<3h", data, q + 4)),
                        "source_offset": q,
                    }
                )
            q += width
        p += size
    return result


def primitives(room, script):
    """Generate simple furniture proposals, preserving dynamic model ownership."""
    items = []
    solids = [b for b in room["boundaries"] if b["flags"] & 0x100]
    circles = [b for b in room["boundaries"] if b["type"] & 255 == 3]
    for a in script["anchors"]:
        if a.get("conditional", False):
            continue
        if a["kind"] not in ("door", "typewriter", "itembox"):
            continue
        x, z, w, d = a["zone"]
        cx, cz = x + w / 2, z + d / 2
        if a["kind"] == "door":
            if a["transition_type"] >= 15 or not solids:
                continue
            # Snap a closed door panel to the nearest collision wall edge.
            candidates = []
            for b in solids:
                for axis, value, lo, hi, coord, along in [
                    ("x", b["xmin"], b["zmin"], b["zmax"], cx, cz),
                    ("x", b["xmax"], b["zmin"], b["zmax"], cx, cz),
                    ("z", b["zmin"], b["xmin"], b["xmax"], cz, cx),
                    ("z", b["zmax"], b["xmin"], b["xmax"], cz, cx),
                ]:
                    if lo <= along <= hi:
                        candidates.append((abs(coord - value), axis, value, b["index"]))
            if not candidates:
                continue
            distance, axis, value, boundary = min(candidates)
            if distance > 1800:
                continue
            # Extend toward the trigger centre so it is in front of the wall.
            value += 25 if (cx if axis == "x" else cz) > value else -25
            width = min(1800, max(700, d if axis == "x" else w))
            bounds = (
                [value - 12, cz - width / 2, value + 12, cz + width / 2, -3000, 0]
                if axis == "x"
                else [cx - width / 2, value - 12, cx + width / 2, value + 12, -3000, 0]
            )
            items.append(
                {
                    "kind": "door_panel",
                    "box": bounds,
                    "hide_with": f"boundary_{boundary}",
                    "source_offset": a["source_offset"],
                    "estimated": "Height, width, snap and closed panel thickness",
                }
            )
        elif a["kind"] == "typewriter":
            if circles:
                b = min(
                    circles,
                    key=lambda b: (
                        ((b["xmin"] + b["xmax"]) / 2 - cx) ** 2
                        + ((b["zmin"] + b["zmax"]) / 2 - cz) ** 2
                    ),
                )
                cx = (b["xmin"] + b["xmax"]) / 2
                cz = (b["zmin"] + b["zmax"]) / 2
            items.append(
                {
                    "kind": "typewriter_table",
                    "center": [cx, cz],
                    "source_offset": a["source_offset"],
                    "estimated": "Table and machine dimensions; interaction-zone-to-surface placement",
                }
            )
        else:
            models = [
                m
                for m in script["anchors"]
                if m["kind"] == "dynamic_model" and not m.get("conditional", False)
            ]
            if models:
                model = min(
                    models,
                    key=lambda m: (
                        (m["position"][0] - cx) ** 2 + (m["position"][2] - cz) ** 2
                    ),
                )
                px, py, pz = model["position"]
                items.append(
                    {
                        "kind": "chest_body",
                        "box": [px - 1080, pz, px + 1080, pz + 1440, py, 0],
                        "source_offset": a["source_offset"],
                        "estimated": "Body width/depth and orientation; original moving model excluded",
                    }
                )
    # Furniture overlapping a cutaway wall follows that wall's visibility.
    # This is a footprint heuristic, not evidence of the original object shape.
    for item in items:
        if "hide_with" in item:
            continue
        if "center" in item:
            cx, cz = item["center"]
            x0, z0, x1, z1 = cx - 650, cz - 400, cx + 650, cz + 400
        else:
            x0, z0, x1, z1 = item["box"][:4]
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        overlapping = [
            b
            for b in solids
            if b["type"] & 255 in (1, 5)
            and min(x1, b["xmax"]) > max(x0, b["xmin"])
            and min(z1, b["zmax"]) > max(z0, b["zmin"])
        ]
        if overlapping:
            wall = min(
                overlapping,
                key=lambda b: (
                    max(b["xmin"] - cx, 0, cx - b["xmax"]) ** 2
                    + max(b["zmin"] - cz, 0, cz - b["zmax"]) ** 2
                ),
            )
            item["hide_with"] = f"boundary_{wall['index']}"
            item["estimated"] += "; cutaway follows nearest overlapping wall"
    return items


def fit_detected_primitives(items, room, detections):
    """Fit vertical dimensions only when a detection contains the script anchor.

    3D centres and horizontal extents remain explicit proposals. Single-view boxes
    cannot establish object depth, nor validate the detector's semantic label.
    """
    import numpy as np
    from room3d_reconstruct import project, projection

    notes = []
    for item in items:
        label = {
            "door_panel": "door",
            "typewriter_table": "table",
            "chest_body": "storage chest",
        }[item["kind"]]
        if "box" in item:
            x0, z0, x1, z1, top, base = item["box"]
            x, z = (x0 + x1) / 2, (z0 + z1) / 2
        else:
            x, z = item["center"]
            top, base = -1550, 0
        estimates = []
        for camera, record in zip(room["cameras"], detections, strict=True):
            uv, depth = project([[x, (top + base) / 2, z]], camera)
            if depth[0] <= camera["fov"] * 2:
                continue
            for detection in sorted(
                record["detections"], key=lambda d: d["score"], reverse=True
            ):
                if label not in detection["label"] or detection["score"] < 0.4:
                    continue
                a, b, c, d = detection["box"]
                if not (a <= uv[0, 0] <= c and b <= uv[0, 1] <= d):
                    continue
                p = projection(camera)
                row = p[1] - b * p[2]
                if abs(row[1]) < 1e-6:
                    continue
                y = -(row[0] * x + row[2] * z + row[3]) / row[1]
                if -7000 < y < -300:
                    estimates.append(float(y))
                    break  # One independent height observation per camera.
        if not estimates:
            continue
        candidate = float(np.median(estimates))
        # Disagreeing views remain an unresolved proposal.
        accepted = max(estimates) - min(estimates) <= 600
        notes.append(
            {
                "kind": item["kind"],
                "top_y": candidate,
                "views": len(estimates),
                "accepted": accepted,
                "qualification": "Single-view height estimate"
                if len(estimates) == 1
                else "Cross-view height estimate",
            }
        )
        if accepted:
            if "box" in item:
                item["box"][4] = candidate
            else:
                item["table_top"] = candidate
    return notes


def fit_ceiling(room, detections):
    """Fit a horizontal ceiling only for a collision-supported rectangular room."""
    import cv2
    import numpy as np
    from room3d_preview import camera_basis, clip_near

    walls = [
        b for b in room["boundaries"] if b["flags"] & 0x100 and b["type"] & 255 == 1
    ]
    sx, sz = room["quadrant_split"]
    left = [
        b["xmax"] for b in walls if b["xmax"] <= sx and b["zmin"] <= sz <= b["zmax"]
    ]
    right = [
        b["xmin"] for b in walls if b["xmin"] >= sx and b["zmin"] <= sz <= b["zmax"]
    ]
    front = [
        b["zmax"] for b in walls if b["zmax"] <= sz and b["xmin"] <= sx <= b["xmax"]
    ]
    back = [
        b["zmin"] for b in walls if b["zmin"] >= sz and b["xmin"] <= sx <= b["xmax"]
    ]
    if not all((left, right, front, back)):
        return {"accepted": False, "reason": "No enclosing rectangle"}
    x0, x1, z0, z1 = max(left), min(right), max(front), min(back)
    if any(
        min(b["xmax"], x1) > max(b["xmin"], x0)
        and min(b["zmax"], z1) > max(b["zmin"], z0)
        for b in walls
    ):
        return {"accepted": False, "reason": "Interior wall splits the rectangle"}
    targets = []
    for camera, record in zip(room["cameras"], detections, strict=True):
        choices = [
            d
            for d in record["detections"]
            if d["label"] == "ceiling" and d["score"] >= 0.4
        ]
        if choices:
            targets.append((camera, max(choices, key=lambda d: d["score"])["box"]))
    if len(targets) < 2:
        return {"accepted": False, "reason": "Need ceiling evidence in two cameras"}
    scores = []
    frame = np.array([[0, 0], [320, 0], [320, 240], [0, 240]], np.float32)
    for height in range(3000, 9001, 100):
        corners = np.array(
            [
                [x0, -height, z0],
                [x0, -height, z1],
                [x1, -height, z1],
                [x1, -height, z0],
            ],
            float,
        )
        errors = []
        for camera, box in targets:
            view = (corners - camera["origin"]) @ camera_basis(camera).T
            clipped = np.array(clip_near(view, camera["fov"] * 2))
            if len(clipped) < 3:
                errors.append(1000)
                continue
            uv = (clipped[:, :2] / clipped[:, 2:3] * camera["fov"] + [160, 120]).astype(
                np.float32
            )
            _, polygon = cv2.intersectConvexConvex(cv2.convexHull(uv), frame)
            if polygon is None:
                errors.append(1000)
                continue
            pts = polygon.reshape(-1, 2)
            errors.append(
                abs(float(pts[:, 1].max()) - box[3])
                + 0.1
                * (
                    abs(float(pts[:, 0].min()) - box[0])
                    + abs(float(pts[:, 0].max()) - box[2])
                )
            )
        scores.append((sum(errors) / len(errors), height))
    error, height = min(scores)
    accepted = error < 15 and 3000 < height < 9000
    return {
        "accepted": accepted,
        "height": height,
        "mean_edge_error_px": error,
        "views": len(targets),
        "box": [x0, z0, x1, z1, -height - 50, -height],
        "qualification": "Estimated ceiling from detector boxes and rectangular-wall hypothesis",
    }
