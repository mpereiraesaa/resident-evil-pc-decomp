"""Creative stair/landing proposals anchored to multi-view edge hypotheses."""

import copy
import math

import numpy as np
from room3d_shapes import generate, prism


def stair_proportions(stairs):
    """Measure the flight excluding the horizontal landing, in game units."""
    p = stairs["shape"]["parameters"]
    flight = p["run"] - p.get("landing", 0)
    rise = p["rise"] / p["steps"]
    tread = flight / p["steps"]
    slope = math.degrees(math.atan2(p["rise"], flight))
    return {
        "step_rise": rise,
        "step_depth": tread,
        "flight_slope_degrees": slope,
        "plausible": 120 <= rise <= 240 and tread >= 200 and 15 <= slope <= 42,
        "qualification": "Procedural scale prior, not recovered dimensions or traversal proof",
    }


def regular_stair_variants(stairs):
    """Keep the footprint and pose; redistribute the flight into regular steps."""
    shape = stairs["shape"]
    p = shape["parameters"]
    seen = set()
    for landing in (p.get("landing", 0), 0):
        for target_rise in (160, 180, 220):
            steps = int(np.clip(round(p["rise"] / target_rise), 2, 64))
            key = (landing, steps)
            if key in seen:
                continue
            seen.add(key)
            candidate = generate(
                "staircase", {**p, "landing": landing, "steps": steps},
                shape["origin"], shape["yaw"],
            )
            if stair_proportions(candidate)["plausible"]:
                yield candidate


def feasible_stair_seed(stairs):
    """Project a noisy initialization into the procedural feasible range.

    Preserve width, base pose, total rise and landing; the upper endpoint can
    move. This is an estimated alternative, never a correction of evidence.
    """
    if stairs.get("level_prior") or stair_proportions(stairs)["plausible"]:
        return None
    shape = stairs["shape"]
    p = shape["parameters"]
    lower = max(2, math.ceil(p["rise"] / 240))
    upper = min(64, math.floor(p["rise"] / 120))
    if lower > upper:
        return None
    steps = int(np.clip(round(p["rise"] / 180), lower, upper))
    flight = p["run"] - p.get("landing", 0)
    # Leave margin from the hard inclination limits so local search can move.
    slope = math.atan2(p["rise"], flight)
    slope = float(np.clip(slope, math.radians(18), math.radians(40)))
    flight = max(p["rise"] / math.tan(slope), steps * 200)
    candidate = generate("staircase", {**p, "steps": steps,
                         "run": flight + p.get("landing", 0)},
                         shape["origin"], shape["yaw"])
    if not stair_proportions(candidate)["plausible"]:
        return None
    candidate["initialization"] = "feasible_" + stairs.get("initialization", "seed")
    candidate["interpretation"] = {
        "style": "classic_mansion", "source": "regular_step_prior",
        "qualification": "Estimated feasible initialization; upper endpoint and traversal unverified",
    }
    return candidate


def regularize_stairs(stairs, scores, evaluate):
    """Select plausible detail using training scores; final scene gates are external."""
    report = {"before": stair_proportions(stairs), "trials": [], "selected": None}
    if report["before"]["plausible"]:
        return stairs, scores, report
    eligible = []
    for candidate in regular_stair_variants(stairs):
        after = evaluate(candidate)
        keep = bool(after) and len(after) == len(scores) and all(
            a["loss"] <= b["loss"] + 0.1
            and a["silhouette_iou"] >= 0.35
            and a.get("contact_valid", True)
            for a, b in zip(after, scores, strict=True)
        )
        index = len(report["trials"])
        report["trials"].append({
            "shape": candidate["shape"], "scores": after, "eligible": keep,
            "proportions": stair_proportions(candidate),
        })
        if keep:
            eligible.append((float(np.mean([s["loss"] for s in after])),
                             index, candidate, after))
    if not eligible:
        return stairs, scores, report
    _, index, candidate, after = min(eligible, key=lambda item: item[0])
    report["selected"] = index
    candidate["interpretation"] = {
        "style": "classic_mansion", "source": "regular_step_prior",
        "qualification": "Procedural proportions; not recovered original step geometry",
    }
    return candidate, after, report


def stair_landings(edge, floor_y=0, width_fraction=0.8):
    """Infer extent on both sides; the observed edge is not a recovered floor."""
    if not 0.4 <= width_fraction <= 1:
        raise ValueError("Width fraction must be within 0.4..1")
    endpoints = np.asarray(edge["endpoints"], float)
    center = endpoints.mean(0)
    center[1] = edge["height"]
    across = endpoints[1] - endpoints[0]
    across[1] = 0
    length = np.linalg.norm(across)
    rise = floor_y - center[1]
    if length < 600 or rise < 400:
        return
    across /= length
    width = min(2400.0, length * width_fraction)
    for side in (-1, 1):
        along = np.array([-across[2], 0, across[0]]) * side
        yaw = math.atan2(along[0], along[2])
        for gradient in (0.5, 0.7):
            run = rise / gradient
            origin = center - along * run
            origin[1] = floor_y
            stairs = generate(
                "staircase",
                {
                    "width": width,
                    "run": run,
                    "rise": rise,
                    "steps": int(np.clip(round(rise / 180), 2, 64)),
                    "landing": 0,
                },
                origin,
                yaw,
            )
            for depth in (1000.0, 2000.0):
                landing = landing_for_stairs(stairs, depth)
                yield {
                    "stairs": stairs,
                    "landing": landing,
                    "side": side,
                    "depth": depth,
                    "gradient": gradient,
                    "edge": edge,
                    "qualification": "Proposed assembly, not validated geometry",
                }


def landing_for_stairs(stairs, depth=1000):
    """Attach a proposed slab to a flight independently of its pose source."""
    shape = stairs["shape"]
    p = shape["parameters"]
    yaw = shape["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    center = np.array([0, -p["rise"], p["run"]]) @ rotation + shape["origin"]
    vertices = (
        np.asarray(prism([-p["width"] / 2, 0, 0], [p["width"] / 2, 150, depth]))
        @ rotation
        + center
    )
    lo, hi = vertices.min(0), vertices.max(0)
    return {
        "kind": "landing",
        "parts": [vertices.tolist()],
        "box": [lo[0], lo[2], hi[0], hi[2], lo[1], hi[1]],
        "estimated": "Creative landing attached to flight; structural support and traversal unverified",
    }


def gallery_for_stairs(stairs, bounds, walkway_width=2000, thickness=150, minimum_landing_depth=1000):
    """Propose a perimeter gallery joined to a stair top, with an open flight.

    The collision envelope bounds a creative layout; it does not recover walls
    or floor extent. No room ID, model depth or hand-authored room coordinates
    participate. The caller must validate imagery, overhead and traversal.
    """
    from room3d_floor import (
        clip_halfplane,
        floor_patches,
        subtract_convex,
        validated_polygon,
    )

    limits = np.asarray(bounds, float)
    if limits.shape != (4,) or not np.isfinite(limits).all():
        raise ValueError("Expected finite XZ bounds")
    x0, z0, x1, z1 = limits
    sizes = np.array([x1 - x0, z1 - z0])
    if (sizes <= 0).any():
        raise ValueError("Gallery bounds must have positive extent")
    if (not np.isfinite([walkway_width, thickness, minimum_landing_depth]).all()
            or not 0 < walkway_width < min(sizes) / 2
            or thickness <= 0 or minimum_landing_depth <= 0):
        raise ValueError("Gallery width must leave a central void; slab and landing must be positive")
    if stairs.get("kind") != "staircase":
        raise ValueError("Gallery requires a parametric staircase")
    shape = stairs["shape"]
    p = shape["parameters"]
    # Reuse the generator's parameter/pose validation instead of trusting a box.
    generate("staircase", p, shape["origin"], shape["yaw"])
    origin = np.asarray(shape["origin"], float)
    along = np.array([math.sin(shape["yaw"]), math.cos(shape["yaw"])])
    across = np.array([along[1], -along[0]])
    start = origin[[0, 2]]
    top = start + along * p["run"]
    half = across * p["width"] / 2
    opening = validated_polygon([start - half, start + half, top + half, top - half])
    lo, hi = limits[:2], limits[2:]
    if ((top - half < lo).any() or (top - half > hi).any()
            or (top + half < lo).any() or (top + half > hi).any()):
        raise ValueError("Stair top must fit inside the gallery envelope")

    def exit_distance(lower, upper):
        hits = [(upper[i] - top[i]) / along[i] if along[i] > 0
                else (lower[i] - top[i]) / along[i]
                for i in range(2) if abs(along[i]) > 1e-9]
        return min(hits)

    available = exit_distance(lo, hi)
    if available <= 200:
        raise ValueError("Stair top leaves no room for a supported exit")
    inner_lo, inner_hi = lo + walkway_width, hi - walkway_width
    inner = validated_polygon([inner_lo, [inner_hi[0], inner_lo[1]],
                               inner_hi, [inner_lo[0], inner_hi[1]]])
    inside_void = bool(((top > inner_lo) & (top < inner_hi)).all())
    distance = exit_distance(inner_lo, inner_hi) if inside_void else 0
    depth = min(available, max(minimum_landing_depth,
                              distance + min(walkway_width / 2, minimum_landing_depth)))
    connector = validated_polygon([top - half, top + half,
                                   top + along * depth + half, top + along * depth - half])
    outer = validated_polygon([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    for a, b in zip(outer, np.roll(outer, -1, axis=0), strict=True):
        connector = clip_halfplane(connector, a, b, True)
        if not len(connector):
            raise ValueError("Gallery connector was clipped away")
    # One coarse cell leaves only polygon boundaries; export tessellation is
    # separate. Subtracting the flight also opens any perimeter strip it crosses.
    patches = floor_patches(limits, [{"polygon": inner}, {"polygon": opening}],
                            tile_size=float(max(sizes)))
    if not patches:
        raise ValueError("No perimeter gallery remains outside the flight")
    # Partition the overlap rather than stacking coplanar surfaces where the
    # connector enters the perimeter strip (which would cause z-fighting).
    patches = [fragment for patch in patches
               for fragment in subtract_convex(validated_polygon(patch), connector)]
    patches.extend(subtract_convex(connector, opening))
    height = float(origin[1] - p["rise"])
    parts = []
    for patch in patches:
        polygon = validated_polygon(patch)
        # Clipping can retain a vertex on a straight edge. Remove it before
        # midpoint subdivision: the exporter's reversed bottom-face fan would
        # otherwise contain a zero-area triangle through three edge points.
        incoming = polygon - np.roll(polygon, 1, axis=0)
        outgoing = np.roll(polygon, -1, axis=0) - polygon
        turns = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        lengths = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
        polygon = validated_polygon(polygon[np.abs(turns) > 1e-10 * lengths])
        center = polygon.mean(0)
        # Split each convex polygon into proper quadrilateral prisms. This
        # preserves the exporter's eight-vertex parts without degenerate faces.
        for i, point in enumerate(polygon):
            quad = [point, (point + polygon[(i + 1) % len(polygon)]) / 2,
                    center, (polygon[i - 1] + point) / 2]
            parts.append([[float(x), y, float(z)]
                          for y in (height, height + thickness) for x, z in quad])
    return {
        "kind": "landing", "parts": parts,
        "box": [float(x0), float(z0), float(x1), float(z1), height, height + thickness],
        "gallery": {"bounds": limits.tolist(), "walkway_width": float(walkway_width),
                    "thickness": float(thickness), "connector_depth": float(depth),
                    "flight_opening": opening.tolist()},
        "estimated": "Creative perimeter gallery from collision envelope and stair endpoint; original layout, wall support and traversal unverified",
    }


def corroborated_stair_assemblies(report, floor_y=0, excluded_cameras=()):
    """Bounded connected proposals at independently observed upper edges.

    The edge constrains the upper endpoint as well as its height. A proposed
    landing is estimated geometry, never independent evidence of support.
    Excluded evaluation cameras cannot contribute either source or checks.
    """
    excluded = set(excluded_cameras)
    selected = []
    for edge in report.get("hypotheses", []):
        used = set(edge["cameras"]) | set(edge.get("corroborating_cameras", []))
        endpoints = np.asarray(edge["endpoints"], float)
        if (
            not edge.get("corroborated")
            or len(used) < 3
            or used & excluded
            or endpoints.shape != (2, 3)
            or not np.isfinite(endpoints).all()
            or not 400 <= floor_y - edge["height"] <= 12000
        ):
            continue
        center = endpoints.mean(0)
        direction = endpoints[1] - endpoints[0]
        length = np.linalg.norm(direction)
        if length < 600:
            continue
        direction /= length
        if any(
            np.linalg.norm(center - prior_center) < 250
            and abs(direction @ prior_direction) > 0.98
            for prior_center, prior_direction in selected
        ):
            continue
        selected.append((center, direction))
        for fraction in (0.5, 0.8):
            for assembly in stair_landings(edge, floor_y, fraction):
                if not stair_proportions(assembly["stairs"])["plausible"]:
                    continue
                yield {
                    **assembly,
                    "source_cameras": sorted(used),
                    "width_fraction": fraction,
                    "qualification": "Creative connected assembly anchored to a corroborated edge; identity, landing extent and structural support unverified",
                }
        if len(selected) == 2:
            break


def ceiling_hypotheses(recipe, upper_y, clearances=(1800, 2500, 3500)):
    """Move only explicitly estimated enclosing slabs above an upper-level prior.

    This proposes room geometry; it does not certify headroom or extend walls.
    Callers must check all cameras and reject unsupported changes.
    """
    if not math.isfinite(upper_y):
        raise ValueError("Upper level must be finite")
    targets = [
        i
        for i, item in enumerate(recipe.get("primitives", []))
        if item.get("kind") == "ceiling" and item.get("estimated") and "box" in item
    ]
    for clearance in clearances:
        if not math.isfinite(clearance) or clearance <= 0:
            raise ValueError("Clearance must be finite and positive")
        candidate = copy.deepcopy(recipe)
        changed = []
        for index in targets:
            box = candidate["primitives"][index]["box"]
            if box[5] <= upper_y - clearance:
                continue
            thickness = box[5] - box[4]
            if thickness <= 0:
                raise ValueError("Invalid ceiling thickness")
            box[5] = upper_y - clearance
            box[4] = box[5] - thickness
            changed.append(index)
        if changed:
            yield (
                candidate,
                {
                    "clearance": clearance,
                    "changed_ceilings": changed,
                    "qualification": "Estimated enclosure hypothesis; walls and headroom need validation",
                },
            )


def zoned_ceiling(recipe, index, axis, cut, lower_y):
    """Split an inferred slab, lowering its positive-axis side with a fascia."""
    if axis not in (0, 2):
        raise ValueError("Ceiling split axis must be X or Z")
    candidate = copy.deepcopy(recipe)
    item = candidate["primitives"][index]
    if item.get("kind") != "ceiling" or not item.get("estimated"):
        raise ValueError("Only inferred ceilings can be split")
    x0, z0, x1, z1, y0, y1 = item["box"]
    lo = np.array([x0, y0, z0], float)
    hi = np.array([x1, y1, z1], float)
    if (
        not np.isfinite([cut, lower_y]).all()
        or not lo[axis] < cut < hi[axis]
        or lower_y <= y1
    ):
        raise ValueError("Invalid ceiling split")
    thickness = y1 - y0
    if thickness <= 0:
        raise ValueError("Invalid slab thickness")
    first_hi = hi.copy()
    first_hi[axis] = cut
    second_lo = lo.copy()
    second_lo[axis] = cut
    second_lo[1] = lower_y - thickness
    second_hi = hi.copy()
    second_hi[1] = lower_y
    fascia_lo = lo.copy()
    fascia_lo[axis] = max(lo[axis], cut - thickness / 2)
    fascia_hi = hi.copy()
    fascia_hi[axis] = min(hi[axis], cut + thickness / 2)
    fascia_hi[1] = lower_y
    item["parts"] = [
        prism(lo, first_hi),
        prism(second_lo, second_hi),
        prism(fascia_lo, fascia_hi),
    ]
    item["box"][5] = lower_y
    item["ceiling_zones"] = {
        "axis": axis,
        "cut": float(cut),
        "upper_y": float(y1),
        "lower_y": float(lower_y),
    }
    return candidate


def raised_ceiling_patch(recipe, index, footprint, upper_y):
    """Raise a rectangular patch of an inferred slab, closing its four edges."""
    candidate = copy.deepcopy(recipe)
    item = candidate["primitives"][index]
    if item.get("kind") != "ceiling" or not item.get("estimated") or item.get("parts"):
        raise ValueError("Expected an unsplit inferred ceiling slab")
    x0, z0, x1, z1, y0, y1 = item["box"]
    a, b, c, d = footprint
    if not np.isfinite([a, b, c, d, upper_y]).all() or upper_y >= y0:
        raise ValueError("Invalid raised ceiling patch")
    a, b, c, d = max(x0, a), max(z0, b), min(x1, c), min(z1, d)
    thickness = y1 - y0
    if a >= c or b >= d or thickness <= 0:
        raise ValueError("Patch must overlap a positive-thickness slab")
    parts = []
    for left, front, right, back in (
        (x0, z0, a, z1), (c, z0, x1, z1),
        (a, z0, c, b), (a, d, c, z1),
    ):
        if right > left and back > front:
            parts.append(prism([left, y0, front], [right, y1, back]))
    parts.append(prism([a, upper_y - thickness, b], [c, upper_y, d]))
    # Fascias stay within the patch; callers reserve lateral clearance.
    for left, front, right, back in (
        (a, b, min(c, a + thickness), d),
        (max(a, c - thickness), b, c, d),
        (a, b, c, min(d, b + thickness)),
        (a, max(b, d - thickness), c, d),
    ):
        parts.append(prism([left, upper_y, front], [right, y1, back]))
    item["parts"] = parts
    item["box"][4] = upper_y - thickness
    item["ceiling_patch"] = {"footprint": [a, b, c, d], "upper_y": upper_y}
    return candidate


def stair_ceiling_patches(recipe, stairs):
    """Bounded stair-footprint proposals without room-specific coordinates."""
    box = stairs["box"]
    upper = stairs["shape"]["origin"][1] - stairs["shape"]["parameters"]["rise"]
    for index, item in enumerate(recipe.get("primitives", [])):
        if item.get("kind") != "ceiling" or not item.get("estimated") or item.get("parts"):
            continue
        for margin in (250, 750):
            footprint = [box[0] - margin, box[1] - margin,
                         box[2] + margin, box[3] + margin]
            for clearance in (1800, 2500):
                try:
                    candidate = raised_ceiling_patch(recipe, index, footprint, upper - clearance)
                except ValueError:
                    continue
                yield candidate, {"ceiling": index, "margin": margin,
                                  "clearance": clearance,
                                  "qualification": "Local inferred overhead; scene checks required"}
