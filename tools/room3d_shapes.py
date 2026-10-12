"""Reusable geometric priors, independent of room IDs and image textures.

Coordinates use game Y (negative upwards). Parts are closed eight-vertex
prisms accepted by batch_room3d.generate_mesh. Unseen geometry is inferred.
"""

import math
from itertools import pairwise

import numpy as np

VERSION = 1


def prism(lower, upper):
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all():
        raise ValueError("Expected finite 3D bounds")
    if np.any(hi <= lo):
        raise ValueError("Prism dimensions must be positive")
    x, y, z = lo
    X, Y, Z = hi
    return [
        [x, y, z],
        [X, y, z],
        [X, Y, z],
        [x, Y, z],
        [x, y, Z],
        [X, y, Z],
        [X, Y, Z],
        [x, Y, Z],
    ]


def beam(start, end, radius):
    a, b = np.asarray(start, float), np.asarray(end, float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
        raise ValueError("Expected finite beam endpoints")
    length = np.linalg.norm(b - a)
    if length <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Beam needs positive length and radius")
    direction = (b - a) / length
    helper = np.eye(3)[np.argmin(np.abs(direction))]
    u = np.cross(direction, helper)
    u *= radius / np.linalg.norm(u)
    v = np.cross(direction, u)
    return np.array(
        [
            p + su * u + sv * v
            for p in (a, b)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    ).tolist()


def _positive(params, name):
    value = float(params[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _count(params, name, minimum, maximum):
    value = params[name]
    if (
        isinstance(value, bool)
        or int(value) != value
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} outside integer range {minimum}..{maximum}")
    return int(value)


def local_parts(kind, p):
    """Generate in local frame: X across, Z along; base at origin."""
    parts = []
    if kind == "column":
        height, width = (_positive(p, k) for k in ("height", "width"))
        base = float(p.get("base_height", height * 0.08))
        capital = float(p.get("capital_height", height * 0.08))
        if not np.isfinite([base, capital]).all() or min(base, capital) <= 0 or base + capital >= height:
            raise ValueError("Column base and capital must leave a positive shaft")
        radius = width / 2
        # Six quadrilateral sectors form a twelve-sided shaft while retaining
        # the eight-vertex part format used by the room mesh exporter.
        for start in np.arange(6) * math.pi / 3:
            ring = [[0, 0], *[[radius * math.cos(a), radius * math.sin(a)]
                              for a in start + np.arange(3) * math.pi / 6]]
            parts.append([[x, y, z] for y in (-height + capital, -base) for x, z in ring])
        half = width * 0.75
        parts.append(prism([-half, -base, -half], [half, 0, half]))
        parts.append(prism([-half, -height, -half], [half, -height + capital, half]))
    elif kind == "ladder":
        width, height, thickness, stand_off = (
            _positive(p, name) for name in ("width", "height", "thickness", "stand_off")
        )
        rungs = _count(p, "rungs", 2, 64)
        if (width <= 4 * thickness or height <= 2 * thickness * (rungs + 1)
                or stand_off < thickness):
            raise ValueError("Ladder needs open rung spacing, rail clearance and positive wall stand-off")
        half_span = (width - thickness) / 2
        # Rails end on the base/top planes. Rungs intersect their center lines;
        # four brackets connect the rails to the local wall plane Z=0.
        for x in (-half_span, half_span):
            parts.append(beam([x, 0, stand_off], [x, -height, stand_off], thickness / 2))
        for rung in range(rungs):
            y = -height * (rung + 1) / (rungs + 1)
            parts.append(prism([-half_span, y - thickness / 2, stand_off - thickness / 2],
                               [half_span, y + thickness / 2, stand_off + thickness / 2]))
        for x in (-half_span, half_span):
            for y in (-height * .1, -height * .9):
                parts.append(beam([x, y, 0], [x, y, stand_off], thickness / 2))
    elif kind == "staircase":
        width, run, rise = (_positive(p, k) for k in ("width", "run", "rise"))
        steps = _count(p, "steps", 2, 64)
        landing = float(p.get("landing", 0))
        if not math.isfinite(landing) or not 0 <= landing < run:
            raise ValueError("Landing must be shorter than the run")
        structure = p.get("structure", "solid")
        if structure not in ("solid", "suspended"):
            raise ValueError("Unknown staircase structure")
        slab = float(p.get("slab_depth", 100))
        if structure == "suspended" and (not math.isfinite(slab) or not 0 < slab <= 240):
            raise ValueError("Suspended staircase slab depth must be within 0..240")
        flight = run - landing
        for i in range(steps):
            top = -rise * (i + 1) / steps
            bottom = min(0, top + rise / steps + slab) if structure == "suspended" else 0
            parts.append(
                prism(
                    [-width / 2, top, flight * i / steps],
                    [width / 2, bottom, flight * (i + 1) / steps],
                )
            )
        if landing:
            bottom = min(0, -rise + slab) if structure == "suspended" else 0
            parts.append(prism([-width / 2, -rise, flight], [width / 2, bottom, run]))
    elif kind == "railing":
        if p.get('profile') == 'handrail_segments':
            segments = np.asarray(p['segments'], float)
            if (segments.ndim != 3 or segments.shape[1:] != (2, 3)
                    or not 1 <= len(segments) <= 8 or not np.isfinite(segments).all()):
                raise ValueError('Expected one to eight finite handrail segments')
            thickness = _positive(p, 'thickness')
            return [beam(start, end, thickness/2) for start, end in segments]
        run, height, thickness = (
            _positive(p, k) for k in ("run", "height", "thickness")
        )
        profile = p.get('profile', 'balusters')
        if profile not in ('balusters', 'handrail'):
            raise ValueError('Unknown railing profile')
        posts = _count(p, "posts", 2, 64) if profile == 'balusters' else 0
        rise = float(p.get("rise", 0))
        if not math.isfinite(rise):
            raise ValueError("Railing rise must be finite")
        for i in range(posts):
            t = i / (posts - 1)
            parts.append(
                beam(
                    [0, -rise * t, run * t],
                    [0, -rise * t - height, run * t],
                    thickness / 2,
                )
            )
        parts.append(beam([0, -height, 0], [0, -rise - height, run], thickness / 2))
    elif kind == "lamp":
        height, radius, thickness = (
            _positive(p, k) for k in ("height", "radius", "thickness")
        )
        arms = _count(p, "arms", 1, 12)
        shade = _positive(p, "shade")
        profile = p.get("profile", "shade")
        if profile not in ("shade", "candle", "candelabra", "wall_sconce", "wall_candelabra", "table_lamp", "ceiling_fan", "wall_linear"):
            raise ValueError("Unknown lamp profile")
        if profile == "wall_linear":
            if not 1 <= arms <= 3:
                raise ValueError("Linear fixture needs one to three tubes")
            housing_height = height
            if (radius * 2 < housing_height * 3 or thickness >= shade
                    or ((arms-1)*1.25+.7)*shade > housing_height):
                raise ValueError("Linear housing needs elongated proportions and thin supports")
            anchor = np.array([0, -height, 0])
            parts.append(beam(anchor, anchor + [0, 0, thickness], thickness / 2))
            # Back plane at Z=0, using the same corners as wall plate checks.
            parts.append(prism([-radius, -height-housing_height/2, -thickness],
                               [radius, -height+housing_height/2, 0]))
            for tube in range(arms):
                y = -height + (tube - (arms-1)/2) * shade * 1.25
                parts.append(beam([-radius+shade, y, shade*.35],
                                  [radius-shade, y, shade*.35], shade * .35))
            return parts
        if profile == "ceiling_fan":
            if not 3 <= arms <= 6:
                raise ValueError("Ceiling fan needs three to six blades")
            width = _positive(p, "blade_width")
            light_drop = _positive({"light_drop": p.get("light_drop", height * .35)}, "light_drop")
            if (thickness >= height * .2 or shade >= light_drop
                    or not thickness < light_drop < height - thickness
                    or shade >= radius * .5 or not thickness * 2 < width <= radius * .35
                    or thickness * 4 >= radius * .2):
                raise ValueError("Fan needs separate stem, hub, light and thin radial blades")
            # First beam starts at the ceiling attachment, as for other lamps.
            # Local origin is the bottom of the light. Negative Y is upwards.
            hub_y = -light_drop
            parts.append(beam([0, -height, 0], [0, hub_y, 0], thickness / 2))
            hub = max(shade * .4, thickness * 2)
            parts.append(prism([-hub, hub_y-thickness, -hub],
                               [hub, hub_y+thickness, hub]))
            parts.append([
                [sx * shade * scale / 2, y, sz * shade * scale / 2]
                for y, scale in ((hub_y+thickness, .7), (0, 1))
                for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ])
            for i in range(arms):
                angle = math.tau * i / arms
                along = np.array([math.cos(angle), 0, math.sin(angle)])
                across = np.array([-math.sin(angle), 0, math.cos(angle)])
                # Tapered blades are convex closed prisms, not thin open sheets.
                ring = [(hub*.8, -width*.3), (radius, -width*.5),
                        (radius, width*.5), (hub*.8, width*.3)]
                parts.append([(along*r + across*w + [0, y, 0]).tolist()
                              for y in (hub_y-thickness/2, hub_y+thickness/2)
                              for r, w in ring])
            return parts
        if profile == "table_lamp":
            if arms != 1 or height <= shade + thickness:
                raise ValueError("Table lamp needs one arm and room below its shade")
            parts.append(beam([0, -thickness, 0], [0, -height + shade, 0], thickness / 2))
            parts.append(prism([-radius, -thickness, -radius], [radius, 0, radius]))
            parts.append([
                [sx * shade * scale / 2, y, sz * shade * scale / 2]
                for y, scale in ((-height, .55), (-height + shade, 1))
                for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ])
            return parts
        if profile in ("wall_sconce", "wall_candelabra"):
            if profile == "wall_sconce" and arms != 1:
                raise ValueError("Wall sconce requires one arm")
            if profile == "wall_candelabra" and not 2 <= arms <= 4:
                raise ValueError("Wall candelabra requires two to four lights")
            if radius < (shade + thickness) / 2:
                raise ValueError("Wall shade must clear its attachment plane")
            anchor = np.array([0, -height, 0])
            arm_rise = float(p.get("arm_rise", -.65))
            if not math.isfinite(arm_rise) or not -1 <= arm_rise <= 1:
                raise ValueError("Wall lamp arm rise must be within -1..1")
            # Negative Y points up. Preserve the legacy downward arm by
            # default, while permitting observed upward-facing wall fixtures.
            tip = (anchor + [0, -height * arm_rise, radius]
                   if "arm_rise" in p else np.array([0, -height * .35, radius]))
            # Keep the attachment as the first beam's start, matching the
            # lamp_attachment diagnostic. Local +Z points away from the wall.
            parts.append(beam(anchor, tip, thickness / 2))
            parts.append(prism(
                [-thickness * 1.5, -height - height * 0.1, -thickness],
                [thickness * 1.5, -height + height * 0.1, 0],
            ))
            tilt = float(p.get("shade_tilt", 0))
            if not math.isfinite(tilt) or abs(tilt) > math.pi / 2:
                raise ValueError("Shade tilt must be within a half turn")
            c, s = math.cos(tilt), math.sin(tilt)
            rotation = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
            offsets = [0.0]
            if profile == "wall_candelabra":
                span = _positive(p, "span")
                if span / (arms - 1) <= shade + thickness:
                    raise ValueError("Wall lights must leave space between shades")
                offsets = np.linspace(-span / 2, span / 2, arms)
            # Stable part roles: stem, plate, all light shades, then branches.
            # The legacy single-light geometry remains byte-for-byte identical.
            for offset in offsets:
                light = tip + [offset, 0, 0]
                rings = []
                for y, scale in ((-shade, 0.55), (0, 1)):
                    rings.extend(
                        (light + [sx * shade * scale / 2, y, sz * shade * scale / 2]).tolist()
                        for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                    )
                tilted = (np.asarray(rings)-light) @ rotation + light
                if tilted[:, 2].min() < thickness / 2 - 1e-8:
                    raise ValueError("Tilted shade must clear its attachment plane")
                parts.append(tilted.tolist())
            if profile == "wall_candelabra":
                for offset in offsets:
                    if abs(offset) <= 1e-8:
                        continue
                    control = tip + [offset / 2, abs(offset) * .3, 0]
                    end = tip + [offset, 0, 0]
                    curve = [(1-t)**2 * tip + 2*(1-t)*t*control + t*t*end
                             for t in np.linspace(0, 1, 4)]
                    parts.extend(beam(a, b, thickness / 2) for a, b in pairwise(curve))
            return parts
        parts.append(beam([0, -height, 0], [0, -height * 0.35, 0], thickness / 2))
        for i in range(arms):
            angle = math.tau * i / arms
            tip = np.array(
                [radius * math.cos(angle), -height * 0.35, radius * math.sin(angle)]
            )
            if profile == "candelabra":
                root = np.array([0, -height * 0.35, 0])
                control = tip * [0.45, 0, 0.45] + [0, root[1] + radius * 0.45, 0]
                curve = [
                    (1 - t) ** 2 * root + 2 * (1 - t) * t * control + t * t * tip
                    for t in np.linspace(0, 1, 5)
                ]
                parts.extend(beam(a, b, thickness / 2) for a, b in pairwise(curve))
                parts.append(beam(tip, tip + [0, -shade, 0], thickness * 0.65))
                continue
            parts.append(beam([0, -height * 0.35, 0], tip, thickness / 2))
            if profile == "candle":
                # A narrow upright light replaces the broad lampshade. Keeping
                # it a separate family avoids fitting candle chandeliers with
                # opaque pyramids merely because they overlap a coarse mask.
                parts.append(beam(tip, tip + [0, -shade, 0], thickness * 0.65))
                continue
            # Truncated square shade: structural prior, not recovered design.
            rings = []
            for y, scale in ((-shade, 0.55), (0, 1)):
                rings.extend(
                    (tip + [sx * shade * scale / 2, y, sz * shade * scale / 2]).tolist()
                    for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                )
            parts.append(rings)
    else:
        raise ValueError(f"Unknown shape family: {kind}")
    return parts


def generate(kind, parameters, origin=(0, 0, 0), yaw=0):
    """Serializable primitive with explicit family parameters and estimated parts."""
    origin = np.asarray(origin, float)
    if origin.shape != (3,) or not np.isfinite(origin).all() or not math.isfinite(yaw):
        raise ValueError("Invalid object pose")
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    parts = np.asarray(local_parts(kind, parameters)) @ rotation + origin
    lo, hi = parts.reshape(-1, 3).min(0), parts.reshape(-1, 3).max(0)
    result = {
        "kind": kind,
        "parts": parts.tolist(),
        "box": [lo[0], lo[2], hi[0], hi[2], lo[1], hi[1]],
        "shape": {
            "version": VERSION,
            "parameters": dict(parameters),
            "origin": origin.tolist(),
            "yaw": float(yaw),
        },
        "estimated": "Parametric prior; hidden surfaces and unobserved parts are inferred",
    }
    if kind == "lamp" and parameters.get("profile") == "wall_candelabra":
        lights = int(parameters["arms"])
        result["part_roles"] = ["support", "support"] + ["shade"] * lights + [
            "support"] * (len(parts) - 2 - lights)
    if kind == "ladder":
        result["part_roles"] = ["rail"] * 2 + ["rung"] * int(parameters["rungs"]) + ["bracket"] * 4
    if kind == "lamp" and parameters.get("profile") == "wall_linear":
        result["part_roles"] = ["support", "support"] + ["shade"] * int(parameters["arms"])
    if kind == "lamp" and parameters.get("profile") == "ceiling_fan":
        result["part_roles"] = ["support", "support", "shade"] + ["support"] * int(parameters["arms"])
    return result
