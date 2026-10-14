#!/usr/bin/env python3
"""Hand-authored ROOM10B (mansion 1F right stairs) mesh with generated textures.

Coordinates are RE1 room units (negative Y is up). The walkable outline is the
complement of ROOM10B0.RDT's blocking rectangles: an L-shaped corridor, a
double-height stair hall and a short east branch. Everything else - stair
layout, heights, furniture, materials and lighting - is an authored
interpretation inspired by the four fixed views, not recovered geometry.

The game's Room3D loader uses point sampling with clamped texture addressing and
no lighting, so large surfaces are cut into tile-aligned cells with UVs inside
0..1 and the lighting is baked into vertex colours. Textures are procedural and
written beside the mesh; no proprietary images are needed. Run from any
directory.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- layout --
# Blocking rectangles of ROOM10B0.RDT leave this walkable outline.
WEST, CORR_EAST = 1926, 4410           # corridor and junction side walls
SOUTH, CORR_END = 6102, 13644          # south door wall, start of the hall
JUNC_NORTH = 17100                     # hall edge where the stair block begins
STUB_EAST, STUB_NORTH = 3222, 20232    # north stub beside the stair foot
EAST_END = 10566                       # east branch end wall
FLOOR = 0
CORR_CEIL = -5300                      # corridor ceiling, below the 2F slab
UPPER = -6000                          # 2F landing level
HALL_CEIL = -11400                     # double-height stair hall
RISE, RUN, STEPS = 300, 420, 10        # one flight climbs 3000 over 4200
LOWER_Z0, LOWER_Z1 = JUNC_NORTH, 18666     # lower flight, open side south
UPPER_Z0, UPPER_Z1 = 18666, STUB_NORTH     # upper flight, against north wall
FOOT_X = STUB_EAST                     # first riser
LAND_X = FOOT_X + STEPS * RUN          # 7422: intermediate landing west edge
LAND_Y = -STEPS * RISE                 # -3000
SPINE_Z0, SPINE_Z1 = 18546, 18786      # wall between the two flights
# Fitted to the four globe sprites the game overlays in camera 2 (1.5 px rms).
CHANDELIER = (4700.0, 15600.0)
GLOBE_Y = -6360.0
GLOBE_RADIUS, GLOBE_PHASE = 950.0, 1.249
WINDOW = (5400.0, -9200.0, 1400.0)     # fanlight centre x, sill y, radius
SCONCES = ((WEST, -3900.0, 6600.0), (WEST, -3400.0, 14800.0))
CELL = 600.0                           # max lighting cell edge

MATERIALS = {
    "parquet": {"map": "parquet.png", "tile": (1200, 1200)},
    "plaster": {"map": "plaster.png", "tile": (1000, 1000)},
    "wood": {"map": "wood.png", "tile": (900, 900)},
    "panel": {"map": "panel.png", "tile": (1050, 1500)},
    "tread": {"map": "wood.png", "tile": (900, 900)},
    "door_white": {"map": "door_white.png", "tile": None},
    "door_dark": {"map": "door_dark.png", "tile": None},
    "painting": {"map": "painting.png", "tile": None},
    "window": {"map": "window.png", "tile": None},
    "glow": {"kd": (1.0, 0.98, 0.90), "emissive": True},
    "iron": {"kd": (0.20, 0.21, 0.24)},
    "brass": {"kd": (0.48, 0.38, 0.18)},
}


# --------------------------------------------------------------- lighting --
def globe_positions():
    cx, cz = CHANDELIER
    return [(cx + GLOBE_RADIUS * math.cos(GLOBE_PHASE + k * math.pi / 2), GLOBE_Y,
             cz + GLOBE_RADIUS * math.sin(GLOBE_PHASE + k * math.pi / 2)) for k in range(4)]


LIGHTS = (
    [(g, (1.0, 0.92, 0.74), 0.19, 6000.0) for g in globe_positions()]
    + [((x + 300, y, z), (1.0, 0.90, 0.72), 0.26, 1900.0) for x, y, z in SCONCES]
    + [((WINDOW[0], WINDOW[1] - 700, STUB_NORTH - 100), (0.75, 0.80, 0.95), 0.16, 6500.0)]
)
AMBIENT = (0.13, 0.105, 0.08)


def illumination(point, normal):
    """Authored night-time bake: blue ambient, white chandelier, warm sconces."""
    nx, ny, nz = normal
    up = max(0.0, -ny)
    color = [a * (0.85 + 0.3 * up) for a in AMBIENT]
    for (lx, ly, lz), tint, strength, reach in LIGHTS:
        dx, dy, dz = lx - point[0], ly - point[1], lz - point[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        lambert = (nx * dx + ny * dy + nz * dz) / dist
        wrap = max(0.0, lambert + 0.35) / 1.35
        falloff = 1.0 / (1.0 + (dist / reach) ** 2)
        for k in range(3):
            color[k] += tint[k] * strength * wrap * falloff
    return tuple(min(1.0, max(0.0, c)) for c in color)


# ------------------------------------------------------------------- mesh --
def unit(v):
    length = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / length for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def splits(start, length, tile):
    """Cell boundaries along one axis: tile multiples plus lighting cells."""
    end = start + length
    marks = {start, end}
    if tile:
        first = math.floor(start / tile) + 1
        marks.update(k * tile for k in range(first, math.ceil(end / tile)))
    ordered = sorted(marks)
    result = [ordered[0]]
    for a, b in zip(ordered, ordered[1:]):
        pieces = max(1, math.ceil((b - a) / CELL))
        result.extend(a + (b - a) * i / pieces for i in range(1, pieces + 1))
    return result


class Mesh:
    def __init__(self, name):
        self.lines = [
            "# Hand-authored by tools/build_room10b_mesh.py; RE1 room units, -Y up.",
            "# Layout from ROOM10B0.RDT collision; appearance is an authored interpretation.",
            f"mtllib {name}.mtl",
        ]
        self.vertex_count = 0
        self.uv_count = 0
        self.triangles = 0
        self.group = None

    def set_group(self, group):
        if group != self.group:
            self.lines.append("g " + group if group else "g")
            self.group = group

    def emit(self, points, uvs, material, normal):
        spec = MATERIALS[material]
        refs = []
        for point, uv in zip(points, uvs):
            if spec.get("emissive"):
                color = (1.0, 1.0, 1.0)
            else:
                color = illumination(point, normal)
            self.vertex_count += 1
            self.uv_count += 1
            self.lines.append("v %.2f %.2f %.2f %.4f %.4f %.4f" % (*point, *color))
            self.lines.append("vt %.5f %.5f" % uv)
            refs.append(f"{self.vertex_count}/{self.uv_count}")
        self.lines.append(f"usemtl {material}")
        self.lines.append("f " + " ".join(refs))
        self.triangles += len(points) - 2

    def rect(self, origin, eu, ev, lu, lv, material, flip=False):
        """Planar rectangle: origin plus lu along unit eu and lv along unit ev.

        Texture tiles are phased on world coordinates so neighbouring rectangles
        line up; each emitted cell stays inside one tile (clamped sampler).
        """
        eu, ev = unit(eu), unit(ev)
        normal = cross(ev, eu) if flip else cross(eu, ev)
        tile = MATERIALS[material].get("tile")
        su0, sv0 = dot(origin, eu), dot(origin, ev)
        tu, tv = tile if tile else (None, None)
        us = splits(su0, lu, tu)
        vs = splits(sv0, lv, tv)

        def uv_of(s, s0, length, size):
            if size:
                base = math.floor(s0 / size + 1e-9) * size
                return (s - base) / size
            return (s - s0) / length

        for a, b in zip(us, us[1:]):
            for c, d in zip(vs, vs[1:]):
                corners = []
                uvs = []
                for s, t in ((a, c), (b, c), (b, d), (a, d)):
                    corners.append(tuple(origin[k] + eu[k] * (s - su0) + ev[k] * (t - sv0) for k in range(3)))
                    uvs.append((min(1.0, uv_of(s, a, lu, tu) if tu else uv_of(s, su0, lu, None)),
                                min(1.0, uv_of(t, c, lv, tv) if tv else uv_of(t, sv0, lv, None))))
                if flip:
                    corners.reverse()
                    uvs.reverse()
                self.emit(corners, uvs, material, normal)

    def poly(self, points, material, flip=False):
        """Arbitrary planar polygon; the whole texture is stretched across it."""
        p0, p1, p2 = points[0], points[1], points[-1]
        normal = unit(cross(tuple(p1[k] - p0[k] for k in range(3)),
                            tuple(p2[k] - p0[k] for k in range(3))))
        if flip:
            normal = tuple(-c for c in normal)
        axis = unit(tuple(p1[k] - p0[k] for k in range(3)))
        other = unit(cross(normal, axis))
        us = [dot(p, axis) for p in points]
        vs = [dot(p, other) for p in points]
        du = (max(us) - min(us)) or 1.0
        dv = (max(vs) - min(vs)) or 1.0
        uvs = [((u - min(us)) / du, (v - min(vs)) / dv) for u, v in zip(us, vs)]
        self.emit(points, uvs, material, normal)

    def box(self, lo, hi, material, omit=()):
        """Axis-aligned box; `lo` holds the smaller coordinates (so the top is y0)."""
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        faces = {
            "top": ((x0, y0, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, False),
            "bottom": ((x0, y1, z0), (1, 0, 0), (0, 0, 1), x1 - x0, z1 - z0, True),
            "south": ((x0, y1, z0), (1, 0, 0), (0, -1, 0), x1 - x0, y1 - y0, False),
            "north": ((x0, y1, z1), (1, 0, 0), (0, -1, 0), x1 - x0, y1 - y0, True),
            "west": ((x0, y1, z0), (0, 0, 1), (0, -1, 0), z1 - z0, y1 - y0, True),
            "east": ((x1, y1, z0), (0, 0, 1), (0, -1, 0), z1 - z0, y1 - y0, False),
        }
        for name, (origin, eu, ev, lu, lv, flip) in faces.items():
            if name in omit or lu <= 0 or lv <= 0:
                continue
            self.rect(origin, eu, ev, lu, lv, material, flip)

    def sphere(self, centre, radius, material, rings=6, segments=10):
        cx, cy, cz = centre
        for i in range(rings):
            t0, t1 = math.pi * i / rings, math.pi * (i + 1) / rings
            for j in range(segments):
                p0, p1 = math.tau * j / segments, math.tau * (j + 1) / segments
                ring = []
                for t, p in ((t0, p0), (t1, p0), (t1, p1), (t0, p1)):
                    ring.append((cx + radius * math.sin(t) * math.cos(p),
                                 cy - radius * math.cos(t),
                                 cz + radius * math.sin(t) * math.sin(p)))
                if i == 0:
                    ring = [ring[0], ring[1], ring[2]]
                elif i == rings - 1:
                    ring = [ring[0], ring[1], ring[3]]
                self.poly(ring, material)

    def cylinder(self, x, z, radius, top, bottom, material, segments=10, caps=True):
        ring = [(x + radius * math.cos(i * math.tau / segments),
                 z + radius * math.sin(i * math.tau / segments)) for i in range(segments)]
        for i, (a, b) in enumerate(ring):
            c, d = ring[(i + 1) % segments]
            self.poly([(a, bottom, b), (a, top, b), (c, top, d), (c, bottom, d)], material)
        if caps:
            self.poly([(a, top, b) for a, b in ring], material)
            self.poly([(a, bottom, b) for a, b in reversed(ring)], material)


# ------------------------------------------------------------- furniture --
def skirting(mesh, x0, z0, x1, z1, axis, face):
    """Skirting plus chair rail along a wall segment; `face` is the open side."""
    depth, height = 55, 260
    if axis == "x":  # wall runs along X at z0; room on +z or -z side
        za, zb = (z0, z0 + depth) if face > 0 else (z0 - depth, z0)
        mesh.box((x0, -height, za), (x1, FLOOR, zb), "wood", omit=("bottom", "south" if face > 0 else "north"))
        za, zb = (z0, z0 + 40) if face > 0 else (z0 - 40, z0)
        mesh.box((x0, -1280, za), (x1, -1200, zb), "wood", omit=("south" if face > 0 else "north",))
    else:  # wall runs along Z at x0; room on +x or -x side
        xa, xb = (x0, x0 + depth) if face > 0 else (x0 - depth, x0)
        mesh.box((xa, -height, z0), (xb, FLOOR, z1), "wood", omit=("bottom", "west" if face > 0 else "east"))
        xa, xb = (x0, x0 + 40) if face > 0 else (x0 - 40, x0)
        mesh.box((xa, -1280, z0), (xb, -1200, z1), "wood", omit=("west" if face > 0 else "east",))


def balustrade(mesh, start, end, side, newels=(True, True), height=950):
    """Handrail on balusters from `start` to `end` (3D, may slope).

    `side` is the unit horizontal direction pointing away from the walkway, used
    to offset nothing but kept for readability of the call sites.
    """
    sx, sy, sz = start
    ex, ey, ez = end
    length = math.hypot(ex - sx, ez - sz)
    dx, dz = (ex - sx) / length, (ez - sz) / length
    count = max(2, int(length // 320))
    for i in range(count + 1):
        f = i / count
        x, y, z = sx + (ex - sx) * f, sy + (ey - sy) * f, sz + (ez - sz) * f
        if (i == 0 and newels[0]) or (i == count and newels[1]):
            continue
        mesh.box((x - 36, y - height + 60, z - 36), (x + 36, y, z + 36), "wood", omit=("bottom",))
    for pos, flag in ((start, newels[0]), (end, newels[1])):
        if not flag:
            continue
        px, py, pz = pos
        mesh.box((px - 100, py - height - 250, pz - 100), (px + 100, py, pz + 100), "wood", omit=("bottom",))
        mesh.box((px - 130, py - height - 340, pz - 130), (px + 130, py - height - 250, pz + 130), "wood")
        mesh.sphere((px, py - height - 440, pz), 110, "wood", rings=4, segments=8)
    # Handrail: a sloped bar built from four quads around the rail line.
    rail_half, rail_depth = 55, 90
    px, pz = -dz, dx  # perpendicular in the floor plane
    top = [(sx + px * rail_half, sy - height, sz + pz * rail_half),
           (ex + px * rail_half, ey - height, ez + pz * rail_half),
           (ex - px * rail_half, ey - height, ez - pz * rail_half),
           (sx - px * rail_half, sy - height, sz - pz * rail_half)]
    bottom = [(p[0], p[1] + rail_depth, p[2]) for p in top]
    mesh.poly(top, "wood")
    mesh.poly(list(reversed(bottom)), "wood")
    mesh.poly([top[0], top[3], bottom[3], bottom[0]], "wood")
    mesh.poly([top[2], top[1], bottom[1], bottom[2]], "wood")


def door(mesh, x0, x1, z, top, material, face, frame_depth=110):
    """Door panel recessed into a wall running along X; `face` is the room side."""
    inner = z + face * 70
    mesh.set_group(None)
    if face > 0:
        mesh.rect((x1, FLOOR, inner), (-1, 0, 0), (0, -1, 0), x1 - x0, -top, material)
    else:
        mesh.rect((x0, FLOOR, inner), (1, 0, 0), (0, -1, 0), x1 - x0, -top, material)
    za, zb = (z, z + frame_depth) if face > 0 else (z - frame_depth, z)
    for xa in (x0 - 90, x1):
        mesh.box((xa, top - 110, za), (xa + 90, FLOOR, zb), "wood", omit=("bottom",))
    mesh.box((x0 - 90, top - 110, za), (x1 + 90, top, zb), "wood")
    # Reveal between the frame and the recessed panel.
    for xa in (x0, x1 - 1):
        pts = [(xa, FLOOR, z), (xa, top, z), (xa, top, inner), (xa, FLOOR, inner)]
        mesh.poly(pts if (xa == x0) == (face > 0) else list(reversed(pts)), "wood")
    mesh.poly([(x0, top, z), (x1, top, z), (x1, top, inner), (x0, top, inner)] if face > 0 else
              [(x1, top, z), (x0, top, z), (x0, top, inner), (x1, top, inner)], "wood")


def door_on_west_wall(mesh, z0, z1, bottom, top, material):
    """Dark 2F doorway recessed into the west wall at the upper landing."""
    inner = WEST - 70
    mesh.rect((inner, bottom, z0), (0, 0, 1), (0, -1, 0), z1 - z0, bottom - top, material)
    for za in (z0 - 90, z1):
        mesh.box((WEST - 110, top - 110, za), (WEST, bottom, za + 90), "wood", omit=("west",))
    mesh.box((WEST - 110, top - 110, z0 - 90), (WEST, top, z1 + 90), "wood", omit=("west",))
    for za in (z0, z1):
        pts = [(WEST, bottom, za), (WEST, top, za), (inner, top, za), (inner, bottom, za)]
        mesh.poly(pts if za == z1 else list(reversed(pts)), "wood")
    mesh.poly([(WEST, top, z0), (WEST, top, z1), (inner, top, z1), (inner, top, z0)], "wood")


def sconce(mesh, x, y, z):
    mesh.box((x, y - 60, z - 70), (x + 130, y + 60, z + 70), "brass", omit=("west",))
    mesh.box((x + 100, y - 20, z - 20), (x + 200, y + 20, z + 20), "brass")
    mesh.sphere((x + 300, y - 40, z), 120, "glow", rings=5, segments=8)


def chandelier(mesh):
    cx, cz = CHANDELIER
    mesh.cylinder(cx, cz, 22, HALL_CEIL, GLOBE_Y - 700, "iron", segments=6, caps=False)
    mesh.cylinder(cx, cz, 180, HALL_CEIL, HALL_CEIL + 60, "iron", segments=10)
    hub_top, hub_bottom = GLOBE_Y - 700, GLOBE_Y - 420
    mesh.cylinder(cx, cz, 120, hub_top, hub_bottom, "iron", segments=8)
    mesh.sphere((cx, hub_bottom + 90, cz), 110, "brass", rings=4, segments=8)
    for gx, gy, gz in globe_positions():
        ax, az = gx - cx, gz - cz
        steps = 5
        for i in range(steps):
            f0, f1 = i / steps, (i + 1) / steps
            drop0 = 260 * math.sin(math.pi * f0) - 120 * f0
            drop1 = 260 * math.sin(math.pi * f1) - 120 * f1
            a = (cx + ax * f0, hub_bottom - 60 + drop0, cz + az * f0)
            b = (cx + ax * f1, hub_bottom - 60 + drop1, cz + az * f1)
            mesh.box((min(a[0], b[0]) - 18, min(a[1], b[1]) - 18, min(a[2], b[2]) - 18),
                     (max(a[0], b[0]) + 18, max(a[1], b[1]) + 18, max(a[2], b[2]) + 18), "iron")
        mesh.cylinder(gx, gz, 70, gy + 180, gy + 300, "brass", segments=8)
        mesh.sphere((gx, gy, gz), 230, "glow", rings=6, segments=10)


# ---------------------------------------------------------------- build --
def build(name):
    mesh = Mesh(name)

    # Floors.
    mesh.set_group("floor")
    # The real backdrop (channel-corrected) shows herringbone parquet throughout.
    mesh.rect((WEST, FLOOR, SOUTH), (1, 0, 0), (0, 0, 1), CORR_EAST - WEST, JUNC_NORTH - SOUTH, "parquet")
    mesh.rect((WEST, FLOOR, JUNC_NORTH), (1, 0, 0), (0, 0, 1), STUB_EAST - WEST, STUB_NORTH - JUNC_NORTH, "parquet")
    mesh.rect((CORR_EAST, FLOOR, CORR_END), (1, 0, 0), (0, 0, 1), EAST_END - CORR_EAST, JUNC_NORTH - CORR_END, "parquet")

    # Ceilings: low corridor, tall hall (hidden for the top-down camera 2).
    mesh.set_group("corridor_ceiling")
    mesh.rect((WEST, CORR_CEIL, SOUTH), (1, 0, 0), (0, 0, 1), CORR_EAST - WEST, CORR_END - SOUTH, "plaster", flip=True)
    mesh.set_group("hall_ceiling")
    mesh.rect((WEST, HALL_CEIL, CORR_END), (1, 0, 0), (0, 0, 1), EAST_END - WEST, STUB_NORTH - CORR_END, "plaster", flip=True)
    mesh.set_group(None)
    # Soffit where the corridor ceiling meets the hall.
    mesh.rect((WEST, CORR_CEIL, CORR_END), (1, 0, 0), (0, -1, 0), CORR_EAST - WEST, CORR_CEIL - HALL_CEIL, "plaster", flip=True)

    # Walls (plaster), each with skirting and chair rail.
    mesh.set_group("walls")
    # West wall, full height along the corridor, hall and stub.
    mesh.rect((WEST, FLOOR, SOUTH), (0, 0, 1), (0, -1, 0), CORR_END - SOUTH, -CORR_CEIL, "plaster")
    mesh.rect((WEST, FLOOR, CORR_END), (0, 0, 1), (0, -1, 0), STUB_NORTH - CORR_END, -HALL_CEIL, "plaster")
    skirting(mesh, WEST, SOUTH, WEST, STUB_NORTH, "z", +1)
    # South wall with the door to the passage.
    mesh.rect((CORR_EAST, FLOOR, SOUTH), (-1, 0, 0), (0, -1, 0), CORR_EAST - WEST, -CORR_CEIL, "plaster")
    skirting(mesh, WEST, SOUTH, 2300 - 90, SOUTH, "x", +1)
    skirting(mesh, 4000 + 90, SOUTH, CORR_EAST, SOUTH, "x", +1)
    # East wall of the corridor up to the hall.
    mesh.rect((CORR_EAST, FLOOR, CORR_END), (0, 0, -1), (0, -1, 0), CORR_END - SOUTH, -CORR_CEIL, "plaster")
    skirting(mesh, CORR_EAST, SOUTH, CORR_EAST, CORR_END, "z", -1)
    # East branch: south wall (with the storeroom door), end wall, north face of stair block.
    mesh.rect((EAST_END, FLOOR, CORR_END), (-1, 0, 0), (0, -1, 0), EAST_END - CORR_EAST, -HALL_CEIL, "plaster")
    skirting(mesh, CORR_EAST, CORR_END, 8600 - 90, CORR_END, "x", +1)
    skirting(mesh, 10400 + 90, CORR_END, EAST_END, CORR_END, "x", +1)
    mesh.rect((EAST_END, FLOOR, JUNC_NORTH), (0, 0, -1), (0, -1, 0), JUNC_NORTH - CORR_END, -HALL_CEIL, "plaster")
    skirting(mesh, EAST_END, CORR_END, EAST_END, JUNC_NORTH, "z", -1)
    # Hall east wall above the landing block and north wall above the stairs.
    mesh.rect((EAST_END, LAND_Y, STUB_NORTH), (0, 0, -1), (0, -1, 0), STUB_NORTH - JUNC_NORTH, LAND_Y - HALL_CEIL, "plaster")
    mesh.rect((STUB_EAST, FLOOR, STUB_NORTH), (1, 0, 0), (0, -1, 0), EAST_END - STUB_EAST, -HALL_CEIL, "plaster")
    mesh.rect((WEST, FLOOR, STUB_NORTH), (1, 0, 0), (0, -1, 0), STUB_EAST - WEST, -HALL_CEIL, "plaster")
    skirting(mesh, WEST, STUB_NORTH, STUB_EAST, STUB_NORTH, "x", -1)
    # Stub ceiling is the underside of the 2F landing slab.
    mesh.rect((WEST, UPPER + 300, JUNC_NORTH), (1, 0, 0), (0, 0, 1), STUB_EAST - WEST, STUB_NORTH - JUNC_NORTH, "plaster", flip=True)
    mesh.rect((WEST, UPPER + 300, JUNC_NORTH), (1, 0, 0), (0, -1, 0), STUB_EAST - WEST, 300, "plaster", flip=True)
    mesh.set_group(None)

    # Doors, painting, sconces.
    door(mesh, 2300, 4000, SOUTH, -3300, "door_white", +1)
    door(mesh, 8600, 10400, CORR_END, -3300, "door_dark", +1)
    door_on_west_wall(mesh, 18100, 19500, UPPER, UPPER - 3300, "door_dark")
    px0, px1, py0, py1 = 10400, 12600, -2400, -4100
    mesh.rect((CORR_EAST - 70, py0, px1), (0, 0, -1), (0, -1, 0), px1 - px0, py0 - py1, "painting")
    for za, zb in ((px0 - 110, px0), (px1, px1 + 110)):
        mesh.box((CORR_EAST - 110, py1 - 110, za), (CORR_EAST, py0, zb), "brass", omit=("east",))
    for ya, yb in ((py1 - 110, py1), (py0, py0 + 110)):
        mesh.box((CORR_EAST - 110, ya, px0 - 110), (CORR_EAST, yb, px1 + 110), "brass", omit=("east",))
    for x, y, z in SCONCES:
        sconce(mesh, x, y, z)

    # Fanlight window on the north wall above the upper flight.
    wx, wy, wr = WINDOW
    mesh.rect((wx - wr - 150, wy, STUB_NORTH - 40), (1, 0, 0), (0, -1, 0), 2 * wr + 300, wr + 150, "window")
    mesh.box((wx - wr - 150, wy, STUB_NORTH - 90), (wx + wr + 150, wy + 90, STUB_NORTH), "wood", omit=("north",))

    # Stairs.
    mesh.set_group("stairs")
    lower_steps = [(FOOT_X + i * RUN, -(i + 1) * RISE, -i * RISE) for i in range(STEPS)]
    upper_steps = [(LAND_X - (i + 1) * RUN, LAND_Y - (i + 1) * RISE, LAND_Y - i * RISE) for i in range(STEPS)]
    for x, top, bottom in lower_steps:
        mesh.box((x, top, LOWER_Z0 + 120), (x + RUN, bottom, SPINE_Z0), "tread",
                 omit=("bottom", "east", "south", "north"))
        mesh.box((x - 30, top, LOWER_Z0 + 120), (x + RUN, top + 40, SPINE_Z0), "tread",
                 omit=("east", "south", "north", "top"))
    for x, top, bottom in upper_steps:
        mesh.box((x, top, SPINE_Z1), (x + RUN, bottom, UPPER_Z1), "tread",
                 omit=("bottom", "west", "south", "north"))
        mesh.box((x, top, SPINE_Z1), (x + RUN + 30, top + 40, UPPER_Z1), "tread",
                 omit=("west", "south", "north", "top"))

    def lower_line(x):
        return -RISE * (x - FOOT_X) / RUN - 250

    def upper_line(x):
        return LAND_Y - RISE * (LAND_X - x) / RUN - 250

    def soffit_line(x):
        return -RISE * (x - FOOT_X) / RUN + RISE + 20

    # String board on the open south side of the lower flight, soffit below it.
    for x, top, bottom in lower_steps:
        lo0, lo1 = min(FLOOR, soffit_line(x)), min(FLOOR, soffit_line(x + RUN))
        mesh.poly([(x, lo0, LOWER_Z0), (x, lower_line(x), LOWER_Z0),
                   (x + RUN, lower_line(x + RUN), LOWER_Z0), (x + RUN, lo1, LOWER_Z0)], "wood")
        mesh.poly([(x, lo0, LOWER_Z0 + 120), (x + RUN, lo1, LOWER_Z0 + 120),
                   (x + RUN, lower_line(x + RUN), LOWER_Z0 + 120), (x, lower_line(x), LOWER_Z0 + 120)], "wood")
        mesh.poly([(x, lo0, LOWER_Z0), (x + RUN, lo1, LOWER_Z0),
                   (x + RUN, lo1, LOWER_Z0 + 120), (x, lo0, LOWER_Z0 + 120)], "wood")
        if soffit_line(x + RUN) < FLOOR:
            xs = x if soffit_line(x) < FLOOR else x + RUN * (FLOOR - soffit_line(x)) / (soffit_line(x + RUN) - soffit_line(x))
            mesh.poly([(xs, min(FLOOR, soffit_line(xs)), LOWER_Z0 + 120), (xs, min(FLOOR, soffit_line(xs)), SPINE_Z0),
                       (x + RUN, lo1, SPINE_Z0), (x + RUN, lo1, LOWER_Z0 + 120)], "plaster")
        mesh.poly([(x, lower_line(x), LOWER_Z0), (x, lower_line(x), LOWER_Z0 + 120),
                   (x + RUN, lower_line(x + RUN), LOWER_Z0 + 120), (x + RUN, lower_line(x + RUN), LOWER_Z0)], "wood")
        mesh.poly([(x + RUN, top, LOWER_Z0 + 120), (x, top, LOWER_Z0 + 120),
                   (x, lower_line(x), LOWER_Z0 + 120), (x + RUN, lower_line(x + RUN), LOWER_Z0 + 120)], "wood")
    mesh.rect((FOOT_X, FLOOR, LOWER_Z0), (1, 0, 0), (0, 0, 1), LAND_X - FOOT_X, SPINE_Z0 - LOWER_Z0, "parquet")
    mesh.poly([(FOOT_X, FLOOR, LOWER_Z0), (FOOT_X, lower_line(FOOT_X), LOWER_Z0),
               (FOOT_X, lower_line(FOOT_X), LOWER_Z0 + 120), (FOOT_X, FLOOR, LOWER_Z0 + 120)], "wood")
    # Spine wall between the flights, rising with the upper flight.
    for x, top, bottom in upper_steps:
        for z, flip in ((SPINE_Z0, True), (SPINE_Z1, False)):
            pts = [(x, FLOOR, z), (x, upper_line(x), z), (x + RUN, upper_line(x + RUN), z), (x + RUN, FLOOR, z)]
            mesh.poly(list(reversed(pts)) if flip else pts, "plaster")
        mesh.poly([(x, upper_line(x), SPINE_Z0), (x, upper_line(x), SPINE_Z1),
                   (x + RUN, upper_line(x + RUN), SPINE_Z1), (x + RUN, upper_line(x + RUN), SPINE_Z0)], "wood")
        mesh.poly([(x + RUN, top, SPINE_Z1), (x, top, SPINE_Z1),
                   (x, upper_line(x), SPINE_Z1), (x + RUN, upper_line(x + RUN), SPINE_Z1)], "wood", flip=True)
    mesh.poly([(FOOT_X, FLOOR, SPINE_Z1), (FOOT_X, upper_line(FOOT_X), SPINE_Z1),
               (FOOT_X, upper_line(FOOT_X), SPINE_Z0), (FOOT_X, FLOOR, SPINE_Z0)], "wood")
    mesh.poly([(FOOT_X, FLOOR, SPINE_Z0 - 120), (FOOT_X, upper_line(FOOT_X), SPINE_Z0 - 120),
               (FOOT_X, upper_line(FOOT_X), SPINE_Z0), (FOOT_X, FLOOR, SPINE_Z0)], "wood")
    # Under the upper flight at the stub end: a panelled cupboard wall.
    mesh.rect((FOOT_X, FLOOR, SPINE_Z1), (0, 0, 1), (0, -1, 0), UPPER_Z1 - SPINE_Z1, -(upper_line(FOOT_X) + 250), "panel", flip=True)
    mesh.poly([(FOOT_X, upper_line(FOOT_X) + 250, SPINE_Z1), (FOOT_X, upper_line(FOOT_X) + 250, UPPER_Z1),
               (FOOT_X, UPPER + 40, UPPER_Z1), (FOOT_X, UPPER + 40, SPINE_Z1)], "wood", flip=True)
    # Intermediate landing and its panelled support block.
    mesh.rect((LAND_X, LAND_Y, LOWER_Z0), (1, 0, 0), (0, 0, 1), EAST_END - LAND_X, STUB_NORTH - LOWER_Z0, "tread")
    mesh.rect((EAST_END, FLOOR, LOWER_Z0), (-1, 0, 0), (0, -1, 0), EAST_END - LAND_X, -LAND_Y, "panel")
    mesh.rect((LAND_X, FLOOR, UPPER_Z1), (0, 0, -1), (0, -1, 0), UPPER_Z1 - LOWER_Z0, -LAND_Y, "panel")
    mesh.rect((LAND_X, LAND_Y + 40, LOWER_Z0), (1, 0, 0), (0, -1, 0), EAST_END - LAND_X, 40, "wood", flip=True)
    # 2F landing over the stub.
    mesh.rect((WEST, UPPER, JUNC_NORTH), (1, 0, 0), (0, 0, 1), STUB_EAST - WEST, STUB_NORTH - JUNC_NORTH, "tread")
    mesh.rect((WEST, UPPER + 300, JUNC_NORTH), (1, 0, 0), (0, -1, 0), STUB_EAST - WEST, 300, "wood", flip=True)
    mesh.rect((STUB_EAST, UPPER + 300, LOWER_Z0), (0, 0, 1), (0, -1, 0), SPINE_Z0 - LOWER_Z0, 300, "wood")
    # Balustrades.
    balustrade(mesh, (FOOT_X + 60, FLOOR, LOWER_Z0 + 60), (LAND_X - 60, LAND_Y, LOWER_Z0 + 60), None)
    balustrade(mesh, (LAND_X - 60, LAND_Y, LOWER_Z0 + 60), (EAST_END - 80, LAND_Y, LOWER_Z0 + 60), None, newels=(False, True))
    balustrade(mesh, (WEST + 80, UPPER, JUNC_NORTH + 60), (STUB_EAST - 60, UPPER, JUNC_NORTH + 60), None)
    balustrade(mesh, (STUB_EAST - 60, UPPER, JUNC_NORTH + 60), (STUB_EAST - 60, UPPER, SPINE_Z0 - 60), None, newels=(False, True))
    # Wall-mounted handrail along the upper flight.
    for x, top, bottom in upper_steps:
        mesh.box((x, upper_line(x) - 650, UPPER_Z1 - 90), (x + RUN, upper_line(x) - 580, UPPER_Z1), "wood", omit=("north",))
    mesh.set_group(None)

    chandelier(mesh)
    return mesh


# ---------------------------------------------------------------- textures --
def value_noise(shape, rng, octaves=4, base_freq=4, persistence=0.55):
    height, width = shape
    total = np.zeros((height, width))
    amplitude, freq, norm = 1.0, base_freq, 0.0
    for _ in range(octaves):
        grid = rng.random((freq + 1, freq + 1))
        grid[-1, :] = grid[0, :]
        grid[:, -1] = grid[:, 0]
        ys = np.linspace(0, freq, height, endpoint=False)
        xs = np.linspace(0, freq, width, endpoint=False)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        fy = ys - y0
        fx = xs - x0
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        a = grid[y0][:, x0]
        b = grid[y0][:, x0 + 1]
        c = grid[y0 + 1][:, x0]
        d = grid[y0 + 1][:, x0 + 1]
        layer = (a * (1 - fx) + b * fx) * (1 - fy)[:, None] + (c * (1 - fx) + d * fx) * fy[:, None]
        total += amplitude * layer
        norm += amplitude
        amplitude *= persistence
        freq *= 2
    return total / norm


def to_image(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8"), "RGB")


def tint(mask, base, color):
    return base * (1 - mask[..., None]) + np.array(color) * mask[..., None]


def make_textures(out):
    rng = np.random.default_rng(10)
    size = 512
    yy, xx = np.mgrid[0:size, 0:size] / size

    # Parquet: herringbone from alternating 45-degree plank bands per column.
    columns = 6
    col = np.floor(xx * columns).astype(int)
    plank_w = 1 / 12
    band_even = np.floor((xx + yy) / plank_w)
    band_odd = np.floor((xx - yy + 2) / plank_w)
    band = np.where(col % 2 == 0, band_even, band_odd)
    key = (band * 31 + col * 17).astype(int)
    shade = (np.sin(key * 12.9898) * 43758.5453) % 1
    grain = value_noise((size, size), rng, octaves=3, base_freq=32, persistence=0.6)
    edge_even = ((xx + yy) % plank_w) < 0.004
    edge_odd = ((xx - yy + 2) % plank_w) < 0.004
    edge = np.where(col % 2 == 0, edge_even, edge_odd) | ((xx * columns) % 1 < 0.012)
    base = 0.34 + 0.16 * shade + 0.07 * grain
    parquet = np.stack([base * 1.45, base * 0.82, base * 0.38], axis=-1)
    parquet = tint(edge.astype(float) * 0.8, parquet, (0.10, 0.07, 0.04))
    to_image(parquet).save(out / "parquet.png")

    # Plaster: pale grey-blue with mottling and fine grain.
    mottle = value_noise((size, size), rng, octaves=3, base_freq=3)
    fine = value_noise((size, size), rng, octaves=2, base_freq=64, persistence=0.5)
    level = 0.56 + 0.12 * mottle + 0.05 * fine
    plaster = np.stack([level * 1.22, level * 1.0, level * 0.74], axis=-1)
    to_image(plaster).save(out / "plaster.png")

    # Wood: dark walnut with wavy grain.
    warp = value_noise((size, size), rng, octaves=3, base_freq=3)
    grain = 0.5 + 0.5 * np.sin((yy * 40 + warp * 6) * math.pi)
    streak = value_noise((size, size), rng, octaves=2, base_freq=48, persistence=0.4)
    level = 0.17 + 0.09 * grain + 0.06 * streak
    wood = np.stack([level * 1.35, level * 0.95, level * 0.62], axis=-1)
    to_image(wood).save(out / "wood.png")

    # Panel: one raised wainscot panel with bevels, on the wood ground.
    panel = wood.copy()
    frame = 0.09
    inner = (xx > frame) & (xx < 1 - frame) & (yy > frame) & (yy < 1 - frame)
    bevel = 0.03
    field = (xx > frame + bevel) & (xx < 1 - frame - bevel) & (yy > frame + bevel) & (yy < 1 - frame - bevel)
    top_left = inner & ~field & ((yy < frame + bevel) | (xx < frame + bevel))
    light = top_left
    panel = tint(((inner & ~field) & ~light).astype(float) * 0.5, panel, (0.05, 0.03, 0.02))
    panel = tint(light.astype(float) * 0.45, panel, (0.55, 0.42, 0.28))
    panel = tint(field.astype(float) * 0.25, panel, (0.12, 0.08, 0.05))
    to_image(panel).save(out / "panel.png")

    # Doors: six-panel white and four-panel dark, drawn as 256x512 images.
    for name, ground, line, dark in (("door_white", (178, 186, 200), (128, 136, 150), (96, 104, 118)),
                                     ("door_dark", (70, 46, 28), (38, 24, 14), (26, 16, 9))):
        image = Image.new("RGB", (256, 512), ground)
        draw = ImageDraw.Draw(image)
        rows = (3 if name == "door_white" else 2)
        for row in range(rows):
            for column in range(2):
                x0 = 22 + column * 112
                y0 = 24 + row * (460 // rows)
                x1 = x0 + 90
                y1 = y0 + (460 // rows) - 24
                draw.rectangle((x0, y0, x1, y1), outline=line, width=4)
                draw.rectangle((x0 + 8, y0 + 8, x1 - 8, y1 - 8), outline=dark, width=2)
        draw.rectangle((0, 0, 255, 511), outline=dark, width=3)
        draw.ellipse((214, 262, 232, 280), fill=(170, 140, 70))
        image.save(out / f"{name}.png")

    # Painting: a dark romantic landscape in a gilt frame (inspired, not copied).
    w, h = 512, 384
    py, px = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    sky = np.stack([0.22 + 0.25 * (1 - py), 0.26 + 0.22 * (1 - py), 0.36 + 0.18 * (1 - py)], axis=-1)
    clouds = value_noise((h, w), rng, octaves=4, base_freq=3)
    sky = tint(np.clip(clouds - 0.45, 0, 1) * 1.6 * (py < 0.55), sky, (0.55, 0.52, 0.50))
    ridge = 0.48 + 0.12 * value_noise((1, w), rng, octaves=3, base_freq=2)[0]
    hills = py > ridge[None, :]
    land = np.stack([0.12, 0.14, 0.10]) + 0.08 * value_noise((h, w), rng, octaves=4, base_freq=6)[..., None]
    scene = np.where(hills[..., None], land, sky)
    river = np.abs(px - 0.5 - 0.15 * np.sin(py * 4)) < 0.03 * (py - 0.5)
    scene = tint((river & hills).astype(float) * 0.8, scene, (0.45, 0.48, 0.55))
    frame_w = 0.08
    frame_mask = (px < frame_w) | (px > 1 - frame_w) | (py < frame_w * w / h) | (py > 1 - frame_w * w / h)
    gilt = np.stack([0.58 + 0.15 * grain[:h, :w], 0.46 + 0.12 * grain[:h, :w], 0.20 + 0.08 * grain[:h, :w]], axis=-1)
    scene = np.where(frame_mask[..., None], gilt, scene)
    inner_edge = ((np.abs(px - frame_w) < 0.006) | (np.abs(px - 1 + frame_w) < 0.006)
                  | (np.abs(py - frame_w * w / h) < 0.008) | (np.abs(py - 1 + frame_w * w / h) < 0.008))
    scene = tint(inner_edge.astype(float), scene, (0.25, 0.18, 0.06))
    to_image(scene).save(out / "painting.png")

    # Fanlight: glowing semicircle with radial mullions, plaster elsewhere.
    w, h = 512, 256
    py, px = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    cx, cy = 0.5, 1.0
    r = np.hypot((px - cx) * 2, (py - cy))
    glow = np.stack([0.70 + 0.25 * (1 - r), 0.80 + 0.18 * (1 - r), 0.98 - 0.02 * r], axis=-1)
    window = np.where((r < 0.9)[..., None], glow, plaster[:h, :w])  # plaster ground matches the wall
    angle = np.arctan2(cy - py, (px - cx) * 2)
    mullion = np.zeros_like(r, dtype=bool)
    for k in range(1, 6):
        mullion |= np.abs(angle - k * math.pi / 6) < 0.025
    mullion |= np.abs(r - 0.55) < 0.015
    mullion |= (r > 0.86) & (r < 0.9)
    window = tint((mullion & (r < 0.9)).astype(float), window, (0.12, 0.08, 0.05))
    arch = (r >= 0.9) & (r < 0.98)
    window = tint(arch.astype(float), window, (0.22, 0.14, 0.08))
    to_image(window).save(out / "window.png")


def write_mtl(out, name):
    lines = ["# Materials for the hand-authored ROOM10B mesh."]
    for material, spec in MATERIALS.items():
        kd = spec.get("kd", (1, 1, 1))
        lines.append(f"newmtl {material}")
        lines.append("Kd %.3f %.3f %.3f" % kd)
        if "map" in spec:
            lines.append(f"map_Kd {spec['map']}")
    (out / f"{name}.mtl").write_text("\n".join(lines) + "\n")


def preview(out, name, rdt):
    sys.path.insert(0, str(ROOT / "tools"))
    from room3d_preview import render
    from survey_room3d import survey

    cameras = survey(Path(rdt))["cameras"]
    for camera in cameras:
        image, _ = render(out / f"{name}.obj", camera, width=640, height=480)
        image.save(out / f"preview-camera{camera['index']}.png")
        print(out / f"preview-camera{camera['index']}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/room3d-handmade/room10b0")
    parser.add_argument("--rdt", type=Path, help="ROOM10B0.RDT; renders preview-camera*.png when given")
    parser.add_argument("--name", default="room10b0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mesh = build(args.name)
    (args.out / f"{args.name}.obj").write_text("\n".join(mesh.lines) + "\n", encoding="ascii")
    (args.out / f"{args.name}.cameras").write_text("hide 2 hall_ceiling\n")
    write_mtl(args.out, args.name)
    make_textures(args.out)
    print(f"{args.out / (args.name + '.obj')}: {mesh.triangles} triangles, {mesh.vertex_count} vertices")
    if args.rdt:
        preview(args.out, args.name, args.rdt)


if __name__ == "__main__":
    main()
