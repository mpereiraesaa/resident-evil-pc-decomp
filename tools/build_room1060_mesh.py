#!/usr/bin/env python3
"""Hand-authored ROOM1060 (mansion main hall) mesh with generated textures.

Coordinates are RE1 room units (negative Y is up). The floor outline, column
positions, north arcade piers, stair block and door openings come from
ROOM1060.RDT's collision rectangles and door records. Storey heights, arches,
gallery, rose window, furniture, materials and lighting are an authored
interpretation inspired by the eight fixed views, not recovered geometry.

Shares its mesh, lighting bake and texture machinery with
tools/room3d_handmade.py. Run from any directory.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room3d_handmade import Mesh, balustrade, compare_sheet, door, preview, tint, to_image, value_noise, write_mtl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- layout --
X0, X1 = 2092, 32105                   # side walls (collision blocks 11 and 0)
Z0, Z1 = 2087, 26827                   # south entrance wall, north wall
CX = (X0 + X1) / 2                     # 17098.5, the entrance/stair axis
FLOOR = 0
SPRING = -3500                         # arch springing above the column capitals
SLAB = -5600                           # gallery slab underside (aisle ceiling)
UPPER = -6000                          # gallery floor
CEIL = -20000                          # hall ceiling (camera 0 sits at -18000)
COL_X = (3970, 30230)                  # colonnade lines, from type-3 collisions
COL_Z = (6100, 10570, 15030, 19520)
COL_R = 520
GAL_W = 2470                           # gallery depth over the side aisles
ARCADE_Z0, ARCADE_Z1 = 22750, 24850    # north pier row
PIERS = ((X0, 4378), (6879, 9507), (12430, 14766), (19026, 21942), (25092, 27108), (30384, X1))
STAIR_X0, STAIR_X1 = 15228, 18864      # tread width between the parapets
PARAPET = 1060                         # parapet thickness (collision strips 18 and 9)
STAIR_Z0, STAIR_Z1 = 16326, 22248      # foot riser to the top landing edge
STEPS = 20
RISE = -UPPER / STEPS                  # 300
RUN = (STAIR_Z1 - STAIR_Z0) / STEPS    # 296.1
RUNNER_X0, RUNNER_X1 = 15850, 18250
CARPET = (10600, 2900, 23600, STAIR_Z0)
ENTRANCE = (15100, 19100, -4600)
ROSE = (CX, -9000, 2800)               # fanlight centre x, chord y, radius
WEST_DOOR = (16000, 17800)
EAST_DOORS = ((10700, 12300, "door_blue"), (18600, 19800, "door_dark"))
CELL = 1500.0

MATERIALS = {
    "marble": {"map": "marble.png", "tile": (2400, 2400)},
    "stone": {"map": "stone.png", "tile": (1600, 1600)},
    "carpet": {"map": "carpet.png", "tile": (4333, 4475)},
    "carpet_border": {"map": "carpet_border.png", "tile": (1200, 600)},
    "runner": {"map": "runner.png", "tile": (2400, 1200)},
    "plaster": {"map": "plaster.png", "tile": (1600, 1600)},
    "panel": {"map": "panel.png", "tile": (1200, 2400)},
    "wood": {"map": "wood.png", "tile": (1200, 1200)},
    "planks": {"map": "planks.png", "tile": (1800, 1800)},
    "coffer": {"map": "coffer.png", "tile": (3000, 3000)},
    "rose": {"map": "rose.png", "tile": None, "emissive": True},
    "glass": {"map": "glass.png", "tile": None, "emissive": True},
    "door_dark": {"map": "door_dark.png", "tile": None},
    "door_blue": {"map": "door_blue.png", "tile": None},
    "entrance": {"map": "entrance.png", "tile": None},
    "glow": {"kd": (1.0, 0.97, 0.88), "emissive": True},
    "brass": {"kd": (0.50, 0.40, 0.20)},
}

SCONCES_1F = [(X0, -3600.0, z) for z in (8300, 12800, 17300, 21800)] + \
             [(X1, -3600.0, z) for z in (8300, 12800, 17300, 21800)]
SCONCES_2F = [(X0, -8800.0, z) for z in (8300, 17300)] + [(X1, -8800.0, z) for z in (8300, 17300)]
SOUTH_LAMPS = [(9000.0, -9000.0, Z0), (25200.0, -9000.0, Z0)]
AISLE_LAMPS = [(x, -3600.0, Z1) for x in (5000.0, 11000.0, 17100.0, 23000.0, 29000.0)]
WINDOWS = [(X0, z) for z in (8300, 17300)] + [(X1, z) for z in (8300, 17300)]


# --------------------------------------------------------------- lighting --
def inward(x):
    return 1.0 if x < CX else -1.0


LIGHTS = (
    [((x + 420 * inward(x), y, z), (1.0, 0.90, 0.72), 0.26, 5000.0) for x, y, z in SCONCES_1F]
    + [((x + 420 * inward(x), y, z), (1.0, 0.90, 0.72), 0.30, 8000.0) for x, y, z in SCONCES_2F]
    + [((x, y, z + 420), (1.0, 0.90, 0.72), 0.30, 8000.0) for x, y, z in SOUTH_LAMPS]
    + [((x, y, z - 420), (1.0, 0.90, 0.72), 0.34, 5500.0) for x, y, z in AISLE_LAMPS]
    + [((ROSE[0], ROSE[1], Z0 + 400), (0.75, 0.80, 1.0), 0.30, 11000.0)]
    + [((x + 300 * inward(x), -11000.0, z), (0.75, 0.82, 1.0), 0.16, 6000.0) for x, z in WINDOWS]
)
AMBIENT = (0.15, 0.125, 0.10)


def illumination(point, normal):
    """Authored warm bake: wall lamps, a cool rose window, dark coffered ceiling."""
    nx, ny, nz = normal
    up = max(0.0, -ny)
    color = [a * (0.8 + 0.35 * up) for a in AMBIENT]
    for (lx, ly, lz), tint_, strength, reach in LIGHTS:
        dx, dy, dz = lx - point[0], ly - point[1], lz - point[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        lambert = (nx * dx + ny * dy + nz * dz) / dist
        wrap = max(0.0, lambert + 0.35) / 1.35
        falloff = 1.0 / (1.0 + (dist / reach) ** 2)
        for k in range(3):
            color[k] += tint_[k] * strength * wrap * falloff
    return tuple(min(1.0, max(0.0, c)) for c in color)


# ---------------------------------------------------------------- pieces --
def wainscot(mesh, origin, eu, length, top, flip):
    """Dark panelling band along a wall base: panel field plus a cap rail."""
    mesh.rect(origin, eu, (0, -1, 0), length, -top, "panel", flip)


def wall_with_dado(mesh, origin, eu, length, bottom, top, flip, dado=2400):
    """Wall from `bottom` (lower) up to `top` with panelling on its lower part."""
    x, y, z = origin
    base = (x, bottom, z)
    wainscot(mesh, base, eu, length, bottom - dado, flip)
    mesh.rect((x, bottom - dado, z), eu, (0, -1, 0), length, (bottom - dado) - top, "plaster", flip)
    nx, nz = (-eu[2], eu[0]) if not flip else (eu[2], -eu[0])  # outward normal in the floor plane
    # Cap rail on the panelling.
    ex, ez = eu[0], eu[2]
    rail_y = bottom - dado
    mesh.box((min(x, x + ex * length) + min(0, nx * 60), rail_y - 90, min(z, z + ez * length) + min(0, nz * 60)),
             (max(x, x + ex * length) + max(0, nx * 60), rail_y, max(z, z + ez * length) + max(0, nz * 60)), "wood")


def arcade_bay(mesh, axis, line, a, b, spring, top, thickness, segments=12):
    """Arched opening between `a` and `b` along `axis` on wall `line`.

    Emits the spandrel strips on both faces and the arch soffit. `axis` is
    "z" for a wall running along Z at x=line, "x" for a wall along X at z=line.
    """
    radius = (b - a) / 2
    centre = (a + b) / 2
    arc = [(centre - radius * math.cos(t * math.pi / segments), spring - radius * math.sin(t * math.pi / segments))
           for t in range(segments + 1)]
    for face in (-1, 1):
        offset = line + face * thickness / 2
        for (s0, y0), (s1, y1) in zip(arc, arc[1:]):
            if axis == "z":
                pts = [(offset, y0, s0), (offset, y1, s1), (offset, top, s1), (offset, top, s0)]
            else:
                pts = [(s0, y0, offset), (s1, y1, offset), (s1, top, offset), (s0, top, offset)]
            mesh.poly(pts if face > 0 else list(reversed(pts)), "stone")
    lo, hi = line - thickness / 2, line + thickness / 2
    for (s0, y0), (s1, y1) in zip(arc, arc[1:]):
        if axis == "z":
            pts = [(lo, y0, s0), (hi, y0, s0), (hi, y1, s1), (lo, y1, s1)]
        else:
            pts = [(s0, y0, lo), (s0, y0, hi), (s1, y1, hi), (s1, y1, lo)]
        mesh.poly(pts, "stone", flip=True)


def column(mesh, x, z, top):
    mesh.box((x - 700, -300, z - 700), (x + 700, FLOOR, z + 700), "stone", omit=("bottom",))
    mesh.cylinder(x, z, COL_R, top + 450, -300, "stone", segments=14, caps=False)
    mesh.box((x - 640, top, z - 640), (x + 640, top + 450, z + 640), "stone")


def sconce(mesh, x, y, z, direction):
    """Three-candle brass wall light; `direction` is the unit vector into the room."""
    dx, dz = direction
    mesh.box((x + min(0, dx * 120) - 70 * abs(dz), y - 220, z + min(0, dz * 120) - 70 * abs(dx)),
             (x + max(0, dx * 120) + 70 * abs(dz), y + 220, z + max(0, dz * 120) + 70 * abs(dx)), "brass")
    for side in (-1, 0, 1):
        ax, az = x + dx * 300, z + dz * 300
        ax += -dz * side * 260
        az += dx * side * 260
        mesh.box((ax - 25, y - 30 + (0 if side else -120), az - 25), (ax + 25, y + 30, az + 25), "brass")
        mesh.sphere((ax, y - 160 + (0 if side else -120), az), 95, "glow", rings=3, segments=6)


def stair(mesh):
    # Treads with the carpet runner, stone on both sides of it.
    for i in range(STEPS):
        z = STAIR_Z0 + i * RUN
        top, bottom = -(i + 1) * RISE, -i * RISE
        for x0, x1, material in ((STAIR_X0, RUNNER_X0, "stone"), (RUNNER_X0, RUNNER_X1, "runner"), (RUNNER_X1, STAIR_X1, "stone")):
            mesh.box((x0, top, z), (x1, bottom, z + RUN), material, omit=("bottom", "north", "west", "east"))
    # Stone parapets stepping up with the flight, balustrade on top.
    for side, (px0, px1) in enumerate(((STAIR_X0 - PARAPET, STAIR_X0), (STAIR_X1, STAIR_X1 + PARAPET))):
        for i in range(STEPS):
            z = STAIR_Z0 + i * RUN
            top = -(i + 1) * RISE - 350
            mesh.box((px0, top, z), (px1, FLOOR, z + RUN), "stone", omit=("bottom", "north"))
        mesh.rect((px0, FLOOR, STAIR_Z0), (1, 0, 0), (0, -1, 0), PARAPET, RISE + 350, "stone", flip=True)
        rail_x = (px0 + px1) / 2
        balustrade(mesh, (rail_x, -RISE - 350, STAIR_Z0 + 150), (rail_x, UPPER - 350, STAIR_Z1 - 150), height=800)
        # Top landing parapet continuing to the north gallery edge.
        mesh.box((px0, UPPER - 350, STAIR_Z1), (px1, UPPER, ARCADE_Z0), "stone", omit=("bottom",))
    # Riser face of the foot and the walls under the parapets at the foot.
    mesh.rect((STAIR_X1, FLOOR, STAIR_Z0), (-1, 0, 0), (0, -1, 0), STAIR_X1 - STAIR_X0, RISE, "stone", flip=True)
    # Landing between the parapets at the top.
    mesh.rect((STAIR_X0, UPPER, STAIR_Z1), (1, 0, 0), (0, 0, 1), STAIR_X1 - STAIR_X0, ARCADE_Z0 - STAIR_Z1, "runner")
    # Under-stair wall closing the arcade gap behind the flight.
    a, b = (PIERS[2][0] + PIERS[2][1]) / 2, (PIERS[3][0] + PIERS[3][1]) / 2
    mesh.rect((a, FLOOR, ARCADE_Z0), (1, 0, 0), (0, -1, 0), b - a, -SLAB, "panel")
    mesh.rect((b, FLOOR, ARCADE_Z1), (-1, 0, 0), (0, -1, 0), b - a, -SLAB, "panel")


def rose_window(mesh):
    cx, cy, r = ROSE
    z = Z0 + 60
    n = 16
    disc = [(cx + r * math.cos(t * math.pi / n), cy - r * math.sin(t * math.pi / n), z) for t in range(n + 1)]
    mesh.poly(list(reversed(disc)), "rose")
    mesh.box((cx - r - 380, cy, Z0), (cx + r + 380, cy + 300, Z0 + 260), "stone", omit=("south",))
    for t in range(n):
        a0, a1 = t * math.pi / n, (t + 1) * math.pi / n
        inner = [(cx + r * math.cos(a), cy - r * math.sin(a)) for a in (a0, a1)]
        outer = [(cx + (r + 380) * math.cos(a), cy - (r + 380) * math.sin(a)) for a in (a0, a1)]
        pts = [(inner[0][0], inner[0][1], z - 20), (outer[0][0], outer[0][1], z - 20),
               (outer[1][0], outer[1][1], z - 20), (inner[1][0], inner[1][1], z - 20)]
        mesh.poly(pts, "stone")
        depth = [(inner[0][0], inner[0][1], z), (inner[0][0], inner[0][1], z + 240),
                 (inner[1][0], inner[1][1], z + 240), (inner[1][0], inner[1][1], z)]
        mesh.poly(depth, "stone", flip=True)


def tall_window(mesh, x, z):
    d = inward(x)
    w, y0, y1 = 800, -8600.0, -13000.0
    face = x + d * 40
    if d > 0:
        mesh.rect((face, y0, z - w), (0, 0, 1), (0, -1, 0), 2 * w, y0 - y1, "glass")
    else:
        mesh.rect((face, y0, z + w), (0, 0, -1), (0, -1, 0), 2 * w, y0 - y1, "glass")
    for za in (z - w - 120, z + w):
        mesh.box((min(x, x + d * 160), y1 - 120, za), (max(x, x + d * 160), y0, za + 120), "stone")
    mesh.box((min(x, x + d * 160), y1 - 120, z - w - 120), (max(x, x + d * 160), y1, z + w + 120), "stone")
    mesh.box((min(x, x + d * 260), y0, z - w - 200), (max(x, x + d * 260), y0 + 160, z + w + 200), "stone")


# ---------------------------------------------------------------- build --
def build(name):
    mesh = Mesh(name, MATERIALS, illumination, header=(
        "# Hand-authored by tools/build_room1060_mesh.py; RE1 room units, -Y up.",
        "# Layout from ROOM1060.RDT collision; appearance is an authored interpretation.",
    ), cell=CELL)

    # Floors: marble everywhere, carpet with a border in the centre-south.
    mesh.set_group("floor")
    cx0, cz0, cx1, cz1 = CARPET
    mesh.rect((X0, FLOOR, Z0), (1, 0, 0), (0, 0, 1), X1 - X0, Z1 - Z0, "marble")
    mesh.rect((cx0, -12, cz0), (1, 0, 0), (0, 0, 1), cx1 - cx0, cz1 - cz0, "carpet")
    for (ax, az, bx, bz, eu, lu, lv) in (
        (cx0 - 600, cz0 - 600, cx1 + 600, cz0, (1, 0, 0), cx1 - cx0 + 1200, 600),
        (cx0 - 600, cz1, cx1 + 600, cz1 + 600, (1, 0, 0), cx1 - cx0 + 1200, 600),
        (cx0 - 600, cz0, cx0, cz1, (0, 0, 1), cz1 - cz0, 600),
        (cx1, cz0, cx1 + 600, cz1, (0, 0, 1), cz1 - cz0, 600),
    ):
        ev = (0, 0, 1) if eu == (1, 0, 0) else (1, 0, 0)
        origin = (ax, -12, az)
        mesh.rect(origin, eu, ev, lu, lv, "carpet_border", flip=(eu != (1, 0, 0)))
    mesh.rect((RUNNER_X0, -14, cz1 + 600), (1, 0, 0), (0, 0, 1), RUNNER_X1 - RUNNER_X0, STAIR_Z0 - (cz1 + 600) + 1, "runner")

    # Ceiling and cornice.
    mesh.set_group("ceiling")
    mesh.rect((X0, CEIL, Z0), (1, 0, 0), (0, 0, 1), X1 - X0, Z1 - Z0, "coffer", flip=True)
    for lo, hi in (((X0, CEIL + 500, Z0), (X0 + 320, CEIL, Z1)), ((X1 - 320, CEIL + 500, Z0), (X1, CEIL, Z1)),
                   ((X0, CEIL + 500, Z0), (X1, CEIL, Z0 + 320)), ((X0, CEIL + 500, Z1 - 320), (X1, CEIL, Z1))):
        mesh.box(lo, hi, "stone", omit=("top",))
    mesh.set_group(None)

    # Side walls: aisle level (panelled), gallery level (panelled), upper plaster.
    for x, eu, flip in ((X0, (0, 0, 1), False), (X1, (0, 0, -1), False)):
        origin_z = Z0 if x == X0 else Z1
        wall_with_dado(mesh, (x, FLOOR, origin_z), eu, Z1 - Z0, FLOOR, SLAB, flip)
        wall_with_dado(mesh, (x, UPPER, origin_z), eu, Z1 - Z0, UPPER, CEIL, flip)
    # South wall (entrance) and north wall (behind the aisle).
    wall_with_dado(mesh, (X1, FLOOR, Z0), (-1, 0, 0), X1 - X0, FLOOR, SLAB, False)
    wall_with_dado(mesh, (X1, UPPER, Z0), (-1, 0, 0), X1 - X0, UPPER, CEIL, False)
    wall_with_dado(mesh, (X0, FLOOR, Z1), (1, 0, 0), X1 - X0, FLOOR, SLAB, False)
    wall_with_dado(mesh, (X0, UPPER, Z1), (1, 0, 0), X1 - X0, UPPER, CEIL, False)
    # Short wall band between aisle ceiling and gallery floor on all walls (slab edge).
    mesh.rect((X0, SLAB, Z0), (0, 0, 1), (0, -1, 0), Z1 - Z0, UPPER - SLAB, "stone")
    mesh.rect((X1, SLAB, Z1), (0, 0, -1), (0, -1, 0), Z1 - Z0, UPPER - SLAB, "stone")
    mesh.rect((X1, SLAB, Z0), (-1, 0, 0), (0, -1, 0), X1 - X0, UPPER - SLAB, "stone")
    mesh.rect((X0, SLAB, Z1), (1, 0, 0), (0, -1, 0), X1 - X0, UPPER - SLAB, "stone")

    # Side colonnades with arches, gallery slabs and balustrades.
    mesh.set_group("gallery")
    for x in COL_X:
        d = inward(x)
        for z in COL_Z:
            column(mesh, x, z, SPRING)
        bays = [(Z0, COL_Z[0] - COL_R)] + [(a + COL_R, b - COL_R) for a, b in zip(COL_Z, COL_Z[1:])] + [(COL_Z[-1] + COL_R, ARCADE_Z0)]
        for a, b in bays:
            arcade_bay(mesh, "z", x, a, b, SPRING, SLAB, 300)
        for z in COL_Z:  # wall piece above each capital
            for face in (-1, 1):
                off = x + face * 150
                pts = [(off, SPRING, z - COL_R), (off, SPRING, z + COL_R), (off, SLAB, z + COL_R), (off, SLAB, z - COL_R)]
                mesh.poly(pts if face > 0 else list(reversed(pts)), "stone")
        inner = x + d * GAL_W
        gx0, gx1 = (X0, inner) if d > 0 else (inner, X1)
        # Slab: gallery floor (planks), aisle ceiling (plaster), inner edge (stone).
        mesh.rect((gx0, UPPER, Z0), (1, 0, 0), (0, 0, 1), gx1 - gx0, ARCADE_Z0 - Z0, "planks")
        mesh.rect((gx0, SLAB, Z0), (1, 0, 0), (0, 0, 1), gx1 - gx0, ARCADE_Z0 - Z0, "plaster", flip=True)
        if d > 0:
            mesh.rect((inner, SLAB, Z0), (0, 0, 1), (0, -1, 0), ARCADE_Z0 - Z0, UPPER - SLAB, "stone")
        else:
            mesh.rect((inner, SLAB, ARCADE_Z0), (0, 0, -1), (0, -1, 0), ARCADE_Z0 - Z0, UPPER - SLAB, "stone")
        rail_x = inner - d * 90
        balustrade(mesh, (rail_x, UPPER, Z0 + 200), (rail_x, UPPER, ARCADE_Z0 - 100), height=950, spacing=420)

    # North arcade: piers, arches, aisle, gallery slab over it.
    arcade_z = (ARCADE_Z0 + ARCADE_Z1) / 2
    pier_x = [(a + b) / 2 for a, b in PIERS[1:-1]]
    for x in pier_x:
        column(mesh, x, arcade_z, SPRING)
        for face in (-1, 1):
            off = arcade_z + face * 150
            pts = [(x - COL_R, SPRING, off), (x + COL_R, SPRING, off), (x + COL_R, SLAB, off), (x - COL_R, SLAB, off)]
            mesh.poly(pts if face < 0 else list(reversed(pts)), "stone")
    for x0, x1 in (PIERS[0], PIERS[-1]):  # wall pilasters
        px0, px1 = (x1 - 700, x1) if x0 == X0 else (x0, x0 + 700)
        mesh.box((px0, SLAB, arcade_z - 350), (px1, FLOOR, arcade_z + 350), "stone", omit=("bottom", "top"))
    openings = [(PIERS[0][1], pier_x[0] - COL_R), (pier_x[0] + COL_R, pier_x[1] - COL_R),
                (pier_x[2] + COL_R, pier_x[3] - COL_R), (pier_x[3] + COL_R, PIERS[-1][0])]
    for a, b in openings:
        arcade_bay(mesh, "x", arcade_z, a, b, SPRING, SLAB, 300)
    # Gallery slab over the piers and aisle, joining the stair landing.
    mesh.rect((X0, UPPER, ARCADE_Z0), (1, 0, 0), (0, 0, 1), X1 - X0, Z1 - ARCADE_Z0, "planks")
    mesh.rect((X0, SLAB, ARCADE_Z1), (1, 0, 0), (0, 0, 1), X1 - X0, Z1 - ARCADE_Z1, "plaster", flip=True)
    mesh.rect((X0, SLAB, ARCADE_Z0), (1, 0, 0), (0, -1, 0), X1 - X0, UPPER - SLAB, "stone", flip=True)
    for a, b in ((X0 + GAL_W, STAIR_X0 - PARAPET), (STAIR_X1 + PARAPET, X1 - GAL_W)):
        balustrade(mesh, (a + 100, UPPER, ARCADE_Z0 - 90), (b - 100, UPPER, ARCADE_Z0 - 90), height=950, spacing=420)
    mesh.set_group(None)

    stair(mesh)

    # Entrance, rose window, side doors.
    ex0, ex1, etop = ENTRANCE
    door(mesh, ex0, ex1, Z0, etop, "entrance", +1, frame_depth=200, frame="stone")
    for xa in (ex0 - 700, ex1 + 300):
        mesh.box((xa, etop - 1500, Z0), (xa + 400, FLOOR, Z0 + 500), "stone", omit=("bottom", "south"))
    mesh.box((ex0 - 700, etop - 1500, Z0), (ex1 + 700, etop - 1100, Z0 + 600), "stone", omit=("south",))
    rose_window(mesh)
    wz0, wz1 = WEST_DOOR
    mesh.rect((X0 + 70, FLOOR, wz0), (0, 0, 1), (0, -1, 0), wz1 - wz0, 3300, "door_dark")
    for za in (wz0 - 90, wz1):
        mesh.box((X0, -3410, za), (X0 + 110, FLOOR, za + 90), "wood", omit=("bottom", "west"))
    mesh.box((X0, -3410, wz0 - 90), (X0 + 110, -3300, wz1 + 90), "wood", omit=("west",))
    for z0, z1, material in EAST_DOORS:
        mesh.rect((X1 - 70, FLOOR, z1), (0, 0, -1), (0, -1, 0), z1 - z0, 3300, material)
        for za in (z0 - 90, z1):
            mesh.box((X1 - 110, -3410, za), (X1, FLOOR, za + 90), "wood", omit=("bottom", "east"))
        mesh.box((X1 - 110, -3410, z0 - 90), (X1, -3300, z1 + 90), "wood", omit=("east",))

    # Lights and windows.
    for x, y, z in SCONCES_1F + SCONCES_2F:
        sconce(mesh, x, y, z, (inward(x), 0))
    for x, y, z in SOUTH_LAMPS:
        sconce(mesh, x, y, z, (0, 1))
    for x, y, z in AISLE_LAMPS:
        sconce(mesh, x, y, z, (0, -1))
    for x, z in WINDOWS:
        tall_window(mesh, x, z)
    return mesh


# ---------------------------------------------------------------- textures --
def make_textures(out):
    rng = np.random.default_rng(1060)
    size = 512
    yy, xx = np.mgrid[0:size, 0:size] / size

    # Marble: pale warm grey with veins; 2x2 tiles per texture with grout.
    veins = value_noise((size, size), rng, octaves=5, base_freq=3)
    vein_lines = np.abs(np.sin((xx * 3 + yy * 2 + veins * 4) * math.pi * 1.5))
    level = 0.74 + 0.08 * veins - 0.18 * np.clip(1 - vein_lines * 6, 0, 1)
    marble = np.stack([level * 1.0, level * 0.98, level * 0.94], axis=-1)
    grout = ((xx * 2) % 1 < 0.012) | ((yy * 2) % 1 < 0.012)
    marble = tint(grout.astype(float) * 0.8, marble, (0.35, 0.33, 0.30))
    to_image(marble).save(out / "marble.png")

    # Stone: warm limestone with light mottling and block joints.
    mottle = value_noise((size, size), rng, octaves=4, base_freq=4)
    level = 0.62 + 0.10 * mottle
    stone = np.stack([level * 1.05, level * 0.98, level * 0.86], axis=-1)
    joints = ((yy * 3) % 1 < 0.01) | (((xx + 0.5 * np.floor(yy * 3)) * 2) % 1 < 0.008)
    stone = tint(joints.astype(float) * 0.6, stone, (0.30, 0.27, 0.22))
    to_image(stone).save(out / "stone.png")

    # Carpet: deep red field with a gold medallion and corner ornaments.
    weave = value_noise((size, size), rng, octaves=3, base_freq=48, persistence=0.5)
    carpet = np.stack([0.50 + 0.10 * weave, 0.12 + 0.04 * weave, 0.10 + 0.03 * weave], axis=-1)
    r = np.hypot(xx - 0.5, yy - 0.5)
    ang = np.arctan2(yy - 0.5, xx - 0.5)
    petal = 0.5 + 0.5 * np.cos(ang * 8)
    medallion = (r < 0.16 + 0.07 * petal) & (r > 0.06 + 0.04 * petal)
    core = r < 0.05
    ring = np.abs(r - 0.30) < 0.012
    corners = np.zeros_like(r, dtype=bool)
    for cxr, cyr in ((0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)):
        corners |= np.hypot(xx - cxr, yy - cyr) < 0.06
    gold = (0.78, 0.62, 0.30)
    cream = (0.85, 0.78, 0.60)
    carpet = tint(medallion.astype(float) * 0.9, carpet, gold)
    carpet = tint(core.astype(float), carpet, cream)
    carpet = tint(ring.astype(float) * 0.8, carpet, cream)
    carpet = tint(corners.astype(float) * 0.8, carpet, gold)
    to_image(carpet).save(out / "carpet.png")

    # Carpet border: dark red band with a gold guard stripe (texture 512x256).
    by, bx = np.mgrid[0:256, 0:512] / np.array([256, 512])[:, None, None]
    border = np.stack([0.32 + 0.05 * weave[:256], 0.08 + 0.02 * weave[:256], 0.07 + 0.02 * weave[:256]], axis=-1)
    stripes = (np.abs(by - 0.25) < 0.03) | (np.abs(by - 0.75) < 0.03)
    border = tint(stripes.astype(float), border, gold)
    to_image(border).save(out / "carpet_border.png")

    # Stair runner: red with a repeating gold scroll along the edges.
    runner = np.stack([0.48 + 0.10 * weave, 0.11 + 0.04 * weave, 0.09 + 0.03 * weave], axis=-1)
    edge = (np.abs(xx - 0.08) < 0.015) | (np.abs(xx - 0.92) < 0.015)
    scroll = (np.abs(xx - 0.5) < 0.06) & (np.abs(np.sin(yy * math.pi * 6)) > 0.8)
    runner = tint((edge | scroll).astype(float) * 0.85, runner, gold)
    to_image(runner).save(out / "runner.png")

    # Plaster: warm cream.
    mottle = value_noise((size, size), rng, octaves=3, base_freq=3)
    fine = value_noise((size, size), rng, octaves=2, base_freq=64, persistence=0.5)
    level = 0.60 + 0.10 * mottle + 0.05 * fine
    plaster = np.stack([level * 1.15, level * 1.02, level * 0.80], axis=-1)
    to_image(plaster).save(out / "plaster.png")

    # Wood and panelling.
    warp = value_noise((size, size), rng, octaves=3, base_freq=3)
    grain = 0.5 + 0.5 * np.sin((yy * 40 + warp * 6) * math.pi)
    streak = value_noise((size, size), rng, octaves=2, base_freq=48, persistence=0.4)
    level = 0.15 + 0.08 * grain + 0.05 * streak
    wood = np.stack([level * 1.35, level * 0.95, level * 0.62], axis=-1)
    to_image(wood).save(out / "wood.png")
    panel = wood.copy()
    frame, bevel = 0.08, 0.03
    inner = (xx > frame) & (xx < 1 - frame) & (yy > frame) & (yy < 1 - frame)
    field = (xx > frame + bevel) & (xx < 1 - frame - bevel) & (yy > frame + bevel) & (yy < 1 - frame - bevel)
    light = inner & ~field & ((yy < frame + bevel) | (xx < frame + bevel))
    panel = tint(((inner & ~field) & ~light).astype(float) * 0.5, panel, (0.05, 0.03, 0.02))
    panel = tint(light.astype(float) * 0.45, panel, (0.55, 0.42, 0.28))
    panel = tint(field.astype(float) * 0.25, panel, (0.12, 0.08, 0.05))
    to_image(panel).save(out / "panel.png")
    boards = np.floor(yy * 6)
    shade = (np.sin(boards * 12.9898) * 43758.5453) % 1
    level = 0.22 + 0.08 * shade + 0.06 * grain * 0.5 + 0.04 * streak
    planks = np.stack([level * 1.35, level * 0.95, level * 0.62], axis=-1)
    planks = tint(((yy * 6) % 1 < 0.015).astype(float), planks, (0.06, 0.04, 0.02))
    to_image(planks).save(out / "planks.png")

    # Coffered ceiling: dark wood grid with recessed warm panels.
    coffer = np.stack([0.20 + 0.04 * mottle, 0.15 + 0.03 * mottle, 0.10 + 0.02 * mottle], axis=-1)
    grid = ((xx * 2) % 1 < 0.08) | ((yy * 2) % 1 < 0.08) | ((xx * 2) % 1 > 0.92) | ((yy * 2) % 1 > 0.92)
    coffer = tint(grid.astype(float), coffer, (0.12, 0.08, 0.05))
    inset = (((xx * 2) % 1 > 0.2) & ((xx * 2) % 1 < 0.8) & ((yy * 2) % 1 > 0.2) & ((yy * 2) % 1 < 0.8))
    coffer = tint(inset.astype(float) * 0.5, coffer, (0.36, 0.28, 0.18))
    to_image(coffer).save(out / "coffer.png")

    # Rose window: radial stained glass, dark tracery, plaster outside the disc.
    ang = np.arctan2(1.0 - yy, xx - 0.5)
    r = np.hypot(xx - 0.5, 1.0 - yy)
    hue = (np.floor((ang + math.pi) / (math.pi / 8)) + np.floor(r * 4)) % 4
    palette = np.array([(0.55, 0.70, 1.0), (0.95, 0.80, 0.45), (0.80, 0.40, 0.45), (0.50, 0.80, 0.65)])
    glass = palette[hue.astype(int)] * (0.75 + 0.25 * value_noise((size, size), rng, octaves=2, base_freq=16))[..., None]
    tracery = (np.abs(np.sin(ang * 8)) < 0.08) | (np.abs((r * 4) % 1 - 0.0) < 0.05) | (r > 0.96)
    rose = np.where((r <= 1)[..., None], glass, plaster)
    rose = tint((tracery & (r <= 1)).astype(float), rose, (0.10, 0.08, 0.06))
    to_image(rose).save(out / "rose.png")

    # Tall window: leaded glass lattice with a pale warm glow (256x512).
    gy, gx = np.mgrid[0:512, 0:256] / np.array([512, 256])[:, None, None]
    lattice = ((gx * 6) % 1 < 0.06) | ((gy * 12) % 1 < 0.04)
    hue = (np.floor(gx * 6) + np.floor(gy * 12)) % 3
    pal = np.array([(0.80, 0.85, 1.0), (0.95, 0.90, 0.70), (0.70, 0.85, 0.80)])
    window = pal[hue.astype(int)] * 0.85
    window = tint(lattice.astype(float), window, (0.12, 0.10, 0.08))
    to_image(window).save(out / "glass.png")

    # Doors.
    for name, ground, line, dark in (("door_dark", (70, 46, 28), (38, 24, 14), (26, 16, 9)),
                                     ("door_blue", (52, 72, 118), (30, 44, 80), (20, 30, 58))):
        image = Image.new("RGB", (256, 512), ground)
        draw = ImageDraw.Draw(image)
        for row in range(2):
            for col in range(2):
                x0, y0 = 22 + col * 112, 24 + row * 230
                draw.rectangle((x0, y0, x0 + 90, y0 + 206), outline=line, width=4)
                draw.rectangle((x0 + 8, y0 + 8, x0 + 82, y0 + 198), outline=dark, width=2)
        draw.rectangle((0, 0, 255, 511), outline=dark, width=3)
        draw.ellipse((214, 262, 232, 280), fill=(170, 140, 70))
        image.save(out / f"{name}.png")
    image = Image.new("RGB", (512, 512), (60, 40, 24))
    draw = ImageDraw.Draw(image)
    for leaf in range(2):
        lx = leaf * 256
        for row in range(3):
            y0 = 20 + row * 160
            draw.rectangle((lx + 24, y0, lx + 232, y0 + 140), outline=(34, 22, 12), width=5)
            draw.rectangle((lx + 36, y0 + 12, lx + 220, y0 + 128), outline=(96, 66, 38), width=2)
        draw.line((lx, 0, lx, 511), fill=(20, 12, 6), width=4)
    draw.ellipse((236, 250, 252, 266), fill=(170, 140, 70))
    draw.ellipse((260, 250, 276, 266), fill=(170, 140, 70))
    image.save(out / "entrance.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/room3d-handmade/room1060")
    parser.add_argument("--rdt", type=Path, help="ROOM1060.RDT; renders preview-camera*.png when given")
    parser.add_argument("--originals", type=Path, help="Directory of channel-corrected camera*-original.png for compare.png")
    parser.add_argument("--name", default="room1060")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mesh = build(args.name)
    (args.out / f"{args.name}.obj").write_text("\n".join(mesh.lines) + "\n", encoding="ascii")
    (args.out / f"{args.name}.cameras").write_text("")
    write_mtl(args.out, args.name, MATERIALS)
    make_textures(args.out)
    print(f"{args.out / (args.name + '.obj')}: {mesh.triangles} triangles, {mesh.vertex_count} vertices")
    if args.rdt:
        preview(args.out, args.name, args.rdt)
        if args.originals:
            print(compare_sheet(args.out, args.name, args.originals, 8))


if __name__ == "__main__":
    main()
