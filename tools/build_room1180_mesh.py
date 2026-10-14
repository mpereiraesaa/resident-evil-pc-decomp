#!/usr/bin/env python3
"""Hand-authored ROOM1180 (mansion 1F storeroom) mesh with generated textures.

Coordinates are RE1 room units (negative Y is up). Walls are the inner faces
of ROOM1180.RDT's blocking rectangles; the chest, crates, barrel and table
footprints come from its type-1 (flags 512) and type-3 collision records and
the door from its door record. The chest lid is a game-owned object model,
so only the static body is modelled. Heights, materials and lighting are an
authored interpretation of the two fixed views. Shares its machinery with
tools/room3d_handmade.py. Run from any directory.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room3d_handmade import Mesh, compare_sheet, door, edge_agreement, preview, tint, to_image, value_noise, write_mtl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- layout --
X0, X1 = 1854, 7344                    # west and east wall faces (blocks 3 and 1)
Z0, Z1 = 1944, 8948                    # south and north wall faces (blocks 5 and 0)
FLOOR = 0
CEIL = -6000                           # south wall/ceiling junction cast from camera 1
DOOR = (5600, 6800, -3350)             # north wall door record zone and panel top
CHEST = (4750, 1944, 7344, 3438, -1410)  # front face from camera 1 (x 4700..east wall); hinge height from omodel
CRATES = (1854, 1944, 4158, 2682)      # block 6 clipped to the room
BARREL = (2400, 2500, 480, -1430)      # yellow drum: circle 7, height cast from camera 1
TABLE = (2556, 8212, 540, -1550)       # foot and top edges cast from camera 0; hidden for camera 1
LAMP = (4995.0, -4284.0, Z0)            # cast from camera 1 onto the south wall
BEAMS_Z = (2450, 4450, 6450, 8450)
CELL = 700.0

MATERIALS = {
    "plaster": {"map": "plaster.png", "tile": (1400, 1400)},
    "planks": {"map": "planks.png", "tile": (1600, 1600)},
    "ceiling": {"map": "ceiling.png", "tile": (1600, 1600)},
    "wood": {"map": "wood.png", "tile": (900, 900)},
    "chest": {"map": "chest.png", "tile": None},
    "crate": {"map": "crate.png", "tile": None},
    "door_dark": {"map": "door_dark.png", "tile": None},
    "iron": {"kd": (0.16, 0.16, 0.17)},
    "drum": {"kd": (0.62, 0.55, 0.22)},
    "brass": {"kd": (0.55, 0.44, 0.22)},
    "paper": {"kd": (0.92, 0.90, 0.82)},
    "enamel": {"kd": (0.12, 0.12, 0.13)},
    "glow": {"kd": (1.0, 0.95, 0.80), "emissive": True},
}

LIGHTS = (
    [((LAMP[0], LAMP[1] + 150, LAMP[2] + 260), (1.0, 0.86, 0.62), 0.95, 4200.0)],
    [((4600.0, -2600.0, 5400.0), (1.0, 0.90, 0.75), 0.18, 5000.0)],  # soft fill
)
AMBIENT = (0.11, 0.085, 0.06)


def illumination(point, normal):
    """One warm wall lamp on the south wall, dark elsewhere."""
    nx, ny, nz = normal
    up = max(0.0, -ny)
    color = [a * (0.8 + 0.4 * up) for a in AMBIENT]
    for group in LIGHTS:
        for (lx, ly, lz), tint_, strength, reach in group:
            dx, dy, dz = lx - point[0], ly - point[1], lz - point[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            lambert = (nx * dx + ny * dy + nz * dz) / dist
            wrap = max(0.0, lambert + 0.3) / 1.3
            falloff = 1.0 / (1.0 + (dist / reach) ** 2)
            for k in range(3):
                color[k] += tint_[k] * strength * wrap * falloff
    return tuple(min(1.0, max(0.0, c)) for c in color)


# ---------------------------------------------------------------- build --
def skirting(mesh, x0, z0, x1, z1, axis, face):
    depth, height = 50, 160
    if axis == "x":
        za, zb = (z0, z0 + depth) if face > 0 else (z0 - depth, z0)
        mesh.box((x0, -height, za), (x1, FLOOR, zb), "wood", omit=("bottom", "south" if face > 0 else "north"))
    else:
        xa, xb = (x0, x0 + depth) if face > 0 else (x0 - depth, x0)
        mesh.box((xa, -height, z0), (xb, FLOOR, z1), "wood", omit=("bottom", "west" if face > 0 else "east"))


def chest(mesh):
    x0, z0, x1, z1, top = CHEST
    # Body sits against the south wall; the lid (game-owned) hinges at the back.
    mesh.box((x0, top, z0), (x1, FLOOR, z1), "chest", omit=("bottom", "south"))
    for y in (top + 40, -120):
        mesh.box((x0 - 10, y, z1), (x1 + 10, y + 70, z1 + 25), "iron")
        for x in (x0 - 10, x1 - 20):
            mesh.box((x, y, z0), (x + 30, y + 70, z1), "iron")
    for x in (x0 + 260, x1 - 320):
        mesh.box((x, top, z1), (x + 60, FLOOR, z1 + 22), "iron")
    cx = (x0 + x1) / 2
    mesh.box((cx - 120, -1000, z1), (cx + 120, -760, z1 + 40), "brass")
    mesh.box((cx - 220, -620, z1 + 30), (cx + 220, -575, z1 + 70), "iron")
    for x in (cx - 220, cx + 180):
        mesh.box((x, -720, z1), (x + 40, -575, z1 + 70), "iron")


def barrel(mesh):
    x, z, r, top = BARREL
    mesh.cylinder(x, z, r, top, FLOOR, "drum", segments=14)
    for y in (top + 180, top + 700, -300):
        mesh.cylinder(x, z, r + 16, y, y + 70, "drum", segments=14, caps=False)


def crates(mesh):
    x0, z0, x1, z1 = CRATES
    # Small boxes and tins between the chest and the drum (camera 1).
    mesh.box((3550, -520, z0), (4150, FLOOR, z0 + 520), "crate", omit=("bottom", "south"))
    mesh.box((3650, -880, z0 + 60), (4050, -520, z0 + 460), "crate", omit=("bottom",))
    for cx, cz, r, h in ((3250, 2350, 120, -620), (3350, 2700, 90, -360), (4350, 2450, 110, -300)):
        mesh.cylinder(cx, cz, r, h, FLOOR, "iron", segments=8)


def table(mesh):
    x, z, r, top = TABLE
    mesh.cylinder(x, z, 330, -110, FLOOR, "wood", segments=12)
    mesh.cylinder(x, z, 60, top + 70, -110, "wood", segments=8, caps=False)
    mesh.cylinder(x, z, r, top, top + 70, "wood", segments=14)
    # Typewriter: base, carriage, keys, platen, paper.
    mesh.box((x - 300, top - 90, z - 260), (x + 290, top, z + 200), "enamel")
    mesh.box((x - 280, top - 220, z + 10), (x + 270, top - 90, z + 190), "enamel")
    for row in range(3):
        for column in range(9):
            kx = x - 230 + column * 55 + (row % 2) * 14
            kz = z - 230 + row * 70
            mesh.cylinder(kx, kz, 18, top - 115 - row * 12, top - 90, "paper", segments=6)
    mesh.box((x - 160, top - 115, z - 258), (x + 140, top - 90, z - 232), "paper")
    mesh.box((x - 300, top - 250, z + 90), (x + 300, top - 195, z + 160), "wood")
    for kx in (x - 340, x + 300):
        mesh.box((kx, top - 258, z + 80), (kx + 40, top - 187, z + 170), "iron")
    mesh.poly([(x - 190, top - 180, z + 40), (x - 190, top - 560, z + 80),
               (x + 160, top - 560, z + 80), (x + 160, top - 180, z + 40)], "paper")


def wall_lamp(mesh):
    x, y, z = LAMP
    mesh.box((x - 100, y - 60, z), (x + 100, y + 60, z + 110), "iron", omit=("south",))
    mesh.box((x - 20, y - 140, z + 90), (x + 20, y - 60, z + 160), "iron")
    mesh.cylinder(x, z + 260, 160, y - 20, y + 120, "iron", segments=10, caps=False)
    mesh.cylinder(x, z + 260, 160, y - 20, y, "iron", segments=10)
    mesh.sphere((x, y + 60, z + 260), 95, "glow", rings=4, segments=8)


def build(name):
    mesh = Mesh(name, MATERIALS, illumination, header=(
        "# Hand-authored by tools/build_room1180_mesh.py; RE1 room units, -Y up.",
        "# Layout from ROOM1180.RDT collision and door record; appearance is an authored interpretation.",
        "# Static chest body only: the original object model supplies its moving lid.",
    ), cell=CELL)

    mesh.set_group("floor")
    mesh.rect((X0, FLOOR, Z0), (0, 0, 1), (1, 0, 0), Z1 - Z0, X1 - X0, "planks", flip=True)
    mesh.set_group("ceiling")
    mesh.rect((X0, CEIL, Z0), (1, 0, 0), (0, 0, 1), X1 - X0, Z1 - Z0, "ceiling", flip=True)
    for z in BEAMS_Z:
        mesh.box((X0, CEIL, z - 70), (X1, CEIL + 170, z + 70), "wood", omit=("top",))
    mesh.box((4560, CEIL, Z0), (4720, CEIL + 170, Z1), "wood", omit=("top",))
    mesh.set_group("walls")
    mesh.rect((X0, FLOOR, Z0), (0, 0, 1), (0, -1, 0), Z1 - Z0, -CEIL, "plaster")
    mesh.rect((X1, FLOOR, Z1), (0, 0, -1), (0, -1, 0), Z1 - Z0, -CEIL, "plaster")
    mesh.rect((X1, FLOOR, Z0), (-1, 0, 0), (0, -1, 0), X1 - X0, -CEIL, "plaster")
    # Camera 1 sits 950 units behind the north wall face, so that wall and its
    # door are hidden for it (see the .cameras sidecar).
    mesh.set_group("north_wall")
    mesh.rect((X0, FLOOR, Z1), (1, 0, 0), (0, -1, 0), X1 - X0, -CEIL, "plaster")
    mesh.set_group("walls")
    skirting(mesh, X0, Z0, X0, Z1, "z", +1)
    skirting(mesh, X1, Z0, X1, Z1, "z", -1)
    skirting(mesh, X0, Z0, X1, Z0, "x", +1)
    dx0, dx1, dtop = DOOR
    mesh.set_group("north_wall")
    skirting(mesh, X0, Z1, dx0 - 90, Z1, "x", -1)
    skirting(mesh, dx1 + 90, Z1, X1, Z1, "x", -1)
    door(mesh, dx0, dx1, Z1, dtop, "door_dark", -1)
    mesh.set_group(None)
    chest(mesh)
    mesh.set_group("drum")
    barrel(mesh)
    mesh.set_group(None)
    crates(mesh)
    mesh.set_group("table")
    table(mesh)
    mesh.set_group(None)
    wall_lamp(mesh)
    return mesh


# ---------------------------------------------------------------- textures --
def make_textures(out):
    rng = np.random.default_rng(1180)
    size = 512
    yy, xx = np.mgrid[0:size, 0:size] / size

    # Plaster: brown-tan with damp stains and fine grain.
    mottle = value_noise((size, size), rng, octaves=4, base_freq=3)
    stains = np.clip(value_noise((size, size), rng, octaves=3, base_freq=2) - 0.55, 0, 1) * 2.0
    fine = value_noise((size, size), rng, octaves=2, base_freq=64, persistence=0.5)
    level = 0.50 + 0.12 * mottle + 0.04 * fine - 0.18 * stains
    plaster = np.stack([level * 1.18, level * 0.96, level * 0.70], axis=-1)
    to_image(plaster).save(out / "plaster.png")

    # Wood grain shared by planks, beams and the barrel.
    warp = value_noise((size, size), rng, octaves=3, base_freq=3)
    grain = 0.5 + 0.5 * np.sin((yy * 36 + warp * 5) * math.pi)
    streak = value_noise((size, size), rng, octaves=2, base_freq=48, persistence=0.4)
    level = 0.20 + 0.09 * grain + 0.06 * streak
    wood = np.stack([level * 1.35, level * 0.95, level * 0.62], axis=-1)
    to_image(wood).save(out / "wood.png")

    def board_texture(count, base, spread, gap_dark):
        boards = np.floor(yy * count)
        shade = (np.sin(boards * 12.9898 + 7.1) * 43758.5453) % 1
        lvl = base + spread * shade + 0.05 * grain + 0.04 * streak
        image = np.stack([lvl * 1.32, lvl * 0.98, lvl * 0.66], axis=-1)
        gap = ((yy * count) % 1 < 0.02)
        return tint(gap.astype(float), image, gap_dark)

    to_image(board_texture(5, 0.26, 0.10, (0.07, 0.05, 0.03))).save(out / "planks.png")
    to_image(board_texture(8, 0.36, 0.10, (0.12, 0.09, 0.06))).save(out / "ceiling.png")

    # Chest: dark planks with black iron straps.
    chest_tex = board_texture(4, 0.16, 0.06, (0.04, 0.03, 0.02))
    straps = ((xx > 0.10) & (xx < 0.16)) | ((xx > 0.84) & (xx < 0.90)) | (yy < 0.06) | (yy > 0.94)
    chest_tex = tint(straps.astype(float), chest_tex, (0.10, 0.10, 0.11))
    to_image(chest_tex).save(out / "chest.png")

    # Crate: boards inside a lighter frame.
    crate = board_texture(6, 0.30, 0.10, (0.10, 0.07, 0.04))
    frame = (xx < 0.08) | (xx > 0.92) | (yy < 0.08) | (yy > 0.92)
    crate = tint(frame.astype(float) * 0.8, crate, (0.46, 0.34, 0.20))
    to_image(crate).save(out / "crate.png")

    # Door: four dark panels.
    image = Image.new("RGB", (256, 512), (72, 48, 28))
    draw = ImageDraw.Draw(image)
    for row in range(2):
        for col in range(2):
            x0, y0 = 22 + col * 112, 24 + row * 230
            draw.rectangle((x0, y0, x0 + 90, y0 + 206), outline=(38, 24, 14), width=4)
            draw.rectangle((x0 + 8, y0 + 8, x0 + 82, y0 + 198), outline=(26, 16, 9), width=2)
    draw.rectangle((0, 0, 255, 511), outline=(26, 16, 9), width=3)
    draw.ellipse((214, 262, 232, 280), fill=(170, 140, 70))
    image.save(out / "door_dark.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/room3d-handmade/room1180")
    parser.add_argument("--rdt", type=Path, help="ROOM1180.RDT; renders preview-camera*.png when given")
    parser.add_argument("--originals", type=Path, help="Directory of camera*-original.png for compare.png")
    parser.add_argument("--name", default="room1180")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mesh = build(args.name)
    (args.out / f"{args.name}.obj").write_text("\n".join(mesh.lines) + "\n", encoding="ascii")
    (args.out / f"{args.name}.cameras").write_text("hide 1 north_wall\nhide 1 table\nhide 0 drum\n")
    write_mtl(args.out, args.name, MATERIALS)
    make_textures(args.out)
    size = (args.out / f"{args.name}.obj").stat().st_size
    print(f"{args.out / (args.name + '.obj')}: {mesh.triangles} triangles, {mesh.vertex_count} vertices, {size} bytes")
    if size > 4 * 1024 * 1024 or mesh.triangles > 50000:
        raise SystemExit("exceeds the Room3D loader limits")
    if args.rdt:
        preview(args.out, args.name, args.rdt)
        if args.originals:
            print(compare_sheet(args.out, args.name, args.originals, 2))
            for camera, precision, recall in edge_agreement(args.out, args.name, args.originals, 2):
                print(f"camera {camera}: edge precision {precision:.2f} recall {recall:.2f}")


if __name__ == "__main__":
    main()
