#!/usr/bin/env python3
"""Evidence helpers for writing room readings.

    room3d_evidence.py blocks ROOM1130 --originals DIR --out sheet.png
        Every original camera view with each collision block's floor outline
        and number drawn on it (blue: walls/solid, red: low furniture), to tell
        which block is which piece of furniture or which wall.

    room3d_evidence.py floor ROOM1130 CAMERA PX PY [PX PY ...] [--y Y]
        The room point under 320x240 pixel coordinates of a camera on the
        plane at height Y (default the floor, 0): bases of furniture, door
        bottoms, rug corners, the floor line of a wall.

The RDT is read from --assets (parent of USA/StageN); the return stages 6 and
7 use their own RDTs (identical cameras to stages 1 and 2).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room3d_preview import camera_basis  # noqa: E402
from survey_room3d import survey  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def project(camera, point):
    t = (np.array(point, float) - camera["origin"]) @ camera_basis(camera).T
    if t[2] < 50:
        return None
    s = t[:2] / t[2] * camera["fov"] + [160, 120]
    return float(s[0]), float(s[1])


def blocks_sheet(info, originals, out, scale=2):
    tiles = []
    for camera in info["cameras"]:
        original = Path(originals) / f"camera{camera['index']}-original.png"
        if not original.exists():
            continue
        image = Image.open(original).convert("RGB").resize((320 * scale, 240 * scale))
        draw = ImageDraw.Draw(image)
        for number, block in enumerate(info["boundaries"]):
            x0, x1, z0, z1 = block["xmin"], block["xmax"], block["zmin"], block["zmax"]
            corners = [project(camera, (x, 0, z)) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
            colour = (255, 60, 60) if block.get("flags") == 512 else (60, 200, 255)
            if all(corners):
                draw.polygon([(u * scale, v * scale) for u, v in corners], outline=colour)
            centre = project(camera, ((x0 + x1) / 2, 0, (z0 + z1) / 2))
            if centre and -25 < centre[0] < 345 and -25 < centre[1] < 265:
                draw.text((centre[0] * scale, centre[1] * scale), str(number), fill=(255, 255, 0))
        draw.text((4, 4), f"camera {camera['index']}", fill=(255, 255, 255))
        tiles.append(image)
    if not tiles:
        raise SystemExit("no camera originals found")
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width * 2, height * ((len(tiles) + 1) // 2)))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % 2) * width, (k // 2) * height))
    sheet.save(out)
    return out


def floor_points(camera, pixels, y=0.0):
    basis = camera_basis(camera)
    origin = np.array(camera["origin"], float)
    points = []
    for px, py in pixels:
        direction = basis.T @ np.array([(px - 160) / camera["fov"], (py - 120) / camera["fov"], 1.0])
        if abs(direction[1]) < 1e-9 or (y - origin[1]) / direction[1] <= 0:
            points.append(None)
            continue
        hit = origin + (y - origin[1]) / direction[1] * direction
        points.append((float(hit[0]), float(hit[2])))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets", help="parent of USA/StageN")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("blocks")
    b.add_argument("room")
    b.add_argument("--originals", type=Path, required=True, help="survey evidence dir with cameraN-original.png")
    b.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("floor")
    f.add_argument("room")
    f.add_argument("camera", type=int)
    f.add_argument("pixels", type=float, nargs="+")
    f.add_argument("--y", type=float, default=0.0)
    args = parser.parse_args()
    room = args.room.upper()
    usa = args.assets / "USA" if (args.assets / "USA").is_dir() else args.assets
    info = survey(usa / f"Stage{room[4]}" / f"{room}.RDT")
    if args.command == "blocks":
        print(blocks_sheet(info, args.originals, args.out))
    else:
        if len(args.pixels) % 2:
            raise SystemExit("pixels come in x y pairs")
        pairs = list(zip(args.pixels[::2], args.pixels[1::2]))
        for (px, py), hit in zip(pairs, floor_points(info["cameras"][args.camera], pairs, args.y)):
            print(f"pixel ({px:.0f},{py:.0f}) -> " + (f"x {hit[0]:.0f} z {hit[1]:.0f}" if hit else "above the horizon"))


if __name__ == "__main__":
    main()
