"""Give the clothed zombies' torn chest texture a shallow 3D cavity."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal
from sculpt_zombie_shoulders_obj import make_displacement as shoulder_displacement


def make_displacement(positions, strength, green=False):
    shoulder = shoulder_displacement(positions, 1.0)
    radius_y = 144.0 if green else 168.0
    radius_z = 108.0 if green else 132.0
    centre_y = 378.0 if green else 360.0

    def displace(position):
        x, y, z = position
        base = shoulder(position)
        front = smoothstep((x - 25.0) / 155.0)
        radius = math.hypot((y - centre_y) / radius_y, (z + 18.0) / radius_z)
        cavity = -72.0 * math.exp(-2.1 * radius * radius)
        rim = 23.0 * math.exp(-0.5 * ((radius - 1.0) / 0.22) ** 2)
        return (base[0] + strength * front * (cavity + rim),
                base[1], base[2])

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["4", "j:4"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        stem = Path(rig["sourcePath"]).stem
        if stem not in {"em1000", "em1100", "em1011", "em1111"}:
            raise ValueError(f"selector {selector} is not a clothed zombie")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(
                                   metadata[2], options.strength,
                                   green=stem in {"em1011", "em1111"}),
                               normal_update=moved_normal)
        print(selector, destination, triangles)


if __name__ == "__main__":
    main()
