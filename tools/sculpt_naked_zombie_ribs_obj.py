"""Add rib and abdomen-wound relief to the exposed zombie torso."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, smoothstep, write_mesh
from sculpt_naked_zombie_shoulders_obj import make_displacement as shoulder_displacement
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    shoulder = shoulder_displacement(positions, 1.0)

    def displace(position):
        x, y, z = position
        base = shoulder(position)
        front = smoothstep((x - 15.0) / 170.0)
        body = smoothstep((y - 35.0) / 110.0)
        body *= 1.0 - smoothstep((y - 625.0) / 90.0)
        az = abs(z)
        ribs = sum(bump(y, height, 35.0) for height in (505.0, 423.0, 342.0))
        rib_ridges = 16.0 * ribs * bump(az, 165.0, 85.0)
        sternum = -8.0 * bump(y, 460.0, 145.0) * bump(z, 0.0, 42.0)
        wound_radius = math.hypot((y - 216.0) / 80.0, z / 80.0)
        abdomen_hole = (-48.0 * math.exp(-2.1 * wound_radius * wound_radius) +
                        16.0 * math.exp(-0.5 * ((wound_radius - 1.0) / 0.22) ** 2))
        dx = strength * front * body * (rib_ridges + sternum + abdomen_hole)
        return (base[0] + dx, base[1], base[2])

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["5", "j:5"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1001", "em1101"}:
            raise ValueError(f"selector {selector} is not a naked zombie")
        source = options.refined_dir / rig["joints"][1]["obj"]
        metadata = parse_obj(source)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(selector, destination, triangles)


if __name__ == "__main__":
    main()
