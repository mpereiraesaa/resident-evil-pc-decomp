"""Round Cerberus's animated front upper legs without moving their joint ends."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    low = min(p[1] for p in positions)
    high = max(p[1] for p in positions)
    if not -390 < low < -380 or not 80 < high < 90:
        raise ValueError(f"unexpected Cerberus front-leg bounds: {low}, {high}")
    center_x = (min(p[0] for p in positions) + max(p[0] for p in positions)) / 2
    center_z = (min(p[2] for p in positions) + max(p[2] for p in positions)) / 2

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (y - low) / (high - low)))
        # Two muscle bellies, with unchanged skin at the shoulder and elbow.
        upper = math.exp(-0.5 * ((u - 0.68) / 0.14) ** 2)
        lower = math.exp(-0.5 * ((u - 0.39) / 0.17) ** 2)
        envelope = math.sin(math.pi * u) ** 2
        swell = strength * envelope * (28.0 * upper + 11.0 * lower)
        dx, dz = x - center_x, z - center_z
        radius = math.hypot(dx, dz)
        if radius < 1.0:
            return (0.0, 0.0, 0.0)
        return (dx / radius * swell, 0.0, dz / radius * swell)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["6", "j:6"])
    options = parser.parse_args()
    if not 0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1002", "em1102"}:
            raise ValueError(f"selector {selector} is not Cerberus")
        for joint in (5, 7):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            output = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(output, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {output} ({triangles} triangles)")


if __name__ == "__main__":
    main()
