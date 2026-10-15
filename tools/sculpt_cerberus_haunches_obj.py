"""Give Cerberus's paired rear thighs a fuller muscular silhouette."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    if not -445 < ymin < -420 or not 140 < ymax < 160:
        raise ValueError(f"not the expected Cerberus hind thigh: {ymin}, {ymax}")
    mid_x = (min(p[0] for p in positions) + max(p[0] for p in positions)) * 0.5
    mid_z = (min(p[2] for p in positions) + max(p[2] for p in positions)) * 0.5
    span = ymax - ymin

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (y - ymin) / span))
        arch = math.sin(math.pi * u) ** 4
        dx = x - mid_x
        dz = z - mid_z
        radius = math.hypot(dx, dz)
        if radius < 1.0:
            return (0.0, 0.0, 0.0)
        fullness = 35.0 * strength * arch
        return (dx / radius * fullness, 0.0, dz / radius * fullness)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["6", "j:6"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1002", "em1102"}:
            raise ValueError(f"selector {selector} is not Cerberus")
        for joint in (9, 12):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
