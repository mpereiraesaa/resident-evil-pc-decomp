"""Round the Chimera's paired upper-arm muscle bellies without moving joints."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    if not -630 < ymin < -610 or not 55 < ymax < 75:
        raise ValueError(f"not the expected Chimera upper arm: {ymin}, {ymax}")
    centre_x = 8.0
    centre_z = (min(p[2] for p in positions) + max(p[2] for p in positions)) * 0.5

    def displace(position):
        x, y, z = position
        root = smoothstep((y + 550.0) / 120.0)
        tip = 1.0 - smoothstep((y + 165.0) / 140.0)
        weight = root * tip
        dx = x - centre_x
        dz = z - centre_z
        radius = math.hypot(dx, dz)
        if radius < 1.0:
            return (0.0, 0.0, 0.0)
        # A slight forward bulge gives the arm relief without opening the
        # shoulder or elbow ends or changing the opposite limb's pose.
        outward = 26.0 * strength * weight
        return ((dx / radius) * outward + 6.0 * strength * weight,
                0.0, (dz / radius) * outward)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["13", "j:13"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1009", "em1109"}:
            raise ValueError(f"selector {selector} is not Chimera")
        for joint in (4, 7):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
