"""Round the adder's articulated body and sharpen its terminal tail."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, joint, strength):
    low = min(p[0] for p in positions)
    high = max(p[0] for p in positions)
    if not -120 < low < -90 or not 9 < high < 15:
        raise ValueError("not the expected adder body segment")

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (x - low) / (high - low)))
        arch = math.sin(math.pi * u) ** 2
        # Keep both hinge rims fixed, adding a soft belly and dorsal ridge
        # between them. The HD scales stay on their original UVs.
        radial = 0.13 * strength * arch
        upper = smoothstep((y + 11.0) / 18.0)
        ridge = 4.8 * strength * arch * upper * math.exp(-0.5 * (z / 15.0) ** 2)
        dy = radial * (y + 8.0) + ridge
        dz = radial * z
        dx = 0.0
        if joint == 10:
            # Only the final piece has no child hinge to preserve at its tip.
            tip = 1.0 - smoothstep((x - low) / 45.0)
            tip *= tip
            dx = -8.0 * strength * tip
            dy -= 0.25 * strength * tip * (y + 5.0)
            dz -= 0.25 * strength * tip * z
        return dx, dy, dz

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["14", "j:14"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em100a", "em110a"}:
            raise ValueError(f"selector {selector} is not an adder")
        for joint in range(2, 11):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], joint,
                                                                options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
