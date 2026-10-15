"""Slim the eight upper spider leg shafts while preserving both joint ends.

The original proximal pieces bulge into broad, nearly round tubes.  Their
local Z axis runs from the body socket to the knee.  This narrows the middle
of that shaft, leaving the socket and knee rings unchanged for animation.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


UPPER_JOINTS = (3, 5, 7, 9, 11, 13, 15, 17)


def make_displacement(positions, joint, strength):
    sign = -1 if joint < 11 else 1
    length = max(sign * p[2] for p in positions)
    if not 900 <= length <= 950:
        raise ValueError(f"unexpected upper leg length: joint {joint}, {length}")

    def displace(position):
        x, y, z = position
        t = max(0.0, min(1.0, sign * z / length))
        # Broad falloff reaches the heavy knee bulge while vanishing at both
        # attachment rings. The original rotation pivots remain untouched.
        taper = math.sin(math.pi * t) ** 0.65 if 0.0 < t < 1.0 else 0.0
        return (-0.35 * strength * taper * x,
                -0.26 * strength * taper * y,
                0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["7", "8", "j:7", "j:8"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1003", "em1004", "em1103", "em1104"}:
            raise ValueError(f"selector {selector} is not a spider")
        for joint in UPPER_JOINTS:
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
