"""Taper the eight distal spider leg segments without moving their hinges.

The limb axis is local Z. The outer half narrows around the measured centre
of its tip, while local Z (and therefore the original foot contact) stays fixed.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


DISTAL_JOINTS = (4, 6, 8, 10, 12, 14, 16, 18)


def make_displacement(positions, joint, strength):
    sign = -1 if joint < 11 else 1
    length = max(sign * p[2] for p in positions)
    if not 1370 <= length <= 1570:
        raise ValueError(f"unexpected spider leg length: joint {joint}, {length}")
    tip = list({p for p in positions if sign * p[2] > 0.85 * length})
    if len(tip) < 30:
        raise ValueError(f"too few tip vertices: joint {joint}")
    centre_x = sum(p[0] for p in tip) / len(tip)
    centre_y = sum(p[1] for p in tip) / len(tip)

    def displace(position):
        x, y, z = position
        t = max(0.0, sign * z / length)
        taper = smoothstep((t - 0.55) / 0.45)
        taper *= taper
        return (-0.43 * strength * taper * (x - centre_x),
                -0.43 * strength * taper * (y - centre_y),
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
        for joint in DISTAL_JOINTS:
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
