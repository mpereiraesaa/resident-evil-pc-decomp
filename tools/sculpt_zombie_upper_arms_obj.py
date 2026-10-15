"""Round the six scenario zombie upper-arm meshes along their actual Z axis.

The torso and elbow ends stay fixed. This retains the white/green coat sleeves
and the naked variant's exposed texture while adding middle-limb volume.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


ZOMBIE_STEMS = {"em1000", "em1001", "em1011",
                "em1100", "em1101", "em1111"}


def make_displacement(positions, strength):
    zmin = min(p[2] for p in positions)
    zmax = max(p[2] for p in positions)
    span = zmax - zmin
    if not 540 < span < 700:
        raise ValueError(f"not the expected zombie upper arm Z span: {span}")

    def section_centre(fraction):
        sample = [p for p in positions
                  if abs((p[2] - zmin) / span - fraction) < 0.09]
        if not sample:
            raise ValueError("missing arm cross-section")
        return ((min(p[0] for p in sample) + max(p[0] for p in sample)) / 2.0,
                (min(p[1] for p in sample) + max(p[1] for p in sample)) / 2.0)

    low_x, low_y = section_centre(0.15)
    high_x, high_y = section_centre(0.85)

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (z - zmin) / span))
        arch = math.sin(math.pi * u) ** 4
        axis_x = low_x + (high_x - low_x) * u
        axis_y = low_y + (high_y - low_y) * u
        return (0.10 * strength * arch * (x - axis_x) +
                8.0 * strength * arch,
                0.075 * strength * arch * (y - axis_y),
                0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*",
                        default=["4", "5", "21", "j:4", "j:5", "j:21"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in ZOMBIE_STEMS or len(rig["joints"]) != 15:
            raise ValueError(f"selector {selector} is not a known zombie EMD")
        for joint in (3, 6):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
