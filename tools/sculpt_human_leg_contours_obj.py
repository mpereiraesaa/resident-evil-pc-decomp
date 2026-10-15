"""Add gentle trouser/leg volume without averaging away garment folds.

The deformation follows each leg's existing cross-section. It is zero at the
hip, knee, ankle, and overlapping joint ends. UVs and animation stay intact.
"""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_human_toes_obj import HUMAN_STEMS
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength, thigh):
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    span = ymax - ymin
    if not 650 < span < 1050:
        raise ValueError(f"not the expected human leg span: {span}")

    def section_centre(fraction):
        sample = [p for p in positions
                  if abs((p[1] - ymin) / span - fraction) < 0.08]
        if not sample:
            raise ValueError("missing leg cross-section")
        return ((min(p[0] for p in sample) + max(p[0] for p in sample)) / 2.0,
                (min(p[2] for p in sample) + max(p[2] for p in sample)) / 2.0)

    low_x, low_z = section_centre(0.15)
    high_x, high_z = section_centre(0.85)

    def displace(position):
        x, y, z = position
        u = max(0.0, min(1.0, (y - ymin) / span))
        # The fourth-power shoulder keeps every attachment end fixed, with a
        # broad middle contour. No vertex is averaged with its neighbours.
        arch = math.sin(math.pi * u) ** 4
        axis_x = low_x + (high_x - low_x) * u
        axis_z = low_z + (high_z - low_z) * u
        x_scale = 0.075 if thigh else 0.065
        z_scale = 0.055 if thigh else 0.045
        return (strength * arch * x_scale * (x - axis_x),
                0.0,
                strength * arch * z_scale * (z - axis_z))

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="+", help="viewer selectors, e.g. 0 1 36 j:51")
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in HUMAN_STEMS or len(rig["joints"]) != 15:
            raise ValueError(f"selector {selector} is not a known human EMD")
        for joint in (3, 4, 6, 7):
            # Kenneth's devoured-state corpse has one 571-unit thigh;
            # its distinct pose needs a separate shape study.
            if Path(rig["sourcePath"]).stem == "em1029" and joint == 3:
                print(f"{selector} joint {joint}: skipped nonstandard thigh")
                continue
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2],
                                                              options.strength,
                                                              joint in (3, 6)),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
