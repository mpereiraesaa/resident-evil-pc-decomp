"""Round distal toes in the separate rigid human foot joints.

The ankle and heel are untouched. The distal five columns are narrowed around
their approximate centerlines, preserving the foot joint animation and UVs.
"""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, smoothstep, write_mesh
from sculpt_tyrant_face_obj import moved_normal


HUMAN_STEMS = {"char10", "char11", "char12", "char13"} | {
    f"em{index:04x}" for index in range(0x1020, 0x102f)
} | {"em1030", "em1031", "em1032", "em1033"}


def make_displacement(positions, strength):
    xmax = max(p[0] for p in positions)
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    zmax = max(abs(p[2]) for p in positions)
    if not 330 <= xmax <= 450 or not 150 <= ymax - ymin <= 270 or not 75 <= zmax <= 190:
        raise ValueError(f"not an expected human foot: {xmax}, {ymin}, {ymax}, {zmax}")
    toe_y = ymin + 0.30 * (ymax - ymin)
    centers = tuple(zmax * 0.68 * i / 2.0 for i in (-2, -1, 0, 1, 2))

    def displace(position):
        x, y, z = position
        distal = smoothstep((x / xmax - 0.68) / 0.32)
        distal *= distal
        toe_z = min(centers, key=lambda centre: abs(z - centre))
        return (7.0 * strength * distal,
                -0.14 * strength * distal * (y - toe_y),
                -0.13 * strength * distal * (z - toe_z))

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
        for joint in (5, 8):  # left and right feet, confirmed by the rig tree
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
