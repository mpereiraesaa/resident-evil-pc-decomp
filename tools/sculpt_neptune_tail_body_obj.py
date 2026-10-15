"""Trial: flatten the repeated bulges of Neptune's tail-body sections."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


BODY_JOINTS = (8, 9, 10, 11)


def make_displacement(positions, strength):
    xmin = min(p[0] for p in positions)
    xmax = max(p[0] for p in positions)
    ymin = min(p[1] for p in positions)
    ymax = max(p[1] for p in positions)
    if not 400.0 < xmax - xmin < 800.0:
        raise ValueError("not a Neptune tail-body section")
    centre_y = (ymin + ymax) * 0.5
    middle_x = (xmin + xmax) * 0.5
    width = (xmax - xmin) * 0.23

    def displace(position):
        x, y, z = position
        scale = -0.12 * strength * bump(x, middle_x, width)
        return (0.0, (y - centre_y) * scale, z * scale)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["15", "j:15"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em100b", "em110b"}:
            raise ValueError(f"selector {selector} is not Neptune")
        for joint in BODY_JOINTS:
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=make_displacement(metadata[2], options.strength),
                                   normal_update=moved_normal)
            print(selector, joint, destination, triangles)


if __name__ == "__main__":
    main()
