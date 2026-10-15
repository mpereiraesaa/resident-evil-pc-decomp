"""Taper Neptune's central body segment without moving its articulated ends."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import bump, parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    xmin, xmax = min(p[0] for p in positions), max(p[0] for p in positions)
    ymin, ymax = min(p[1] for p in positions), max(p[1] for p in positions)
    if not 900 < xmax - xmin < 930 or not 1200 < ymax - ymin < 1300:
        raise ValueError("not Neptune's central body segment")
    center_x = (xmin + xmax) * 0.5
    center_y = (ymin + ymax) * 0.5

    def displace(position):
        x, y, z = position
        scale = -0.10 * strength * bump(x, center_x, (xmax - xmin) * 0.22)
        return (0.0, (y - center_y) * scale, z * scale)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["15", "j:15"])
    options = parser.parse_args()
    if not 0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em100b", "em110b"}:
            raise ValueError(f"selector {selector} is not Neptune")
        source = options.refined_dir / rig["joints"][7]["obj"]
        metadata = parse_obj(source)
        output = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(output, *metadata, 0.1,
                               deformer=make_displacement(metadata[2], options.strength),
                               normal_update=moved_normal)
        print(f"{selector} joint 7 -> {output} ({triangles} triangles)")


if __name__ == "__main__":
    main()
