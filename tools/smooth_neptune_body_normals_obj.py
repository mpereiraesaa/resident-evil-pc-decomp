"""Trial: reduce axial normal seams on Neptune's rigid body segments."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import normalise, parse_obj, smoothstep, write_mesh


BODY_JOINTS = (2, 5, 7, 8, 9, 10, 11)


def make_normal_update(positions, strength):
    xmin = min(position[0] for position in positions)
    xmax = max(position[0] for position in positions)

    def update(position, normal, unused):
        x = position[0]
        edge_distance = min(x - xmin, xmax - x)
        edge = 1.0 - smoothstep((edge_distance - 15.0) / 190.0)
        amount = strength * edge
        return normalise(((1.0 - 0.78 * amount) * normal[0],
                          normal[1], normal[2]))

    return update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["15", "j:15"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.0:
        raise ValueError("strength must be in (0, 1]")
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
                                   deformer=lambda _: (0.0, 0.0, 0.0),
                                   normal_update=make_normal_update(metadata[2], options.strength))
            print(selector, joint, destination, triangles)


if __name__ == "__main__":
    main()
