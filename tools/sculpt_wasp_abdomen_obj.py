"""Round the wasp's striped abdomen (joint 9) without moving its stinger tip."""

import argparse
import json
import math
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    if min(p[0] for p in positions) > -340 or max(abs(p[2]) for p in positions) < 70:
        raise ValueError("not the expected wasp abdomen")

    def displace(position):
        x, y, z = position
        t = max(0.0, min(1.0, (x + 345.0) / 358.0))
        middle = math.sin(math.pi * t) ** 2
        amount = 0.17 * strength * middle
        return (0.0, amount * (y + 10.0), amount * z)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["11", "j:11"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1007", "em1107"}:
            raise ValueError(f"selector {selector} is not a wasp")
        source = options.refined_dir / rig["joints"][9]["obj"]
        metadata = parse_obj(source)
        displacement = make_displacement(metadata[2], options.strength)
        destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
        triangles = write_mesh(destination, *metadata, 0.1,
                               deformer=displacement, normal_update=moved_normal)
        print(f"{selector} {rig['sourcePath']} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
