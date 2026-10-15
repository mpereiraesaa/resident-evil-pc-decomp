"""Give the spider-web's six flat panels a shallow, anchored depth curve."""

import argparse
import json
from pathlib import Path

from sculpt_human_shoulders_obj import parse_obj, write_mesh
from sculpt_tyrant_face_obj import moved_normal


def make_displacement(positions, strength):
    if max(abs(p[0]) for p in positions) > 0.01:
        raise ValueError("not the expected flat spider-web panel")
    ymax = max(abs(p[1]) for p in positions)
    zmax = max(abs(p[2]) for p in positions)
    if not 980 <= ymax <= 1000 or not 1790 <= zmax <= 1810:
        raise ValueError("web panel bounds changed")

    def displace(position):
        _, y, z = position
        yedge = max(0.0, 1.0 - (y / ymax) ** 2)
        zedge = max(0.0, 1.0 - (z / zmax) ** 2)
        return (120.0 * strength * yedge * yedge * zedge * zedge, 0.0, 0.0)

    return displace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("selectors", nargs="*", default=["23", "j:23"])
    options = parser.parse_args()
    if not 0.0 < options.strength <= 1.5:
        raise ValueError("strength must be in (0, 1.5]")
    for selector in options.selectors:
        rig = json.loads((options.refined_dir /
                          f"{selector.replace(':', '_')}_rig.json").read_text())
        if Path(rig["sourcePath"]).stem not in {"em1013", "em1113"}:
            raise ValueError(f"selector {selector} is not the spider web")
        for joint in range(len(rig["joints"])):
            source = options.refined_dir / rig["joints"][joint]["obj"]
            metadata = parse_obj(source)
            displacement = make_displacement(metadata[2], options.strength)
            destination = options.out / source.name.replace("_refined.obj", ".r1mesh")
            triangles = write_mesh(destination, *metadata, 0.1,
                                   deformer=displacement, normal_update=moved_normal)
            print(f"{selector} joint {joint} -> {destination} ({triangles} triangles)")


if __name__ == "__main__":
    main()
